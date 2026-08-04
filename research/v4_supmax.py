"""V4-SUPMAX: exprimir la superficie anisotropa hasta el fondo.

Bloques (python v4_supmax.py <bloque> [k]):
  fino   aniso x k x p en IDW s10, BUDA, k=60
  s3     subsample 3 para las mejores configs
  stab   3 subconjuntos DISJUNTOS de 100 pozos para las top configs (media+-std)
  form   combinacion multi-formacion sobre la mejor config
  trend  C_well con tendencia amortiguada en el prefijo
  interp krige / rbf / linridge en la metrica anisotropa
  blend  barrido de w con pf_ancc cacheado (k=60)
  final  confirmacion k=150 (cv.evaluate) + blend k=150

Reutiliza sur02_lib.Cloud (cache de vecinos LOWO en sur_nb_cache) y
sur03_pred.priors/combine. NO escribe en ficheros de otros agentes: el pf de
pozos nuevos va a v4_pfcache.npz.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, HERE)

import cv as CV                      # noqa: E402
import sur02_lib as L                # noqa: E402
import sur05_blend as B              # noqa: E402
from sur03_pred import priors, combine  # noqa: E402

PFC_MIO = os.path.join(HERE, "v4_pfcache.npz")
NBC_MIO = os.path.join(HERE, "v4_nb_cache")
CAL_TAIL = 500


class CloudNC(L.Cloud):
    """Cloud sin cache de vecinos en disco (para barridos one-shot de theta/aniso
    que inflarian sur_nb_cache; el coste extra es ~0.15 s/pozo)."""

    def neighbors(self, wid, X, Y, kmax=L.KMAX, use_cache=False):
        return super().neighbors(wid, X, Y, kmax, use_cache=False)


def cloud_for(a, dth, subsample=10, cache=True):
    cls = L.Cloud if cache else CloudNC
    return cls(subsample=subsample, aniso=a, theta=L.THETA + np.radians(dth))


def neighbors_local(cloud, wid, X, Y, aniso, kmax=64, use_cache=True):
    """Vecinos LOWO con la metrica anisotropa ROTADA A LA DIRECCION DEL PROPIO
    POZO (extremos de su trayectoria XY, disponible en test). Cache propio."""
    import hashlib
    theta = float(np.arctan2(Y[-1] - Y[0], X[-1] - X[0]))
    h = hashlib.md5(f"loc_s{cloud.subsample}_a{aniso:g}_{wid}_{kmax}".encode()
                    ).hexdigest()[:16]
    f = os.path.join(NBC_MIO, f"{h}.npz")
    if use_cache and os.path.exists(f):
        z = np.load(f)
        return z["d"].astype(np.float64), z["i"]
    c, s_ = np.cos(theta), np.sin(theta)
    R = np.array([[c, s_], [-s_, c]])
    sc = np.array([aniso, 1.0])
    sub = np.where(cloud.wi != cloud.idx[wid])[0]
    from scipy.spatial import cKDTree
    tree = cKDTree((cloud.xy_raw[sub] @ R.T) * sc)
    q = (np.column_stack([X, Y]) @ R.T) * sc
    d, i = tree.query(q, k=kmax, workers=-1)
    gi = sub[i].astype(np.int32)
    if use_cache:
        os.makedirs(NBC_MIO, exist_ok=True)
        np.savez(f, d=d.astype(np.float32), i=gi)
    return d, gi


def make_surf_local(cloud, aniso, k=24, p=2.0, mode="buda", cal_tail=CAL_TAIL, **ckw):
    """Como make_surf pero con anisotropia orientada al pozo evaluado."""
    def fn(df_h, cut, wid):
        X, Y, Z = df_h.X.values, df_h.Y.values, df_h.Z.values
        d, gi = neighbors_local(cloud, wid, X, Y, aniso)
        S = L.idw(cloud, d, gi, k, p)
        base = S - Z[:, None]
        lo = max(0, cut - cal_tail)
        C = np.nanmedian(df_h.TVT_input.values[lo:cut, None] - base[lo:cut], axis=0)
        bad = ~np.isfinite(C)
        if bad.any():
            C[bad] = np.nanmedian(C[~bad]) if (~bad).any() else 0.0
        P = base + C
        out = combine(P, df_h.TVT_input.values, cut, mode, **ckw)
        return np.where(np.isfinite(out), out, P[cut:, 5])
    return fn


# ───────────────────────────── runner ligero ─────────────────────────────────
class Run:
    """Carga los pozos una vez; evalua fn(df_h, cut, wid) -> pred post-PS."""

    def __init__(self, ids):
        self.ids, self.W = [], {}
        for w in ids:
            df, tw, cut = CV.load_well(w)
            if cut < 20 or cut >= len(df):
                continue
            self.ids.append(w)
            self.W[w] = (df[CV.TEST_COLS].copy(), cut, df.TVT.values[cut:])

    def eval(self, fn, keep=False):
        t0 = time.time()
        sse, n, pred = 0.0, 0, {}
        for w in self.ids:
            df_h, cut, tru = self.W[w]
            p = np.asarray(fn(df_h, cut, w), float)
            assert len(p) == len(tru) and np.isfinite(p).all(), w
            sse += float(((tru - p) ** 2).sum())
            n += len(tru)
            if keep:
                pred[w] = p
        rmse = float(np.sqrt(sse / n))
        return (rmse, time.time() - t0, pred) if keep else (rmse, time.time() - t0)


def subsets_disjuntos(n_sets=3, size=100, seed=7):
    """Subconjuntos disjuntos de pozos, ajenos a la seleccion sistematica k=60."""
    ids = np.array(CV.well_ids())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    return [list(perm[i * size:(i + 1) * size]) for i in range(n_sets)]


# ─────────────────────────── constructores ───────────────────────────────────
def make_surf(cloud, k=24, p=2.0, mode="buda", cal_tail=CAL_TAIL, **ckw):
    it = lambda cc, d, gi, X, Y: L.idw(cc, d, gi, k, p)      # noqa: E731

    def fn(df_h, cut, wid):
        P, _ = priors(cloud, df_h, wid, cut, it, cal_tail)
        out = combine(P, df_h.TVT_input.values, cut, mode, **ckw)
        return np.where(np.isfinite(out), out, P[cut:, 5])
    return fn


def make_surf_interp(cloud, it, mode="buda", cal_tail=CAL_TAIL, **ckw):
    """it(cloud, d, gi, X, Y) -> (n,6); para krige/rbf/ridge que necesitan X,Y."""
    def fn(df_h, cut, wid):
        P, _ = priors(cloud, df_h, wid, cut, it, cal_tail)
        out = combine(P, df_h.TVT_input.values, cut, mode, **ckw)
        return np.where(np.isfinite(out), out, P[cut:, 5])
    return fn


def combine_full(P, tvt_in, cut, mode="buda", score_win=2000, topn=None):
    """Como sur03_pred.combine pero devuelve la serie COMPLETA (prefijo+post)."""
    if mode == "buda":
        return P[:, 5]
    good = np.isfinite(P[cut:]).all(0)
    if not good.any():
        good = np.ones(6, bool)
    if mode == "mean":
        return np.nanmean(P[:, good], 1)
    if mode == "median":
        return np.nanmedian(P[:, good], 1)
    lo = max(0, cut - score_win)
    e = tvt_in[lo:cut, None] - P[lo:cut]
    mse = np.nanmean(e ** 2, axis=0)
    mse = np.where(np.isfinite(mse) & good, mse, np.inf)
    if mode == "best":
        return P[:, int(np.argmin(mse))]
    if mode == "topn":
        order = np.argsort(mse)[:topn]
        return np.nanmean(P[:, order], 1)
    if mode == "winv":
        w = 1.0 / np.maximum(mse, 1e-6)
    elif mode == "softmax":
        w = np.exp(-mse / max(np.min(mse), 1e-6))
    else:
        raise ValueError(mode)
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    return np.nansum(P * w, 1)


def make_surf_trend(cloud, k=24, p=2.0, mode="buda", Ldamp=600.0, smax=0.006,
                    slope_win=2000, damp="lin", cal_tail=CAL_TAIL, **ckw):
    """C_well + tendencia del residuo extrapolada AMORTIGUADA.

    e_i = TVT_input_i - prior_i en el prefijo; s = pendiente Theil-Sen de e vs
    distancia horizontal en la ventana slope_win (filas), acotada a |s|<=smax.
      damp=lin: corr(d) = e0 + s*d*exp(-d/L)      (vuelve a e0 a lo lejos)
      damp=sat: corr(d) = e0 + s*L*(1-exp(-d/L))  (satura en e0+s*L)
    """
    from scipy.stats import theilslopes
    it = lambda cc, d, gi, X, Y: L.idw(cc, d, gi, k, p)      # noqa: E731

    def fn(df_h, cut, wid):
        P, _ = priors(cloud, df_h, wid, cut, it, cal_tail)
        full = combine_full(P, df_h.TVT_input.values, cut, mode, **ckw)
        full = np.where(np.isfinite(full), full, P[:, 5])
        X, Y = df_h.X.values, df_h.Y.values
        h = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
        e = df_h.TVT_input.values[:cut] - full[:cut]
        lo_c = max(0, cut - cal_tail)
        e0 = float(np.median(e[lo_c:cut]))
        lo_s = max(0, cut - slope_win)
        hh, ee = h[lo_s:cut], e[lo_s:cut]
        step = max(1, len(hh) // 150)                        # Theil-Sen barato
        hh, ee = hh[::step], ee[::step]
        s = 0.0
        if len(hh) >= 20 and hh[-1] - hh[0] > 100:
            s = float(np.clip(theilslopes(ee, hh)[0], -smax, smax))
        d = h[cut:] - h[cut - 1]
        if damp == "lin":
            corr = e0 + s * d * np.exp(-d / Ldamp)
        else:
            corr = e0 + s * Ldamp * (1.0 - np.exp(-d / Ldamp))
        return full[cut:] + corr
    return fn


# ───────────────────────────── pf cache propio ───────────────────────────────
def pf_para(ids):
    """pf_ancc por pozo: lee sur05_pfcache.npz (solo lectura) y completa lo que
    falte en v4_pfcache.npz (seed fija por pozo para reproducibilidad)."""
    d = {}
    if os.path.exists(B.PFC):
        d.update({k: v for k, v in np.load(B.PFC).items()})
    if os.path.exists(PFC_MIO):
        d.update({k: v for k, v in np.load(PFC_MIO).items()})
    miss = [w for w in ids if w not in d]
    if miss:
        from pf_publico import predict_pf_ancc
        nuevo = {k: v for k, v in np.load(PFC_MIO).items()} if os.path.exists(PFC_MIO) else {}
        for i, w in enumerate(miss):
            np.random.seed(abs(hash(w)) % (2 ** 31))
            df, tw, cut = CV.load_well(w)
            p = np.asarray(predict_pf_ancc(df[CV.TEST_COLS].copy(), tw), float)
            nuevo[w] = d[w] = (p[cut:] if len(p) == len(df) else p).astype(np.float32)
            if i % 20 == 0:
                print(f"  pf {i}/{len(miss)}", flush=True)
        np.savez(PFC_MIO, **nuevo)
    return {w: d[w].astype(np.float64) for w in ids}


def blend_rmse(pred, pf, tru, w):
    b = {x: w * pred[x] + (1 - w) * pf[x] for x in pred}
    return B.pooled(b, tru)


# ─────────────────────────────── bloques ─────────────────────────────────────
if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "fino"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rows = []

    def log(name, rmse, dt, extra=""):
        rows.append({"variante": name, "rmse": rmse, "s_pozo": dt})
        print(f"{name:52s} rmse={rmse:7.3f}  {dt:6.1f}s {extra}", flush=True)

    if block == "fino":
        R = Run(B.wells(K))
        print(f"{len(R.ids)} pozos", flush=True)
        for a in (8, 12, 16, 24, 32):
            c = L.Cloud(subsample=10, aniso=a)
            for k in (16, 24, 32, 48):
                for p in (1.0, 1.5, 2.0, 3.0):
                    r, dt = R.eval(make_surf(c, k=k, p=p))
                    log(f"idw s10 a{a} k{k} p{p:g}", r, dt)
            del c

    elif block == "s3":
        # cfgs: a,k,p,dth
        R = Run(B.wells(K))
        cfgs = [tuple(float(x) for x in s.split(",")) for s in sys.argv[3:]] or \
               [(16, 24, 1.5, 11.0), (48, 24, 2.0, 11.0)]
        for a, k, p, dth in cfgs:
            c = cloud_for(a, dth, subsample=3, cache=False)
            r, dt = R.eval(make_surf(c, k=int(k), p=p))
            log(f"idw s3 a{a:g} k{k:g} p{p:g} dth{dth:+g}", r, dt)
            del c

    elif block == "stab":
        # cfgs: a,k,p,dth — 3 subconjuntos DISJUNTOS de 100 pozos, media+-std
        cfgs = [tuple(float(x) for x in s.split(",")) for s in sys.argv[3:]]
        subs = subsets_disjuntos()
        runners = [Run(s) for s in subs]
        for cfg in cfgs:
            a, k, p, dth = cfg[:4]
            ct = int(cfg[4]) if len(cfg) > 4 else CAL_TAIL
            c = cloud_for(a, dth, cache=False)
            rs = []
            for j, R in enumerate(runners):
                r, dt = R.eval(make_surf(c, k=int(k), p=p, cal_tail=ct))
                rs.append(r)
                log(f"idw s10 a{a:g} k{k:g} p{p:g} dth{dth:+g} ct{ct} sub{j}", r, dt)
            print(f"  => a{a:g} k{k:g} p{p:g} dth{dth:+g} ct{ct}: media={np.mean(rs):.3f}"
                  f" +- {np.std(rs):.3f}", flush=True)
            rows.append({"variante": f"a{a:g}k{k:g}p{p:g}d{dth:+g}ct{ct}_media",
                         "rmse": float(np.mean(rs)), "s_pozo": float(np.std(rs))})
            del c

    elif block == "theta2":
        # rejilla fina dth x aniso alrededor del optimo del barrido grueso
        R = Run(B.wells(K))
        for a in (16, 24, 32, 48, 64):
            for dth in (9.0, 10.0, 11.0, 12.0, 13.0):
                c = cloud_for(a, dth, cache=False)
                for kk, p in ((24, 1.5), (24, 2.0)):
                    r, dt = R.eval(make_surf(c, k=kk, p=p))
                    log(f"idw s10 a{a} k{kk} p{p:g} dth{dth:+g}", r, dt)
                del c

    elif block == "form":
        a, k, p, dth = (float(x) for x in (sys.argv[3:] or ["16,24,1.5,11"])[0].split(","))
        R = Run(B.wells(K))
        c = cloud_for(a, dth)
        for mode in ("buda", "mean", "median", "winv", "winv_sd", "softmax", "best"):
            r, dt = R.eval(make_surf(c, k=int(k), p=p, mode=mode))
            log(f"a{a:g}k{k:g}p{p:g}d{dth:+g} {mode}", r, dt)
        for tn in (2, 3, 4):
            r, dt = R.eval(make_surf(c, k=int(k), p=p, mode="topn", topn=tn))
            log(f"a{a:g}k{k:g}p{p:g}d{dth:+g} top{tn}", r, dt)
        for sw in (500, 1000, 4000):
            r, dt = R.eval(make_surf(c, k=int(k), p=p, mode="winv", score_win=sw))
            log(f"a{a:g}k{k:g}p{p:g}d{dth:+g} winv sw{sw}", r, dt)

    elif block == "trend":
        a, k, p, dth = (float(x) for x in (sys.argv[3:] or ["16,24,1.5,11"])[0].split(","))
        mode = sys.argv[4] if len(sys.argv) > 4 else "buda"
        R = Run(B.wells(K))
        c = cloud_for(a, dth)
        r, dt = R.eval(make_surf(c, k=int(k), p=p, mode=mode))
        log(f"ref {mode} sin tendencia", r, dt)
        for damp in ("lin", "sat"):
            for Ld in (300, 600, 1200, 2500):
                for smax in (0.003, 0.006, 0.012):
                    r, dt = R.eval(make_surf_trend(c, k=int(k), p=p, mode=mode,
                                                   Ldamp=Ld, smax=smax, damp=damp))
                    log(f"trend {damp} L{Ld} smax{smax:g}", r, dt)

    elif block == "theta":
        # OJO: model.py usa theta=-53.08 deg (media de direcciones) y sur02_lib
        # THETA=119.55 deg (PCA) == -60.45 mod 180. Distan 7.4 deg y a aniso
        # altas la diferencia es grande: barrer dth alrededor de sur02.
        a, k, p = (float(x) for x in (sys.argv[3:] or ["16,24,2"])[0].split(","))
        R = Run(B.wells(K))
        for dth in (-10.0, -5.0, 0.0, 3.5, 7.4, 11.0, 15.0, 20.0):
            c = L.Cloud(subsample=10, aniso=a, theta=L.THETA + np.radians(dth))
            r, dt = R.eval(make_surf(c, k=int(k), p=p))
            log(f"a{a:g}k{k:g}p{p:g} dth{dth:+g}", r, dt)
            del c

    elif block == "fino2":
        # re-barrido aniso x k x p con theta corregido (dth en grados)
        dth = float(sys.argv[3]) if len(sys.argv) > 3 else 7.4
        R = Run(B.wells(K))
        for a in (8, 16, 24, 32, 48):
            c = L.Cloud(subsample=10, aniso=a, theta=L.THETA + np.radians(dth))
            for k in (16, 24, 32):
                for p in (1.5, 2.0, 3.0):
                    r, dt = R.eval(make_surf(c, k=k, p=p))
                    log(f"idw s10 a{a} k{k} p{p:g} dth{dth:+g}", r, dt)
            del c

    elif block == "cal":
        a, k, p, dth = (float(x) for x in (sys.argv[3:] or ["16,24,1.5,11"])[0].split(","))
        R = Run(B.wells(K))
        c = cloud_for(a, dth)
        cts = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else \
            [250, 500, 1000, 2000, 100000]
        for ct in cts:
            r, dt = R.eval(make_surf(c, k=int(k), p=p, cal_tail=ct))
            log(f"a{a:g}k{k:g}p{p:g} cal_tail{ct}", r, dt)

    elif block == "local":
        R = Run(B.wells(K))
        c = L.Cloud(subsample=10, aniso=1.0)     # solo se usa xy_raw/s/wi
        for a in (8, 16, 24, 32):
            for k in (16, 24, 32):
                r, dt = R.eval(make_surf_local(c, a, k=k, p=2.0))
                log(f"local s10 a{a} k{k} p2", r, dt)

    elif block == "interp":
        a = float(sys.argv[3]) if len(sys.argv) > 3 else 16
        dth = float(sys.argv[4]) if len(sys.argv) > 4 else 11.0
        R = Run(B.wells(K))
        c = cloud_for(a, dth)
        for k, ng, rg in [(24, 0.5, 500), (24, 0.5, 1500), (24, 2.0, 1500),
                          (24, 0.5, 4000), (32, 0.5, 1500), (32, 2.0, 4000)]:
            it = lambda cc, d, gi, X, Y, k=k, ng=ng, rg=rg: \
                L.krige(cc, d, gi, X, Y, k, ng, 1000.0, rg)   # noqa: E731
            r, dt = R.eval(make_surf_interp(c, it))
            log(f"krige a{a:g} k{k} ng{ng:g} r{rg:g}", r, dt)
        for k, kern, sm, deg in [(24, "linear", 1.0, None), (24, "linear", 30.0, None),
                                 (24, "thin_plate_spline", 30.0, 1),
                                 (32, "linear", 30.0, None),
                                 (32, "thin_plate_spline", 100.0, 1)]:
            it = lambda cc, d, gi, X, Y, k=k, kern=kern, sm=sm, deg=deg: \
                L.rbf_local(cc, d, gi, X, Y, k, kern, sm, deg)  # noqa: E731
            r, dt = R.eval(make_surf_interp(c, it))
            log(f"rbf a{a:g} k{k} {kern} sm{sm:g}", r, dt)
        for k, lam in [(32, 1.0), (32, 10.0), (32, 100.0)]:
            it = lambda cc, d, gi, X, Y, k=k, lam=lam: \
                L.linridge(cc, d, gi, X, Y, k, lam, clip=20.0)  # noqa: E731
            r, dt = R.eval(make_surf_interp(c, it))
            log(f"ridge a{a:g} k{k} lam{lam:g} clip20", r, dt)

    elif block == "blend":
        # variantes finalistas (a,k,p,dth[,mode]) — mezcla con pf_ancc cacheado
        R = Run(B.wells(K))
        tru = {w: R.W[w][2] for w in R.ids}
        pf = pf_para(R.ids)
        print(f"pf_ancc solo = {B.pooled(pf, tru):.3f}", flush=True)
        variantes = {}
        for spec in sys.argv[3:]:
            parts = spec.split(",")
            a, k, p, dth = (float(x) for x in parts[:4])
            mode = parts[4] if len(parts) > 4 else "buda"
            ct = int(parts[5]) if len(parts) > 5 else CAL_TAIL
            c = cloud_for(a, dth)
            r, dt, pred = R.eval(make_surf(c, k=int(k), p=p, mode=mode,
                                           cal_tail=ct), keep=True)
            variantes[f"a{a:g}k{k:g}p{p:g}d{dth:+g}{mode}ct{ct}"] = pred
            print(f"{spec}: sup={r:.3f}", flush=True)
            del c
        for name, pred in variantes.items():
            for w in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
                rb = blend_rmse(pred, pf, tru, w)
                log(f"blend {name} w{w:.2f}", rb, 0.0)

    elif block == "final":
        # confirmacion k=150: superficie sola + blend con pf_ancc
        spec = (sys.argv[3:] or ["16,24,1.5,11"])[0]
        parts = spec.split(",")
        a, k, p, dth = (float(x) for x in parts[:4])
        mode = parts[4] if len(parts) > 4 else "buda"
        ct = int(parts[5]) if len(parts) > 5 else CAL_TAIL
        R = Run(B.wells(K))
        tru = {w: R.W[w][2] for w in R.ids}
        c = cloud_for(a, dth)
        r, dt, pred = R.eval(make_surf(c, k=int(k), p=p, mode=mode, cal_tail=ct),
                             keep=True)
        log(f"FINAL sup a{a:g}k{k:g}p{p:g}d{dth:+g} {mode} ct{ct} k={K}", r, dt)
        pf = pf_para(R.ids)
        log(f"FINAL pf_ancc k={K}", B.pooled(pf, tru), 0.0)
        for w in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
            log(f"FINAL blend w{w:.2f} k={K}", blend_rmse(pred, pf, tru, w), 0.0)

    print()
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(HERE, f"v4_supmax_{block}_k{K}.csv"),
                                  index=False)
