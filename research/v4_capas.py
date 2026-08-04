"""V4-capas: capas finales sobre el mejor blend actual
    base = 0.25 * superficie(BUDA, aniso=16, IDW k=24, LOWO) + 0.75 * pf_ancc

Bloques CLI (python research/v4_capas.py <bloque> [k]):
  sig        firmas geometricas (X,Y muestreadas cada 100 ft de MD) de los 773
             pozos de train -> v4_sig.npz + censo de casi-duplicados en train
  selfcheck  control positivo del contact override con las 3 copias de test/
  cache K    candidatos S,P,G + backtest frac 0.65 + override LOWO
             -> v4_cache_k{K}.npz (+ _meta.csv)
  capas K    cada capa POR SEPARADO sobre el cache + mejores combinaciones
             -> v4_capas_k{K}.csv

Reglas: LOWO estricto (la superficie excluye el pozo; el override excluye el
propio id). El override lleva doble guard: matching geometrico estricto Y
reproduccion del prefijo visible con RMSE<1 ft en >=50 filas => nunca empeora.

RESULTADO FINAL (2026-07-31, sobre cache propio; blend ref con pf 1 realizacion):
  k=60 : ref 12.546 -> stack 11.974 (-0.573)   k=150: ref 11.797 -> stack 11.325
  (-0.472, proxy 12.013 -> 11.474). Capas que entran: irls pre deg2 mix=0.6 +
  rampa tau=1200 + contact override. Descartado: shrink (empeora +1..+9), savgol
  (0.000), seleccion por pozo por backtest (corr bt~real +0.03..+0.23, empeora).
  Controles del override: selfcheck 3 copias test/ -> RMSE 0.005-0.007 ft con
  cobertura 100%; guardcheck 18 twins de train -> 14 rechazados por prermse>=1,
  4 admitidos y en LOS 4 el override MEJORA (1.45 vs 9.1-13.2; 5.70 vs 8.1-8.5).

INTEGRACION (kernel): from v4_capas import Sigs, aplicar_capas
  sigs = Sigs()  # tras build_sig() sobre train/ (7 s, una vez)
  pred = aplicar_capas(df_h, cut, blend_post, sigs=sigs, exclude=None)
  con blend_post = 0.25*S + 0.75*P post-PS. Paridad con el cache: 0.0006 ft.
  Coste re-run: 7 s fijos + ~6 ms/pozo (match 5 ms + irls/rampa 1 ms).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import cv as CV                # noqa: E402
import sur02_lib as L          # noqa: E402
import pf_publico as PF        # noqa: E402

RAW = ROOT / "data/raw"
ANISO = 16.0
K_IDW = 24
CAL_TAIL = 500
W_SUP = 0.25                   # blend de referencia 0.25*S + 0.75*P
FRAC_BT = 0.65                 # corte del backtest leak-free
GRID = 100.0                   # paso MD de las firmas geometricas
SIGF = HERE / "v4_sig.npz"
PFC = HERE / "v4_pfcache.npz"
DIP_WIN = 700

# guard del override
OV_COVER = 0.50                # el match debe cubrir >=50% del rango MD del pozo
OV_DIST = 15.0                 # media |dX|+|dY| en el rango comun (ft)
OV_PRERMSE = 1.0               # reproducir el prefijo visible con RMSE < 1 ft
OV_MINPRE = 50                 # en >= 50 filas


# ────────────────────────── candidatos base ──────────────────────────────────
_CLOUD = None


def _cloud():
    global _CLOUD
    if _CLOUD is None:
        _CLOUD = L.Cloud(subsample=10, aniso=ANISO)
    return _CLOUD


def _cut(df_h):
    m = df_h.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df_h)


def surf_base(df_h, wid):
    """S_BUDA - Z en TODA la trayectoria (sin offset), LOWO, + dist real vecino1."""
    c = _cloud()
    d, gi, dr = c.neighbors(wid, df_h.X.values, df_h.Y.values)
    S = L.idw(c, d, gi, K_IDW)[:, 5]
    return S - df_h.Z.values, dr


def cal_C(base, tvt_in, cut, tail=CAL_TAIL):
    lo = max(0, cut - tail)
    return float(np.nanmedian(tvt_in[lo:cut] - base[lo:cut]))


def hdist(df):
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=DIP_WIN):
    """Recta de dip anclada en el PS (misma forma que research/afinar_dip.py)."""
    z = df_h.Z.values
    tvt_in = df_h.TVT_input.values
    h = hd - hd[cut - 1]
    flat = tvt_in[cut - 1] + (z[cut - 1] - z)
    lo = max(0, cut - win)
    base = tvt_in[lo] + (z[lo] - z[lo:cut])
    r = tvt_in[lo:cut] - base
    hh = h[lo:cut]
    slope = 0.0
    if len(hh) > 10 and hh[-1] != hh[0]:
        dr, dh = r - r[-1], hh - hh[-1]
        ss = float(np.sum(dh * dh))
        slope = float(np.sum(dr * dh) / ss) if ss > 0 else 0.0
    return flat + slope * h


def pf_post(ids, verbose=True):
    """{wid: pf_ancc post-PS}. Reusa sur05_pfcache (solo lectura) y cachea lo
    nuevo en v4_pfcache.npz (fichero propio)."""
    old = {}
    f_old = HERE / "sur05_pfcache.npz"
    if f_old.exists():
        old = dict(np.load(f_old))
    mine = dict(np.load(PFC)) if PFC.exists() else {}
    out, changed = {}, False
    for i, w in enumerate(ids):
        if w in mine:
            out[w] = mine[w]
        elif w in old:
            out[w] = old[w]
        else:
            df, tw, cut = CV.load_well(w)
            df_h = df[CV.TEST_COLS].copy()
            p = np.asarray(PF.predict_pf_ancc(df_h, tw), float)
            out[w] = (p[cut:] if len(p) == len(df) else p).astype(np.float32)
            mine[w] = out[w]
            changed = True
            if verbose and i % 20 == 0:
                print(f"  pf {i}/{len(ids)}", flush=True)
    if changed:
        np.savez(PFC, **mine)
    return {w: out[w].astype(np.float64) for w in ids}


# ───────────────────────── contact override ──────────────────────────────────
def build_sig():
    ids = sorted(p.name.split("__")[0]
                 for p in (RAW / "train").glob("*__horizontal_well.csv"))
    out = {"ids": np.array(ids)}
    t0 = time.time()
    for i, w in enumerate(ids):
        d = pd.read_csv(RAW / "train" / f"{w}__horizontal_well.csv",
                        usecols=["MD", "X", "Y"])
        md = d.MD.values
        g0 = np.ceil(md[0] / GRID) * GRID
        g = np.arange(g0, md[-1] + 1e-9, GRID)
        out[f"x{i}"] = np.interp(g, md, d.X.values).astype(np.float32)
        out[f"y{i}"] = np.interp(g, md, d.Y.values).astype(np.float32)
        out[f"m{i}"] = np.array([g0, float(len(g))])
        if (i + 1) % 100 == 0:
            print(f"  firmas {i+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(SIGF, **out)
    print(f"guardado {SIGF} ({len(ids)} pozos, {time.time()-t0:.0f}s)")


class Sigs:
    def __init__(self):
        z = np.load(SIGF)
        self.ids = [str(x) for x in z["ids"]]
        self.g0 = np.array([z[f"m{i}"][0] for i in range(len(self.ids))])
        self.n = np.array([int(z[f"m{i}"][1]) for i in range(len(self.ids))])
        self.x = [z[f"x{i}"] for i in range(len(self.ids))]
        self.y = [z[f"y{i}"] for i in range(len(self.ids))]
        self.g1 = self.g0 + (self.n - 1) * GRID


def match_geom(md, x, y, S, exclude=None, cover_min=OV_COVER, dist_max=OV_DIST):
    """Mejor pozo de train con la misma geometria: (id, dist, cover) o None.
    dist = media de |dX|+|dY| sobre el rango MD comun (rejillas alineadas)."""
    g0 = np.ceil(md[0] / GRID) * GRID
    g = np.arange(g0, md[-1] + 1e-9, GRID)
    if len(g) < 10:
        return None
    xq = np.interp(g, md, x)
    yq = np.interp(g, md, y)
    span = g[-1] - g[0]
    best = None
    for i, w in enumerate(S.ids):
        if w == exclude:
            continue
        lo = max(g0, S.g0[i])
        hi = min(g[-1], S.g1[i])
        if hi - lo < cover_min * span:
            continue
        a = int(round((lo - g0) / GRID))
        b = int(round((lo - S.g0[i]) / GRID))
        m = int(round((hi - lo) / GRID)) + 1
        # descarte rapido por el primer punto comun
        if abs(float(xq[a]) - float(S.x[i][b])) + abs(float(yq[a]) - float(S.y[i][b])) > 20 * dist_max:
            continue
        dist = float(np.mean(np.abs(xq[a:a + m] - S.x[i][b:b + m]) +
                             np.abs(yq[a:a + m] - S.y[i][b:b + m])))
        if dist < dist_max and (best is None or dist < best[1]):
            best = (w, dist, (hi - lo) / span)
    return best


def override_pred(df_h, cut, mid):
    """TVT reconstruida desde la copia de train, INTERPOLADA POR MD.
    Devuelve (pred_full, mask_cobertura, prefix_rmse, n_prefix) o None."""
    tr = pd.read_csv(RAW / "train" / f"{mid}__horizontal_well.csv",
                     usecols=["MD", "BUDA"])
    md = df_h.MD.values
    S = np.interp(md, tr.MD.values, tr.BUDA.values)
    cov = (md >= tr.MD.values[0] - 0.5) & (md <= tr.MD.values[-1] + 0.5)
    base = S - df_h.Z.values
    tvt_in = df_h.TVT_input.values
    pre = np.where(cov[:cut] & np.isfinite(tvt_in[:cut]))[0]
    if len(pre) < OV_MINPRE:
        return None
    C = float(np.median(tvt_in[pre] - base[pre]))
    r = tvt_in[pre] - (base[pre] + C)
    return base + C, cov, float(np.sqrt(np.mean(r ** 2))), len(pre)


def contact_override(df_h, cut, sigs, exclude=None):
    """Capa 1 completa con doble guard. Devuelve (pred_post | None, info dict)."""
    mm = match_geom(df_h.MD.values, df_h.X.values, df_h.Y.values, sigs,
                    exclude=exclude)
    info = {"hit": 0, "id": "", "dist": np.nan, "cover": np.nan, "prermse": np.nan}
    if mm is None:
        return None, info
    info.update(id=mm[0], dist=mm[1], cover=mm[2])
    op = override_pred(df_h, cut, mm[0])
    if op is None:
        return None, info
    pred, cov, prermse, npre = op
    info["prermse"] = prermse
    if prermse >= OV_PRERMSE:
        return None, info
    info["hit"] = 1
    out = np.where(cov, pred, np.nan)[cut:]
    return out, info


# ─────────────────────────── cache de candidatos ─────────────────────────────
def build_cache(k):
    ids = CV._select(CV.well_ids(), k, CV.SEED)
    sigs = Sigs()
    pf = pf_post(ids)
    cols = ("y", "S", "P", "G", "mds", "MDv", "Zv", "nn", "ov")
    store = {n: [] for n in cols}
    btcols = ("bty", "btS", "btP", "btG", "btmds")
    btst = {n: [] for n in btcols}
    pre = {n: [] for n in ("MDp", "Zp", "Tp")}
    lens, btlens, plens, meta = [], [], [], []
    t0 = time.time()
    done = 0
    for wid in ids:
        df, tw, cut = CV.load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[CV.TEST_COLS].copy()
        tvt_in = df_h.TVT_input.values
        base, dr = surf_base(df_h, wid)
        S = base + cal_C(base, tvt_in, cut)
        hd = hdist(df_h)
        G = geom_pred(df_h, cut, hd)
        store["y"].append(df.TVT.values[cut:].astype(np.float32))
        store["S"].append(S[cut:].astype(np.float32))
        store["P"].append(pf[wid].astype(np.float32))
        store["G"].append(G[cut:].astype(np.float32))
        store["mds"].append((df_h.MD.values[cut:] - df_h.MD.values[cut - 1]).astype(np.float32))
        store["MDv"].append(df_h.MD.values[cut:].astype(np.float32))
        store["Zv"].append(df_h.Z.values[cut:].astype(np.float32))
        store["nn"].append(dr[cut:].astype(np.float32))
        lens.append(len(df) - cut)

        # override LOWO (excluye el propio id) + sanity con self permitido
        ovp, info = contact_override(df_h, cut, sigs, exclude=wid)
        store["ov"].append((ovp if ovp is not None
                            else np.full(len(df) - cut, np.nan)).astype(np.float32))
        _, info_self = contact_override(df_h, cut, sigs, exclude=None)
        m = {"well": wid, "cut": cut, "n": len(df), "n_pred": len(df) - cut,
             "tv0": float(tvt_in[cut - 1]), "z0": float(df_h.Z.values[cut - 1]),
             "md0": float(df_h.MD.values[cut - 1]),
             "ov_hit": info["hit"], "ov_id": info["id"], "ov_dist": info["dist"],
             "ov_prermse": info["prermse"],
             "ovself_hit": info_self["hit"], "ovself_id": info_self["id"],
             "ovself_prermse": info_self["prermse"]}

        # prefijo (cola) para la proyeccion IRLS anclada
        lo = max(0, cut - 1500)
        pre["MDp"].append(df_h.MD.values[lo:cut].astype(np.float32))
        pre["Zp"].append(df_h.Z.values[lo:cut].astype(np.float32))
        pre["Tp"].append(tvt_in[lo:cut].astype(np.float32))
        plens.append(cut - lo)

        # backtest leak-free: enmascarar la cola del prefijo
        c2 = int(round(FRAC_BT * cut))
        if c2 >= 60 and cut - c2 >= 30:
            d2 = df_h.iloc[:cut].copy()
            d2.loc[d2.index[c2:], "TVT_input"] = np.nan
            btS = base[:cut] + cal_C(base[:cut], d2.TVT_input.values, c2)
            btP = np.asarray(PF.predict_pf_ancc(d2, tw), float)
            btG = geom_pred(d2, c2, hd[:cut])
            btst["bty"].append(tvt_in[c2:cut].astype(np.float32))
            btst["btS"].append(btS[c2:cut].astype(np.float32))
            btst["btP"].append(btP.astype(np.float32))
            btst["btG"].append(btG[c2:cut].astype(np.float32))
            btst["btmds"].append((df_h.MD.values[c2:cut] - df_h.MD.values[c2 - 1]).astype(np.float32))
            btlens.append(cut - c2)
        else:
            for n in btcols:
                btst[n].append(np.zeros(0, np.float32))
            btlens.append(0)
        meta.append(m)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    out = {n: np.concatenate(v) for n, v in {**store, **btst, **pre}.items()}
    out["lens"] = np.array(lens)
    out["btlens"] = np.array(btlens)
    out["plens"] = np.array(plens)
    outf = HERE / f"v4_cache_k{k}.npz"
    np.savez_compressed(outf, **out)
    pd.DataFrame(meta).to_csv(str(outf).replace(".npz", "_meta.csv"), index=False)
    dt = time.time() - t0
    print(f"guardado {outf}: {len(lens)} pozos, {sum(lens)} puntos, "
          f"{dt:.0f}s ({dt/len(lens):.2f} s/pozo con backtest)")


class C4:
    """Vista comoda del cache: arrays por punto + slices por pozo + metricas."""

    def __init__(self, k):
        z = np.load(HERE / f"v4_cache_k{k}.npz")
        self.meta = pd.read_csv(HERE / f"v4_cache_k{k}_meta.csv")
        self.lens = z["lens"]
        self.NW = len(self.lens)
        self.wid = np.repeat(np.arange(self.NW), self.lens)
        for n in ("y", "S", "P", "G", "mds", "MDv", "Zv", "nn", "ov"):
            setattr(self, n, z[n].astype(np.float64))
        off = np.concatenate([[0], np.cumsum(self.lens)])
        self.sl = [slice(off[i], off[i + 1]) for i in range(self.NW)]
        self.btlens = z["btlens"]
        self.btwid = np.repeat(np.arange(self.NW), self.btlens)
        for n in ("bty", "btS", "btP", "btG", "btmds"):
            setattr(self, n, z[n].astype(np.float64))
        self.plens = z["plens"]
        poff = np.concatenate([[0], np.cumsum(self.plens)])
        self.psl = [slice(poff[i], poff[i + 1]) for i in range(self.NW)]
        for n in ("MDp", "Zp", "Tp"):
            setattr(self, n, z[n].astype(np.float64))
        self.tv0 = self.meta.tv0.values
        self.z0 = self.meta.z0.values

    def rmse(self, p):
        return float(np.sqrt(np.mean((self.y - p) ** 2)))

    def sse_well(self, p):
        return np.bincount(self.wid, (self.y - p) ** 2, self.NW)

    def lb_proxy(self, p):
        s = self.sse_well(p)
        m = self.lens >= CV.LB_PROXY_NPRED
        return float(np.sqrt(s[m].sum() / self.lens[m].sum())) if m.any() else np.nan

    def expand(self, v):
        return np.asarray(v)[self.wid]


# ────────────────────────────── capas ────────────────────────────────────────
def ramp(c, p, tau):
    """Continuidad: ancla al ultimo TVT conocido, decae con exp(-md/tau)."""
    d0 = np.array([c.tv0[i] - p[c.sl[i]][0] for i in range(c.NW)])
    return p + c.expand(d0) * np.exp(-c.mds / tau)


def shrink(c, p, a):
    """Encoge el delta respecto a la capa plana anclada en el PS."""
    flat = c.expand(c.tv0) + (c.expand(c.z0) - c.Zv)
    return flat + a * (p - flat)


def savgol(c, p, win=17, order=3):
    out = p.copy()
    for s in c.sl:
        n = s.stop - s.start
        w = min(win if win % 2 else win + 1, n if n % 2 else n - 1)
        if w > order + 1:
            out[s] = savgol_filter(p[s], w, order)
    return out


def irls_poly(c, p, deg=3, iters=4, cval=4.0, mix=0.75, dom="post"):
    """Proyeccion robusta de U=TVT+Z: polinomio en MD normalizado, pesos Cauchy
    w=1/(1+(r/(c*s))^2). dom='pre' incluye la cola del prefijo (ancla el nivel)."""
    out = p.copy()
    for i in range(c.NW):
        s = c.sl[i]
        u = p[s] + c.Zv[s]
        x = c.MDv[s]
        if dom == "pre":
            ps_ = c.psl[i]
            x = np.concatenate([c.MDp[ps_], x])
            u = np.concatenate([c.Tp[ps_] + c.Zp[ps_], u])
        if len(x) < deg + 5 or x[-1] == x[0]:
            continue
        t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
        V = np.vander(t, deg + 1)
        w = np.ones(len(t))
        for _ in range(iters):
            beta, *_ = np.linalg.lstsq(V * w[:, None], u * w, rcond=None)
            r = u - V @ beta
            sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            w = 1.0 / (1.0 + (r / (cval * sc)) ** 2)
        fit = (V @ beta)[-(s.stop - s.start):] - c.Zv[s]
        out[s] = mix * fit + (1 - mix) * p[s]
    return out


def apply_override(c, p):
    return np.where(np.isfinite(c.ov), c.ov, p)


# ──────────────────── stack final por pozo (integracion) ─────────────────────
# Parametros ganadores: barrido k=60 + arbitraje fino k=150 (bloque 'fino').
CAPAS = dict(deg=2, mix=0.6, tau=1200.0, pre_rows=1500)


def irls_poly_well(md_pre, z_pre, tvt_pre, md_post, z_post, p_post,
                   deg=3, iters=4, cval=4.0, mix=0.5):
    """IRLS dom='pre' para UN pozo (misma matematica que irls_poly)."""
    p_post = np.asarray(p_post, float)
    x = np.concatenate([md_pre, md_post])
    u = np.concatenate([tvt_pre + z_pre, p_post + z_post])
    if len(x) < deg + 5 or x[-1] == x[0]:
        return p_post
    t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
    V = np.vander(t, deg + 1)
    w = np.ones(len(t))
    for _ in range(iters):
        beta, *_ = np.linalg.lstsq(V * w[:, None], u * w, rcond=None)
        r = u - V @ beta
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / (1.0 + (r / (cval * sc)) ** 2)
    fit = (V @ beta)[-len(p_post):] - z_post
    return mix * fit + (1 - mix) * p_post


def aplicar_capas(df_h, cut, pred_post, sigs=None, exclude=None,
                  deg=CAPAS["deg"], mix=CAPAS["mix"], tau=CAPAS["tau"],
                  pre_rows=CAPAS["pre_rows"]):
    """STACK FINAL por pozo: IRLS(pre) + rampa + contact override.

    pred_post: prediccion base post-PS (p.ej. 0.25*S + 0.75*P).
    sigs: instancia Sigs() compartida (None => sin override).
    exclude: id del pozo si esta en train (LOWO); en el re-run oculto, None.
    Devuelve el array post-PS corregido. Coste ~1 ms/pozo + match ~5 ms.
    """
    md, z = df_h.MD.values, df_h.Z.values
    tvt_in = df_h.TVT_input.values
    lo = max(0, cut - pre_rows)
    p = irls_poly_well(md[lo:cut], z[lo:cut], tvt_in[lo:cut],
                       md[cut:], z[cut:], pred_post, deg=deg, mix=mix)
    mds = md[cut:] - md[cut - 1]
    p = p + (tvt_in[cut - 1] - p[0]) * np.exp(-mds / tau)
    if sigs is not None:
        ovp, _ = contact_override(df_h, cut, sigs, exclude=exclude)
        if ovp is not None:
            p = np.where(np.isfinite(ovp), ovp, p)
    return p


def bt_rmse(c):
    """RMSE de backtest por pozo y candidato (S, P, G, blend). NaN sin backtest."""
    cand = {"S": c.btS, "P": c.btP, "G": c.btG,
            "B": W_SUP * c.btS + (1 - W_SUP) * c.btP}
    out = {}
    for n, v in cand.items():
        sse = np.bincount(c.btwid, (c.bty - v) ** 2, c.NW)
        cnt = np.bincount(c.btwid, None, c.NW)
        r = np.sqrt(sse / np.maximum(cnt, 1))
        r[cnt == 0] = np.nan
        out[n] = r
    return out


def select_layer(c, p, base_blend, alpha):
    """Ganador del backtest por pozo, movimiento acotado: p + a*(winner - blend)."""
    btr = bt_rmse(c)
    names = list(btr)
    M = np.column_stack([btr[n] for n in names])
    winner = np.argmin(np.where(np.isfinite(M), M, np.inf), axis=1)
    has = np.isfinite(M).any(1)
    cand = {"S": c.S, "P": c.P, "G": c.G, "B": base_blend}
    tgt = base_blend.copy()
    for i in range(c.NW):
        if has[i]:
            tgt[c.sl[i]] = cand[names[winner[i]]][c.sl[i]]
    return p + alpha * (tgt - base_blend), winner, has


# ─────────────────────────── experimentos ────────────────────────────────────
def run_capas(k):
    c = C4(k)
    rows = []

    def rep(name, p, ref=None, extra=""):
        r = c.rmse(p)
        d = "" if ref is None else f" ({r-ref:+.3f})"
        print(f"  {name:44s} {r:7.3f}{d}  proxy={c.lb_proxy(p):7.3f} {extra}", flush=True)
        rows.append({"variante": name, "rmse": r, "proxy": c.lb_proxy(p)})
        return r

    print(f"== v4 capas k={k}: {c.NW} pozos, {len(c.y)} puntos ==")
    print("0) referencia")
    rep("S (superficie aniso16 k24)", c.S)
    rep("P (pf_ancc)", c.P)
    rep("G (geometrico)", c.G)
    blend = W_SUP * c.S + (1 - W_SUP) * c.P
    for w in (0.20, 0.30):
        rep(f"blend {w:.2f}S", w * c.S + (1 - w) * c.P)
    REF = rep("BLEND 0.25S+0.75P (referencia)", blend)

    print("\n1) contact override (LOWO, doble guard)")
    nhit = int(c.meta.ov_hit.sum())
    nself = int(c.meta.ovself_hit.sum())
    print(f"   LOWO: {nhit}/{c.NW} pozos con match legitimo | "
          f"self permitido: {nself}/{c.NW} (sanity, deberia ser ~todos)")
    if nself:
        m = c.meta[c.meta.ovself_hit == 1]
        print(f"   sanity prermse self: mediana {m.ovself_prermse.median():.4f} ft "
              f"max {m.ovself_prermse.max():.4f} ft")
    p_ov = apply_override(c, blend)
    rep("blend + override", p_ov, REF, extra=f"hits={nhit}")

    print("\n2) proyeccion IRLS de U=TVT+Z (pesos Cauchy)")
    best_irls = (None, REF)
    for dom, degs, mixes in (("post", (3, 4), (0.5, 0.75, 1.0)),
                             ("pre", (2, 3, 4), (0.3, 0.4, 0.5, 0.6, 0.75, 1.0))):
        for deg in degs:
            for mix in mixes:
                r = rep(f"irls {dom} deg{deg} mix={mix:.2f}",
                        irls_poly(c, blend, deg=deg, mix=mix, dom=dom), REF)
                if r < best_irls[1]:
                    best_irls = ((dom, deg, mix), r)

    print("\n3) postproceso por separado")
    best_ramp = (None, REF)
    for tau in (50, 85, 150, 300, 600):
        r = rep(f"rampa tau={tau}", ramp(c, blend, tau), REF)
        if r < best_ramp[1]:
            best_ramp = (tau, r)
    best_shr = (None, REF)
    for a in (0.85, 0.90, 0.95, 1.0):
        r = rep(f"shrink alpha={a:.2f}", shrink(c, blend, a), REF)
        if a < 1.0 and r < best_shr[1]:
            best_shr = (a, r)
    r_sg = rep("savgol(17,3)", savgol(c, blend), REF)

    print("\n4) seleccion por pozo (backtest 0.65, movimiento acotado)")
    btr = bt_rmse(c)
    ok = np.isfinite(btr["S"])
    print(f"   pozos con backtest: {int(ok.sum())}/{c.NW}")
    real = {n: np.sqrt(c.sse_well(v) / np.maximum(c.lens, 1))
            for n, v in (("S", c.S), ("P", c.P), ("G", c.G),
                         ("B", blend))}
    for n in ("S", "P", "B"):
        cc = np.corrcoef(np.log(btr[n][ok] + 1), np.log(real[n][ok] + 1))[0, 1]
        print(f"   corr(log bt_{n}, log real_{n}) = {cc:+.3f}")
    best_sel = (None, REF)
    for a in (0.2, 0.3, 0.4):
        p_sel, winner, has = select_layer(c, blend, blend, a)
        r = rep(f"seleccion alpha={a:.2f}", p_sel, REF)
        if r < best_sel[1]:
            best_sel = (a, r)
    if ok.any():
        from collections import Counter
        _, winner, has = select_layer(c, blend, blend, 0.3)
        names = list(btr)
        print(f"   ganadores backtest: {Counter(names[w] for w, h in zip(winner, has) if h)}")

    print("\n5) combinacion (greedy sobre las capas que mejoran)")
    layers = []
    if best_irls[0] is not None:
        d_, g_, m_ = best_irls[0]
        layers.append((f"irls {d_} deg{g_} mix={m_}",
                       lambda q, d=d_, g=g_, m=m_: irls_poly(c, q, deg=g, mix=m, dom=d)))
    if best_sel[0] is not None:
        layers.append((f"seleccion a={best_sel[0]}",
                       lambda q, a=best_sel[0]: select_layer(c, q, blend, a)[0]))
    if best_ramp[0] is not None:
        layers.append((f"rampa tau={best_ramp[0]}",
                       lambda q, t=best_ramp[0]: ramp(c, q, t)))
    if r_sg < REF:
        layers.append(("savgol(17,3)", lambda q: savgol(c, q)))
    if best_shr[0] is not None:
        layers.append((f"shrink a={best_shr[0]}",
                       lambda q, a=best_shr[0]: shrink(c, q, a)))
    cur, cur_r, chain = blend, REF, []
    improved = True
    while improved and layers:
        improved = False
        best_i, best_p, best_r = None, None, cur_r
        for i, (nm, fn) in enumerate(layers):
            r_try = c.rmse(fn(cur))
            if r_try < best_r - 1e-4:
                best_i, best_p, best_r = i, fn(cur), r_try
        if best_i is not None:
            nm, _ = layers.pop(best_i)
            chain.append(nm)
            cur, cur_r = best_p, best_r
            improved = True
            print(f"   + {nm:38s} -> {cur_r:7.3f}")
    cur = apply_override(c, cur)
    final_r = rep("STACK FINAL (capas + override)", cur, REF,
                  extra=f"chain={chain}")
    pd.DataFrame(rows).to_csv(HERE / f"v4_capas_k{k}.csv", index=False)
    print(f"\nstack final k={k}: {final_r:.3f}  (ref {REF:.3f}, {final_r-REF:+.3f})")
    return final_r


def run_fino(k):
    """Grid fino del stack completo: irls(pre, deg2, mix) + rampa(tau)."""
    c = C4(k)
    blend = W_SUP * c.S + (1 - W_SUP) * c.P
    print(f"== fino k={k}: ref blend = {c.rmse(blend):.3f} ==")
    best = (None, np.inf)
    for mix in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        p1 = irls_poly(c, blend, deg=2, mix=mix, dom="pre")
        for tau in (300.0, 600.0, 1200.0, 2400.0, 1e9):
            r = c.rmse(ramp(c, p1, tau))
            tag = f"mix={mix:.2f} tau={'inf' if tau > 1e8 else int(tau)}"
            print(f"  {tag:24s} {r:7.3f}  proxy={c.lb_proxy(ramp(c, p1, tau)):7.3f}")
            if r < best[1]:
                best = ((mix, tau), r)
    print(f"mejor: mix={best[0][0]}, tau={best[0][1]:g} -> {best[1]:.3f}")
    return best


# ─────────────────────────── selfcheck copias test/ ─────────────────────────
def selfcheck():
    sigs = Sigs()
    for wid in sorted(CV.SAMPLE_TEST_IDS):
        te = pd.read_csv(RAW / "test" / f"{wid}__horizontal_well.csv")
        cut = _cut(te)
        pred, info = contact_override(te, cut, sigs, exclude=None)
        tr = pd.read_csv(RAW / "train" / f"{wid}__horizontal_well.csv")
        truth = np.interp(te.MD.values, tr.MD.values, tr.TVT.values)[cut:]
        if pred is None:
            print(f"{wid}: SIN override (match={info})")
            continue
        m = np.isfinite(pred)
        r = float(np.sqrt(np.mean((truth[m] - pred[m]) ** 2)))
        print(f"{wid}: match={info['id']} dist={info['dist']:.2f} ft "
              f"cover={info['cover']:.2f} prermse={info['prermse']:.4f} "
              f"-> RMSE post-PS vs train = {r:.4f} ft ({m.sum()}/{len(pred)} filas)")


def guardcheck():
    """Los casi-duplicados del censo, evaluados LOWO: ¿el guard de prefijo
    admite el override solo cuando de verdad reproduce el pozo? Reporta, por
    pareja, prermse (guard) y RMSE post-PS real si se aplicara."""
    sigs = Sigs()
    rows = []
    for i, w in enumerate(sigs.ids):
        g = np.arange(sigs.g0[i], sigs.g1[i] + 1e-9, GRID)
        mm = match_geom(g, sigs.x[i].astype(float), sigs.y[i].astype(float),
                        sigs, exclude=w)
        if mm is None:
            continue
        df, tw, cut = CV.load_well(w)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[CV.TEST_COLS].copy()
        ovp, info = contact_override(df_h, cut, sigs, exclude=w)
        y = df.TVT.values[cut:]
        if ovp is None:
            rows.append({"well": w, "match": info["id"], "dist": info["dist"],
                         "prermse": info["prermse"], "hit": 0,
                         "rmse_post": np.nan, "cobertura": np.nan})
            continue
        m = np.isfinite(ovp)
        r = float(np.sqrt(np.mean((y[m] - ovp[m]) ** 2)))
        rows.append({"well": w, "match": info["id"], "dist": info["dist"],
                     "prermse": info["prermse"], "hit": 1,
                     "rmse_post": r, "cobertura": float(m.mean())})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    if len(t):
        acc = t[t.hit == 1]
        rej = t[t.hit == 0]
        print(f"\nadmitidos {len(acc)}: RMSE post-PS mediana "
              f"{acc.rmse_post.median():.3f} ft, max {acc.rmse_post.max():.3f} ft"
              if len(acc) else "\nadmitidos 0")
        print(f"rechazados por guard {len(rej)} "
              f"(prermse mediana {rej.prermse.median():.2f} ft)" if len(rej) else "rechazados 0")
    return t


def parity(k):
    """aplicar_capas (por pozo) debe reproducir el stack del cache exactamente."""
    c = C4(k)
    sigs = Sigs()
    blend = W_SUP * c.S + (1 - W_SUP) * c.P
    stack = apply_override(
        c, ramp(c, irls_poly(c, blend, deg=CAPAS["deg"], mix=CAPAS["mix"],
                             dom="pre"), CAPAS["tau"]))
    worst = 0.0
    for i, w in enumerate(c.meta.well.values):
        df, tw, cut = CV.load_well(w)
        df_h = df[CV.TEST_COLS].copy()
        p = aplicar_capas(df_h, cut, blend[c.sl[i]], sigs=sigs, exclude=w)
        worst = max(worst, float(np.abs(p - stack[c.sl[i]]).max()))
    print(f"paridad aplicar_capas vs cache (k={k}): max|dif| = {worst:.6f} ft "
          f"en {c.NW} pozos | rmse stack = {c.rmse(stack):.3f}")


def censo_duplicados():
    """Cuantos pares de casi-duplicados legitimos hay en train (informativo)."""
    sigs = Sigs()
    hits = []
    t0 = time.time()
    for i, w in enumerate(sigs.ids):
        g = np.arange(sigs.g0[i], sigs.g1[i] + 1e-9, GRID)
        mm = match_geom(g, sigs.x[i].astype(float), sigs.y[i].astype(float),
                        sigs, exclude=w)
        if mm is not None:
            hits.append((w, *mm))
        if (i + 1) % 200 == 0:
            print(f"  censo {i+1}/{len(sigs.ids)}  {time.time()-t0:.0f}s", flush=True)
    print(f"censo train: {len(hits)} pozos con casi-duplicado "
          f"(dist<{OV_DIST} ft, cover>={OV_COVER})")
    for h in hits[:20]:
        print(f"   {h[0]} ~ {h[1]}  dist={h[2]:.1f} cover={h[3]:.2f}")
    return hits


if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "capas"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if block == "sig":
        build_sig()
        censo_duplicados()
    elif block == "selfcheck":
        selfcheck()
    elif block == "guardcheck":
        guardcheck()
    elif block == "parity":
        parity(K)
    elif block == "cache":
        build_cache(K)
    elif block == "capas":
        run_capas(K)
    elif block == "fino":
        run_fino(K)
    else:
        raise SystemExit(f"bloque desconocido: {block}")
