"""v9-divers: recuperar pf_z y beam media-7 como miembros del blend adaptativo IVW.

Base = v8 "s2d multiplic." (ss=S_MULT 2D de v8_s2d_tablas.npz, sp=sig_p_1d):
control exacto 9.699 k60 / 8.949 k150. Sobre esa base se anaden miembros extra
con cap (+gate opcional), el mismo esquema que gano en blend3 con el geometrico:

    p_e = 1/sig_e(md_since)^2       (sig_e ajustada en los 200 fit wells de v7)
    w_e = min(p_e / (ps+pp+sum p_e), cap_e)   [0 si md_since >= gate_e]
    wr  = 1 - sum w_e
    w_s = wr * clip(ps/(ps+pp), 0.05, 0.9) ;  w_p = wr - w_s
    pred = w_s*S + w_p*P64 + sum w_e * extra_e

Con todos los caps a 0 se reproduce v8 EXACTO (pareado, mismo cache PF S=64 via
v8_blend3_sig_k*.npz). Miembros: S=superficie, P64=PF ANCC media 64 semillas,
Z=pf_z media S_Z semillas (semilla numba explicita, deterministico),
B=beam media-7 (wrapper arreglado: ban_lib.beam_all == v4_multiseed.beam7_of),
G=geometrico dip700 (gated, de blend3). Variantes anti-colinealidad PF~PFZ:
"pzavg" promedia P64 y Z en un solo miembro M con sig_m propia; "cov" Markowitz
con las rho(md) medidas y shrink.

Uso:
  python research/v9_divers_ivw.py fit        # rz, rb en los 200 fit wells (~8 min)
  python research/v9_divers_ivw.py curvas     # sig_z, sig_b, sig_m + rho(md) pares
  python research/v9_divers_ivw.py cache 60   # anade z8 y beam7 al cache de senales
  python research/v9_divers_ivw.py sweep 60   # barrido caps/gates desde cache (pareado)
  python research/v9_divers_ivw.py eval 60 "z=0.3@2000,b=0.1,g=0.2@1000" ...
  python research/v9_divers_ivw.py oracle     # NNLS por pozo 5 miembros (techo)
  python research/v9_divers_ivw.py all        # fit+curvas+cache+sweep (60 y 150)
"""
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import nnls

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod                                   # noqa: E402
import v7_wadapt as v7                               # noqa: E402
import v8_sig2d_wadapt as v8                         # noqa: E402  (solo _interp2d)
from ban_lib import beam_all                         # noqa: E402
from cv import evaluate, load_well                   # noqa: E402
from pf_publico import _tw, run_pf_z                 # noqa: E402

NN_BINS, MD_BINS = v7.NN_BINS, v7.MD_BINS
W_CAP = (0.05, 0.9)
S_Z = 8                                    # semillas pf_z (miembro = media)
TABLAS = R / "v8_s2d_tablas.npz"
B3FIT = R / "v8_blend3_fitdata.npz"
B3CURV = R / "v8_blend3_curvas.npz"
B3SIG = {60: R / "v8_blend3_sig_k60.npz", 150: R / "v8_blend3_sig_k150.npz"}
FITD = R / "v9_divers_fitdata.npz"
CURV = R / "v9_divers_curvas.npz"
SIGC = {60: R / "v9_divers_sig_k60.npz", 150: R / "v9_divers_sig_k150.npz"}

Z_T, B_T = [], []


@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)


def z_seeds(df_h, tw, S=S_Z):
    """(S, n_pred) pasadas de pf_z con semilla numba explicita (deterministico)."""
    t, g = _tw(tw)
    t0 = time.time()
    out = []
    for s in range(1, S + 1):
        _seed_numba(s)
        p, _ = run_pf_z(df_h, t, g)
        out.append(np.asarray(p, np.float32))
    Z_T.append((time.time() - t0) / S)
    return np.stack(out)


def beam_m7(df_h, tw):
    t0 = time.time()
    b = beam_all(df_h, tw).mean(0)
    B_T.append(time.time() - t0)
    return b


# ------------------------------------------------------------------ fit
def fit():
    """Residuos con signo de Z (media S_Z) y B (media-7) en los fit wells de v7.

    Mismo bucle/orden que v8_blend3.fit: la concatenacion queda ALINEADA punto a
    punto con rs/rp/rg de v8_blend3_fitdata.npz (se asserta contra mds).
    """
    fw = v7.fit_wells()
    rz, rb, mds, zsd = [], [], [], []
    t0 = time.time()
    for i, wid in enumerate(fw):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[cvmod.TEST_COLS].copy()
        y = df.TVT.values[cut:]
        md = df.MD.values
        zz = z_seeds(df_h, tw)
        rz.append(y - zz.mean(0))
        rb.append(y - beam_m7(df_h, tw))
        zsd.append(zz.std(0))
        mds.append(md[cut:] - md[cut - 1])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fw)}  {time.time()-t0:.0f}s "
                  f"(z {np.mean(Z_T):.2f}s/semilla, beam7 {np.mean(B_T):.2f}s)", flush=True)
    rz, rb, mds, zsd = map(np.concatenate, (rz, rb, mds, zsd))
    b3 = np.load(B3FIT)
    assert len(rz) == len(b3["rs"]) and np.allclose(mds, b3["mds"]), \
        "desalineado con v8_blend3_fitdata"
    np.savez(FITD, rz=rz, rb=rb, zsd=zsd)
    print(f"guardado {FITD} ({len(rz)} puntos)")
    print(f"COSTE fit: z {np.mean(Z_T):.3f}s/pozo/semilla, beam7 {np.mean(B_T):.3f}s/pozo")


# ------------------------------------------------------------------ curvas
def curvas():
    b3 = np.load(B3FIT)
    d9 = np.load(FITD)
    rs, rp, rg, mds = b3["rs"], b3["rp"], b3["rg"], b3["mds"]
    rz, rb = d9["rz"], d9["rb"]
    rm = 0.5 * (rp + rz)                       # miembro M = (P+Z)/2
    sig_z = v7._binned_rms(mds, rz ** 2, MD_BINS)
    sig_b = v7._binned_rms(mds, rb ** 2, MD_BINS)
    sig_m = v7._binned_rms(mds, rm ** 2, MD_BINS)
    sig_g = np.load(B3CURV)["sig_g"]

    idx = np.clip(np.digitize(mds, MD_BINS) - 1, 0, len(MD_BINS) - 2)

    def rho(a, b):
        out = np.full(len(MD_BINS) - 1, np.nan)
        for k in range(len(MD_BINS) - 1):
            m = idx == k
            if m.sum() > 200:
                out[k] = np.corrcoef(a[m], b[m])[0, 1]
        return out

    res = {"sig_z": sig_z, "sig_b": sig_b, "sig_m": sig_m, "sig_g": sig_g}
    names = {"s": rs, "p": rp, "z": rz, "b": rb, "g": rg}
    keys = list(names)
    for i, a in enumerate(keys):
        for c in keys[i + 1:]:
            res[f"rho_{a}{c}"] = rho(names[a], names[c])
    np.savez(CURV, **res)
    tb = np.load(TABLAS)
    print("md_bins:          ", MD_BINS[:-1].astype(int))
    print("sig_p_1d (P64ref):", np.round(tb["sig_p_1d"], 1))
    print("sig_z(md_since):  ", np.round(sig_z, 1))
    print("sig_b(md_since):  ", np.round(sig_b, 1))
    print("sig_m(md_since):  ", np.round(sig_m, 1))
    print("sig_g(md_since):  ", np.round(sig_g, 1))
    for q in ("pz", "pb", "zb", "sz", "sb", "pg", "zg", "bg"):
        key = f"rho_{q}" if f"rho_{q}" in res else f"rho_{q[::-1]}"
        print(f"rho_{q}(md):       ", np.round(res[key], 2))
    print(f"guardado {CURV}")


# ------------------------------------------------------------------ cache eval
def build_cache(k):
    z8 = np.load(B3SIG[k])
    wids = sorted({f.split("_", 1)[1] for f in z8.files})
    store, t0 = {}, time.time()
    for i, wid in enumerate(wids):
        df, tw, cut = load_well(wid)
        df_h = df[cvmod.TEST_COLS].copy()
        store[f"z_{wid}"] = z_seeds(df_h, tw)          # (S_Z, n)
        store[f"b_{wid}"] = beam_m7(df_h, tw).astype(np.float32)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(wids)}  {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(SIGC[k], **store)
    print(f"guardado {SIGC[k]} ({len(wids)} pozos)")
    print(f"COSTE cache: z {np.mean(Z_T):.3f}s/pozo/semilla, beam7 {np.mean(B_T):.3f}s/pozo")


def load_cache(k, s_z=S_Z):
    b3, d9 = np.load(B3SIG[k]), np.load(SIGC[k])
    wids = sorted({f.split("_", 1)[1] for f in b3.files})
    out = {}
    for w in wids:
        d = {f: b3[f"{f}_{w}"].astype(np.float64) for f in ("prior", "nn", "pf", "geo", "md", "y")}
        d["z"] = d9[f"z_{w}"][:s_z].mean(0).astype(np.float64)
        d["b"] = d9[f"b_{w}"].astype(np.float64)
        out[w] = d
    return out


# ------------------------------------------------------------------ pesos
class Curvas:
    def __init__(self):
        tb = np.load(TABLAS)
        self.ss2 = v8._interp2d(NN_BINS, MD_BINS, tb["s_mult"])
        self.sig_p_1d = tb["sig_p_1d"]
        self.c = dict(np.load(CURV).items())

    def ss(self, nn, md):
        return self.ss2(nn, md)

    def sig(self, name, md):
        key = {"p": None, "z": "sig_z", "b": "sig_b", "g": "sig_g", "m": "sig_m"}[name]
        vals = self.sig_p_1d if key is None else self.c[key]
        return v7._interp_curve(md, MD_BINS, vals)

    def rho(self, a, b, md):
        key = f"rho_{a}{b}" if f"rho_{a}{b}" in self.c else f"rho_{b}{a}"
        return v7._interp_curve(md, MD_BINS, np.nan_to_num(self.c[key]))


def blend_pred(d, cv_, extras=(), pzavg=False, cov=None):
    """extras: lista de (nombre, cap, gate) con nombre en {z,b,g}.

    pzavg=True: el miembro central es M=(P+Z)/2 con sig_m (Z deja de ser extra).
    cov=(miembros, shrink): Markowitz w prop Sigma^-1 1 sobre esos miembros.
    """
    nn, md = d["nn"], d["md"]
    sig_of = {"p": d["pf"], "z": d["z"], "b": d["b"], "g": d["geo"],
              "m": 0.5 * (d["pf"] + d["z"])}
    if cov is not None:
        miembros, shrink = cov
        sigs = [cv_.ss(nn, md) if m == "s" else cv_.sig(m, md) for m in miembros]
        n, q = len(nn), len(miembros)
        S = np.empty((n, q, q))
        for i in range(q):
            S[:, i, i] = sigs[i] ** 2
            for j in range(i + 1, q):
                r = cv_.rho(miembros[i], miembros[j], md) * shrink
                S[:, i, j] = S[:, j, i] = r * sigs[i] * sigs[j]
        w = np.linalg.solve(S, np.ones((n, q, 1)))[:, :, 0]
        w = np.clip(w, 0.0, None)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        pred = np.zeros(n)
        for i, m in enumerate(miembros):
            pred += w[:, i] * (d["prior"] if m == "s" else sig_of[m])
        return pred

    ss = cv_.ss(nn, md)
    sp = cv_.sig("m" if pzavg else "p", md)
    ps, pp = 1.0 / ss ** 2, 1.0 / sp ** 2
    pe, active = [], []
    for name, cap, gate in extras:
        p = 1.0 / cv_.sig(name, md) ** 2
        if gate is not None:
            p = np.where(md < gate, p, 0.0)
        pe.append(p); active.append((name, cap))
    tot = ps + pp + sum(pe) if pe else ps + pp
    w_e = [np.minimum(p / tot, cap) for p, (_, cap) in zip(pe, active)]
    wr = 1.0 - sum(w_e) if w_e else 1.0
    w_s = wr * np.clip(ps / (ps + pp), *W_CAP)
    w_p = wr - w_s
    pred = w_s * d["prior"] + w_p * sig_of["m" if pzavg else "p"]
    for w, (name, _) in zip(w_e, active):
        pred += w * sig_of[name]
    return pred


def rmse_of(cache, cv_, **kw):
    sse = n = 0.0
    for d in cache.values():
        e = d["y"] - blend_pred(d, cv_, **kw)
        sse += float((e ** 2).sum()); n += len(e)
    return float(np.sqrt(sse / n))


# ------------------------------------------------------------------ sweep
def sweep(k):
    cache, cv_ = load_cache(k), Curvas()
    ref = rmse_of(cache, cv_)
    print(f"== sweep k={k} (pareado, {len(cache)} pozos) ==")
    print(f"  control v8 s2d-mult (caps=0)          rmse={ref:7.3f}")

    def p(tag, **kw):
        r = rmse_of(cache, cv_, **kw)
        print(f"  {tag:38s} rmse={r:7.3f}  d={r-ref:+.3f}", flush=True)
        return r

    print("-- {S,P,Z}: cap_z x gate_z --")
    for cap in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        for gate in (None, 1000, 2000, 3000, 5000):
            p(f"z cap={cap:.2f} gate={gate}", extras=[("z", cap, gate)])
    print("-- {S,P,Z,B}: mejor z + cap_b x gate_b --")
    for capz in (0.15, 0.30):
        for capb in (0.05, 0.10, 0.15, 0.20, 0.30):
            for gateb in (None, 1500, 3000):
                p(f"z {capz:.2f} + b cap={capb:.2f} gate={gateb}",
                  extras=[("z", capz, None), ("b", capb, gateb)])
    print("-- {S,P,Z,B,G}: + geometrico gated (blend3) --")
    for capz in (0.15, 0.30):
        for capb in (0.10, 0.20):
            for capg, gg in ((0.2, 1000), (0.3, 1250)):
                p(f"z {capz:.2f} b {capb:.2f} + g {capg:.2f}@{gg}",
                  extras=[("z", capz, None), ("b", capb, None), ("g", capg, gg)])
    print("-- PZ-avg: M=(P+Z)/2 con sig_m --")
    p("pzavg", pzavg=True)
    for capb in (0.05, 0.10, 0.20):
        p(f"pzavg + b cap={capb:.2f}", pzavg=True, extras=[("b", capb, None)])
    for capb in (0.10, 0.20):
        p(f"pzavg + b {capb:.2f} + g 0.2@1000", pzavg=True,
          extras=[("b", capb, None), ("g", 0.2, 1000)])
    print("-- cov Markowitz --")
    for mm in ("spz", "spzb", "spzbg"):
        for sh in (1.0, 0.8, 0.5):
            p(f"cov {mm} shrink={sh}", cov=(tuple(mm), sh))
    print("-- Z con menos semillas (media 4) --")
    c4 = load_cache(k, s_z=4)
    r = rmse_of(c4, cv_, extras=[("z", 0.3, None)])
    print(f"  z cap=0.30 con media4                  rmse={r:7.3f}  d={r-ref:+.3f}")


# ------------------------------------------------------------------ eval cv
def parse_spec(spec):
    kw = {"extras": []}
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "pzavg":
            kw["pzavg"] = True
        elif tok.startswith("cov="):
            mm, _, sh = tok[4:].partition("@")
            kw["cov"] = (tuple(mm), float(sh) if sh else 1.0)
        else:
            name, _, val = tok.partition("=")
            cap, _, gate = val.partition("@")
            kw["extras"].append((name, float(cap), float(gate) if gate else None))
    return kw


def eval_k(k, specs):
    cv_ = Curvas()
    cache = {}
    for kk in (60, 150):
        if SIGC[kk].exists():
            cache.update(load_cache(kk))

    def make(kw):
        return lambda df_h, tw, wid=None: blend_pred(cache[wid], cv_, **kw)

    print(f"== eval k={k} via cv.evaluate (pareado, mismo cache PF S=64) ==")
    r = evaluate(make({}), k=k, verbose=False)
    print(f"  {'control v8 s2d-mult':38s} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}", flush=True)
    for spec in specs:
        r = evaluate(make(parse_spec(spec)), k=k, verbose=False)
        print(f"  {spec:38s} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}", flush=True)


# ------------------------------------------------------------------ oraculo
def oracle():
    """NNLS por pozo con los 5 miembros nuevos vs los 5 de orc01 (mismos pozos)."""
    from orclib import Cache as OrcCache
    orc = OrcCache()
    for k in (60, 150):
        cache = load_cache(k)
        rows = {"nnls": [], "convexo": [], "ls": [], "orc_nnls": [], "orc_ls": []}
        for wid, d in cache.items():
            A = np.column_stack([d["prior"], d["pf"], d["z"], d["b"], d["geo"]])
            y = d["y"]
            w, _ = nnls(A, y)
            rows["nnls"].append((float(((A @ w - y) ** 2).sum()), len(y)))
            s = w.sum()
            wc = w / s if s > 0 else np.full(5, 0.2)
            rows["convexo"].append((float(((A @ wc - y) ** 2).sum()), len(y)))
            wl = np.linalg.lstsq(A, y, rcond=None)[0]
            rows["ls"].append((float(((A @ wl - y) ** 2).sum()), len(y)))
            i = orc.widx[wid]; sl = orc.sl(i)
            Ao = np.column_stack([orc.d[p][sl] for p in ("surf", "ancc", "pfz", "beam", "geom")])
            yo = orc.d["y"][sl]
            wo, _ = nnls(Ao, yo)
            rows["orc_nnls"].append((float(((Ao @ wo - yo) ** 2).sum()), len(yo)))
            wol = np.linalg.lstsq(Ao, yo, rcond=None)[0]
            rows["orc_ls"].append((float(((Ao @ wol - yo) ** 2).sum()), len(yo)))

        def pooled(rr):
            return float(np.sqrt(sum(s for s, _ in rr) / sum(n for _, n in rr)))

        print(f"== ORACULO k={k} ({len(cache)} pozos) ==")
        print(f"  NNLS/pozo 5 nuevos {{S,P64,Z8,B7,G}}:  {pooled(rows['nnls']):6.3f}")
        print(f"  convexo/pozo 5 nuevos:                {pooled(rows['convexo']):6.3f}")
        print(f"  LS libre/pozo 5 nuevos:               {pooled(rows['ls']):6.3f}")
        print(f"  NNLS/pozo 5 orc01 (1 semilla):        {pooled(rows['orc_nnls']):6.3f}")
        print(f"  LS libre/pozo 5 orc01:                {pooled(rows['orc_ls']):6.3f}", flush=True)


# ==================================================================== main
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "fit":
        fit()
    elif cmd == "curvas":
        curvas()
    elif cmd == "cache":
        build_cache(int(sys.argv[2]))
    elif cmd == "sweep":
        sweep(int(sys.argv[2]))
    elif cmd == "eval":
        eval_k(int(sys.argv[2]), sys.argv[3:])
    elif cmd == "oracle":
        oracle()
    else:
        if not FITD.exists():
            fit()
        curvas()
        for k in (60, 150):
            if not SIGC[k].exists():
                build_cache(k)
        sweep(60)
        sweep(150)
        print("DONE_V9_DIVERS")
