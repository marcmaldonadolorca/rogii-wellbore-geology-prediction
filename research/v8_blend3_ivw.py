"""v8: blend de 3 senales por varianza inversa — superficie + PF + geometrico.

Tercer miembro: geometrico anclado (baseline.predict con dip_win=700).
Esquema de pesos por punto (cap_g=0 reproduce EXACTAMENTE el adaptativo v7):
    ps=1/sig_s(nn)^2  pp=1/sig_p(md_since)^2  pg=1/sig_g(md_since)^2
    w_g = min(pg/(ps+pp+pg), cap_g)        [opcional: w_g=0 si md_since>=GATE]
    w_s = (1-w_g) * clip(ps/(ps+pp), 0.05, 0.9)
    w_p = 1 - w_g - w_s
    pred = w_s*superficie + w_p*PF + w_g*geo

Los errores de PF y geo estan correlacionados cerca del PS (ambos anclados al
ultimo punto conocido): fit() mide la correlacion de residuos por tramos de
md_since; si es alta el IVW ingenuo sobrepondera al par anclado, de ahi el cap
y el gate. Variante extra "cov": pesos Markowitz w prop Sigma^-1 1 con las
correlaciones medidas (clip de negativos + renormalizacion).

Uso:
  python research/v8_blend3_ivw.py fit        # sig_g + corr residuos (200 fit wells, ~6 min)
  python research/v8_blend3_ivw.py cache 60   # cachea senales de los pozos eval k=60
  python research/v8_blend3_ivw.py sweep 60   # barrido rapido caps/gate desde cache (pareado)
  python research/v8_blend3_ivw.py eval 60    # confirmacion via cv.evaluate (pareado)
"""
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod                                  # noqa: E402
import v7_wadapt as v7                              # noqa: E402
from baseline import predict as geo_predict         # noqa: E402
from cv import evaluate, load_well, _select, well_ids  # noqa: E402
from model import _prior                            # noqa: E402

DIP_WIN = 700
NN_BINS, MD_BINS = v7.NN_BINS, v7.MD_BINS
FITDATA = R / "v8_blend3_fitdata.npz"
CURVAS = R / "v8_blend3_curvas.npz"
SIGC = {60: R / "v8_blend3_sig_k60.npz", 150: R / "v8_blend3_sig_k150.npz"}
W_CAP = (0.05, 0.9)          # clip v7 del reparto superficie/PF


# ------------------------------------------------------------------ fit
def fit():
    """Residuos CON SIGNO de las 3 senales en los 200 pozos de ajuste de v7."""
    fw = v7.fit_wells()
    rs, rp, rg, nns, mds = [], [], [], [], []
    t0 = time.time()
    for i, wid in enumerate(fw):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[cvmod.TEST_COLS].copy()
        prior, nn = _prior(df_h, v7.field(), wid, cut)
        pfp = v7.pf_of(df_h, tw, wid, v7.S_FIT)
        geo = geo_predict(df_h, cut, DIP_WIN)
        y = df.TVT.values[cut:]
        md = df.MD.values
        rs.append(y - prior[cut:]); rp.append(y - pfp); rg.append(y - geo)
        nns.append(nn[cut:]); mds.append(md[cut:] - md[cut - 1])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fw)}  {time.time()-t0:.0f}s", flush=True)
    np.savez(FITDATA, rs=np.concatenate(rs), rp=np.concatenate(rp),
             rg=np.concatenate(rg), nns=np.concatenate(nns),
             mds=np.concatenate(mds))
    print(f"guardado {FITDATA}")
    build_curvas()


def build_curvas():
    z = np.load(FITDATA)
    rs, rp, rg, nns, mds = z["rs"], z["rp"], z["rg"], z["nns"], z["mds"]
    c7 = np.load(v7.CURVAS)
    sig_g = v7._binned_rms(mds, rg ** 2, MD_BINS)

    # correlaciones de residuos por tramo de md_since
    idx = np.clip(np.digitize(mds, MD_BINS) - 1, 0, len(MD_BINS) - 2)
    def corr_curve(a, b):
        out = np.full(len(MD_BINS) - 1, np.nan)
        for k in range(len(MD_BINS) - 1):
            m = idx == k
            if m.sum() > 200:
                out[k] = np.corrcoef(a[m], b[m])[0, 1]
        return out
    rho_pg = corr_curve(rp, rg)
    rho_sg = corr_curve(rs, rg)
    rho_sp = corr_curve(rs, rp)
    np.savez(CURVAS, sig_s=c7["sig_s"], sig_p=c7["sig_p"], sig_g=sig_g,
             nn_bins=NN_BINS, md_bins=MD_BINS,
             rho_pg=rho_pg, rho_sg=rho_sg, rho_sp=rho_sp)
    print("sig_s(nn):        ", np.round(c7["sig_s"], 1))
    print("sig_p(md_since):  ", np.round(c7["sig_p"], 1))
    print("sig_g(md_since):  ", np.round(sig_g, 1))
    print("rho(PF,geo) md:   ", np.round(rho_pg, 2))
    print("rho(surf,geo) md: ", np.round(rho_sg, 2))
    print("rho(surf,PF) md:  ", np.round(rho_sp, 2))
    print(f"guardado {CURVAS}")


# ------------------------------------------------------------------ cache eval
def build_cache(k):
    ids = _select(well_ids(), k, cvmod.SEED)
    store, t0 = {}, time.time()
    for i, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[cvmod.TEST_COLS].copy()
        prior, nn = _prior(df_h, v7.field(), wid, cut)
        pfp = v7.pf_of(df_h, tw, wid, v7.S_EVAL)   # cache S64 o fallback determinista
        geo = geo_predict(df_h, cut, DIP_WIN)
        md = df_h.MD.values
        store[f"prior_{wid}"] = prior[cut:]
        store[f"nn_{wid}"] = nn[cut:]
        store[f"pf_{wid}"] = pfp
        store[f"geo_{wid}"] = geo
        store[f"md_{wid}"] = md[cut:] - md[cut - 1]
        store[f"y_{wid}"] = df.TVT.values[cut:]
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    np.savez(SIGC[k], **store)
    print(f"guardado {SIGC[k]} ({len(store)//6} pozos)")


def load_cache(k):
    z = np.load(SIGC[k])
    wids = sorted({key.split("_", 1)[1] for key in z.files})
    return {w: {f: z[f"{f}_{w}"] for f in ("prior", "nn", "pf", "geo", "md", "y")}
            for w in wids}


# ------------------------------------------------------------------ pesos
def weights3(nn, mdsince, cap_g, gate=None, curvas=None):
    c = curvas if curvas is not None else np.load(CURVAS)
    ss = v7._interp_curve(nn, NN_BINS, c["sig_s"])
    sp = v7._interp_curve(mdsince, MD_BINS, c["sig_p"])
    sg = v7._interp_curve(mdsince, MD_BINS, c["sig_g"])
    ps, pp, pg = 1 / ss ** 2, 1 / sp ** 2, 1 / sg ** 2
    w_g = np.minimum(pg / (ps + pp + pg), cap_g)
    if gate is not None:
        w_g = np.where(mdsince < gate, w_g, 0.0)
    w_s = (1 - w_g) * np.clip(ps / (ps + pp), *W_CAP)
    w_p = 1 - w_g - w_s
    return w_s, w_p, w_g


def weights_cov(nn, mdsince, curvas=None, shrink=1.0):
    """Markowitz: w prop Sigma^-1 1 con rho medidas (shrink: rho*=shrink)."""
    c = curvas if curvas is not None else np.load(CURVAS)
    ss = v7._interp_curve(nn, NN_BINS, c["sig_s"])
    sp = v7._interp_curve(mdsince, MD_BINS, c["sig_p"])
    sg = v7._interp_curve(mdsince, MD_BINS, c["sig_g"])
    rho = {q: v7._interp_curve(mdsince, MD_BINS,
                               np.nan_to_num(c[f"rho_{q}"])) * shrink
           for q in ("pg", "sg", "sp")}
    n = len(ss)
    S = np.empty((n, 3, 3))
    S[:, 0, 0] = ss ** 2; S[:, 1, 1] = sp ** 2; S[:, 2, 2] = sg ** 2
    S[:, 0, 1] = S[:, 1, 0] = rho["sp"] * ss * sp
    S[:, 0, 2] = S[:, 2, 0] = rho["sg"] * ss * sg
    S[:, 1, 2] = S[:, 2, 1] = rho["pg"] * sp * sg
    w = np.linalg.solve(S, np.ones((n, 3, 1)))[:, :, 0]
    w = np.clip(w, 0.0, None)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return w[:, 0], w[:, 1], w[:, 2]


def blend_pred(d, cap_g, gate=None, curvas=None, cov=False, shrink=1.0):
    if cov:
        w_s, w_p, w_g = weights_cov(d["nn"], d["md"], curvas, shrink)
    else:
        w_s, w_p, w_g = weights3(d["nn"], d["md"], cap_g, gate, curvas)
    return w_s * d["prior"] + w_p * d["pf"] + w_g * d["geo"]


# ------------------------------------------------------------------ sweep
def rmse_of(cache, **kw):
    curvas = dict(np.load(CURVAS).items())
    sse = n = 0.0
    for d in cache.values():
        e = d["y"] - blend_pred(d, curvas=curvas, **kw)
        sse += float((e ** 2).sum()); n += len(e)
    return float(np.sqrt(sse / n))


def sweep(k):
    cache = load_cache(k)
    ref = rmse_of(cache, cap_g=0.0)
    print(f"== sweep k={k} (pareado, {len(cache)} pozos) ==")
    print(f"  cap_g=0.00 (== adaptativo v7)      rmse={ref:7.3f}")
    for cap in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.66, 1.00):
        r = rmse_of(cache, cap_g=cap)
        print(f"  cap_g={cap:4.2f}                        rmse={r:7.3f}  d={r-ref:+.3f}")
    for cap in (0.10, 0.20, 0.30, 0.40, 0.50, 0.66, 1.00):
        for gate in (250, 500, 750, 1000, 1250, 1500, 2000, 3000, 5000):
            r = rmse_of(cache, cap_g=cap, gate=gate)
            print(f"  cap_g={cap:4.2f} gate={gate:5d}             rmse={r:7.3f}  d={r-ref:+.3f}")
    for sh in (1.0, 0.8, 0.5):
        r = rmse_of(cache, cap_g=None, cov=True, shrink=sh)
        print(f"  cov shrink={sh:3.1f}                    rmse={r:7.3f}  d={r-ref:+.3f}")


# ------------------------------------------------------------------ eval harness
def make_predictor(cap_g, gate=None):
    curvas = dict(np.load(CURVAS).items())
    cache = {}
    for k in (60, 150):
        if SIGC[k].exists():
            cache.update(load_cache(k))

    def predict(df_h, tw, wid=None):
        return blend_pred(cache[wid], cap_g=cap_g, gate=gate, curvas=curvas)

    return predict


def eval_k(k, variants):
    print(f"== eval k={k} via cv.evaluate (pareado, mismo cache de senales) ==")
    for name, cap, gate in variants:
        r = evaluate(make_predictor(cap, gate), k=k, verbose=False)
        print(f"  {name:28s} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}",
              flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        fit()
    elif cmd == "curvas":
        build_curvas()
    elif cmd == "cache":
        build_cache(int(sys.argv[2]))
    elif cmd == "sweep":
        sweep(int(sys.argv[2]))
    elif cmd == "eval":
        k = int(sys.argv[2])
        caps = [("adapt2 (ref v7)", 0.0, None)]
        for arg in sys.argv[3:]:
            cap, _, gate = arg.partition("@")
            caps.append((f"blend3 cap={cap}" + (f" gate={gate}" if gate else ""),
                         float(cap), float(gate) if gate else None))
        eval_k(k, caps)
