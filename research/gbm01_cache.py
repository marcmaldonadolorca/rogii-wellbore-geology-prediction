"""Cache por PUNTO post-PS de todos los candidatos + features, para el GBM arbitro.

Para cada pozo del pool guarda, en las filas post-PS:
  - candidatos (todos como DELTA respecto a last_known_tvt = TVT_input[cut-1]):
    superficie, superficie anclada en PS, superficie+HMM, pf_ancc, pf_z,
    7 beams, geometrico con dip, plano (solo -dZ)
  - incertidumbres: std de los PF, nn_dist y dispersion ponderada de la superficie
  - contexto: md_since, hd_since, rolling GR, derivadas de trayectoria, GR del
    typewell evaluado en los candidatos
  - meta por pozo: cut, n_pred, saltos en PS, y BACKTEST leak-free de cada
    candidato sobre la cola del prefijo (donde TVT_input SI se conoce)

Todo con el df en formato test exacto (MD,X,Y,Z,GR,TVT_input) y LOWO estricto.

Uso:  python research/gbm01_cache.py <n_pool>   (default 350)
Salida: research/gbm01_cache.npz + research/gbm01_meta.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from cv import TEST_COLS, SEED, load_well, well_ids, _select  # noqa: E402
import model as M  # noqa: E402
import pf_publico as P  # noqa: E402

BT_FRACS = (0.5, 0.75)
DIP_WIN = 700
NBEAM = 7

ROW = ["y", "dsurf", "dsurfa", "dhmm", "dancc", "dpfz", "dgeom", "dflat",
       "sancc", "spfz", "nn", "sstd", "md_since", "hd_since",
       "gr", "gr5", "gr21", "gr51", "gr101", "dzdm", "dxydm",
       "grtw_ancc", "grtw_surf", "grtw_pfz"] + [f"dbm{c}" for c in range(NBEAM)]


def interp2(field, X, Y, exclude=None, k=M.K_NEIGH):
    """Como SurfaceField.interp pero devuelve tambien la dispersion ponderada."""
    tree, vals = field._tree_without(exclude)
    dist, ind = tree.query(np.column_stack([X, Y]), k=k, workers=-1)
    w = 1.0 / np.maximum(dist, 1e-3) ** 2
    w /= w.sum(1, keepdims=True)
    v = vals[ind]
    mu = (w * v).sum(1)
    sd = np.sqrt(np.maximum((w * (v - mu[:, None]) ** 2).sum(1), 0.0))
    return mu, dist[:, 0], sd


def hdist(df):
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=DIP_WIN):
    """Recta de dip anclada en PS (research/afinar_dip.py: anchor700)."""
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
    return flat + slope * h, slope


def candidates(df_h, tw, wid, cut, field, hd):
    """dict nombre -> array de TVT sobre TODAS las filas (post-PS relleno)."""
    n = len(df_h)
    out = {}
    s, nn, sd = interp2(field, df_h.X.values, df_h.Y.values, exclude=wid)
    lo = max(0, cut - M.CAL_TAIL)
    c = np.median(df_h.TVT_input.values[lo:cut] + df_h.Z.values[lo:cut] - s[lo:cut])
    prior = s - df_h.Z.values + c
    out["surf"] = prior
    out["nn"], out["sstd"] = nn, sd
    # anclada exactamente en el PS (continuidad forzada)
    out["surfa"] = prior + (df_h.TVT_input.values[cut - 1] - prior[cut - 1])
    hmm = prior.copy()
    hmm[cut:] = prior[cut:] + M.hmm_refine(df_h, tw, prior, cut, sigma_r=0.01, dec=10)
    out["hmm"] = hmm

    t_, g_ = P._tw(tw)
    a, sa = P.run_pf_ancc(df_h, t_, g_)
    b, sb = P.run_pf_z(df_h, t_, g_)
    for k, (v, sv) in (("ancc", (a, sa)), ("pfz", (b, sb))):
        full = np.empty(n); full[:cut] = df_h.TVT_input.values[:cut]; full[cut:] = v
        out[k] = full
        fs = np.zeros(n); fs[cut:] = sv
        out["s" + k] = fs
    for cfg in range(NBEAM):
        pb = P.predict_beam(df_h, tw, cfg)
        full = np.empty(n); full[:cut] = df_h.TVT_input.values[:cut]; full[cut:] = pb
        out[f"bm{cfg}"] = full
    gp, slope = geom_pred(df_h, cut, hd)
    out["geom"] = gp
    out["flat"] = df_h.TVT_input.values[cut - 1] + (df_h.Z.values[cut - 1] - df_h.Z.values)
    out["_slope"] = slope
    out["_twtvt"], out["_twgr"] = t_, g_
    return out


CAND = ["surf", "surfa", "hmm", "ancc", "pfz", "geom", "flat"] + [f"bm{c}" for c in range(NBEAM)]


def one_well(wid, field):
    df, tw, cut = load_well(wid)
    if cut < 80 or cut >= len(df) - 10:
        return None
    df_h = df[TEST_COLS].copy()
    hd = hdist(df_h)
    C = candidates(df_h, tw, wid, cut, field, hd)
    lk = float(df_h.TVT_input.values[cut - 1])
    sl = slice(cut, len(df))

    md, z, x, y_, gr = (df_h.MD.values, df_h.Z.values, df_h.X.values,
                        df_h.Y.values, df_h.GR.values)
    grs = pd.Series(gr)
    roll = {w: grs.rolling(w, center=True, min_periods=1).mean().to_numpy() for w in (5, 21, 51, 101)}
    dmd = np.maximum(np.diff(md, prepend=md[0] - 1.0), 1e-6)
    dzdm = pd.Series(np.diff(z, prepend=z[0]) / dmd).rolling(21, center=True, min_periods=1).mean().to_numpy()
    dxy = np.hypot(np.diff(x, prepend=x[0]), np.diff(y_, prepend=y_[0])) / dmd
    dxydm = pd.Series(dxy).rolling(21, center=True, min_periods=1).mean().to_numpy()

    r = {"y": df.TVT.values[sl]}
    for c in CAND:
        r["d" + c] = C[c][sl] - lk
    r["sancc"], r["spfz"] = C["sancc"][sl], C["spfz"][sl]
    r["nn"], r["sstd"] = C["nn"][sl], C["sstd"][sl]
    r["md_since"] = md[sl] - md[cut - 1]
    r["hd_since"] = hd[sl] - hd[cut - 1]
    r["gr"] = gr[sl]
    for w in (5, 21, 51, 101):
        r[f"gr{w}"] = roll[w][sl]
    r["dzdm"], r["dxydm"] = dzdm[sl], dxydm[sl]
    tt, tg = C["_twtvt"], C["_twgr"]
    for c in ("ancc", "surf", "pfz"):
        r["grtw_" + c] = np.interp(C[c][sl], tt, tg)

    # ---- backtest leak-free en la cola del prefijo ----
    m = {"well": wid, "cut": cut, "n": len(df), "n_pred": len(df) - cut,
         "lk": lk, "slope": C["_slope"],
         "nn_pre": float(np.median(C["nn"][:cut])),
         "sstd_pre": float(np.median(C["sstd"][:cut])),
         "gr_sig": float(P._gr_sig(df_h, tt, tg)),
         "gr_pre_m": float(np.nanmean(gr[:cut])), "gr_pre_s": float(np.nanstd(gr[:cut])),
         "tw_span": float(tt[-1] - tt[0]),
         "z_pre_s": float(np.std(z[max(0, cut - 500):cut]))}
    for c in CAND:
        m["d0" + c] = lk - float(C[c][cut - 1])
    for f in BT_FRACS:
        c2 = int(round(f * cut))
        tag = f"bt{int(f*100)}"
        if c2 < 60 or cut - c2 < 40:
            for c in CAND:
                m[f"{tag}_{c}"] = np.nan
            m[tag + "_n"] = 0
            continue
        d2 = df_h.iloc[:cut].copy()
        d2.loc[d2.index[c2:], "TVT_input"] = np.nan
        C2 = candidates(d2, tw, wid, c2, field, hd[:cut])
        yb = df_h.TVT_input.values[c2:cut]
        for c in CAND:
            m[f"{tag}_{c}"] = float(np.sqrt(np.mean((yb - C2[c][c2:cut]) ** 2)))
        m[tag + "_n"] = cut - c2
        m[tag + "_md"] = float(md[cut - 1] - md[c2])
    return r, m


def main():
    npool = int(sys.argv[1]) if len(sys.argv) > 1 else 350
    ids = well_ids()
    pool = sorted(set(_select(ids, 60, SEED)) | set(_select(ids, 150, SEED)))
    rng = np.random.default_rng(11)
    rest = [w for w in ids if w not in set(pool)]
    extra = rng.choice(rest, size=max(0, npool - len(pool)), replace=False)
    pool = sorted(set(pool) | set(extra))
    print(f"pool = {len(pool)} pozos", flush=True)

    field = M.SurfaceField()
    print(f"campo listo: {len(field.s)} puntos", flush=True)
    store = {k: [] for k in ROW}
    meta, lens = [], []
    t0 = time.time()
    for j, wid in enumerate(pool):
        out = one_well(wid, field)
        if out is None:
            continue
        r, m = out
        for k in ROW:
            store[k].append(r[k].astype(np.float32))
        lens.append(len(r["y"]))
        meta.append(m)
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(pool)}  {time.time()-t0:.0f}s", flush=True)
    d = {k: np.concatenate(v) for k, v in store.items()}
    d["lens"] = np.array(lens)
    np.savez(ROOT / "research/gbm01_cache.npz", **d)
    pd.DataFrame(meta).to_csv(ROOT / "research/gbm01_meta.csv", index=False)
    print(f"guardado: {len(lens)} pozos, {d['lens'].sum()} puntos, {time.time()-t0:.0f}s")
    for c in CAND:
        print(f"  RMSE {c:6s}: {np.sqrt(np.mean((d['y'] - (d['d'+c] + np.repeat([m['lk'] for m in meta], lens)))**2)):8.3f}")


if __name__ == "__main__":
    main()
