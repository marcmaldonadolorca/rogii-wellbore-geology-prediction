"""V4-GBM: LightGBM que ARBITRA entre candidatos por punto (linea del publico 9.9).

Dataset por punto post-PS. Target dTVT = TVT - last_known_tvt. Features en delta
respecto a last_known_tvt:
  candidatos  : S (superficie aniso16 k24, LOWO), P (pf_ancc), Z (pf_z),
                B (beam cfg0), G (geometrico dip anclado win700)
  confianzas  : nn_real (cKDTree SIN escala, LOWO), nn_aniso (proxy del interp),
                std del PF (ancc y z), y el RMSE de CADA candidato re-prediciendo
                el tramo [0.65*cut : cut] del propio prefijo (leak-free)
  desacuerdos : |P-S|, |P-Z|, S-G, S-B, std y media del stack de candidatos
  contexto    : md_since, hd_since, frac, dz, dz/dMD, GR rolling 21/101,
                GR vs typewell evaluado en el candidato superficie, slopes prefijo

Pozos de entrenamiento: 380 del pool de 574 que NO estan en cv._select k=60 ni
k=150 => la evaluacion con cv.py es sobre pozos jamas vistos por el GBM.

Uso (en orden):
  python research/v4_gbm.py cache train     -> v4_gbm_train.npz   (~8 min)
  python research/v4_gbm.py cache eval60    -> v4_gbm_eval60.npz
  python research/v4_gbm.py cache eval150   -> v4_gbm_eval150.npz
  python research/v4_gbm.py fit             -> modelo + OOF (GroupKFold 5 por pozo)
  python research/v4_gbm.py sweep           -> postproceso barrido en OOF, aplica a eval
  python research/v4_gbm.py perm            -> permutation importance (eval60)
  python research/v4_gbm.py cv 60|150       -> cv.evaluate LIVE (confirmacion final)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from cv import SEED, TEST_COLS, _select, load_well, well_ids  # noqa: E402
import model as M  # noqa: E402
import pf_publico as PF  # noqa: E402

HERE = ROOT / "research"
ANISO = 16.0
THETA_OPT = 2.278489      # sur02_lib.THETA (PCA) + 11 grados — v4_supmax, confirmado k150
PF_SEEDS = 16             # PF multiseed media (v4_multiseed); 1 = comportamiento antiguo
DIP_WIN = 700
BT_FRAC = 0.65
N_TRAIN_WELLS = 380
LGB_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=96, min_child_samples=60, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0,
                  reg_alpha=0.05, max_bin=255, verbose=-1, n_jobs=6, seed=42)
N_ROUNDS, EARLY = 3000, 120

_field = None
_real_tree_cache = {}


def get_field():
    global _field
    if _field is None:
        _field = M.SurfaceField(aniso=ANISO, theta=THETA_OPT)
    return _field


def real_nn(field, wid, X, Y):
    """Distancia REAL (sin escala anisotropa) al vecino mas cercano, LOWO."""
    if wid in field.idx:
        key = wid
        if _real_tree_cache.get("wid") != key:
            keep = field.wid != field.idx[wid]
            _real_tree_cache.update(wid=key, tree=cKDTree(field.xy_raw[keep]))
        tree = _real_tree_cache["tree"]
    else:
        if "full" not in _real_tree_cache:
            _real_tree_cache["full"] = cKDTree(field.xy_raw)
        tree = _real_tree_cache["full"]
    d, _ = tree.query(np.column_stack([X, Y]), k=1, workers=4)
    return d


def hdist(df):
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=DIP_WIN):
    """Recta de dip anclada en el PS (misma matematica que research/afinar_dip anchor700)."""
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
    """S,P,Z,B,G sobre las filas post-cut + extras (nn, stds, slope)."""
    s_full, nn_aniso = M._prior(df_h, field, wid, cut)
    t, g = PF._tw(tw)
    if PF_SEEDS > 1:
        from ban_lib import run_pf_ancc_multi
        pts, _ll = run_pf_ancc_multi(df_h, t, g,
                                     seeds=np.arange(1, PF_SEEDS + 1, dtype=np.int64))
        P, Pstd = pts.mean(0).astype(float), pts.std(0).astype(float)
    else:
        P, Pstd = PF.run_pf_ancc(df_h, t, g)
    Zp, Zstd = PF.run_pf_z(df_h, t, g)
    bs, mc, es, r, _ = PF.BEAMS[0]
    B = PF.beam_search(df_h.GR.values[cut:], t, g,
                       float(df_h.TVT_input.values[cut - 1]), bs, mc, es, r)
    Gfull, slope = geom_pred(df_h, cut, hd)
    return dict(S=s_full[cut:], P=np.asarray(P, float), Z=np.asarray(Zp, float),
                B=np.asarray(B, float), G=Gfull[cut:],
                Pstd=np.asarray(Pstd, float), Zstd=np.asarray(Zstd, float),
                nn_aniso=nn_aniso[cut:], slope=slope, tw=(t, g))


def backtest_rmse(df_h, tw, wid, cut, field, hd):
    """RMSE leak-free de cada candidato re-prediciendo [c2:cut] del prefijo."""
    c2 = int(round(BT_FRAC * cut))
    out = {f"bt{n}": np.nan for n in "SPZBG"}
    out["btBlend"] = np.nan
    if c2 < 60 or cut - c2 < 30:
        return out
    d2 = df_h.iloc[:cut].copy()
    d2.loc[d2.index[c2:], "TVT_input"] = np.nan
    y = df_h.TVT_input.values[c2:cut]
    try:
        c = candidates(d2, tw, wid, c2, field, hd[:cut])
    except Exception:
        return out
    for n in "SPZBG":
        out[f"bt{n}"] = float(np.sqrt(np.mean((y - c[n]) ** 2)))
    blend = 0.25 * c["S"] + 0.75 * c["P"]
    out["btBlend"] = float(np.sqrt(np.mean((y - blend) ** 2)))
    return out


def build_well(df_h, tw, wid, field):
    """DataFrame de features por punto post-PS + aux. None si el pozo no vale."""
    m = df_h.TVT_input.isna()
    cut = int(m.idxmax()) if m.any() else len(df_h)
    if cut < 20 or cut >= len(df_h):
        return None
    hd = hdist(df_h)
    c = candidates(df_h, tw, wid, cut, field, hd)
    bt = backtest_rmse(df_h, tw, wid, cut, field, hd)
    n = len(df_h) - cut
    lk = float(df_h.TVT_input.values[cut - 1])
    t, g = c["tw"]

    # calibracion afin GR pozo vs typewell en el prefijo
    gr = df_h.GR.values
    g_pref = np.interp(df_h.TVT_input.values[:cut], t, g)
    ok = np.isfinite(gr[:cut]) & np.isfinite(g_pref)
    a, b, pfx_gr = 1.0, 0.0, np.nan
    if ok.sum() >= 40:
        A = np.column_stack([g_pref[ok], np.ones(ok.sum())])
        a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
        pfx_gr = float(np.std(gr[:cut][ok] - (a * g_pref[ok] + b)))

    # slope de los ultimos 50 puntos del prefijo (TVT_input vs hd)
    slp50 = 0.0
    hh, tt = hd[max(0, cut - 50):cut], df_h.TVT_input.values[max(0, cut - 50):cut]
    if len(hh) > 10 and hh[-1] != hh[0]:
        slp50 = float(np.polyfit(hh - hh[0], tt - tt[0], 1)[0])

    S, P, Zc, B, G = (c[k] for k in "SPZBG")
    stack = np.stack([S, P, Zc, B, G])
    grs = pd.Series(gr)
    r21 = grs.rolling(21, center=True, min_periods=1)
    r101 = grs.rolling(101, center=True, min_periods=1)
    mdv = df_h.MD.values
    zv = df_h.Z.values
    dz_dmd = pd.Series(np.gradient(zv, np.maximum(mdv, 1e-9))).rolling(
        21, center=True, min_periods=1).mean().values
    twgr_S = a * np.interp(S, t, g) + b
    twgr_P = a * np.interp(P, t, g) + b
    grpost = gr[cut:]
    nn_real = real_nn(field, wid, df_h.X.values[cut:], df_h.Y.values[cut:])
    sc = lambda v: np.full(n, np.float32(v))

    X = pd.DataFrame({
        "dS": S - lk, "dP": P - lk, "dZ": Zc - lk, "dB": B - lk, "dG": G - lk,
        "dPS": P - S, "dPZ": P - Zc, "dSG": S - G, "dSB": S - B,
        "cstd": stack.std(0), "cmean": stack.mean(0) - lk,
        "nn_real": nn_real, "nn_aniso": c["nn_aniso"],
        "pstd": c["Pstd"], "zstd": c["Zstd"],
        "md_since": mdv[cut:] - mdv[cut - 1],
        "hd_since": hd[cut:] - hd[cut - 1],
        "frac": np.arange(n) / max(n - 1, 1),
        "dz": zv[cut:] - zv[cut - 1], "dzdmd": dz_dmd[cut:],
        "gr": grpost,
        "grm21": r21.mean().values[cut:], "grs21": r21.std().values[cut:],
        "grm101": r101.mean().values[cut:], "grs101": r101.std().values[cut:],
        "gres_S": grpost - twgr_S, "twgr_S": twgr_S, "twgr_P": twgr_P,
        "gr_anchor": grpost - (a * np.interp(lk, t, g) + b),
        "lk": sc(lk),
        "btS": sc(bt["btS"]), "btP": sc(bt["btP"]), "btZ": sc(bt["btZ"]),
        "btB": sc(bt["btB"]), "btG": sc(bt["btG"]), "btBlend": sc(bt["btBlend"]),
        "btPS_ratio": sc(bt["btP"] / (bt["btS"] + 1e-6)),
        "pfx_gr": sc(pfx_gr), "cutn": sc(cut), "npred": sc(n),
        "slp700": sc(c["slope"]), "slp50": sc(slp50),
        "tw_span": sc(float(t.max() - t.min())),
    }).astype(np.float32)
    aux = dict(wid=wid, lk=lk, cut=cut)
    return X, aux


FEATS = None  # se fija con la primera build


def build_cache(which):
    ids_all = well_ids()
    k60 = set(_select(ids_all, 60, SEED))
    k150 = set(_select(ids_all, 150, SEED))
    if which == "train":
        pool = [w for w in ids_all if w not in (k60 | k150)]
        pick = np.round(np.linspace(0, len(pool) - 1, N_TRAIN_WELLS)).astype(int)
        ids = [pool[i] for i in pick]
    elif which == "eval60":
        ids = sorted(k60)
    elif which == "eval150":
        ids = sorted(k150)
    else:
        raise SystemExit(f"cache {which}?")
    field = get_field()
    Xs, ys, wells, lks = [], [], [], []
    t0 = time.time()
    for j, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[TEST_COLS].copy()
        out = build_well(df_h, tw, wid, field)
        if out is None:
            continue
        X, aux = out
        Xs.append(X)
        ys.append((df.TVT.values[cut:] - aux["lk"]).astype(np.float32))
        wells.append(np.full(len(X), j, np.int32))
        lks.append(np.full(len(X), aux["lk"], np.float32))
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    Xall = pd.concat(Xs, ignore_index=True)
    outf = HERE / f"v4_gbm_{which}.npz"
    np.savez_compressed(outf, X=Xall.values.astype(np.float32),
                        cols=np.array(list(Xall.columns)),
                        y=np.concatenate(ys), well=np.concatenate(wells),
                        lk=np.concatenate(lks), ids=np.array(ids))
    print(f"guardado {outf}: {Xall.shape} pozos={len(Xs)} "
          f"({(time.time()-t0)/max(len(Xs),1):.2f}s/pozo)")


def load_cache(which):
    z = np.load(HERE / f"v4_gbm_{which}.npz", allow_pickle=False)
    X = pd.DataFrame(z["X"], columns=[str(c) for c in z["cols"]])
    return X, z["y"].astype(np.float64), z["well"], z["lk"].astype(np.float64)


def pooled_rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def base_delta(X, variant):
    """Delta base al que el GBM añade su salida. 'abs' = 0 (target dTVT crudo);
    'resid' = blend 0.25S+0.75P (el GBM aprende la correccion al blend)."""
    if variant == "resid":
        # blend optimo con superficie theta+11 y PF multiseed (v4_supmax + suite)
        return 0.45 * X.dS.values.astype(np.float64) + 0.55 * X.dP.values.astype(np.float64)
    if variant == "resid_s2d":
        # base = blend adaptativo con sigma_s 2D multiplicativa (v8_sig2d)
        z = np.load(HERE / "v8_s2d_tablas.npz")
        nnb, mdb = z["nn_bins"], z["md_bins"]
        smult, sp1 = z["s_mult"], z["sig_p_1d"]
        def cent(bins):
            return 0.5 * (bins[:-1] + np.minimum(bins[1:], bins[-2] * 2))
        cr, cc = cent(nnb), cent(mdb)
        ri = np.clip(np.searchsorted(cr, X.nn_aniso.values), 0, len(cr) - 1)
        ci = np.clip(np.searchsorted(cc, X.md_since.values), 0, len(cc) - 1)
        ss = smult[ri, ci]
        sp = np.interp(X.md_since.values, cc, sp1)
        w = np.clip(sp**2 / (ss**2 + sp**2), 0.05, 0.9)
        return w * X.dS.values.astype(np.float64) + (1 - w) * X.dP.values.astype(np.float64)
    if variant == "resid_adapt":
        # base = blend adaptativo por varianza inversa (v7_wadapt, curvas en v7_curvas.npz)
        z = np.load(HERE / "v7_curvas.npz")
        nnb, mdb, ss_b, sp_b = z["nn_bins"], z["md_bins"], z["sig_s"], z["sig_p"]
        def icurve(x, bins, vals):
            c = 0.5 * (bins[:-1] + np.minimum(bins[1:], bins[-2] * 2))
            return np.interp(x, c, vals)
        ss = icurve(X.nn_aniso.values, nnb, ss_b)
        sp = icurve(X.md_since.values, mdb, sp_b)
        w = np.clip(sp**2 / (ss**2 + sp**2), 0.05, 0.9)
        return w * X.dS.values.astype(np.float64) + (1 - w) * X.dP.values.astype(np.float64)
    return np.zeros(len(X))


def fit(variant="abs"):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    X, y, well, lk = load_cache("train")
    print(f"train: {X.shape}, {len(np.unique(well))} pozos, variant={variant}")
    b = base_delta(X, variant)
    t = y - b
    oof = np.zeros(len(y))
    iters = []
    t0 = time.time()
    for f, (tr, va) in enumerate(GroupKFold(5).split(X, t, well)):
        dtr = lgb.Dataset(X.iloc[tr], label=t[tr])
        dva = lgb.Dataset(X.iloc[va], label=t[va], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, valid_sets=[dva], num_boost_round=N_ROUNDS,
                      callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  fold{f}: rmse_d={pooled_rmse(y[va], b[va] + oof[va]):.3f} "
              f"iter={m.best_iteration} {time.time()-t0:.0f}s", flush=True)
    print(f"OOF rmse (delta total): {pooled_rmse(y, b + oof):.3f}")
    best_it = int(np.median(iters) * 1.1)
    final = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=t), num_boost_round=best_it)
    final.save_model(str(HERE / f"v4_gbm_model_{variant}.txt"))
    np.savez_compressed(HERE / f"v4_gbm_oof_{variant}.npz",
                        oof=(b + oof).astype(np.float32),
                        y=y.astype(np.float32), well=well,
                        md=X.md_since.values, dP=X.dP.values)
    print(f"final: {best_it} iters -> v4_gbm_model_{variant}.txt  ({time.time()-t0:.0f}s)")


def postprocess(d, dP, md, well, alpha, tau, w_pf, sg):
    out = d * (1.0 - w_pf) + dP * w_pf
    if tau:
        out = out * (1.0 - np.exp(-np.maximum(md, 0.0) / tau))
    out = out * alpha
    if sg:
        out = out.copy()
        for w in np.unique(well):
            i = np.where(well == w)[0]
            v = out[i]
            wl = min(17, len(v) if len(v) % 2 else len(v) - 1)
            if wl >= 5:
                out[i] = savgol_filter(v, wl, 3)
    return out


def sweep(variant="abs"):
    import lightgbm as lgb
    z = np.load(HERE / f"v4_gbm_oof_{variant}.npz")
    oof, y, well, md, dP = (z[k].astype(np.float64) for k in ("oof", "y", "well", "md", "dP"))
    print(f"[{variant}] OOF crudo: {pooled_rmse(y, oof):.3f}  | pf_ancc solo: {pooled_rmse(y, dP):.3f}")
    best, br = None, np.inf
    rows = []
    for alpha in (0.8, 0.85, 0.9, 0.95, 1.0):
        for tau in (None, 50.0, 85.0, 150.0):
            for w_pf in (0.0, 0.05, 0.1, 0.15, 0.2):
                for sg in (False, True):
                    r = pooled_rmse(y, postprocess(oof, dP, md, well, alpha, tau, w_pf, sg))
                    rows.append((alpha, tau, w_pf, sg, r))
                    if r < br:
                        br, best = r, dict(alpha=alpha, tau=tau, w_pf=w_pf, sg=sg)
    print(f"mejor postproceso en OOF: {best} -> {br:.3f}")
    for r in sorted(rows, key=lambda x: x[-1])[:8]:
        print(f"   alpha={r[0]:.2f} tau={r[1]} w_pf={r[2]:.2f} sg={r[3]} -> {r[4]:.3f}")
    (HERE / f"v4_gbm_post_{variant}.json").write_text(json.dumps(best))

    booster = lgb.Booster(model_file=str(HERE / f"v4_gbm_model_{variant}.txt"))
    for which in ("eval60", "eval150"):
        f = HERE / f"v4_gbm_{which}.npz"
        if not f.exists():
            continue
        X, ye, we, lke = load_cache(which)
        d = base_delta(X, variant) + booster.predict(X)
        raw = pooled_rmse(ye, d)
        post = pooled_rmse(ye, postprocess(d, X.dP.values.astype(np.float64),
                                           X.md_since.values.astype(np.float64),
                                           we, **best))
        refS = pooled_rmse(ye, X.dS.values)
        refP = pooled_rmse(ye, X.dP.values)
        refB = pooled_rmse(ye, 0.25 * X.dS.values + 0.75 * X.dP.values)
        print(f"{which}: GBM crudo {raw:.3f} | GBM+post {post:.3f} || "
              f"refs S {refS:.3f} P {refP:.3f} blend .25/.75 {refB:.3f}", flush=True)


def perm(variant="abs"):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(HERE / f"v4_gbm_model_{variant}.txt"))
    X, y, well, lk = load_cache("eval60")
    b = base_delta(X, variant)
    base = pooled_rmse(y, b + booster.predict(X))
    rng = np.random.default_rng(0)
    rows = []
    for c in X.columns:
        Xp = X.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        rows.append((c, pooled_rmse(y, base_delta(Xp, variant) + booster.predict(Xp)) - base))
    rows.sort(key=lambda r: -r[1])
    print(f"base eval60 (crudo, {variant}): {base:.3f}")
    for c, d in rows:
        print(f"  {c:12s} {d:+8.3f}")


def make_cv_predictor(variant="abs"):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(HERE / f"v4_gbm_model_{variant}.txt"))
    post = json.loads((HERE / f"v4_gbm_post_{variant}.json").read_text())
    field = get_field()

    def predict(df_h, tw, wid=None):
        out = build_well(df_h, tw, wid, field)
        assert out is not None
        X, aux = out
        d = base_delta(X, variant) + booster.predict(X)
        d = postprocess(d, X.dP.values.astype(np.float64),
                        X.md_since.values.astype(np.float64),
                        np.zeros(len(X)), **post)
        return aux["lk"] + d

    return predict


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    var = sys.argv[3] if len(sys.argv) > 3 else (
        sys.argv[2] if cmd in ("fit", "sweep", "perm") and len(sys.argv) > 2 else "abs")
    if cmd == "cache":
        build_cache(sys.argv[2])
    elif cmd == "fit":
        fit(var)
    elif cmd == "sweep":
        sweep(var)
    elif cmd == "perm":
        perm(var)
    elif cmd == "cv":
        from cv import evaluate
        evaluate(make_cv_predictor(var), k=int(sys.argv[2]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
