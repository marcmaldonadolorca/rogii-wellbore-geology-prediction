"""v6 pesos por pozo — rig de evaluacion sobre los caches del GBM (v4_gbm_*.npz).

Miembros del blend (deltas vs last-known, convexos => equivalen al TVT):
  S = superficie aniso16 k24 theta+11 (LOWO)   P = PF ancc multiseed (S=16 en el
  cache; en eval se puede sustituir por P64 = mean S=64 de v4_multiseed_k*.npz)
  G = geometrico dip anclado win700.

Variantes (todas sin tocar el TVT real del pozo evaluado):
  invbt   : w_i ∝ bt_i^-alpha con los RMSE de backtest de cola enmascarada
            (BT_FRAC=0.65 del cache) + shrinkage lam hacia (0.45,0.55,0)
  reg     : regresion (ridge / LightGBM) de los pesos NNLS optimos reales,
            ajustada en los 380 pozos train (disjuntos de eval60/eval150)
  punto   : w(nn_real, md_since) por punto via softmax lineal (6 params),
            ajustado en los 380 pozos por SSE pooled

Baselines del rig (P64): eval150 9.978 | referencia oficial cv 10.200 (theta viejo).

Uso: python research/v6_pesos_rig.py [all|invbt|reg|punto|oracle]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
W_GLOBAL = np.array([0.45, 0.55, 0.0])          # (S, P, G)
LAMS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

FEATS = ["log_nn", "log_nn_aniso", "pstd", "cstd", "btS", "btP", "btG",
         "btBlend", "log_btPS", "log_npred", "log_cutn", "tw_span", "pfx_gr",
         "slp50", "slp700", "m_dPS", "m_dSG", "md_max"]


def load_set(which, p64=False):
    """Lista de pozos: dict(wid, y, A=(n,3) miembros S,P,G, feats)."""
    z = np.load(HERE / f"v4_gbm_{which}.npz")
    X = pd.DataFrame(z["X"], columns=[str(c) for c in z["cols"]])
    y, w, lk = z["y"].astype(np.float64), z["well"], z["lk"].astype(np.float64)
    ids = [str(i) for i in z["ids"]]
    ms = None
    if p64:
        tag = {"eval60": "k60", "eval150": "k150"}[which]
        ms = np.load(HERE / f"v4_multiseed_{tag}.npz")
    wells = []
    for j in np.unique(w):
        m = w == j
        wid = ids[j]
        S = X.dS.values[m].astype(np.float64)
        P = X.dP.values[m].astype(np.float64)
        G = X.dG.values[m].astype(np.float64)
        if ms is not None:
            P = ms[f"{wid}_pred"].astype(np.float64) - lk[m]
        f = {
            "log_nn": np.log1p(np.median(X.nn_real.values[m])),
            "log_nn_aniso": np.log1p(np.median(X.nn_aniso.values[m])),
            "pstd": float(np.mean(X.pstd.values[m])),
            "cstd": float(np.mean(X.cstd.values[m])),
            "btS": float(X.btS.values[m][0]), "btP": float(X.btP.values[m][0]),
            "btG": float(X.btG.values[m][0]),
            "btBlend": float(X.btBlend.values[m][0]),
            "log_btPS": np.log(max(float(X.btPS_ratio.values[m][0]), 1e-3)),
            "log_npred": np.log(float(X.npred.values[m][0])),
            "log_cutn": np.log(float(X.cutn.values[m][0])),
            "tw_span": float(X.tw_span.values[m][0]),
            "pfx_gr": float(X.pfx_gr.values[m][0]),
            "slp50": abs(float(X.slp50.values[m][0])),
            "slp700": abs(float(X.slp700.values[m][0])),
            "m_dPS": float(np.mean(np.abs(X.dPS.values[m]))),
            "m_dSG": float(np.mean(np.abs(X.dSG.values[m]))),
            "md_max": float(X.md_since.values[m].max()) / 1e3,
        }
        wells.append(dict(wid=wid, y=y[m], A=np.column_stack([S, P, G]), feats=f,
                          nn_med=float(np.median(X.nn_real.values[m])),
                          nn_pt=X.nn_real.values[m].astype(np.float64),
                          md_pt=X.md_since.values[m].astype(np.float64)))
    return wells


def pooled(wells, w_of):
    """RMSE pooled aplicando w_of(well) -> (3,) o (n,3)."""
    sse = n = 0.0
    for wl in wells:
        w = w_of(wl)
        p = (wl["A"] * w).sum(1) if np.ndim(w) == 2 else wl["A"] @ w
        sse += float(((wl["y"] - p) ** 2).sum())
        n += len(wl["y"])
    return float(np.sqrt(sse / n))


def nnls_w(A, y):
    """Pesos NNLS normalizados a suma 1 (convexos salvo renorm)."""
    w, _ = nnls(A, y)
    s = w.sum()
    return w / s if s > 0 else W_GLOBAL.copy()


def clip_norm(w):
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 1e-9 else W_GLOBAL.copy()


def shrink_eval(wells, w_by_wid, label, rows, tag):
    for lam in LAMS:
        r = pooled(wells, lambda wl: clip_norm(
            (1 - lam) * W_GLOBAL + lam * w_by_wid[wl["wid"]]))
        rows.append({"variante": f"{label} lam={lam:.1f}", "set": tag, "rmse": r})
        print(f"  {label:28s} lam={lam:.1f} -> {r:8.3f}", flush=True)


def feat_mat(wells):
    return np.array([[wl["feats"][f] for f in FEATS] for wl in wells])


# ------------------------------------------------------------------ variantes
def var_invbt(wells, rows, tag):
    for alpha in (1.0, 2.0, 4.0):
        for use_g in (False, True):
            w_by = {}
            for wl in wells:
                bt = np.array([wl["feats"]["btS"], wl["feats"]["btP"],
                               wl["feats"]["btG"]])
                if not np.isfinite(bt[:2]).all() or (bt[:2] <= 0).any():
                    w_by[wl["wid"]] = W_GLOBAL.copy(); continue
                if not use_g or not np.isfinite(bt[2]) or bt[2] <= 0:
                    bt = bt[:2]
                v = bt ** -alpha
                v = v / v.sum()
                w_by[wl["wid"]] = np.array([v[0], v[1], v[2] if len(v) > 2 else 0.0])
            shrink_eval(wells, w_by, f"invbt a={alpha:g} G={int(use_g)}", rows, tag)


def oracle_targets(wells):
    return {wl["wid"]: nnls_w(wl["A"], wl["y"]) for wl in wells}


def var_reg(train, evals, rows):
    """ridge y LightGBM: features pozo -> pesos NNLS oraculo del train."""
    tw = oracle_targets(train)
    Xtr = feat_mat(train)
    Ytr = np.array([tw[wl["wid"]] for wl in train])          # (380,3)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xn = (Xtr - mu) / sd

    # ridge multioutput sobre (wS, wG); wP = 1 - wS - wG
    lam_r = 10.0
    A = Xn.T @ Xn + lam_r * np.eye(Xn.shape[1])
    B = np.linalg.solve(A, Xn.T @ Ytr[:, [0, 2]])
    b0 = Ytr[:, [0, 2]].mean(0) - (Xn.mean(0) @ B)

    import lightgbm as lgb
    models = []
    for c in (0, 2):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=15,
                              min_child_samples=25, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=2.0, verbose=-1,
                              seed=42, n_jobs=4)
        m.fit(Xtr, Ytr[:, c])
        models.append(m)

    def w_ridge(F):
        z = (F - mu) / sd
        sg = z @ B + b0
        return np.array([sg[0], 1.0 - sg[0] - sg[1], sg[1]])

    def w_gbm(F):
        s, g = models[0].predict(F[None])[0], models[1].predict(F[None])[0]
        return np.array([s, 1.0 - s - g, g])

    for tag, wells in evals.items():
        for name, fw in (("reg-ridge", w_ridge), ("reg-lgbm", w_gbm)):
            w_by = {wl["wid"]: clip_norm(fw(np.array([wl["feats"][f] for f in FEATS])))
                    for wl in wells}
            print(f"[{tag}] {name}: wS med={np.median([w_by[w['wid']][0] for w in wells]):.2f}")
            shrink_eval(wells, w_by, name, rows, tag)
    return dict(mu=mu.tolist(), sd=sd.tolist(), B=B.tolist(), b0=b0.tolist())


def var_punto(train, evals, rows):
    """softmax lineal por punto: scores_S/G = c0+c1*z_nn+c2*z_md (P referencia)."""
    def prep(wl):
        return np.column_stack([np.log1p(wl["nn_pt"]), wl["md_pt"] / 1e3])
    Ztr = np.concatenate([prep(wl) for wl in train])
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9

    def weights(theta, Z):
        z = (Z - mu) / sd
        sS = theta[0] + z @ theta[1:3]
        sG = theta[3] + z @ theta[4:6]
        e = np.exp(np.column_stack([sS, np.zeros(len(z)), sG])
                   - np.maximum.reduce([sS, np.zeros(len(z)), sG])[:, None])
        return e / e.sum(1, keepdims=True)

    def loss(theta):
        sse = n = 0.0
        for wl in train:
            W = weights(theta, prep(wl))
            p = (wl["A"] * W).sum(1)
            sse += float(((wl["y"] - p) ** 2).sum()); n += len(wl["y"])
        return np.sqrt(sse / n)

    x0 = np.array([np.log(0.45 / 0.55), 0, 0, -4.0, 0, 0])
    res = minimize(loss, x0, method="Powell",
                   options=dict(maxiter=4000, xtol=1e-3, ftol=1e-4))
    print(f"punto: train rmse {res.fun:.3f}  theta={np.round(res.x, 3)}", flush=True)
    for tag, wells in evals.items():
        r = pooled(wells, lambda wl: weights(res.x, prep(wl)))
        rows.append({"variante": "punto softmax(nn,md)", "set": tag, "rmse": r})
        print(f"  [{tag}] punto softmax -> {r:8.3f}", flush=True)
    return dict(theta=res.x.tolist(), mu=mu.tolist(), sd=sd.tolist())


def var_oracle(evals, rows):
    for tag, wells in evals.items():
        tw = oracle_targets(wells)
        r = pooled(wells, lambda wl: tw[wl["wid"]])
        rows.append({"variante": "ORACULO nnls pozo", "set": tag, "rmse": r})
        print(f"  [{tag}] oraculo nnls (S,P,G) -> {r:8.3f}")


def strat_nn(wells, w_of, label):
    """RMSE por quintil de nn_med: el regimen aislado no debe colapsar."""
    nn = np.array([wl["nn_med"] for wl in wells])
    qs = np.quantile(nn, [0.2, 0.4, 0.6, 0.8])
    grp = np.digitize(nn, qs)
    out = []
    for g in range(5):
        sub = [wl for wl, gg in zip(wells, grp) if gg == g]
        out.append(pooled(sub, w_of))
    print(f"  {label}: quintiles nn {' '.join(f'{v:.2f}' for v in out)}")
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    train = load_set("train")
    evals = {}
    for which, tag in (("eval60", "k60_P16"), ("eval150", "k150_P16")):
        evals[tag] = load_set(which)
    for which, tag in (("eval60", "k60_P64"), ("eval150", "k150_P64")):
        evals[tag] = load_set(which, p64=True)

    rows = []
    for tag, wells in evals.items():
        r = pooled(wells, lambda wl: W_GLOBAL)
        rows.append({"variante": "BASE fija .45/.55", "set": tag, "rmse": r})
        print(f"[{tag}] base fija -> {r:8.3f}")

    if cmd in ("all", "oracle"):
        var_oracle(evals, rows)
    if cmd in ("all", "invbt"):
        for tag, wells in evals.items():
            print(f"-- invbt {tag}")
            var_invbt(wells, rows, tag)
    params = {}
    if cmd in ("all", "reg"):
        params["reg"] = var_reg(train, evals, rows)
    if cmd in ("all", "punto"):
        params["punto"] = var_punto(train, evals, rows)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "v6_pesos_rig.csv", index=False)
    if params:
        (HERE / "v6_pesos_params.json").write_text(json.dumps(params, indent=1))
    print("\nTOP por set:")
    for tag in evals:
        d = df[df.set == tag].sort_values("rmse")
        print(d.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
