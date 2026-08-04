"""v6_offset: correlaciones univariantes + 3 estimadores del offset por pozo.

Entrena SOLO con los pozos split=train (400, disjuntos de k60/k150).
Objetivo exacto: minimizar sum_w n_pred_w * (off_true_w - off_hat_w)^2
(la reduccion del SSE pooled del blend es n*[off^2 - (off-off_hat)^2]).

Modelos:
  (a) shrink : off_hat = lambda * off_bt_X  (barrido de lambda y de la variante X)
  (b) huber  : HuberRegressor multivariante sobre features estandarizadas + gamma
  (c) lgbm   : LightGBM por pozo, KFold sobre pozos (1 fila/pozo = GroupKFold) + gamma

gamma (shrink global) se elige CONSERVADOR: el minimo gamma cuyo MSE OOF esta
a <=0.5% del optimo (mejor quedarse corto que sobrecorregir).

Salida: research/v6_offset_estimator.json (+ v6_offset_lgbm.txt) con TODO lo
necesario para reproducir off_hat en el kernel (imputacion, escalado, coefs).

Uso: .venv/bin/python research/v6_offset_fit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
DS = R / "v6_offset_ds.csv"

FEATS = ["off_bt", "off_bt_tail", "off_bt_end", "bt_slope", "bt_rmse", "bt_len", "has_bt",
         "sstd_mean", "sstd_p90", "ll_mean_n", "ll_std_n", "ll_rng_n",
         "nn_mean", "nn_p90", "nnr_mean", "nnr_p90",
         "dis_mean", "dis_abs", "dis_slope", "dis_std",
         "dip30", "dip200", "gr_res_std", "gr_corr",
         "n_pred", "cut_len", "ps_frac", "md_post"]
CLIP = 15.0          # tope de |off_hat| en ft (la mediana de |off_true| es ~4)


def wmse(y, yh, w):
    return float(np.sum(w * (y - yh) ** 2) / np.sum(w))


def gamma_cons(y, oof, w, tol=0.005):
    """gamma conservador: el MINIMO gamma con MSE OOF <= (1+tol)*optimo."""
    gs = np.arange(0.0, 1.21, 0.05)
    mses = np.array([wmse(y, g * oof, w) for g in gs])
    g_opt = float(gs[mses.argmin()])
    ok = np.where(mses <= mses.min() * (1 + tol))[0]
    return float(gs[ok[0]]), g_opt, mses


def kfold_idx(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    per = rng.permutation(n)
    return [(np.setdiff1d(per, f), f) for f in np.array_split(per, k)]


def main():
    df = pd.read_csv(DS)
    tr = df[df.split == "train"].reset_index(drop=True)
    ev = df[df.split == "eval"].reset_index(drop=True)
    print(f"train={len(tr)} eval={len(ev)} pozos")
    y, w = tr.off_true.values, tr.n_pred.values.astype(float)
    base = wmse(y, 0 * y, w)
    print(f"offset RMS ponderado (baseline off_hat=0): {np.sqrt(base):.3f} ft "
          f"| |off_true| mediana {tr.off_true.abs().median():.2f} ft\n")

    # imputacion con medianas de TRAIN (se guarda para el kernel)
    med = {c: float(tr[c].median()) for c in FEATS}
    Xtr = tr[FEATS].fillna(med).values.astype(float)

    # ---------------- 1. correlaciones univariantes ----------------
    print("== correlacion univariante feature -> off_true (train, n=%d) ==" % len(tr))
    rows = []
    for j, c in enumerate(FEATS):
        x = Xtr[:, j]
        pe = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else 0.0
        sp = pd.Series(x).corr(pd.Series(y), method="spearman")
        rows.append((c, pe, sp))
    for c, pe, sp in sorted(rows, key=lambda r: -abs(r[1])):
        mark = " <<<" if abs(pe) >= 0.3 else ""
        print(f"  {c:12s} pearson {pe:+.3f}  spearman {sp:+.3f}{mark}")

    folds = kfold_idx(len(tr))
    out = {"imputation": med, "clip": CLIP, "feats": FEATS, "models": {}}

    # ---------------- 2a. shrinkage simple ----------------
    print("\n== (a) shrinkage off_hat = lambda * off_bt_X ==")
    best_a = None
    for c in ("off_bt", "off_bt_tail", "off_bt_end"):
        x = tr[c].values
        for lam in np.arange(0.0, 1.51, 0.05):
            m = wmse(y, np.clip(lam * x, -CLIP, CLIP), w)
            if best_a is None or m < best_a[2]:
                best_a = (c, float(lam), m)
        lam_c = float(np.arange(0, 1.51, .05)[np.argmin(
            [wmse(y, np.clip(l_ * x, -CLIP, CLIP), w) for l_ in np.arange(0, 1.51, .05)])])
        print(f"  {c:12s} lambda* {lam_c:.2f} -> RMS resid "
              f"{np.sqrt(wmse(y, np.clip(lam_c*x,-CLIP,CLIP), w)):.3f} ft")
    c_a, lam_a, mse_a = best_a
    # conservador: recorta el lambda un 15%
    lam_fin = round(lam_a * 0.85, 3)
    mse_fin = wmse(y, np.clip(lam_fin * tr[c_a].values, -CLIP, CLIP), w)
    print(f"  MEJOR: {c_a} lambda*={lam_a:.2f} (in-sample {np.sqrt(mse_a):.3f}); "
          f"FINAL lambda={lam_fin:.3f} -> {np.sqrt(mse_fin):.3f} ft")
    out["models"]["shrink"] = {"feature": c_a, "lam": lam_fin,
                               "rms_insample": float(np.sqrt(mse_fin))}

    # ---------------- 2b. Huber multivariante ----------------
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    oof_h = np.zeros(len(tr))
    for itr, iva in folds:
        h = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500)
        h.fit(Xs[itr], y[itr], sample_weight=w[itr])
        oof_h[iva] = h.predict(Xs[iva])
    oof_h = np.clip(oof_h, -CLIP, CLIP)
    g_h, gopt_h, _ = gamma_cons(y, oof_h, w)
    print(f"\n== (b) huber ==  OOF RMS {np.sqrt(wmse(y, oof_h, w)):.3f} | "
          f"gamma cons {g_h:.2f} (opt {gopt_h:.2f}) -> {np.sqrt(wmse(y, g_h*oof_h, w)):.3f} ft")
    hf = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500)
    hf.fit(Xs, y, sample_weight=w)
    out["models"]["huber"] = {
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "coef": hf.coef_.tolist(), "intercept": float(hf.intercept_),
        "gamma": g_h, "rms_oof": float(np.sqrt(wmse(y, g_h * oof_h, w)))}

    # ---------------- 2c. LightGBM ----------------
    import lightgbm as lgb
    P = dict(objective="regression", learning_rate=0.03, num_leaves=15,
             min_child_samples=25, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
             n_estimators=1500, verbosity=-1, seed=42)
    oof_g, rounds = np.zeros(len(tr)), []
    for itr, iva in folds:
        m = lgb.LGBMRegressor(**P)
        m.fit(Xtr[itr], y[itr], sample_weight=w[itr],
              eval_set=[(Xtr[iva], y[iva])], eval_sample_weight=[w[iva]],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof_g[iva] = m.predict(Xtr[iva], num_iteration=m.best_iteration_)
        rounds.append(m.best_iteration_ or P["n_estimators"])
    oof_g = np.clip(oof_g, -CLIP, CLIP)
    g_g, gopt_g, _ = gamma_cons(y, oof_g, w)
    print(f"== (c) lgbm ==   OOF RMS {np.sqrt(wmse(y, oof_g, w)):.3f} | "
          f"gamma cons {g_g:.2f} (opt {gopt_g:.2f}) -> {np.sqrt(wmse(y, g_g*oof_g, w)):.3f} ft"
          f" | rounds {rounds}")
    nfin = max(int(np.mean(rounds)), 50)
    mf = lgb.LGBMRegressor(**{**P, "n_estimators": nfin})
    mf.fit(Xtr, y, sample_weight=w)
    mf.booster_.save_model(str(R / "v6_offset_lgbm.txt"))
    imp = sorted(zip(FEATS, mf.feature_importances_), key=lambda t: -t[1])[:10]
    print("  importancias:", ", ".join(f"{c}:{v}" for c, v in imp))
    out["models"]["lgbm"] = {"file": "v6_offset_lgbm.txt", "gamma": g_g,
                             "n_rounds": nfin,
                             "rms_oof": float(np.sqrt(wmse(y, g_g * oof_g, w)))}

    # ---------------- resumen ----------------
    print(f"\nbaseline {np.sqrt(base):.3f} | shrink {np.sqrt(mse_fin):.3f} | "
          f"huber(OOF,g) {np.sqrt(wmse(y, g_h*oof_h, w)):.3f} | "
          f"lgbm(OOF,g) {np.sqrt(wmse(y, g_g*oof_g, w)):.3f}  [RMS ft ponderado]")
    with open(R / "v6_offset_estimator.json", "w") as f:
        json.dump(out, f, indent=1)
    print("guardado research/v6_offset_estimator.json + v6_offset_lgbm.txt")


if __name__ == "__main__":
    sys.exit(main())
