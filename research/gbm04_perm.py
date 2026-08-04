"""Permutation importance sobre OOF: cuanto sube el RMSE al barajar cada feature.

Para cada fold usa el modelo que NO vio esos pozos y baraja la columna en las
filas de validacion (stride 20). Delta positivo = la feature aporta de verdad.

Uso: python research/gbm04_perm.py [tag]   (tag = res | raw)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import gbm_lib as L  # noqa: E402
from gbm02_fit import load_cache, NFOLD  # noqa: E402

STRIDE = 20


def main():
    import lightgbm as lgb
    tag = sys.argv[1] if len(sys.argv) > 1 else "res"
    d = np.load(ROOT / "research/gbm02_oof.npz")
    wfold = d["wfold"]
    wells = load_cache()
    rng = np.random.default_rng(0)

    Xs, ys, fs = [], [], []
    for i, (m, r) in enumerate(wells):
        f = L.features(r, m).iloc[::STRIDE]
        t = (r["y"].astype(np.float64) - m["lk"])[::STRIDE]
        if tag == "res":
            t = t - f["blend"].values
        Xs.append(f.astype(np.float32)); ys.append(t); fs.append(np.full(len(f), wfold[i]))
    X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys); fold = np.concatenate(fs)
    models = [lgb.Booster(model_file=str(ROOT / f"research/gbm02_{tag}_f{k}.txt")) for k in range(NFOLD)]

    base = np.empty(len(X))
    for k in range(NFOLD):
        mk = fold == k
        base[mk] = models[k].predict(X[mk])
    b0 = float(np.sqrt(np.mean((y - base) ** 2)))
    print(f"RMSE OOF base (stride {STRIDE}, target {tag}) = {b0:.4f}  rows={len(X)}")

    rows = []
    for c in X.columns:
        orig = X[c].values.copy()
        X[c] = rng.permutation(orig)
        p = np.empty(len(X))
        for k in range(NFOLD):
            mk = fold == k
            p[mk] = models[k].predict(X[mk])
        rows.append({"f": c, "delta": float(np.sqrt(np.mean((y - p) ** 2))) - b0})
        X[c] = orig
    imp = pd.DataFrame(rows).sort_values("delta", ascending=False)
    imp.to_csv(ROOT / f"research/gbm04_perm_{tag}.csv", index=False)
    print(imp.head(30).to_string(index=False))
    print("\ncola (aportan 0 o ruido):")
    print(imp.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
