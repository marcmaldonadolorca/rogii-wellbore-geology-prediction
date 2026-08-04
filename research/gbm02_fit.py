"""GBM arbitro: entrena LightGBM sobre el cache por punto y mide OOF honesto.

GroupKFold(5) POR POZO. Entrena con filas submuestreadas (stride) de los pozos
de train del fold; predice TODAS las filas post-PS de los pozos held-out del
fold => OOF a resolucion completa, comparable punto a punto con cv.evaluate.

Dos targets:
  raw : dTVT              -> pred = lk + gbm
  res : dTVT - blend      -> pred = lk + blend + gbm   (blend = 0.25 surf + 0.75 ancc)

Uso: python research/gbm02_fit.py [stride] [n_estimators]
Salida: research/gbm02_oof.npz (predicciones OOF + fold por pozo) y modelos .txt
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import gbm_lib as L  # noqa: E402
from gbm01_cache import ROW  # noqa: E402

NFOLD = 5
PARAMS = dict(objective="regression", metric="l2", learning_rate=0.05,
              num_leaves=63, min_child_samples=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              num_threads=8, verbose=-1, seed=42)


def load_cache():
    d = np.load(ROOT / "research/gbm01_cache.npz")
    meta = pd.read_csv(ROOT / "research/gbm01_meta.csv")
    lens = d["lens"]
    off = np.concatenate([[0], np.cumsum(lens)])
    wells = []
    for i in range(len(lens)):
        sl = slice(int(off[i]), int(off[i + 1]))
        r = {k: d[k][sl] for k in ROW}
        wells.append((meta.iloc[i].to_dict(), r))
    return wells


def build_all(wells, stride):
    """Matriz de entrenamiento submuestreada + indices de pozo."""
    Xs, ys, gs = [], [], []
    for i, (m, r) in enumerate(wells):
        f = L.features(r, m)
        t = r["y"].astype(np.float64) - m["lk"]
        Xs.append(f.iloc[::stride].astype(np.float32))
        ys.append(t[::stride])
        gs.append(np.full(len(Xs[-1]), i))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(gs)


def main():
    import lightgbm as lgb
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    nest = int(sys.argv[2]) if len(sys.argv) > 2 else 700
    t0 = time.time()
    wells = load_cache()
    print(f"pozos={len(wells)} puntos={sum(len(r['y']) for _, r in wells)}", flush=True)
    X, y, g = build_all(wells, stride)
    print(f"train rows={len(X)} feats={X.shape[1]}  {time.time()-t0:.0f}s", flush=True)

    gkf = GroupKFold(n_splits=NFOLD)
    fold_of = np.zeros(len(wells), dtype=int)
    for k, (_, va) in enumerate(gkf.split(np.arange(len(wells)), groups=np.arange(len(wells)))):
        fold_of[va] = k
    wfold = np.array([fold_of[i] for i in range(len(wells))])
    rowfold = wfold[g]

    models = {}
    for tag in ("raw", "res"):
        for k in range(NFOLD):
            tr = rowfold != k
            yy = y if tag == "raw" else y - X["blend"].values.astype(np.float64)
            ds = lgb.Dataset(X[tr], label=yy[tr])
            m = lgb.train(PARAMS, ds, num_boost_round=nest)
            models[(tag, k)] = m
            print(f"  {tag} fold{k} listo  {time.time()-t0:.0f}s", flush=True)
        models[(tag, "last")] = models[(tag, NFOLD - 1)]

    # ---- OOF a resolucion completa ----
    out = {t: [] for t in ("raw", "res")}
    base = {n: [] for n in ("blend", "surf", "ancc")}
    ys, wid_idx = [], []
    for i, (m, r) in enumerate(wells):
        f = L.features(r, m)
        k = wfold[i]
        Xi = f.astype(np.float32)
        out["raw"].append(models[("raw", k)].predict(Xi))
        out["res"].append(f["blend"].values + models[("res", k)].predict(Xi))
        base["blend"].append(f["blend"].values)
        base["surf"].append(f["dsurf"].values)
        base["ancc"].append(f["dancc"].values)
        ys.append(r["y"].astype(np.float64) - m["lk"])
        wid_idx.append(np.full(len(f), i))
    res = {f"pred_{t}": np.concatenate(v).astype(np.float32) for t, v in out.items()}
    for n, v in base.items():
        res[f"base_{n}"] = np.concatenate(v).astype(np.float32)
    res["y"] = np.concatenate(ys).astype(np.float32)
    res["wi"] = np.concatenate(wid_idx).astype(np.int32)
    res["wfold"] = wfold
    res["lens"] = np.array([len(r["y"]) for _, r in wells])
    np.savez(ROOT / "research/gbm02_oof.npz", **res)
    for tag in ("raw", "res"):
        for k in range(NFOLD):
            models[(tag, k)].save_model(str(ROOT / f"research/gbm02_{tag}_f{k}.txt"))
    X.columns.to_series().to_csv(ROOT / "research/gbm02_feats.csv", index=False, header=False)

    rm = lambda v: float(np.sqrt(np.mean((res["y"] - v) ** 2)))
    print(f"\n== pooled sobre los {len(wells)} pozos del pool (OOF) ==")
    for n in ("base_surf", "base_ancc", "base_blend", "pred_raw", "pred_res"):
        print(f"  {n:10s} {rm(res[n]):8.3f}")

    # importancia
    imp = pd.DataFrame({"f": X.columns})
    for tag in ("raw", "res"):
        imp[tag] = np.mean([models[(tag, k)].feature_importance("gain") for k in range(NFOLD)], axis=0)
    imp = imp.sort_values("res", ascending=False)
    imp.to_csv(ROOT / "research/gbm02_imp.csv", index=False)
    print("\ntop 25 gain (target res):")
    print(imp.head(25).to_string(index=False))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
