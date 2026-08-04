"""GBM con validacion OOF sobre los 770 pozos (no sobre subconjuntos).

Motivacion (ROG-011): la eleccion entre modelos finales medida en k=60/k=150
mintio; el veredicto sobre 770 invirtio la recomendacion. Ademas el GBM del v7
solo entrena con 380 pozos habiendo 574 libres.

Aqui: features de los 770 pozos -> GroupKFold(5) por pozo -> prediccion OOF de
CADA pozo con un modelo que no lo vio -> RMSE pooled directamente comparable con
el 9.151 de v7 y el 9.792 de v8. El modelo final se entrena con los 770.

Uso: python research/v9e_gbm770.py cache | oof | final
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import v4_gbm as G                                    # noqa: E402
from cv import TEST_COLS, load_well, well_ids         # noqa: E402

CACHE = R / "v9e_cache770.npz"
MODEL = R / "v9e_model770.txt"


def build_cache():
    ids = well_ids()
    Xs, ys, ws, t0 = [], [], [], time.time()
    field = G.get_field()
    for i, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        out = G.build_well(df[TEST_COLS].copy(), tw, wid, field)
        if out is None:
            continue
        X, aux = out[0], out[1]
        lk = aux["lk"]
        Xs.append(X)
        ys.append(df.TVT.values[cut:] - lk)      # target = dTVT
        ws.append(np.full(len(X), i))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    X = pd.concat(Xs, ignore_index=True)
    feats = [c for c in X.columns if c not in ("wid",)]   # FEATS se fija al build
    np.savez_compressed(CACHE, X=X[feats].values.astype(np.float32),
                        y=np.concatenate(ys), w=np.concatenate(ws),
                        feats=np.array(feats))
    print(f"cache: {X.shape} de {len(set(np.concatenate(ws)))} pozos "
          f"({time.time()-t0:.0f}s)")


def _load():
    z = np.load(CACHE, allow_pickle=True)
    return (pd.DataFrame(z["X"], columns=list(z["feats"])), z["y"], z["w"])


def _base(X):
    """Base adaptativa 1D, la del v7 (mejor sobre 770)."""
    zz = np.load(R / "v7_curvas.npz")
    nnb, mdb, ss_b, sp_b = zz["nn_bins"], zz["md_bins"], zz["sig_s"], zz["sig_p"]

    def ic(x, bins, vals):
        c = 0.5 * (bins[:-1] + np.minimum(bins[1:], bins[-2] * 2))
        return np.interp(x, c, vals)

    ss = ic(X.nn_aniso.values, nnb, ss_b)
    sp = ic(X.md_since.values, mdb, sp_b)
    w = np.clip(sp**2 / (ss**2 + sp**2), 0.05, 0.9)
    return w * X.dS.values.astype(np.float64) + (1 - w) * X.dP.values.astype(np.float64)


def oof():
    X, y, w = _load()
    base = _base(X)
    resid = y - base                       # el GBM corrige el blend adaptativo
    oof_pred = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for f, (tr, va) in enumerate(gkf.split(X, resid, groups=w)):
        m = lgb.train(G.LGB_PARAMS, lgb.Dataset(X.iloc[tr], resid[tr]),
                      num_boost_round=G.N_ROUNDS,
                      valid_sets=[lgb.Dataset(X.iloc[va], resid[va])],
                      callbacks=[lgb.early_stopping(G.EARLY, verbose=False)])
        oof_pred[va] = m.predict(X.iloc[va])
        print(f"  fold{f}: iters={m.best_iteration} "
              f"rmse_fold={np.sqrt(np.mean((resid[va]-oof_pred[va])**2)):.3f}", flush=True)
    for nombre, p in [("blend adapt 1D (base)", base),
                      ("GBM OOF sobre 770", base + oof_pred)]:
        print(f"  {nombre:24s} RMSE = {np.sqrt(np.mean((y - p)**2)):.4f} ft")
    print("\nreferencias sobre los MISMOS 770 pozos: v7 = 9.151 | v8 = 9.792")


def final():
    X, y, w = _load()
    resid = y - _base(X)
    m = lgb.train(G.LGB_PARAMS, lgb.Dataset(X, resid), num_boost_round=300)
    m.save_model(str(MODEL))
    print(f"modelo final ({m.num_trees()} arboles) -> {MODEL}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cache"
    {"cache": build_cache, "oof": oof, "final": final}[cmd]()
