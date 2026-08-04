"""SUR-05: evaluador con predicciones cacheadas + blend 0.25/0.75 con pf_ancc.

El numero que decide la mision es el blend, no la superficie sola. pf_ancc es
caro (~1-2 s/pozo) y NO depende de la superficie: se calcula una vez por pozo y
se cachea; despues cualquier variante de superficie se mezcla en milisegundos.

La metrica reproduce exactamente la de cv.evaluate (RMSE pooled de dTVT sobre
los mismos pozos, mismo criterio de descarte cut<20).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, HERE)

import cv as CV  # noqa: E402

PFC = os.path.join(HERE, "sur05_pfcache.npz")


def wells(k=60):
    ids = CV.well_ids()
    ids = CV._select(ids, k, CV.SEED) if k else ids
    out = []
    for w in ids:
        df, tw, cut = CV.load_well(w)
        if cut < 20 or cut >= len(df):
            continue
        out.append(w)
    return out


def truth(ids):
    return {w: CV.load_well(w)[0].TVT.values[CV.load_well(w)[2]:] for w in ids}


def pooled(pred, tru):
    sse = sum(float(((tru[w] - pred[w]) ** 2).sum()) for w in pred)
    n = sum(len(tru[w]) for w in pred)
    return float(np.sqrt(sse / n))


def per_well(pred, tru):
    return pd.DataFrame([{"well": w, "n": len(tru[w]),
                          "rmse": float(np.sqrt(((tru[w] - pred[w]) ** 2).mean()))}
                         for w in pred]).sort_values("rmse", ascending=False)


def pf_cache(ids, path=PFC):
    """{wid: pf_ancc post-PS} cacheado en disco."""
    d = dict(np.load(path)) if os.path.exists(path) else {}
    miss = [w for w in ids if w not in d]
    if miss:
        from pf_publico import predict_pf_ancc
        for i, w in enumerate(miss):
            df, tw, cut = CV.load_well(w)
            df_h = df[CV.TEST_COLS].copy()
            p = np.asarray(predict_pf_ancc(df_h, tw), float)
            d[w] = (p[cut:] if len(p) == len(df) else p).astype(np.float32)
            if i % 10 == 0:
                print(f"  pf {i}/{len(miss)}", flush=True)
        np.savez(path, **d)
    return {w: d[w].astype(np.float64) for w in ids}


def collect(predict_fn, ids):
    """{wid: array post-PS} llamando a predict_fn(df_h, tw, wid)."""
    out = {}
    for w in ids:
        df, tw, cut = CV.load_well(w)
        df_h = df[CV.TEST_COLS].copy()
        p = np.asarray(predict_fn(df_h, tw, w), float)
        out[w] = p[cut:] if len(p) == len(df) else p
    return out


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ids = wells(k)
    print(f"{len(ids)} pozos")
    tru = truth(ids)
    pf = pf_cache(ids)
    print(f"pf_ancc                     {pooled(pf, tru):7.3f}")
    from model import make_predictor
    sur = collect(make_predictor(use_hmm=False), ids)
    print(f"superficie (model.py)       {pooled(sur, tru):7.3f}")
    for a in (0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5):
        b = {w: a * sur[w] + (1 - a) * pf[w] for w in ids}
        print(f"blend {a:.2f}*sup+{1-a:.2f}*pf     {pooled(b, tru):7.3f}")
