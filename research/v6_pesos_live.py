"""v6 pesos — confirmacion LIVE con cv.evaluate del blend por punto punto3:

  w(punto) = softmax de scores lineales en z = [log1p(nn_real), md/1e3, sstd]
  miembros: S = superficie aniso16 k24 theta=2.278489 (LOWO)
            P = PF ancc multiseed S=64 media (cache disco v4_multiseed_cache)
            G = dip anclado win700
  params: research/v6_pesos_punto2_params.json (ajustados en 380 pozos train,
  jamas en los de eval) -> honesto en LOWO.

Uso: python research/v6_pesos_live.py [k] [variante]   (k=150, punto3 nn,md,pstd)
"""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import cv as cvmod  # noqa: E402
from ban_lib import run_pf_ancc_multi  # noqa: E402
from cv import evaluate  # noqa: E402
from model import _prior  # noqa: E402
from pf_publico import _tw  # noqa: E402
from v4_gbm import geom_pred, get_field, hdist, real_nn  # noqa: E402

cvmod.load_well = lru_cache(maxsize=None)(cvmod.load_well)
CACHE_DIR = HERE / "v4_multiseed_cache"
S_SEEDS = 64


def pf64(df_h, tw, wid):
    f = CACHE_DIR / f"pf_{wid}_S{S_SEEDS}_N600.npz"
    if f.exists():
        z = np.load(f)
        pts = z["pts"].astype(np.float64)
    else:
        t, g = _tw(tw)
        pts, ll = run_pf_ancc_multi(df_h, t, g,
                                    seeds=np.arange(1, S_SEEDS + 1, dtype=np.int64))
        np.savez_compressed(f, pts=pts.astype(np.float32), ll=ll)
        pts = pts.astype(np.float64)
    return pts.mean(0), pts.std(0)


def make_predictor(variante):
    p = json.loads((HERE / "v6_pesos_punto2_params.json").read_text())[variante]
    theta = np.array(p["theta"]); mu = np.array(p["mu"]); sd = np.array(p["sd"])
    k = len(p["cols"])
    field = get_field()

    def predict(df_h, tw, wid):
        m = df_h.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df_h)
        S = _prior(df_h, field, wid, cut)[0][cut:]
        P, sstd = pf64(df_h, tw, wid)
        G = geom_pred(df_h, cut, hdist(df_h))[0][cut:]
        nn = real_nn(field, wid, df_h.X.values[cut:], df_h.Y.values[cut:])
        md = df_h.MD.values[cut:] - df_h.MD.values[cut - 1]
        Z = np.column_stack([np.log1p(nn), md / 1e3, sstd])[:, p["cols"][:k]]
        z = (Z - mu) / sd
        sS = theta[0] + z @ theta[1:1 + k]
        sG = theta[1 + k] + z @ theta[2 + k:2 + 2 * k]
        mx = np.maximum.reduce([sS, np.zeros(len(z)), sG])
        e = np.exp(np.column_stack([sS, np.zeros(len(z)), sG]) - mx[:, None])
        W = e / e.sum(1, keepdims=True)
        return W[:, 0] * S + W[:, 1] * np.asarray(P) + W[:, 2] * G

    return predict


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    var = sys.argv[2] if len(sys.argv) > 2 else "punto3 nn,md,pstd"
    t0 = time.time()
    res = evaluate(make_predictor(var), k=k)
    print(f"[{var}] k={k} LIVE rmse={res['rmse']:.3f} "
          f"proxy={res['rmse_lb_proxy']:.3f} ({time.time()-t0:.0f}s)")
