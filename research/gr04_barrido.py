"""Barrido fino del HMM alrededor del optimo conocido (sigma_r=.01, dec=10,
sigma_prior=25, sigma_init=2, span=80  ->  17.681).

Barre sigma_prior (5/10/25/50/inf), sigma_gr (5/10/20/None), dec (1/2/5/10),
span (20/40/80) y sigma_r. Paralelo por pozo con multiprocessing.
"""
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grlib import Wells, hmm, pooled                                  # noqa: E402

_W = None


def _init():
    global _W
    _W = Wells()


def _one(args):
    wid, kw = args
    w = _W.get(wid)
    cut = w["cut"]
    r = w["tvt"][cut:] - w["prior"][cut:]
    rh = hmm(w, **kw)
    return float(((r - rh) ** 2).sum()), len(r)


def run(ids, kw, pool):
    out = pool.map(_one, [(w, kw) for w in ids], chunksize=4)
    sse = np.array([o[0] for o in out]); n = np.array([o[1] for o in out])
    return pooled(sse, n)


CONFIGS = []
BASE = dict(sigma_r=0.01, dec=10, sigma_prior=25.0, sigma_init=2.0, span=80.0)
for sp in (5.0, 10.0, 25.0, 50.0, np.inf):
    CONFIGS.append(("sigma_prior", sp, {**BASE, "sigma_prior": sp}))
for sg in (5.0, 10.0, 20.0, None):
    CONFIGS.append(("sigma_gr", sg, {**BASE, "sigma_gr": sg}))
for dc in (1, 2, 5, 10, 20):
    CONFIGS.append(("dec", dc, {**BASE, "dec": dc}))
for sn in (20.0, 40.0, 80.0, 120.0):
    CONFIGS.append(("span", sn, {**BASE, "span": sn}))
for sr in (0.0, 0.003, 0.01, 0.03, 0.1):
    CONFIGS.append(("sigma_r", sr, {**BASE, "sigma_r": sr}))
for si in (0.5, 2.0, 8.0, 30.0):
    CONFIGS.append(("sigma_init", si, {**BASE, "sigma_init": si}))


def main():
    ids = Wells().ids
    res = []
    with Pool(6, initializer=_init) as pool:
        print(f"BASE (sigma_r=.01 dec=10 sp=25 si=2 span=80): {run(ids, BASE, pool):.3f}\n",
              flush=True)
        for name, val, kw in CONFIGS:
            v = run(ids, kw, pool)
            res.append({"eje": name, "valor": str(val), "rmse": v})
            print(f"  {name:<12s} = {str(val):<8s} -> {v:7.3f}", flush=True)
    pd.DataFrame(res).to_csv(Path(__file__).resolve().parent / "gr04_barrido.csv", index=False)


if __name__ == "__main__":
    main()
