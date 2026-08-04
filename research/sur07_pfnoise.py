"""SUR-07: ruido de realizacion del particle filter.

run_pf_ancc usa np.random dentro de numba SIN semilla: cada ejecucion da un
resultado distinto. Si el ruido de realizacion es comparable a las mejoras que
buscamos, comparar dos superficies con UNA realizacion del PF es enganoso.
Se generan R realizaciones y se mide la dispersion del RMSE pooled (k=60).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, HERE)

import cv as CV        # noqa: E402
import sur05_blend as B  # noqa: E402
from pf_publico import predict_pf_ancc  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ids = B.wells(60)
tru = B.truth(ids)
rs = []
for r in range(R):
    path = os.path.join(HERE, f"sur07_pf_r{r}.npz")
    if os.path.exists(path):
        d = dict(np.load(path))
    else:
        d = {}
        for w in ids:
            df, tw, cut = CV.load_well(w)
            p = np.asarray(predict_pf_ancc(df[CV.TEST_COLS].copy(), tw), float)
            d[w] = (p[cut:] if len(p) == len(df) else p).astype(np.float32)
        np.savez(path, **d)
    pr = {w: d[w].astype(np.float64) for w in ids}
    rs.append(B.pooled(pr, tru))
    print(f"realizacion {r}: pf_ancc = {rs[-1]:.3f}", flush=True)

rs = np.array(rs)
print(f"pf_ancc: media {rs.mean():.3f}  sd {rs.std(ddof=1):.3f}  rango [{rs.min():.3f},{rs.max():.3f}]")

# media de realizaciones (PF promediado) y su blend con la superficie de model.py
avg = {}
for r in range(R):
    d = dict(np.load(os.path.join(HERE, f"sur07_pf_r{r}.npz")))
    for w in ids:
        avg[w] = avg.get(w, 0.0) + d[w].astype(np.float64) / R
print(f"pf_ancc promediado x{R}: {B.pooled(avg, tru):.3f}")
np.savez(os.path.join(HERE, "sur07_pf_avg.npz"), **{w: v.astype(np.float32) for w, v in avg.items()})

from model import make_predictor  # noqa: E402
sur = B.collect(make_predictor(use_hmm=False), ids)
for tag, pf in [("1 realizacion", {w: np.load(os.path.join(HERE, "sur07_pf_r0.npz"))[w].astype(np.float64) for w in ids}),
                (f"promedio x{R}", avg)]:
    for a in (0.2, 0.25, 0.3, 0.35, 0.4):
        b = {w: a * sur[w] + (1 - a) * pf[w] for w in ids}
        print(f"blend {a:.2f} ({tag}): {B.pooled(b, tru):.3f}")
