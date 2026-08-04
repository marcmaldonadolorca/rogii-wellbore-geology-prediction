"""Calibra el motor de barrido contra cv.evaluate y contra model.py.

1) model.py make_predictor(use_hmm=False)  -> referencia 20.325
2) predictor equivalente montado sobre surf_lib, via cv.evaluate
3) el mismo por el motor surf_eval.eval_configs
Los tres deben coincidir salvo la fase del submuestreo.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import evaluate  # noqa: E402
from surf_eval import CAL_TAIL, eval_configs, show  # noqa: E402
from surf_lib import Cloud, LocalField, idw  # noqa: E402

CLOUD = Cloud()


def make_pred(sub=10, k=16, j=5, p=2.0):
    def predict(df_h, tw, wid=None):
        m = df_h.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df_h)
        X = df_h.X.values.astype(float); Y = df_h.Y.values.astype(float)
        Z = df_h.Z.values.astype(float); ti = df_h.TVT_input.values.astype(float)
        lf = LocalField(CLOUD, wid, X, Y, subsample=sub)
        q, d, i = lf.query(X, Y, j, k)
        s = idw(lf, q, d, i, j, p=p)
        lo = max(0, cut - CAL_TAIL)
        C = np.median(ti[lo:cut] + Z[lo:cut] - s[lo:cut])
        return (s - Z + C)[cut:]
    return predict


if __name__ == "__main__":
    t0 = time.time()
    print("== 1) model.py superficie sola (referencia) ==")
    from model import SurfaceField, make_predictor
    evaluate(make_predictor(use_hmm=False, field=SurfaceField()), k=150)
    print(f"  [{time.time()-t0:.0f}s]")

    t0 = time.time()
    print("\n== 2) surf_lib via cv.evaluate (sub=10,k=16,idw p=2,BUDA) ==")
    evaluate(make_pred(), k=150)
    print(f"  [{time.time()-t0:.0f}s]")

    t0 = time.time()
    print("\n== 3) motor surf_eval, misma config ==")
    res = eval_configs([dict(name="base idw k16 sub10 BUDA", sub=10, k=16,
                             method="idw", kw={"p": 2.0}, frms=[5])],
                       k_wells=150, cloud=CLOUD, verbose=False)
    show(res)
    print(f"  [{time.time()-t0:.0f}s]")
