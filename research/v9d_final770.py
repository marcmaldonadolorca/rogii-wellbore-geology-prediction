"""Veredicto final: v8 (blend adaptativo 2D) vs v7 (GBM sobre base adapt) sobre
los 770 pozos. Con n=770 el umbral de ruido pareado baja a ~0.15 ft (frente a
0.35 en k=150), asi que esta es la comparacion con maxima potencia para elegir
los 2 envios finales.

Uso: python research/v9d_final770.py [v8|v7]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

from cv import evaluate                               # noqa: E402
import v9b_sigp64 as V9                               # noqa: E402


def main(cual):
    if cual == "v8":
        pred = V9.make_pred(V9.TAB["sig_p_1d"])
        nombre = "v8 blend adaptativo 2D"
    else:
        import v4_gbm as G
        pred = G.make_cv_predictor("resid_adapt")
        nombre = "v7 GBM sobre base adapt"
    t0 = time.time()
    r = evaluate(pred, k=None, verbose=True)
    r["per_well"].to_csv(R / f"v9d_perwell_{cual}.csv", index=False)
    print(f"\n{nombre}: {r['rmse']:.4f} ft en {r['n_wells']} pozos "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v8")
