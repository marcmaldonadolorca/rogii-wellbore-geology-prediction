"""Estructura del error: w optimo por (nn_dist, distancia desde PS) y pesos
optimos 3-way. Exploratorio sobre los mismos 150 pozos (los numeros honestos
salen en mix05 con ajuste en pozos disjuntos)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = np.load(ROOT / "research/mix01_cache.npz")
META = pd.read_csv(ROOT / "research/mix01_meta.csv")
lens = D["lens"]
wid = np.repeat(np.arange(len(lens)), lens)
y, A, B, C = (D[k].astype(np.float64) for k in ("y", "A", "B", "C"))
nn, md_since, hd_since = (D[k].astype(np.float64) for k in ("nn", "md_since", "hd_since"))
NW = len(lens)


def rmse(p, m=None):
    e = (y - p) if m is None else (y[m] - p[m])
    return float(np.sqrt(np.mean(e ** 2)))


def wopt(p, q, m):
    """w que minimiza |y-(p+w(q-p))| en la mascara m."""
    d = (q - p)[m]
    den = float(d @ d)
    return float(((y[m] - p[m]) @ d) / den) if den > 0 else np.nan


nnb = np.digitize(nn, [100, 300, 600, 1000])
mdb = np.digitize(md_since, [500, 1500, 3000, 5000])
nnl = ["<100", "100-300", "300-600", "600-1k", ">1k"]
mdl = ["<500", "500-1.5k", "1.5-3k", "3-5k", ">5k"]

print("== w optimo (peso a C; resto a A) por (nn_dist x md_since) ==")
rows = []
for i, ln in enumerate(nnl):
    r = {"nn": ln}
    for j, lm in enumerate(mdl):
        m = (nnb == i) & (mdb == j)
        r[lm] = round(wopt(A, C, m), 2) if m.sum() > 200 else np.nan
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== RMSE de A y C por (nn x md_since) ==")
for nm, p in (("A", A), ("C", C)):
    rows = []
    for i, ln in enumerate(nnl):
        r = {"nn": ln}
        for j, lm in enumerate(mdl):
            m = (nnb == i) & (mdb == j)
            r[lm] = round(rmse(p, m), 1) if m.sum() > 200 else np.nan
        rows.append(r)
    print(f"-- {nm} --"); print(pd.DataFrame(rows).to_string(index=False))

print("\n== n puntos por celda ==")
print(pd.crosstab(pd.Series(nnb).map(dict(enumerate(nnl))),
                  pd.Series(mdb).map(dict(enumerate(mdl)))).to_string())

print("\n== pesos fijos 3-way (minimos cuadrados, sin restriccion) ==")
X = np.column_stack([A, B, C])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
print("  w(A,B,C) libres =", np.round(coef, 3), " rmse =", round(rmse(X @ coef), 3))
X1 = np.column_stack([A, B, C, np.ones_like(A)])
c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
print("  + intercept      =", np.round(c1, 3), " rmse =", round(rmse(X1 @ c1), 3))
# convexos en rejilla
best = None
for wa in np.arange(0, 1.01, 0.05):
    for wb in np.arange(0, 1.01 - wa + 1e-9, 0.05):
        wc = 1 - wa - wb
        r = rmse(wa * A + wb * B + wc * C)
        if best is None or r < best[0]:
            best = (r, wa, wb, wc)
print(f"  convexo optimo   = A{best[1]:.2f} B{best[2]:.2f} C{best[3]:.2f}  rmse={best[0]:.3f}")

print("\n== continuidad en PS y shrink ==")
# salto de cada predictor en el primer punto post-PS respecto al ultimo conocido
first = np.concatenate([[0], np.cumsum(lens)[:-1]])
for nm, p in (("A", A), ("B", B), ("C", C)):
    print(f"  {nm}: err medio |y-p| en 1er punto post-PS = {np.mean(np.abs(y[first]-p[first])):.3f} ft")
