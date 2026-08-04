"""Barridos de combinacion sobre el cache de mix01 (k=150, LOWO estricto).

1) blend fijo w*B+(1-w)*A y w*C+(1-w)*A
2) blend por nn_dist (sigmoide) por pozo y por punto
3) oraculos (por pozo / por punto) como techo
4) desglose por bins de nn_dist
"""
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


def rmse(p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def per_well_sse(p):
    e2 = (y - p) ** 2
    return np.bincount(wid, e2, NW), np.bincount(wid, None, NW)


def lb_proxy(p):
    """pooled solo en pozos con n_pred >= 5147 (definicion de cv.py)."""
    sse, n = per_well_sse(p)
    m = META.n_pred.values >= 5147
    return float(np.sqrt(sse[m].sum() / n[m].sum()))


def report(name, p, extra=""):
    print(f"{name:44s} rmse={rmse(p):7.3f}  lbproxy={lb_proxy(p):7.3f} {extra}")
    return rmse(p)


def bins_table(preds, names):
    edges = [0, 100, 300, 600, 1000, 1e9]
    labs = ["<100", "100-300", "300-600", "600-1000", ">1000"]
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labs):
        m = (nn >= lo) & (nn < hi)
        r = {"bin": lab, "npts": int(m.sum()), "frac%": 100 * m.mean()}
        for nm, p in zip(names, preds):
            r[nm] = float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    print(f"pozos={NW} puntos={len(y)}")
    for nm, p in (("A geometrico", A), ("B superficie", B), ("C sup+HMM", C)):
        report(nm, p)

    print("\n== oraculos (techo, NO implementable) ==")
    for nm, (p, q) in (("min(A,B) por punto", (A, B)), ("min(A,C) por punto", (A, C))):
        e = np.where(np.abs(y - p) < np.abs(y - q), p, q)
        report(nm, e)
    for nm, (p, q) in (("min(A,B) por pozo", (A, B)), ("min(A,C) por pozo", (A, C))):
        sp, _ = per_well_sse(p); sq, _ = per_well_sse(q)
        sel = (sp <= sq)[wid]
        report(nm, np.where(sel, p, q))
    # oraculo de peso continuo por pozo (w optimo por pozo, techo del blend)
    for nm, (p, q) in (("blend w* por pozo (A,B)", (A, B)), ("blend w* por pozo (A,C)", (A, C))):
        d = q - p
        num = np.bincount(wid, (y - p) * d, NW)
        den = np.bincount(wid, d * d, NW)
        w = np.clip(np.where(den > 0, num / np.maximum(den, 1e-12), 1.0), 0, 1)
        report(nm, p + w[wid] * d)

    print("\n== 1) blend fijo ==")
    best = {}
    for nm, (p, q) in (("w*B+(1-w)*A", (B, A)), ("w*C+(1-w)*A", (C, A)), ("w*C+(1-w)*B", (C, B))):
        tab = [(w, rmse(w * p + (1 - w) * q)) for w in np.arange(0, 1.001, 0.05)]
        w0, r0 = min(tab, key=lambda t: t[1])
        print(f"  {nm:14s} mejor w={w0:.2f} rmse={r0:.3f}   " +
              " ".join(f"{w:.1f}:{r:.2f}" for w, r in tab[::2]))
        best[nm] = (w0, r0)

    print("\n== 2) blend por nn_dist: w(d)=sigmoide, peso a la SUPERFICIE ==")
    # w = 1/(1+exp((d-d0)/s)) -> w~1 cerca (superficie), w~0 lejos (geometrico)
    res = []
    for base, bn in ((B, "B"), (C, "C")):
        for percy, pn in ((nn, "punto"), (np.repeat(np.array([np.median(nn[wid == i]) for i in range(NW)]), lens), "pozo")):
            for d0 in (200, 300, 400, 500, 700, 900, 1200, 1600):
                for s in (30, 80, 150, 300, 600):
                    w = 1.0 / (1.0 + np.exp((percy - d0) / s))
                    res.append((bn, pn, d0, s, rmse(w * base + (1 - w) * A)))
    rdf = pd.DataFrame(res, columns=["base", "modo", "d0", "s", "rmse"]).sort_values("rmse")
    print(rdf.head(12).to_string(index=False))
    print("  mejor por (base,modo):")
    print(rdf.groupby(["base", "modo"]).head(1).to_string(index=False))

    print("\n== 4) bins de nn_dist ==")
    print(bins_table([A, B, C], ["A", "B", "C"]).to_string(index=False))

    rdf.to_csv(ROOT / "research/mix02_sigmoide.csv", index=False)


if __name__ == "__main__":
    main()
