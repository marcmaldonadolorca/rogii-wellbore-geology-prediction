"""Donde se concentra el MSE del mejor modelo (0.25*surf + 0.75*ancc).

Desglose por: bins de md_since, bins de nn_dist, longitud del tramo (n_pred),
y top-20 de pozos por contribucion al SSE total. Sobre los 770 pozos (y k=60).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orclib import Cache, PRED, rmse, subset  # noqa: E402

BEST = dict(surf=0.25, ancc=0.75)


def bins_tabla(c, m, pred, x, edges, nombre):
    y = c.d["y"]
    e2 = (y - pred) ** 2
    rows = []
    tot = e2[m].sum()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = m & (x >= lo) & (x < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        s = float(e2[sel].sum())
        rows.append({"bin": f"[{lo:g},{hi:g})", "n_pts": n, "pct_pts": 100 * n / m.sum(),
                     "rmse": float(np.sqrt(s / n)), "pct_sse": 100 * s / tot})
    t = pd.DataFrame(rows)
    print(f"\n--- {nombre} ---")
    print(t.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    return t


def main():
    c = Cache()
    y = c.d["y"]
    mix = c.blend(**BEST)
    out = {}

    for tag, ids in (("TODOS", None), ("k=60", subset(60))):
        m = c.mask(ids)
        print(f"\n########## {tag}  RMSE mix = {rmse(y, mix, m):.3f} "
              f"(surf {rmse(y, c.d['surf'], m):.3f} | ancc {rmse(y, c.d['ancc'], m):.3f}) ##########")
        out[f"md_{tag}"] = bins_tabla(c, m, mix, c.d["md_since"],
                                      [0, 500, 1000, 2000, 3000, 5000, 8000, 1e9],
                                      "MSE por md_since (ft desde PS)")
        out[f"nn_{tag}"] = bins_tabla(c, m, mix, c.d["nn"],
                                      [0, 100, 300, 600, 1000, 2000, 1e9],
                                      "MSE por nn_dist (ft al vecino de la nube)")
        # longitud del tramo: covariable por pozo, expandida a puntos
        npred_pt = np.repeat(c.meta.n_pred.values, c.lens)
        out[f"len_{tag}"] = bins_tabla(c, m, mix, npred_pt.astype(float),
                                       [0, 2000, 4000, 5147, 7000, 10000, 1e9],
                                       "MSE por longitud del tramo (n_pred)")

    # --- concentracion por pozo ---------------------------------------------
    e2 = (y - mix) ** 2
    sse = np.bincount(c.wpt, weights=e2, minlength=len(c.lens))
    n = c.lens.astype(float)
    per = c.meta.copy()
    per["sse"] = sse
    per["rmse"] = np.sqrt(sse / n)
    for p in PRED:
        s2 = np.bincount(c.wpt, weights=(y - c.d[p]) ** 2, minlength=len(c.lens))
        per[f"rmse_{p}"] = np.sqrt(s2 / n)
    per["pct_sse"] = 100 * per.sse / per.sse.sum()
    per = per.sort_values("sse", ascending=False).reset_index(drop=True)
    per["cum_pct"] = per.pct_sse.cumsum()
    per.to_csv(HERE / "orc03_perwell.csv", index=False)

    print("\n########## CONCENTRACION POR POZO (770) ##########")
    nw = len(per)
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = int(round(frac * nw))
        print(f"  top {frac*100:4.0f}% de pozos ({k:3d}) aportan {per.pct_sse[:k].sum():5.1f}% del SSE")
    print("\n  top-20 pozos por SSE:")
    cols = ["well", "n_pred", "ps_frac", "nn_med", "rmse", "pct_sse", "cum_pct",
            "rmse_surf", "rmse_ancc", "rmse_pfz", "rmse_beam", "rmse_geom"]
    print(per[cols].head(20).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    # que pasaria si en esos pozos usaramos el mejor de los 5 (oraculo local)
    for k in (20, 50, 100):
        top = per.head(k)
        alt = np.minimum.reduce([top[f"rmse_{p}"].values ** 2 * top.n_pred.values for p in PRED])
        gan = top.sse.sum() - alt.sum()
        tot = per.sse.sum()
        print(f"\n  si en el top-{k} acertaramos el mejor de los 5: RMSE "
              f"{np.sqrt((tot - gan)/n.sum()):.3f} (desde {np.sqrt(tot/n.sum()):.3f})")

    # correlacion de la contribucion con covariables
    print("\n  correlacion rmse_pozo vs covariables (Spearman):")
    for col in ("n_pred", "ps_frac", "nn_med", "md_len", "hd_len"):
        print(f"    {col:10s} {per.rmse.corr(per[col], method='spearman'):+.3f}")


if __name__ == "__main__":
    main()
