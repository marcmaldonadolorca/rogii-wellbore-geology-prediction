"""Distribucion del punto PS (Prediction Start) en train vs los 3 pozos muestra de test.

Por pozo: n filas, indice de corte (primera fila con TVT_input NaN), fraccion
cut/n, longitud MD total y post-PS, y n de puntos a predecir.
Salida: research/ps_stats.csv + resumen por stdout.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data/raw"


def stats(split):
    rows = []
    for f in sorted((RAW / split).glob("*__horizontal_well.csv")):
        wid = f.name.split("__")[0]
        df = pd.read_csv(f)
        m = df.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df)
        md = df.MD.values
        rows.append({
            "well": wid, "n": len(df), "cut": cut,
            "ps_frac": cut / len(df),
            "md_total": md[-1] - md[0],
            "md_post": md[-1] - md[cut - 1] if cut < len(df) else 0.0,
            "n_pred": len(df) - cut,
        })
    return pd.DataFrame(rows)


tr = stats("train")
te = stats("test")
tr.to_csv(Path(__file__).parent / "ps_stats.csv", index=False)

q = tr.ps_frac.quantile([.05, .1, .25, .5, .75, .9, .95]).round(3)
print(f"train: {len(tr)} pozos")
print("ps_frac train quantiles:\n", q.to_string())
print(f"ps_frac media {tr.ps_frac.mean():.3f} | std {tr.ps_frac.std():.3f}")
print(f"n filas: mediana {tr.n.median():.0f} | p10 {tr.n.quantile(.1):.0f} | p90 {tr.n.quantile(.9):.0f}")
print(f"n_pred: mediana {tr.n_pred.median():.0f} | p90 {tr.n_pred.quantile(.9):.0f}")
print("\ntest muestra:")
print(te[["well", "n", "cut", "ps_frac", "n_pred"]].to_string(index=False))
lo, hi = te.ps_frac.min(), te.ps_frac.max()
frac_in = ((tr.ps_frac >= lo) & (tr.ps_frac <= hi)).mean()
print(f"\nfraccion de train con ps_frac dentro del rango test [{lo:.3f},{hi:.3f}]: {frac_in:.1%}")
print(f"fraccion de train con ps_frac <= 0.35: {(tr.ps_frac <= 0.35).mean():.1%}")
