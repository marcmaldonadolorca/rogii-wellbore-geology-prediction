"""SUR-08: anatomia del regimen "lejos" (>1000 ft al pozo vecino), que aporta el
69% del SSE de la superficie.

Preguntas:
  1) cuantos puntos/pozos son y como se reparten
  2) el residuo TVT - prior alli, ¿es DERIVA (tendencia suave) o ruido?
     -> si es deriva, un modelo de tendencia/dip regional puede arreglarlo
  3) ¿que parte de la deriva se ve ya en el prefijo pre-PS?
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, HERE)

import cv as CV        # noqa: E402
import sur02_lib as L  # noqa: E402
import sur05_blend as B  # noqa: E402

ids = B.wells(60)
c = L.Cloud(subsample=10)
rows, allr = [], []
for w in ids:
    df, tw, cut = CV.load_well(w)
    X, Y, Z = df.X.values, df.Y.values, df.Z.values
    d, gi, dr = c.neighbors(w, X, Y)
    S = L.idw(c, d, gi, 16)[:, 5]
    base = S - Z
    C = np.median(df.TVT_input.values[max(0, cut - 500):cut] - base[max(0, cut - 500):cut])
    pr = base + C
    e = df.TVT.values - pr
    post = slice(cut, len(df))
    md = df.MD.values
    rows.append({
        "well": w, "n_post": len(df) - cut, "dr_med": float(np.median(dr[post])),
        "dr_ps": float(dr[cut]), "rmse": float(np.sqrt((e[post] ** 2).mean())),
        "e_ps": float(e[cut]),
        "e_fin": float(e[-1]),
        "sd_detrend": float(np.std(e[post] - np.polyval(np.polyfit(md[post], e[post], 1), md[post]))),
        "pend_post": float(np.polyfit(md[post], e[post], 1)[0] * 1000),
        "pend_pre": float(np.polyfit(md[max(0, cut - 2000):cut], e[max(0, cut - 2000):cut], 1)[0] * 1000)
        if cut > 100 else np.nan,
        "sd_pre": float(np.std(e[max(0, cut - 2000):cut])),
        "largo": float(md[-1] - md[cut]),
    })
    allr.append(np.column_stack([dr[post], e[post], md[post] - md[cut]]))

per = pd.DataFrame(rows)
per.to_csv(os.path.join(HERE, "sur08_perwell.csv"), index=False)
A = np.vstack(allr)
print(f"puntos post-PS: {len(A)}")
b = np.digitize(A[:, 0], [0, 100, 300, 600, 1000, 1e9]) - 1
for i, lab in enumerate(["<100", "100-300", "300-600", "600-1000", ">1000"]):
    m = b == i
    print(f"  {lab:9s} n={m.sum():7d} ({100*m.mean():5.1f}%)  rmse={np.sqrt((A[m,1]**2).mean()):6.1f}"
          f"  sse_share={100*(A[m,1]**2).sum()/(A[:,1]**2).sum():5.1f}%")

print("\npozos ordenados por SSE aportado:")
per["sse"] = per.rmse ** 2 * per.n_post
per = per.sort_values("sse", ascending=False)
tot = per.sse.sum()
print(per.head(12)[["well", "n_post", "dr_med", "rmse", "e_ps", "e_fin", "sd_detrend",
                    "pend_post", "pend_pre", "sd_pre", "largo"]].to_string(index=False))
print(f"\ntop-5 pozos = {100*per.sse.head(5).sum()/tot:.1f}% del SSE ; "
      f"top-10 = {100*per.sse.head(10).sum()/tot:.1f}%")

far = per[per.dr_med > 1000]
print(f"\npozos con dr_med>1000: {len(far)}  ({100*far.sse.sum()/tot:.1f}% del SSE)")
print(f"  rmse medio {far.rmse.mean():.1f}  |  sd tras quitar la recta en MD: {far.sd_detrend.mean():.1f}")
print(f"  correlacion pendiente pre-PS vs post-PS: "
      f"{np.corrcoef(far.pend_pre, far.pend_post)[0,1]:.3f}  (todos: "
      f"{np.corrcoef(per.pend_pre.fillna(0), per.pend_post)[0,1]:.3f})")
print(f"  |pendiente post| mediana {far.pend_post.abs().median():.1f} ft/1000ft ; "
      f"todos {per.pend_post.abs().median():.1f}")
