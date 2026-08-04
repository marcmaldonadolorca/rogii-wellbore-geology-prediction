"""Por que la ventana importa tanto: perfil del residuo a lo largo del prefijo,
barrido fino ventana x estimador, y mezclas ancla/nivel.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cal02_diag import load, pooled  # noqa: E402
from cal03_sweep import level  # noqa: E402

W = load()

# ---- 1. sesgo del nivel segun donde se mire dentro del prefijo -------------
print("== nivel estimado en la ventana [cut-a, cut-b] vs c* = media(e post) ==")
print(f"{'ventana':>18s} {'bias_rms':>9s} {'bias_med':>9s} {'rmse_pooled':>11s} {'cobertura':>9s}")
for a, b in [(100, 0), (250, 0), (500, 0), (750, 0), (1000, 0), (1500, 0), (2000, 0),
             (3000, 0), (10 ** 9, 0), (500, 0), (1000, 500), (1500, 1000), (2000, 1500),
             (10 ** 9, 2000)]:
    bias, errs, nwin = [], [], []
    for d in W:
        c = d["cut"]
        lo, hi = max(0, c - a), max(0, c - b)
        if hi - lo < 20:
            continue
        cc = np.median(d["ein"][lo:hi])
        e = d["e"][c:]
        bias.append(cc - e.mean())
        errs.append(e - cc)
        nwin.append(hi - lo)
    bias = np.array(bias)
    print(f"  [-{a:>9}, -{b:<5}] {np.sqrt((bias**2).mean()):9.3f} "
          f"{np.median(np.abs(bias)):9.3f} {pooled(errs):11.3f} {np.median(nwin):9.0f}")

# ---- 2. barrido fino ventana x estimador ----------------------------------
print("\n== ventana x estimador (rmse pooled) ==")
ests = ["median", "mean", "trim10", "huber"]
wins = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 10 ** 9]
tab = []
for win in wins:
    row = {"win": win}
    for est in ests:
        errs = []
        for d in W:
            c = d["cut"]
            lo = max(0, c - win)
            v = d["ein"][lo:c].astype(np.float64)
            cc = level(v, np.ones(len(v)), est)
            errs.append(d["e"][c:] - cc)
        row[est] = pooled(errs)
    tab.append(row)
t = pd.DataFrame(tab)
print(t.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

# ---- 3. mezcla nivel largo + ancla en PS ----------------------------------
print("\n== c = alpha*median(win) + (1-alpha)*e[cut-1] ==")
for win in (1000, 1500, 10 ** 9):
    line = [f"  win={win:>10}"]
    for al in (0.0, 0.25, 0.5, 0.75, 1.0):
        errs = []
        for d in W:
            c = d["cut"]
            cc = al * np.median(d["ein"][max(0, c - win):c]) + (1 - al) * d["ein"][c - 1]
            errs.append(d["e"][c:] - cc)
        line.append(f"a={al:.2f}:{pooled(errs):7.3f}")
    print(" ".join(line))

# ---- 4. de donde viene la ganancia: por pozo -------------------------------
r500, r1000 = [], []
rows = []
for d in W:
    c = d["cut"]
    c5 = np.median(d["ein"][max(0, c - 500):c])
    c1 = np.median(d["ein"][max(0, c - 1000):c])
    e = d["e"][c:]
    rows.append({"well": d["wid"], "npred": len(e), "cut": c,
                 "rmse500": float(np.sqrt(((e - c5) ** 2).mean())),
                 "rmse1000": float(np.sqrt(((e - c1) ** 2).mean())),
                 "sse500": float(((e - c5) ** 2).sum()),
                 "sse1000": float(((e - c1) ** 2).sum())})
df = pd.DataFrame(rows)
df["dsse"] = df.sse1000 - df.sse500
df = df.sort_values("dsse")
print("\n== pozos que mas ganan con win=1000 (dsse<0) ==")
print(df.head(6)[["well", "cut", "npred", "rmse500", "rmse1000"]].to_string(index=False))
print("== pozos que mas pierden ==")
print(df.tail(4)[["well", "cut", "npred", "rmse500", "rmse1000"]].to_string(index=False))
print(f"\npozos que mejoran: {(df.dsse < 0).sum()}/{len(df)} | "
      f"sse total 500={df.sse500.sum():.3e} 1000={df.sse1000.sum():.3e}")
top = df.head(3)
print(f"quitando los 3 mejores: 500={np.sqrt((df.sse500.sum()-top.sse500.sum())/(df.npred.sum()-top.npred.sum())):.3f} "
      f"1000={np.sqrt((df.sse1000.sum()-top.sse1000.sum())/(df.npred.sum()-top.npred.sum())):.3f}")
df.to_csv(ROOT / "research" / "cal04_win_perwell.csv", index=False)
