"""Cache propio de la nube de superficies: (X, Y, 6 formaciones, wid) a resolucion 1/1.

Escribe research/surf_cloud.npz  (float32, ~5.1M filas x 8 cols -> ~180 MB).
Tambien imprime el diagnostico de NaN por formacion (ANCC tiene NaN en 7 pozos).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data/raw"
OUT = Path(__file__).resolve().parent / "surf_cloud.npz"
FRM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def main():
    files = sorted((RAW / "train").glob("*__horizontal_well.csv"))
    names, XY, S, W = [], [], [], []
    nan_report = []
    for i, p in enumerate(files):
        d = pd.read_csv(p, usecols=["X", "Y"] + FRM, dtype=np.float32)
        names.append(p.name.split("__")[0])
        nn = d[FRM].isna().sum().values
        if nn.any():
            nan_report.append((p.name.split("__")[0], len(d), dict(zip(FRM, nn))))
        XY.append(d[["X", "Y"]].values)
        S.append(d[FRM].values)
        W.append(np.full(len(d), i, np.int32))
    xy = np.concatenate(XY)
    s = np.concatenate(S)
    w = np.concatenate(W)
    np.savez(OUT, xy=xy, s=s, wid=w, names=np.array(names), frm=np.array(FRM))
    print(f"filas={len(xy)} pozos={len(names)}  -> {OUT}")
    print("NaN por pozo (solo pozos con algun NaN):")
    for n, ln, dd in nan_report:
        print(f"  {n} nrows={ln} {dd}")
    print(f"total pozos con NaN: {len(nan_report)}")


if __name__ == "__main__":
    main()
