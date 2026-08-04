"""Afinado fino del dip lineal pre-PS alrededor de lin500 (muestra 200 / --full 770).
Variantes: ventanas 400-800, pesos exponenciales (recencia), ajuste anclado en PS,
Theil-Sen subsampleado.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TEST_IDS = {"000d7d20", "00bbac68", "00e12e8b"}
SEED, N_SAMPLE = 0, 200


def wells():
    return sorted(p.name.split("__")[0] for p in (RAW / "train").glob("*__horizontal_well.csv"))


def cut_of(df):
    m = df.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df)


def resid_h(tvt_in, z, hd, cut, win):
    lo = max(0, cut - win)
    base = tvt_in[lo] + (z[lo] - z[lo:cut])
    resid = tvt_in[lo:cut] - base
    h = hd[lo:cut] - hd[lo]
    return resid, h


def main():
    full = "--full" in sys.argv
    rng = np.random.default_rng(SEED)
    all_wells = [w for w in wells() if w not in TEST_IDS]
    ids = all_wells if full else sorted(rng.choice(all_wells, size=N_SAMPLE, replace=False))

    names = ["lin400", "lin500", "lin600", "lin700", "lin800",
             "wexp1500_500", "wexp1000_300", "anchor500", "anchor700", "theilsen500"]
    sq = {v: 0.0 for v in names}
    npt = 0

    for wid in ids:
        df = pd.read_csv(RAW / "train" / f"{wid}__horizontal_well.csv")
        cut = cut_of(df)
        if cut < 20 or cut >= len(df):
            continue
        z, tvt_in, tvt = df.Z.values, df.TVT_input.values, df.TVT.values
        step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                        np.diff(df.Y.values, prepend=df.Y.values[0]))
        hd = np.cumsum(step) - np.cumsum(step)[cut - 1]
        flat = tvt_in[cut - 1] + (z[cut - 1] - z)
        hpost = hd[cut:]
        y = tvt[cut:]
        npt += len(y)

        slopes = {}
        for win in (400, 500, 600, 700, 800):
            r, h = resid_h(tvt_in, z, hd, cut, win)
            slopes[f"lin{win}"] = np.polyfit(h, r, 1)[0] if len(h) > 10 and h[-1] != 0 else 0.0
        # pesos exponenciales por recencia sobre ventana ancha
        for win, tau, name in ((1500, 500.0, "wexp1500_500"), (1000, 300.0, "wexp1000_300")):
            r, h = resid_h(tvt_in, z, hd, cut, win)
            if len(h) > 10 and h[-1] != 0:
                w = np.exp(-(h[-1] - h) / tau)
                slopes[name] = np.polyfit(h, r, 1, w=np.sqrt(w))[0]
            else:
                slopes[name] = 0.0
        # anclado: recta forzada a pasar por el punto PS (ultimo pre-PS)
        for win in (500, 700):
            r, h = resid_h(tvt_in, z, hd, cut, win)
            name = f"anchor{win}"
            if len(h) > 10 and h[-1] != 0:
                dr, dh = r - r[-1], h - h[-1]
                slopes[name] = float(np.sum(dr * dh) / np.sum(dh * dh))
            else:
                slopes[name] = 0.0
        # Theil-Sen subsampleado en win 500
        r, h = resid_h(tvt_in, z, hd, cut, 500)
        if len(h) > 20:
            k = min(len(h), 120)
            idx = rng.choice(len(h), size=k, replace=False)
            hi, ri = h[idx], r[idx]
            dh = hi[:, None] - hi[None, :]
            dr = ri[:, None] - ri[None, :]
            m = np.abs(dh) > 50
            slopes["theilsen500"] = float(np.median(dr[m] / dh[m])) if m.any() else 0.0
        else:
            slopes["theilsen500"] = 0.0

        for v in names:
            sq[v] += float(np.sum((y - (flat[cut:] + slopes[v] * hpost)) ** 2))

    tag = "770 pozos" if full else f"muestra {len(ids)}"
    print(f"== RMSE global ({tag}, {npt} puntos) ==")
    for v in sorted(names, key=lambda v: sq[v]):
        print(f"  {v:14s} {np.sqrt(sq[v] / npt):7.2f} ft")


if __name__ == "__main__":
    main()
