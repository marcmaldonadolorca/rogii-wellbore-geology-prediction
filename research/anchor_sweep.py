"""Barrido de ventana del dip anclado en PS (el ganador de afinar_dip).
Muestra 200 pozos por defecto; --full = 770.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TEST_IDS = {"000d7d20", "00bbac68", "00e12e8b"}
SEED, N_SAMPLE = 0, 200
WINS = [600, 700, 800, 1000, 1200, 1500]
SHRINKS = [("anchor700_s09", 700, 0.9), ("anchor1000_s09", 1000, 0.9)]


def wells():
    return sorted(p.name.split("__")[0] for p in (RAW / "train").glob("*__horizontal_well.csv"))


def cut_of(df):
    m = df.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df)


def anchor_slope(tvt_in, z, hd, cut, win):
    lo = max(0, cut - win)
    base = tvt_in[lo] + (z[lo] - z[lo:cut])
    r = tvt_in[lo:cut] - base
    h = hd[lo:cut] - hd[lo]
    if len(h) <= 10 or h[-1] == 0:
        return 0.0
    dr, dh = r - r[-1], h - h[-1]
    den = np.sum(dh * dh)
    return float(np.sum(dr * dh) / den) if den else 0.0


def main():
    full = "--full" in sys.argv
    rng = np.random.default_rng(SEED)
    all_wells = [w for w in wells() if w not in TEST_IDS]
    ids = all_wells if full else sorted(rng.choice(all_wells, size=N_SAMPLE, replace=False))

    names = [f"anchor{w}" for w in WINS] + [n for n, _, _ in SHRINKS]
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
        hpost, y = hd[cut:], tvt[cut:]
        npt += len(y)

        for w in WINS:
            s = anchor_slope(tvt_in, z, hd, cut, w)
            sq[f"anchor{w}"] += float(np.sum((y - (flat[cut:] + s * hpost)) ** 2))
        for name, w, k in SHRINKS:
            s = k * anchor_slope(tvt_in, z, hd, cut, w)
            sq[name] += float(np.sum((y - (flat[cut:] + s * hpost)) ** 2))

    tag = "770 pozos" if full else f"muestra {len(ids)}"
    print(f"== RMSE global ({tag}, {npt} puntos) ==")
    for v in sorted(names, key=lambda v: sq[v]):
        print(f"  {v:16s} {np.sqrt(sq[v] / npt):7.2f} ft")


if __name__ == "__main__":
    main()
