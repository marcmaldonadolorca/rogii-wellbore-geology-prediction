"""A) Sensibilidad del detector de faults (ventanas 5/20/50/200 muestras).
B) Variantes baratas del dip pre-PS vs baseline lin500, misma muestra de 200 pozos.
   Con --full evalua las variantes sobre los 770 pozos (comparable al 39.4).
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


def slope_fit(tvt_in, z, hd, cut, win, deg=1):
    lo = max(0, cut - win)
    base = tvt_in[lo] + (z[lo] - z[lo:cut])
    resid = tvt_in[lo:cut] - base
    h = hd[lo:cut] - hd[lo]
    if len(h) <= 10 or h[-1] == 0:
        return None
    return np.polyfit(h, resid, deg)  # coefs mayor->menor


def main():
    full = "--full" in sys.argv
    rng = np.random.default_rng(SEED)
    all_wells = [w for w in wells() if w not in TEST_IDS]
    ids = all_wells if full else sorted(rng.choice(all_wells, size=N_SAMPLE, replace=False))

    fault_wins = [5, 20, 50, 200]
    fault_max = {w: [] for w in fault_wins}   # max salto por pozo
    variants = ["flat", "lin300", "lin500", "lin1000", "lin2000", "lin_all",
                "quad500", "quad1000", "quad2000", "shrink08", "damped3000",
                "blend500_2000", "avg3win"]
    sq = {v: 0.0 for v in variants}
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

        # A) faults: residuo flat post-PS, salto max por ventana
        if not full:
            r = (tvt + z)[cut:]
            for w in fault_wins:
                if len(r) > w:
                    fault_max[w].append(float(np.max(np.abs(r[w:] - r[:-w]))))

        # B) variantes
        slopes = {}
        for win in (300, 500, 1000, 2000, 10 ** 9):
            c = slope_fit(tvt_in, z, hd, cut, win, 1)
            slopes[win] = c[0] if c is not None else 0.0
        preds = {
            "flat": flat[cut:],
            "lin300": flat[cut:] + slopes[300] * hpost,
            "lin500": flat[cut:] + slopes[500] * hpost,
            "lin1000": flat[cut:] + slopes[1000] * hpost,
            "lin2000": flat[cut:] + slopes[2000] * hpost,
            "lin_all": flat[cut:] + slopes[10 ** 9] * hpost,
            "shrink08": flat[cut:] + 0.8 * slopes[500] * hpost,
            "blend500_2000": flat[cut:] + 0.5 * (slopes[500] + slopes[2000]) * hpost,
            "avg3win": flat[cut:] + (slopes[500] + slopes[1000] + slopes[2000]) / 3 * hpost,
        }
        L = 3000.0
        preds["damped3000"] = flat[cut:] + slopes[500] * L * (1 - np.exp(-hpost / L))
        for win, name in ((500, "quad500"), (1000, "quad1000"), (2000, "quad2000")):
            c = slope_fit(tvt_in, z, hd, cut, win, 2)
            if c is None:
                preds[name] = preds["flat"]
            else:
                lo = max(0, cut - win)
                h = hpost - (hd[lo] - hd[lo])  # h medido desde lo: hpost - hd[lo] + 0
                hq = hpost - hd[lo]
                base_ps = np.polyval(c, hd[cut - 1] - hd[lo])
                preds[name] = flat[cut:] + np.polyval(c, hq) - base_ps
        for v in variants:
            sq[v] += float(np.sum((y - preds[v]) ** 2))

    tag = "770 pozos" if full else f"muestra {len(ids)} pozos"
    print(f"== B) RMSE global por variante ({tag}, {npt} puntos) ==")
    for v in sorted(variants, key=lambda v: sq[v]):
        print(f"  {v:14s} {np.sqrt(sq[v] / npt):7.2f} ft")

    if not full:
        print("\n== A) max salto |dTVT+dZ| post-PS por pozo, segun ventana ==")
        for w in fault_wins:
            a = np.array(fault_max[w])
            print(f"  win={w:3d} muestras: p50={np.median(a):6.2f} p90={np.quantile(a,.9):6.2f} "
                  f"max={a.max():7.2f} | pozos>20ft: {(a>20).mean()*100:5.1f}% "
                  f"| >50ft: {(a>50).mean()*100:4.1f}%")


if __name__ == "__main__":
    main()
