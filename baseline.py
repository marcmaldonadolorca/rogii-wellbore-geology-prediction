"""Baseline honesto para ROGII wellbore: geologia plana + dip estimado pre-PS.

Metrica oficial: RMSE de dTVT = manualTVT - predictedTVT sobre todos los puntos
predichos (unidades: pies). Validacion local con los pozos de train que NO son
los 3 de test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data/raw"
TEST_IDS = {"000d7d20", "00bbac68", "00e12e8b"}


def wells(split):
    return sorted(p.name.split("__")[0] for p in (RAW / split).glob("*__horizontal_well.csv"))


def load(split, wid):
    return pd.read_csv(RAW / split / f"{wid}__horizontal_well.csv")


def cut_of(df):
    """Indice del punto Prediction Start = primera fila sin TVT_input."""
    m = df.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df)


def predict(df, cut, dip_win=300):
    """TVT plano desde PS, mas correccion de dip lineal ajustada pre-PS."""
    z, tvt_in = df.Z.values, df.TVT_input.values
    flat = tvt_in[cut - 1] + (z[cut - 1] - z)

    # distancia horizontal acumulada desde PS
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    hd = np.cumsum(step) - np.cumsum(step)[cut - 1]

    pred = flat.copy()
    if dip_win:
        # residuo del modelo plano en la ventana pre-PS -> pendiente vs distancia horizontal
        lo = max(0, cut - dip_win)
        base = tvt_in[lo] + (z[lo] - z[lo:cut])
        resid = tvt_in[lo:cut] - base
        h = hd[lo:cut] - hd[lo]
        if len(h) > 10 and h[-1] != 0:
            slope = np.polyfit(h, resid, 1)[0]
            pred = flat + slope * hd
    return pred[cut:]


def main():
    dip_win = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    err, rows = [], []
    for wid in wells("train"):
        if wid in TEST_IDS:
            continue
        df = load("train", wid)
        cut = cut_of(df)
        if cut < 20 or cut >= len(df):
            continue
        d = df.TVT.values[cut:] - predict(df, cut, dip_win)
        err.append(d)
        rows.append((wid, len(d), float(np.sqrt(np.mean(d**2)))))

    all_err = np.concatenate(err)
    per = pd.DataFrame(rows, columns=["well", "n", "rmse"])
    print(f"dip_win={dip_win} | pozos={len(per)} puntos={len(all_err)}")
    print(f"RMSE global (metrica oficial): {np.sqrt(np.mean(all_err**2)):.3f} ft")
    print(f"RMSE por pozo: mediana {per.rmse.median():.3f} | p90 {per.rmse.quantile(.9):.3f} | max {per.rmse.max():.3f}")


if __name__ == "__main__":
    main()
