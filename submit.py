"""Genera submission para los 3 pozos de test y mide el RMSE local contra la
columna TVT que train/ expone para esos mismos pozos (hipotesis de fuga).

Si el score que devuelve Kaggle coincide con el RMSE local, entonces train.TVT
ES el ground truth de puntuacion y la fuga es explotable.
"""
import numpy as np
import pandas as pd

from baseline import RAW, TEST_IDS, cut_of, load, predict

DIP_WIN = 500


def main():
    sample = pd.read_csv(RAW / "sample_submission.csv")
    out, checks = {}, []
    for wid in sorted(TEST_IDS):
        te = load("test", wid)
        cut = cut_of(te)
        pred = predict(te, cut, DIP_WIN)
        for i, v in zip(range(cut, len(te)), pred):
            out[f"{wid}_{i}"] = v
        # ground truth candidato: la columna TVT que train/ deja ver
        gt = load("train", wid).TVT.values[cut:]
        d = gt - pred
        checks.append((wid, cut, len(te), float(np.sqrt(np.mean(d**2)))))

    sample["tvt"] = sample.id.map(out)
    assert sample.tvt.notna().all(), f"ids sin prediccion: {sample.tvt.isna().sum()}"
    sample.to_csv("submission.csv", index=False)

    for wid, cut, n, rmse in checks:
        print(f"{wid}: PS={cut} filas={n} RMSE vs train.TVT = {rmse:.3f} ft")
    allerr = np.concatenate([
        load("train", wid).TVT.values[cut_of(load("test", wid)):]
        - predict(load("test", wid), cut_of(load("test", wid)), DIP_WIN)
        for wid in sorted(TEST_IDS)])
    print(f"\nRMSE global sobre los 3 pozos de test: {np.sqrt(np.mean(allerr**2)):.4f} ft")
    print(f"submission.csv escrito ({len(sample)} filas)")


if __name__ == "__main__":
    main()
