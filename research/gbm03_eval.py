"""Evaluacion honesta del GBM arbitro + barridos de postproceso, desde el OOF.

Todo se calcula sobre las predicciones OOF a resolucion completa (gbm02_oof.npz):
cada pozo lo predice el modelo del fold que NO lo vio. Reproduce exactamente el
pooled RMSE de cv.evaluate cuando se restringe al subconjunto de pozos de k.

Uso: python research/gbm03_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cv import well_ids, _select, SEED  # noqa: E402


def pooled(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return float(np.sqrt(np.mean((y - p) ** 2)))


def per_well(arr, lens):
    off = np.concatenate([[0], np.cumsum(lens)])
    return [arr[off[i]:off[i + 1]] for i in range(len(lens))]


def apply_pw(pred_w, fn):
    return np.concatenate([fn(p, i) for i, p in enumerate(pred_w)])


def main():
    d = np.load(ROOT / "research/gbm02_oof.npz")
    meta = pd.read_csv(ROOT / "research/gbm01_meta.csv")
    lens = d["lens"]
    y = d["y"].astype(np.float64)
    wells = meta.well.values
    ids = well_ids()
    s60 = set(_select(ids, 60, SEED))
    s150 = set(_select(ids, 150, SEED))
    in60 = np.repeat(np.isin(wells, list(s60)), lens)
    in150 = np.repeat(np.isin(wells, list(s150)), lens)
    md_w = None

    names = ["base_surf", "base_ancc", "base_blend", "pred_raw", "pred_res"]
    print("pooled RMSE (OOF, resolucion completa)")
    print(f"{'variante':14s} {'pool':>8s} {'k=60':>8s} {'k=150':>8s}")
    for n in names:
        p = d[n].astype(np.float64)
        print(f"{n:14s} {pooled(y,p):8.3f} {pooled(y,p,in60):8.3f} {pooled(y,p,in150):8.3f}")

    # md_since por punto para la rampa: se recupera del cache de features
    c = np.load(ROOT / "research/gbm01_cache.npz")
    md_since = c["md_since"].astype(np.float64)
    dancc = d["base_ancc"].astype(np.float64)

    best_tag, best_v = None, 1e9
    for tag in ("pred_raw", "pred_res"):
        p0 = d[tag].astype(np.float64)
        print(f"\n=== postproceso sobre {tag} (base pool={pooled(y,p0):.3f}) ===")
        print("shrink alpha:")
        for a in (0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05):
            print(f"   a={a:5.2f} pool={pooled(y,a*p0):7.3f} k60={pooled(y,a*p0,in60):7.3f} k150={pooled(y,a*p0,in150):7.3f}")
        print("rampa 1-exp(-md/tau):")
        for tau in (30, 60, 85, 150, 300, 600):
            r = 1.0 - np.exp(-md_since / tau)
            print(f"   tau={tau:4d} pool={pooled(y,r*p0):7.3f} k60={pooled(y,r*p0,in60):7.3f} k150={pooled(y,r*p0,in150):7.3f}")
        print("savgol por pozo:")
        pw = per_well(p0, lens)
        for win, po in ((17, 3), (51, 3), (201, 3), (801, 3)):
            sg = np.concatenate([savgol_filter(p, min(win, len(p) - (1 - len(p) % 2)), po)
                                 if len(p) > win else p for p in pw])
            print(f"   win={win:4d} pool={pooled(y,sg):7.3f} k60={pooled(y,sg,in60):7.3f} k150={pooled(y,sg,in150):7.3f}")
        print("mezcla con pf_ancc:")
        for w in (0.0, 0.05, 0.1, 0.15, 0.25, 0.4):
            p = (1 - w) * p0 + w * dancc
            print(f"   w={w:4.2f} pool={pooled(y,p):7.3f} k60={pooled(y,p,in60):7.3f} k150={pooled(y,p,in150):7.3f}")
        if pooled(y, p0) < best_v:
            best_v, best_tag = pooled(y, p0), tag

    # per-well: donde gana/pierde el GBM contra el blend
    pwy = per_well(y, lens)
    pwb = per_well(d["base_blend"].astype(np.float64), lens)
    pwg = per_well(d[best_tag].astype(np.float64), lens)
    rows = [{"well": wells[i], "n": lens[i],
             "rmse_blend": float(np.sqrt(np.mean((pwy[i] - pwb[i]) ** 2))),
             "rmse_gbm": float(np.sqrt(np.mean((pwy[i] - pwg[i]) ** 2)))} for i in range(len(lens))]
    pw = pd.DataFrame(rows)
    pw["gain"] = pw.rmse_blend - pw.rmse_gbm
    pw.to_csv(ROOT / "research/gbm03_perwell.csv", index=False)
    print(f"\npozos donde el GBM mejora: {(pw.gain>0).mean()*100:.0f}%  "
          f"mediana gain {pw.gain.median():.2f} ft")
    print(pw.sort_values("gain").head(5).to_string(index=False))
    print(pw.sort_values("gain").tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
