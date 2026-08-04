"""Utilidades comunes para el estudio de blend adaptativo (ad*)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRACS = (0.5, 0.65, 0.75)
NAMES = ("S", "H", "P", "Z", "B", "G")


class Cache:
    """Cache de un conjunto de pozos: arrays post-PS concatenados + backtests."""

    def __init__(self, path):
        self.path = Path(path)
        d = np.load(self.path)
        self.d = d
        self.lens = d["lens"]
        self.NW = len(self.lens)
        self.wid = np.repeat(np.arange(self.NW), self.lens)
        for n in list(NAMES) + ["y", "nn", "md_since", "hd_since", "MDv", "Zv"]:
            setattr(self, n, d[n].astype(np.float64))
        for n in ("Pspr", "Pstd"):
            setattr(self, n, d[n].astype(np.float64) if n in d else None)
        mf = str(self.path).replace(".npz", "_meta.csv")
        if not Path(mf).exists():
            mf = str(self.path).replace("cache", "meta").replace("_meta.csv", "_meta.csv")
        self.meta = pd.read_csv(mf) if Path(mf).exists() else None
        # backtests
        self.bt = {}
        for f in FRACS:
            ft = str(f).replace(".", "")
            key = f"btlens_{ft}"
            if key not in d:
                continue
            bl = d[key]
            b = {"lens": bl, "wid": np.repeat(np.arange(self.NW), bl)}
            for n in list(NAMES) + ["y", "md_since"]:
                kk = f"bt{n}_{ft}"
                if kk in d:
                    b[n] = d[kk].astype(np.float64)
            for n in ("Pspr", "Pstd"):
                kk = f"bt{n}_{ft}"
                if kk in d:
                    b[n] = d[kk].astype(np.float64)
            self.bt[f] = b

    # ---- metricas
    def rmse(self, p, m=None):
        e = (self.y - p) if m is None else (self.y[m] - p[m])
        return float(np.sqrt(np.mean(e ** 2)))

    def sse_well(self, p):
        return np.bincount(self.wid, (self.y - p) ** 2, self.NW)

    def n_well(self):
        return self.lens.astype(float)

    def rmse_well(self, p):
        return np.sqrt(self.sse_well(p) / np.maximum(self.n_well(), 1))

    def lb_proxy(self, p):
        s = self.sse_well(p)
        m = self.lens >= 5147
        return float(np.sqrt(s[m].sum() / self.lens[m].sum())) if m.any() else float("nan")

    def expand(self, v):
        """array por pozo -> array por punto."""
        return np.asarray(v)[self.wid]

    def per_well_med(self, x):
        """mediana por pozo de un array por punto, expandida a puntos."""
        med = np.array([np.median(x[self.wid == i]) for i in range(self.NW)])
        return med, med[self.wid]

    def wopt(self, p, q, m=None):
        """w que minimiza ||y-(p+w(q-p))||, opcionalmente en la mascara m."""
        d = (q - p) if m is None else (q - p)[m]
        r = (self.y - p) if m is None else (self.y - p)[m]
        den = float(d @ d)
        return float(r @ d / den) if den > 0 else np.nan

    def wopt_well(self, p, q):
        d = q - p
        num = np.bincount(self.wid, (self.y - p) * d, self.NW)
        den = np.bincount(self.wid, d * d, self.NW)
        return np.where(den > 1e-9, num / np.maximum(den, 1e-12), 0.0)


def report(c, name, p, ref=None):
    r = c.rmse(p)
    d = "" if ref is None else f"  ({r-ref:+.3f})"
    print(f"  {name:52s} {r:7.3f}{d}")
    return r
