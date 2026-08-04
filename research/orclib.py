"""Utilidades comunes de los oraculos: carga del cache maestro y RMSE pooled."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv import well_ids, _select, SEED, LB_PROXY_NPRED  # noqa: E402

PRED = ("geom", "surf", "ancc", "pfz", "beam")


class Cache:
    """Cache maestro (orc01) con vistas por pozo."""

    def __init__(self, tag=""):
        f = ROOT / f"research/orc01_cache{tag}.npz"
        z = np.load(f)
        self.d = {k: z[k].astype(np.float64) for k in z.files if k != "lens"}
        self.lens = z["lens"]
        self.meta = pd.read_csv(str(f).replace(".npz", "_meta.csv"))
        self.off = np.concatenate([[0], np.cumsum(self.lens)])
        self.wells = self.meta.well.values
        self.widx = {w: i for i, w in enumerate(self.wells)}
        # id de pozo por punto
        self.wpt = np.repeat(np.arange(len(self.lens)), self.lens)

    def sl(self, i):
        return slice(self.off[i], self.off[i + 1])

    def mask(self, ids=None):
        """Mascara booleana de puntos para un subconjunto de pozos."""
        if ids is None:
            return np.ones(len(self.d["y"]), bool)
        keep = np.zeros(len(self.lens), bool)
        for w in ids:
            if w in self.widx:
                keep[self.widx[w]] = True
        return keep[self.wpt]

    def blend(self, **w):
        """Combinacion lineal de predictores: blend(surf=.25, ancc=.75)."""
        out = np.zeros(len(self.d["y"]))
        for k, v in w.items():
            out += v * self.d[k]
        return out


def rmse(y, p, m=None):
    if m is not None:
        y, p = y[m], p[m]
    return float(np.sqrt(np.mean((y - p) ** 2)))


def subset(k):
    return set(_select(well_ids(), k, SEED))


def per_well_sse(c, pred, m=None):
    """(sse, n) por pozo para un vector de prediccion de longitud completa."""
    e2 = (c.d["y"] - pred) ** 2
    if m is not None:
        e2 = np.where(m, e2, 0.0)
    sse = np.bincount(c.wpt, weights=e2, minlength=len(c.lens))
    n = np.bincount(c.wpt, weights=(m if m is not None else np.ones(len(e2))).astype(float),
                    minlength=len(c.lens))
    return sse, n
