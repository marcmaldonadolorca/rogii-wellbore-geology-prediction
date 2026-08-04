"""SUR-04: interpolacion con UN punto por pozo vecino (en vez de k puntos crudos).

Diagnostico del IDW actual: a subsample 10 los puntos de un mismo pozo estan a
10 ft; el pozo vecino mas cercano esta a ~211-292 ft. Los k=16 vecinos de una
consulta salen casi todos del MISMO pozo vecino -> el IDW es practicamente
"copia el valor del pozo mas cercano" y el ajuste de plano local es colineal.

Aqui se toma el punto MAS CERCANO DE CADA POZO vecino: M muestras de M pozos
distintos, repartidas en 2D. Con eso el IDW promedia informacion independiente y
el plano local queda bien condicionado.

LOWO es trivial: excluir un pozo = no consultar su arbol.
"""
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, HERE)

import sur02_lib as L  # noqa: E402


class PerWellField:
    """773 arboles (uno por pozo) + prefiltro de pozos candidatos."""

    def __init__(self, subsample=10, mcand=80):
        z = np.load(L.CLOUD)
        self.xy = z["xy"][::subsample]
        self.s = z["s"][::subsample].astype(np.float64)
        self.wi = z["wi"][::subsample]
        self.ids = [str(x) for x in z["ids"]]
        self.idx = {n: i for i, n in enumerate(self.ids)}
        self.nw = len(self.ids)
        self.mcand = mcand
        order = np.argsort(self.wi, kind="stable")
        self.xy, self.s, self.wi = self.xy[order], self.s[order], self.wi[order]
        bnd = np.searchsorted(self.wi, np.arange(self.nw + 1))
        self.sl = [slice(bnd[i], bnd[i + 1]) for i in range(self.nw)]
        self.trees = [cKDTree(self.xy[s_]) for s_ in self.sl]
        # centroides para el prefiltro grueso de candidatos
        self.cxy = np.array([self.xy[s_].mean(0) for s_ in self.sl])
        self.ctree = cKDTree(self.cxy)

    def sample(self, wid, X, Y, mwell=24):
        """(dist, vals, wells) con el punto mas cercano de cada uno de los
        `mwell` pozos mas proximos, excluyendo wid. dist (n,m), vals (n,m,6)."""
        q = np.column_stack([X, Y])
        self_i = self.idx.get(wid, -1)
        # candidatos: pozos cuyo centroide esta cerca del centroide de la consulta
        cand = self.ctree.query(q.mean(0)[None, :],
                                k=min(self.mcand, self.nw))[1].ravel()
        cand = np.array([c for c in cand if c != self_i])
        D = np.empty((len(q), len(cand)))
        I = np.empty((len(q), len(cand)), np.int64)
        for j, c in enumerate(cand):
            d, i = self.trees[c].query(q, k=1, workers=-1)
            D[:, j] = d
            I[:, j] = i + self.sl[c].start
        o = np.argsort(D, axis=1)[:, :mwell]
        r = np.arange(len(q))[:, None]
        return D[r, o], self.s[I[r, o]], cand[o]


def pw_idw(F, D, V, m=12, p=2.0):
    D, V = D[:, :m], V[:, :m]
    w = (1.0 / np.maximum(D, 1e-3) ** p)[:, :, None] * np.isfinite(V)
    s = w.sum(1, keepdims=True)
    w = np.divide(w, s, out=np.zeros_like(w), where=s > 0)
    return np.einsum("nkf,nkf->nf", w, np.nan_to_num(V))


def pw_plane(F, D, V, XY, PXY, m=12, p=2.0, lam=1e-2, clip=None):
    """Plano local ponderado con ridge, sobre M muestras de M pozos distintos."""
    D, V, PXY = D[:, :m], V[:, :m], PXY[:, :m]
    n, k = D.shape
    w = (1.0 / np.maximum(D, 1e-3) ** p)[:, :, None] * np.isfinite(V)
    s = w.sum(1, keepdims=True)
    w = np.divide(w, s, out=np.zeros_like(w), where=s > 0)
    V0 = np.nan_to_num(V)
    base = np.einsum("nkf,nkf->nf", w, V0)
    dx = PXY - XY[:, None, :]
    h = np.maximum(np.linalg.norm(dx, axis=2).mean(1), 1e-6)[:, None, None]
    A = np.concatenate([np.ones((n, k, 1)), dx / h], 2)
    reg = np.diag([0.0, lam, lam])
    out = np.empty((n, 6))
    for f in range(6):
        Aw = A * w[:, :, f, None]
        G = np.einsum("nki,nkj->nij", Aw, A) + reg
        b = np.einsum("nki,nk->ni", Aw, V0[:, :, f])
        out[:, f] = np.linalg.solve(G, b[:, :, None])[:, 0, 0]
    if clip is not None:
        out = base + np.clip(out - base, -clip, clip)
    return np.where(np.isfinite(out), out, base)


if __name__ == "__main__":
    import pandas as pd
    t0 = time.time()
    F = PerWellField(subsample=10)
    print(f"773 arboles: {time.time()-t0:.1f}s")
    df = pd.read_csv("/home/ftpx100/work/active/kaggle-rogii/data/raw/train/"
                     "000d7d20__horizontal_well.csv")
    t0 = time.time()
    D, V, W = F.sample("000d7d20", df.X.values, df.Y.values, mwell=24)
    print(f"muestreo 1 pozo: {time.time()-t0:.2f}s  d1 med={np.median(D[:,0]):.0f} "
          f"d12 med={np.median(D[:,11]):.0f}  pozos distintos={len(np.unique(W))}")
    S = pw_idw(F, D, V, 12)
    print("sd(TVT-(S-Z)) pw_idw m12:", np.nanstd(df.TVT.values - (S[:, 5] - df.Z.values)))
