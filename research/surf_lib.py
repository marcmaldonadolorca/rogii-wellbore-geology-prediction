"""Libreria de interpoladores de superficie para ROGII (research).

Objetivo: sustituir el IDW 1/d^2 k=16 de model.py por algo mejor, midiendo cada
paso. Todo con LEAVE-ONE-WELL-OUT exacto.

Truco de rendimiento (LOWO barato):
  reconstruir un cKDTree de 5M puntos por pozo es inviable. Como el pozo a
  predecir ocupa una caja pequenya del campo, se filtra la nube a la caja del
  pozo + MARGIN y se excluye el propio pozo. El arbol local tiene ~1e5 puntos y
  se construye en ~0.1 s. Es EXACTO mientras la distancia al vecino k-esimo sea
  << MARGIN (se comprueba con assert_margin()).

Interpoladores (todos sobre los mismos k vecinos del arbol local):
  idw     w = 1/d^p
  plane   ajuste de plano local ponderado con RIDGE sobre el gradiente
  krige   kriging ordinario local, variograma exponencial/esferico + nugget
  rbf     scipy RBFInterpolator (thin_plate_spline / linear) con smoothing
Cualquiera admite detrend=True: se interpola el residuo respecto a una
tendencia global cuadratica ajustada a los 773 pozos.
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
NPZ = HERE / "surf_cloud.npz"
FRM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
MARGIN = 12000.0        # ft de holgura de la caja local


class Cloud:
    """Nube completa (X, Y, S_1..S_6) de los 773 pozos de train, float32."""

    def __init__(self):
        d = np.load(NPZ, allow_pickle=True)
        self.xy = np.ascontiguousarray(d["xy"], dtype=np.float32)
        self.s = np.ascontiguousarray(d["s"], dtype=np.float32)
        self.wid = np.ascontiguousarray(d["wid"])
        self.names = [str(x) for x in d["names"]]
        self.frm = [str(x) for x in d["frm"]]
        self.idx = {n: i for i, n in enumerate(self.names)}
        del d
        self.valid = np.isfinite(self.s)                        # (n, 6)
        self.x0 = float(self.xy[:, 0].mean()); self.xs = float(self.xy[:, 0].std())
        self.y0 = float(self.xy[:, 1].mean()); self.ys = float(self.xy[:, 1].std())
        self.trend = np.array([
            np.linalg.lstsq(self._poly(self.xy[self.valid[:, j], 0].astype(np.float64),
                                       self.xy[self.valid[:, j], 1].astype(np.float64)),
                            self.s[self.valid[:, j], j].astype(np.float64), rcond=None)[0]
            for j in range(self.s.shape[1])])                   # (6, 6)

    def _poly(self, X, Y):
        u = (X - self.x0) / self.xs
        v = (Y - self.y0) / self.ys
        return np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])

    def trend_at(self, X, Y):
        return self._poly(np.asarray(X, float), np.asarray(Y, float)) @ self.trend.T


class LocalField:
    """Contexto de interpolacion para UN pozo: arbol local sin ese pozo.

    aniso: factor de escala del eje Y (tras rotar theta rad) antes del KDTree.
    """

    def __init__(self, cloud, wid, qx, qy, subsample=10, aniso=1.0, theta=0.0,
                 margin=MARGIN, detrend=False):
        c = self.cloud = cloud
        self.aniso, self.theta = aniso, theta
        wex = c.idx.get(wid, -1)
        x0, x1 = qx.min() - margin, qx.max() + margin
        y0, y1 = qy.min() - margin, qy.max() + margin
        m = ((c.xy[:, 0] >= x0) & (c.xy[:, 0] <= x1) &
             (c.xy[:, 1] >= y0) & (c.xy[:, 1] <= y1) & (c.wid != wex))
        if subsample > 1:
            m &= (np.arange(len(m)) % subsample) == 0
        self.sub_xy = c.xy[m].astype(np.float64)
        self.sub_s = c.s[m].astype(np.float64)
        self.sub_valid = c.valid[m]
        self.margin = margin
        self.detrend = detrend
        if detrend:
            self.sub_v = self.sub_s - c.trend_at(self.sub_xy[:, 0], self.sub_xy[:, 1])
        else:
            self.sub_v = self.sub_s
        self.pts = self._warp(self.sub_xy[:, 0], self.sub_xy[:, 1])
        self.tree = cKDTree(self.pts)
        self.n_all_valid = self.sub_valid.all(0)                # (6,) bool
        self._trees_f = {}
        self.max_d = 0.0

    def _warp(self, X, Y):
        if self.theta:
            ct, st = np.cos(self.theta), np.sin(self.theta)
            X, Y = ct * X + st * Y, -st * X + ct * Y
        return np.ascontiguousarray(np.column_stack([X, Y * self.aniso]))

    def tree_for(self, j):
        if self.n_all_valid[j]:
            return self.tree, None
        if j not in self._trees_f:
            keep = np.flatnonzero(self.sub_valid[:, j])
            self._trees_f[j] = (cKDTree(self.pts[keep]), keep)
        return self._trees_f[j]

    def query(self, qx, qy, j, k):
        tree, keep = self.tree_for(j)
        q = self._warp(np.asarray(qx, float), np.asarray(qy, float))
        kk = min(k, tree.n)
        dist, ind = tree.query(q, k=kk, workers=-1)
        if kk == 1:
            dist, ind = dist[:, None], ind[:, None]
        if keep is not None:
            ind = keep[ind]
        self.max_d = max(self.max_d, float(dist.max()))
        return q, dist, ind

    def assert_margin(self):
        lim = self.margin * min(1.0, self.aniso)
        assert self.max_d < 0.6 * lim, f"margen insuficiente: {self.max_d:.0f} vs {lim:.0f}"

    def finish(self, out, qx, qy, j):
        if self.detrend:
            out = out + self.cloud.trend_at(qx, qy)[:, j]
        return out


# ---------------------------------------------------------------- interpoladores

def idw(lf, q, dist, ind, j, p=2.0, **kw):
    w = 1.0 / np.maximum(dist, 1e-3) ** p
    w /= w.sum(1, keepdims=True)
    return (w * lf.sub_v[ind, j]).sum(1)


def plane(lf, q, dist, ind, j, lam=1.0, p=2.0, **kw):
    """S ~ a + gx*u + gy*v ponderado por 1/d^p, con ridge lam sobre (gx, gy).

    u, v = desplazamiento al punto de consulta / distancia media de los vecinos
    => lam adimensional; lam -> inf degenera exactamente en IDW.
    """
    v = lf.sub_v[ind, j]                                        # (n, k)
    d = q[:, None, :] - lf.pts[ind]                             # (n, k, 2)
    L = np.maximum(dist.mean(1), 1e-6)[:, None, None]
    d = d / L
    w = 1.0 / np.maximum(dist, 1e-3) ** p
    w /= w.sum(1, keepdims=True)
    B = np.concatenate([np.ones(d.shape[:2] + (1,)), d], axis=2)
    Bw = B * w[:, :, None]
    A = np.einsum("nki,nkj->nij", Bw, B)
    A[:, 1, 1] += lam
    A[:, 2, 2] += lam
    b = np.einsum("nki,nk->ni", Bw, v)
    return np.linalg.solve(A, b)[:, 0]


def krige(lf, q, dist, ind, j, nugget=1.0, sill=100.0, rng=1000.0, model="exp", **kw):
    """Kriging ordinario local con los k vecinos."""
    P = lf.pts[ind]                                             # (n,k,2)
    dij = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)

    def gam(h):
        if model == "sph":
            hh = np.minimum(h / rng, 1.0)
            g = sill * (1.5 * hh - 0.5 * hh**3)
        elif model == "gau":
            g = sill * (1.0 - np.exp(-(h / rng) ** 2))
        else:
            g = sill * (1.0 - np.exp(-h / rng))
        return np.where(h > 0, nugget + g, 0.0)

    n, k = dist.shape
    A = np.zeros((n, k + 1, k + 1))
    A[:, :k, :k] = gam(dij)
    A[:, :k, k] = 1.0
    A[:, k, :k] = 1.0
    A[:, np.arange(k), np.arange(k)] = 0.0
    A[:, k, k] = 0.0
    A[:, :k, :k] += 1e-6 * np.eye(k)
    b = np.ones((n, k + 1))
    b[:, :k] = gam(dist)
    lam = np.linalg.solve(A, b)
    return (lam[:, :k] * lf.sub_v[ind, j]).sum(1)


def rbf(lf, q, dist, ind, j, kernel="thin_plate_spline", smoothing=1.0, degree=None, **kw):
    from scipy.interpolate import RBFInterpolator
    tree, keep = lf.tree_for(j)
    pts = lf.pts if keep is None else lf.pts[keep]
    v = lf.sub_v[:, j] if keep is None else lf.sub_v[keep, j]
    kk = min(dist.shape[1], len(pts))
    ex = {} if degree is None else {"degree": degree}
    itp = RBFInterpolator(pts, v, neighbors=kk, kernel=kernel, smoothing=smoothing, **ex)
    return itp(q)


METHODS = {"idw": idw, "plane": plane, "krige": krige, "rbf": rbf}
