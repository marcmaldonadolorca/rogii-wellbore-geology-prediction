"""SUR-02: nube multi-formacion + vecinos LOWO precomputados + interpoladores.

Eficiencia: para un (subsample, anisotropia) dado los VECINOS de cada pozo
evaluado no dependen ni de k (basta pedir KMAX y recortar) ni del interpolador
ni de la formacion. Se calculan UNA vez por pozo con exclusion exacta del propio
pozo (LOWO) y se cachean en disco; los barridos de k / formaciones / pesos /
interpolador son casi gratis despues.
En el re-run de Kaggle no hay cache: se construye el arbol y se consulta igual
(coste medido: ~0.11 s por pozo a subsample 10).

NaN: ANCC falta en 7 pozos ENTEROS y EGFDL en 1. Se gestiona con mascara de
validez por formacion (pesos a cero + renormalizacion), sin rellenar nada, para
no introducir fugas.
"""
import hashlib
import os
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD = os.path.join(HERE, "sur01_cloud.npz")
CACHE = os.path.join(HERE, "sur_nb_cache")
FORM = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
KMAX = 64
THETA = np.arctan2(0.86994462, -0.49314942)   # direccion principal (PCA, sur01)


class Cloud:
    """Nube (X,Y) + las 6 superficies, submuestreada, con metrica anisotropa.

    aniso: factor por el que se ESTIRA el eje PARALELO a las trayectorias antes
    del KDTree. aniso>1 => los vecinos a lo largo del propio pozo se "alejan" y
    el arbol prefiere vecinos perpendiculares (de otros pozos).
    """

    def __init__(self, subsample=10, aniso=1.0, theta=THETA):
        z = np.load(CLOUD)
        self.xy_raw = np.ascontiguousarray(z["xy"][::subsample])
        self.s = np.ascontiguousarray(z["s"][::subsample].astype(np.float64))
        self.wi = np.ascontiguousarray(z["wi"][::subsample])
        self.ok = np.isfinite(self.s)                 # (N,6) validez
        self.s0 = np.where(self.ok, self.s, 0.0)      # sin NaN, para einsum
        self.ids = [str(x) for x in z["ids"]]
        self.idx = {n: i for i, n in enumerate(self.ids)}
        self.subsample, self.aniso, self.theta = subsample, aniso, theta
        c, s_ = np.cos(theta), np.sin(theta)
        self.R = np.array([[c, s_], [-s_, c]])        # -> (paralelo, perp)
        self.scale = np.array([aniso, 1.0])
        self.xy = np.ascontiguousarray((self.xy_raw @ self.R.T) * self.scale)

    def transform(self, X, Y):
        return (np.column_stack([X, Y]) @ self.R.T) * self.scale

    def key(self):
        return f"s{self.subsample}_a{self.aniso:g}_t{self.theta:.4f}"

    def neighbors(self, wid, X, Y, kmax=KMAX, use_cache=True):
        """(d, gi, dr): distancia anisotropa, indice global y distancia REAL al
        vecino 1, de los kmax vecinos mas cercanos EXCLUYENDO el pozo wid."""
        h = hashlib.md5(f"{self.key()}_{wid}_{kmax}".encode()).hexdigest()[:16]
        f = os.path.join(CACHE, f"{h}.npz")
        if use_cache and os.path.exists(f):
            z = np.load(f)
            return z["d"].astype(np.float64), z["i"], z["dr"]
        sub = np.where(self.wi != self.idx[wid])[0]
        tree = cKDTree(self.xy[sub])
        d, i = tree.query(self.transform(X, Y), k=kmax, workers=-1)
        gi = sub[i].astype(np.int32)
        dr = np.hypot(self.xy_raw[gi[:, 0], 0] - X,
                      self.xy_raw[gi[:, 0], 1] - Y).astype(np.float32)
        if use_cache:
            os.makedirs(CACHE, exist_ok=True)
            np.savez(f, d=d.astype(np.float32), i=gi, dr=dr)
        return d, gi, dr


# ───────────────────────────── interpoladores ────────────────────────────────
# Todos devuelven (n, 6): una superficie interpolada por formacion.

def _wmask(cloud, d, gi, k, p):
    """Pesos IDW (n,k,6) ya enmascarados por validez y normalizados."""
    d, gi = d[:, :k], gi[:, :k]
    w = (1.0 / np.maximum(d, 1e-3) ** p)[:, :, None] * cloud.ok[gi]
    s = w.sum(1, keepdims=True)
    return np.divide(w, s, out=np.zeros_like(w), where=s > 0), gi


def idw(cloud, d, gi, k=16, p=2.0):
    w, gi = _wmask(cloud, d, gi, k, p)
    return np.einsum("nkf,nkf->nf", w, cloud.s0[gi])


def linridge(cloud, d, gi, X, Y, k=32, lam=1e-2, p=2.0, clip=None):
    """Plano local ponderado con RIDGE sobre el gradiente.

    S ~ a + b*u + c*v con (u,v) = (vecino - consulta)/h normalizados,
    h = distancia media de los k vecinos. Penalizacion lam*(b^2+c^2).
    lam -> inf recupera exactamente IDW. clip acota |correccion| en ft.
    """
    w, gi = _wmask(cloud, d, gi, k, p)                # (n,k,6)
    n, kk = w.shape[0], w.shape[1]
    q = cloud.transform(X, Y)
    dx = cloud.xy[gi] - q[:, None, :]
    h = np.maximum(np.linalg.norm(dx, axis=2).mean(1), 1e-6)[:, None, None]
    A = np.concatenate([np.ones((n, kk, 1)), dx / h], 2)   # (n,k,3)
    base = np.einsum("nkf,nkf->nf", w, cloud.s0[gi])
    out = np.empty((n, 6))
    reg = np.diag([0.0, lam, lam])
    for f in range(6):
        Aw = A * w[:, :, f, None]
        G = np.einsum("nki,nkj->nij", Aw, A) + reg
        b = np.einsum("nki,nk->ni", Aw, cloud.s0[gi][:, :, f])
        out[:, f] = np.linalg.solve(G, b[:, :, None])[:, 0, 0]
    if clip is not None:
        out = base + np.clip(out - base, -clip, clip)
    return np.where(np.isfinite(out), out, base)


def rbf_local(cloud, d, gi, X, Y, k=32, kernel="thin_plate_spline",
              smoothing=0.0, degree=None):
    """RBF local (scipy) reutilizando el solve cuando el vecindario no cambia."""
    from scipy.interpolate import RBFInterpolator
    gi = gi[:, :k]
    q = cloud.transform(X, Y)
    out = np.empty((len(X), 6))
    _, first, inv = np.unique(gi.astype(np.int64), axis=0,
                              return_index=True, return_inverse=True)
    inv = inv.ravel()
    for u in range(len(first)):
        rows = np.where(inv == u)[0]
        nb = gi[first[u]]
        vals = cloud.s[nb]
        for f in range(6):
            m = np.isfinite(vals[:, f])
            if m.sum() < 6:
                out[rows, f] = np.nanmean(vals[:, f]) if m.any() else np.nan
                continue
            try:
                r = RBFInterpolator(cloud.xy[nb][m], vals[m, f], kernel=kernel,
                                    smoothing=smoothing, degree=degree)
                out[rows, f] = r(q[rows])
            except Exception:
                out[rows, f] = vals[m, f].mean()
    return out


def krige(cloud, d, gi, X, Y, k=24, nugget=1.0, sill=1000.0, rango=1500.0):
    """Kriging ordinario local, variograma exponencial + nugget.
    gamma(h) = nugget + (sill-nugget)*(1-exp(-3h/rango)), gamma(0)=0.
    Los pesos NO dependen de la formacion (solo de la geometria) -> un solo
    sistema por fila; la validez se aplica renormalizando despues.
    """
    d, gi = d[:, :k], gi[:, :k]
    n = len(X)
    P = cloud.xy[gi]
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)

    def gam(h):
        return np.where(h <= 0, 0.0,
                        nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * h / rango)))

    A = np.zeros((n, k + 1, k + 1))
    A[:, :k, :k] = gam(D)
    A[:, :k, k] = 1.0
    A[:, k, :k] = 1.0
    A[:, np.arange(k), np.arange(k)] = 1e-8
    b = np.empty((n, k + 1))
    b[:, :k] = gam(d)
    b[:, k] = 1.0
    lam = np.linalg.solve(A, b[:, :, None])[:, :k, 0]
    w = lam[:, :, None] * cloud.ok[gi]
    s = w.sum(1, keepdims=True)
    w = np.divide(w, s, out=np.zeros_like(w), where=np.abs(s) > 1e-9)
    return np.einsum("nkf,nkf->nf", w, cloud.s0[gi])


if __name__ == "__main__":
    import pandas as pd
    t0 = time.time()
    c = Cloud(subsample=10)
    print(f"nube s=10: {len(c.xy)} pts  ({time.time()-t0:.1f}s)")
    df = pd.read_csv("/home/ftpx100/work/active/kaggle-rogii/data/raw/train/"
                     "000d7d20__horizontal_well.csv")
    t0 = time.time()
    d, gi, dr = c.neighbors("000d7d20", df.X.values, df.Y.values, use_cache=False)
    print(f"vecinos LOWO 1 pozo: {time.time()-t0:.2f}s  d0 med={np.median(dr):.0f} ft")
    for name, fn in [("idw", lambda: idw(c, d, gi, 16)),
                     ("ridge", lambda: linridge(c, d, gi, df.X.values, df.Y.values, 32)),
                     ("krige", lambda: krige(c, d, gi, df.X.values, df.Y.values, 24)),
                     ("rbf", lambda: rbf_local(c, d, gi, df.X.values, df.Y.values, 24))]:
        t0 = time.time()
        S = fn()
        # error del prior BUDA con C calibrada en todo el prefijo (chequeo rapido)
        e = df.TVT.values - (S[:, 5] - df.Z.values)
        print(f"{name:6s} {time.time()-t0:6.2f}s  sd(TVT-(S-Z))={np.nanstd(e):7.2f}")
