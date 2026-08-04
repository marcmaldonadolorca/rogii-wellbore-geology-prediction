"""ROGII wellbore: superficie geologica como prior + GR como corrector bayesiano.

Base fisica (medida, residuo 0.0065 ft = el redondeo de los datos):

    TVT_i = S(X_i, Y_i) - Z_i + C_well

TVT no es una serie a extrapolar: es la altura de una superficie geologica
evaluada en la trayectoria. Hay dos incognitas: S (global, de los 773 pozos de
train) y C_well (constante por pozo, calibrable con el prefijo pre-PS).

Dos etapas:
  1) SurfaceField: interpola S en (X,Y) por IDW leave-one-well-out.
     Solo -> ~24.7 ft local (vs 39.4 del baseline geometrico).
  2) hmm_refine: el residuo r = TVT_real - TVT_prior se estima con un HMM 1D
     resuelto por forward-backward, donde la EMISION es el GR del horizontal
     contra GR(TVT) del typewell. La media posterior es el estimador optimo
     para RMSE; los particle filters publicos solo filtran hacia delante.

Uso:
    from cv import evaluate
    from model import make_predictor
    evaluate(make_predictor(use_hmm=True), k=150)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

RAW = Path(__file__).resolve().parent / "data/raw"
SURF = "BUDA"          # superficie de referencia: 0 NaN en los 773 pozos
SUBSAMPLE = 10         # 1 de cada N filas por pozo para la nube de referencia
K_NEIGH = 24           # vecinos IDW (con anisotropia el optimo sube a 24)
ANISO = 5.0            # estiramiento del eje paralelo a las trayectorias
CAL_TAIL = 500         # calibrar C_well con las ultimas N filas pre-PS (mejor que todas)


class SurfaceField:
    """Nube (X, Y, S) de todos los pozos de train + IDW con exclusion del propio pozo."""

    def __init__(self, subsample=SUBSAMPLE, surf=SURF, aniso=ANISO, theta=None):
        xs, ys, zs, wid, dirs = [], [], [], [], []
        for i, p in enumerate(sorted((RAW / "train").glob("*__horizontal_well.csv"))):
            d = pd.read_csv(p, usecols=["X", "Y", surf])[::subsample].dropna()
            xs.append(d.X.values); ys.append(d.Y.values); zs.append(d[surf].values)
            wid.append(np.full(len(d), i))
            if len(d) > 2:      # direccion del pozo, para la anisotropia
                dirs.append([d.X.values[-1] - d.X.values[0], d.Y.values[-1] - d.Y.values[0]])
            self.names = getattr(self, "names", []) + [p.name.split("__")[0]]
        self.xy_raw = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.s = np.concatenate(zs)
        self.wid = np.concatenate(wid)
        self.idx = {n: i for i, n in enumerate(self.names)}

        # Anisotropia: se rota a la direccion principal de las trayectorias y se
        # ESTIRA el eje paralelo. Asi los k vecinos salen de pozos distintos (una
        # seccion transversal) en vez de k puntos del mismo pozo vecino a lo largo.
        if theta is None:
            v = np.array(dirs, float)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            v[v[:, 0] < 0] *= -1                      # sentido irrelevante
            theta = float(np.arctan2(*np.flip(v.mean(0))))
        self.theta, self.aniso = theta, aniso
        c, s_ = np.cos(theta), np.sin(theta)
        self.R = np.array([[c, s_], [-s_, c]])        # -> (paralelo, perpendicular)
        self.scale = np.array([aniso, 1.0])
        self.xy = (self.xy_raw @ self.R.T) * self.scale
        self.tree = cKDTree(self.xy)

    def _t(self, X, Y):
        return (np.column_stack([X, Y]) @ self.R.T) * self.scale

    def _tree_without(self, wid):
        """Arbol y valores excluyendo un pozo (LOWO exacto). Cachea el ultimo:
        cv.evaluate recorre pozo a pozo, asi que se reconstruye una vez por pozo."""
        if wid is None or wid not in self.idx:
            return self.tree, self.s
        if getattr(self, "_cache_wid", None) != wid:
            keep = self.wid != self.idx[wid]
            self._cache_wid = wid
            self._cache = (cKDTree(self.xy[keep]), self.s[keep])
        return self._cache

    def interp(self, X, Y, exclude=None, k=K_NEIGH):
        """S interpolada por IDW(1/d^2). `exclude` = id de pozo a excluir (LOWO)."""
        tree, vals = self._tree_without(exclude)
        dist, ind = tree.query(self._t(X, Y), k=k, workers=-1)
        w = 1.0 / np.maximum(dist, 1e-3) ** 2
        w /= w.sum(1, keepdims=True)
        return (w * vals[ind]).sum(1), dist[:, 0]


def _prior(df_h, field, wid, cut):
    """TVT prior desde la superficie + offset calibrado en la cola del prefijo."""
    s, nn = field.interp(df_h.X.values, df_h.Y.values, exclude=wid)
    lo = max(0, cut - CAL_TAIL)
    c = np.median(df_h.TVT_input.values[lo:cut] + df_h.Z.values[lo:cut] - s[lo:cut])
    return s - df_h.Z.values + c, nn


def hmm_refine(df_h, tw, prior, cut, span=80.0, step=0.5, sigma_r=0.06,
               sigma_gr=None, dec=10, sigma_init=2.0, sigma_prior=25.0):
    """Corrige el prior con el GR: HMM 1D sobre el residuo r = TVT - prior.

    Estado: r discretizado en [-span, span]. Transicion: difusion gaussiana
    (el error de la superficie varia despacio). Emision: N(GR_hw; a*GR_tw(prior+r)+b).
    Devuelve la MEDIA POSTERIOR de r (optima para RMSE) via forward-backward.
    """
    tw = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(tw) < 20:
        return np.zeros(len(df_h) - cut)
    tvt_tw, gr_tw = tw.TVT.values, tw.GR.values

    gr = df_h.GR.values
    md = df_h.MD.values
    # calibracion afin del GR del pozo contra el typewell, ajustada en el prefijo
    known = df_h.TVT_input.values[:cut]
    g_pref = np.interp(known, tvt_tw, gr_tw)
    ok = np.isfinite(gr[:cut]) & np.isfinite(g_pref)
    if ok.sum() < 40:
        return np.zeros(len(df_h) - cut)
    A = np.column_stack([g_pref[ok], np.ones(ok.sum())])
    a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
    resid = gr[:cut][ok] - (a * g_pref[ok] + b)
    sg = float(np.clip(np.std(resid), 5.0, 60.0)) if sigma_gr is None else sigma_gr

    grid = np.arange(-span, span + step, step)
    rows = np.arange(cut, len(df_h), dec)          # decimado: r es suave en MD
    # residuo conocido EXACTAMENTE en el punto PS: el estado arranca anclado ahi
    r0 = float(df_h.TVT_input.values[cut - 1] - prior[cut - 1])
    logp = np.zeros((len(rows), len(grid)))
    for j, i in enumerate(rows):
        if not np.isfinite(gr[i]):
            continue
        pred_gr = a * np.interp(prior[i] + grid, tvt_tw, gr_tw) + b
        logp[j] = -0.5 * ((gr[i] - pred_gr) / sg) ** 2
    # prior debil de superficie: r no se aleja arbitrariamente de r0
    logp -= 0.5 * ((grid - r0) / sigma_prior) ** 2
    emis = np.exp(logp - logp.max(1, keepdims=True))

    # difusion por paso: sigma_r (ft de residuo por ft de MD recorrido)
    dmd = np.diff(md[rows], prepend=md[rows[0]])
    sig_states = np.maximum(sigma_r * np.abs(dmd) / step, 1e-3)

    n = len(rows)
    alpha = np.empty((n, len(grid)))
    f = np.exp(-0.5 * ((grid - r0) / sigma_init) ** 2)   # anclado en el PS
    f /= f.sum()
    for j in range(n):
        if j:
            f = gaussian_filter1d(f, sig_states[j], mode="nearest")
        f = f * emis[j]
        s = f.sum()
        f = f / s if s > 0 else np.ones(len(grid)) / len(grid)
        alpha[j] = f

    beta = np.ones((n, len(grid)))
    bk = np.ones(len(grid)) / len(grid)
    for j in range(n - 2, -1, -1):
        bk = gaussian_filter1d(bk * emis[j + 1], sig_states[j + 1], mode="nearest")
        s = bk.sum()
        bk = bk / s if s > 0 else np.ones(len(grid)) / len(grid)
        beta[j] = bk

    post = alpha * beta
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    r_hat = post @ grid
    # el residuo estimado es suave: se interpola a todas las filas post-PS
    return np.interp(np.arange(cut, len(df_h)), rows, r_hat)


def hmm2_refine(df_h, tw, prior, cut, span=80.0, step=0.5, dec=10, vmax=3,
                sigma_v=0.35, sigma_gr=None, sigma_init=2.0, sigma_prior=25.0):
    """Como hmm_refine pero con VELOCIDAD en el estado (r, v).

    El GR no localiza globalmente (hay alias), pero si rastrea: lo que informa es
    la consistencia de la trayectoria, no cada punto. Un estado con velocidad
    puede seguir un dip persistente; el random-walk de hmm_refine no.

    v se discretiza en CELDAS de r por paso (enteros), asi que la prediccion es
    un np.roll exacto por fila. Estado: (n_v, n_r).
    """
    tw = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    n_out = len(df_h) - cut
    if len(tw) < 20:
        return np.zeros(n_out)
    tvt_tw, gr_tw = tw.TVT.values, tw.GR.values
    gr = df_h.GR.values

    g_pref = np.interp(df_h.TVT_input.values[:cut], tvt_tw, gr_tw)
    ok = np.isfinite(gr[:cut]) & np.isfinite(g_pref)
    if ok.sum() < 40:
        return np.zeros(n_out)
    A = np.column_stack([g_pref[ok], np.ones(ok.sum())])
    a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
    sg = float(np.clip(np.std(gr[:cut][ok] - (a * g_pref[ok] + b)), 5.0, 60.0)) \
        if sigma_gr is None else sigma_gr

    grid = np.arange(-span, span + step, step)
    rows = np.arange(cut, len(df_h), dec)
    r0 = float(df_h.TVT_input.values[cut - 1] - prior[cut - 1])
    ng, nv = len(grid), 2 * vmax + 1
    vcells = np.arange(-vmax, vmax + 1)

    logp = np.zeros((len(rows), ng))
    for j, i in enumerate(rows):
        if np.isfinite(gr[i]):
            logp[j] = -0.5 * ((gr[i] - (a * np.interp(prior[i] + grid, tvt_tw, gr_tw) + b)) / sg) ** 2
    logp -= 0.5 * ((grid - r0) / sigma_prior) ** 2
    emis = np.exp(logp - logp.max(1, keepdims=True))

    def step_fwd(m):
        """Predice: desplaza cada fila v por sus celdas y difunde en v."""
        out = np.empty_like(m)
        for k, s in enumerate(vcells):
            out[k] = np.roll(m[k], s)
            if s > 0:
                out[k, :s] = m[k, 0]
            elif s < 0:
                out[k, s:] = m[k, -1]
        return gaussian_filter1d(out, sigma_v, axis=0, mode="nearest")

    n = len(rows)
    init_r = np.exp(-0.5 * ((grid - r0) / sigma_init) ** 2)
    m = np.repeat(init_r[None, :], nv, 0) / (init_r.sum() * nv)
    alpha = np.empty((n, nv, ng))
    for j in range(n):
        if j:
            m = step_fwd(m)
        m = m * emis[j][None, :]
        s_ = m.sum()
        m = m / s_ if s_ > 0 else np.full((nv, ng), 1.0 / (nv * ng))
        alpha[j] = m

    beta = np.ones((n, nv, ng))
    bk = np.full((nv, ng), 1.0 / (nv * ng))
    for j in range(n - 2, -1, -1):
        bk = step_fwd(bk * emis[j + 1][None, :])
        s_ = bk.sum()
        bk = bk / s_ if s_ > 0 else np.full((nv, ng), 1.0 / (nv * ng))
        beta[j] = bk

    post = (alpha * beta).sum(1)                 # marginal en r
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    return np.interp(np.arange(cut, len(df_h)), rows, post @ grid)


def make_predictor(use_hmm=True, field=None, hmm=None, **hmm_kw):
    """Devuelve predict(df_h, tw, wid) listo para cv.evaluate."""
    field = field or SurfaceField()

    def predict(df_h, tw, wid=None):
        m = df_h.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df_h)
        prior, _ = _prior(df_h, field, wid, cut)
        out = prior[cut:]
        if use_hmm:
            fn = hmm2_refine if hmm == "v" else hmm_refine
            out = out + fn(df_h, tw, prior, cut, **hmm_kw)
        return out

    return predict


if __name__ == "__main__":
    import sys

    from cv import evaluate

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    fld = SurfaceField()
    print("== superficie sola (sin GR) ==")
    evaluate(make_predictor(use_hmm=False, field=fld), k=k)
    print("\n== superficie + HMM sobre GR ==")
    evaluate(make_predictor(use_hmm=True, field=fld), k=k)
