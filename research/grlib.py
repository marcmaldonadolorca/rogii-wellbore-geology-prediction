"""Utilidades comunes de la linea GR: carga del cache y HMM sobre arrays.

El HMM aqui es MATEMATICAMENTE IDENTICO a model.hmm_refine, pero recibe arrays
en vez de DataFrames para poder barrer parametros sin recalcular el prior LOWO.
Verificado contra model.hmm_refine en research/gr02_check.py.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

CACHE = Path(__file__).resolve().parent / "gr01_cache.npz"


class Wells:
    """Acceso perezoso al cache de pozos."""

    def __init__(self, path=CACHE):
        self.z = np.load(path, allow_pickle=False)
        self.ids = [str(x) for x in self.z["ids"]]

    def get(self, wid):
        z = self.z
        return dict(
            wid=wid,
            md=z[f"{wid}/md"].astype(np.float64),
            gr=z[f"{wid}/gr"].astype(np.float64),
            tvt=z[f"{wid}/tvt"].astype(np.float64),
            tvt_in=z[f"{wid}/tvt_in"].astype(np.float64),
            prior=z[f"{wid}/prior"].astype(np.float64),
            nn=z[f"{wid}/nn"].astype(np.float64),
            cut=int(z[f"{wid}/cut"]),
            tw_tvt=z[f"{wid}/tw_tvt"].astype(np.float64),
            tw_gr=z[f"{wid}/tw_gr"].astype(np.float64),
        )

    def __iter__(self):
        for w in self.ids:
            yield self.get(w)


def calib(w):
    """(a, b, sg) de la calibracion afin GR_hw ~ a*GR_tw(TVT)+b en el prefijo."""
    cut, gr = w["cut"], w["gr"]
    if len(w["tw_tvt"]) < 20:
        return None
    gp = np.interp(w["tvt_in"][:cut], w["tw_tvt"], w["tw_gr"])
    ok = np.isfinite(gr[:cut]) & np.isfinite(gp)
    if ok.sum() < 40:
        return None
    A = np.column_stack([gp[ok], np.ones(ok.sum())])
    a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
    sg = float(np.clip(np.std(gr[:cut][ok] - (a * gp[ok] + b)), 5.0, 60.0))
    return a, b, sg


def emis_logp(w, cal, grid, rows, sigma_gr=None):
    """log-verosimilitud gaussiana (sin el prior de superficie) en (rows, grid)."""
    a, b, sg = cal
    if sigma_gr is not None:
        sg = sigma_gr
    gr, prior = w["gr"], w["prior"]
    logp = np.zeros((len(rows), len(grid)))
    for j, i in enumerate(rows):
        if np.isfinite(gr[i]):
            pg = a * np.interp(prior[i] + grid, w["tw_tvt"], w["tw_gr"]) + b
            logp[j] = -0.5 * ((gr[i] - pg) / sg) ** 2
    return logp


def fb(emis, sig_states, grid, r0, sigma_init):
    """Forward-backward con difusion gaussiana; devuelve la media posterior."""
    n = len(emis)
    alpha = np.empty_like(emis)
    f = np.exp(-0.5 * ((grid - r0) / sigma_init) ** 2)
    f /= f.sum()
    for j in range(n):
        if j:
            f = gaussian_filter1d(f, sig_states[j], mode="nearest")
        f = f * emis[j]
        s = f.sum()
        f = f / s if s > 0 else np.ones(len(grid)) / len(grid)
        alpha[j] = f
    beta = np.ones_like(emis)
    bk = np.ones(len(grid)) / len(grid)
    for j in range(n - 2, -1, -1):
        bk = gaussian_filter1d(bk * emis[j + 1], sig_states[j + 1], mode="nearest")
        s = bk.sum()
        bk = bk / s if s > 0 else np.ones(len(grid)) / len(grid)
        beta[j] = bk
    post = alpha * beta
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    return post @ grid


def hmm(w, span=80.0, step=0.5, sigma_r=0.01, sigma_gr=None, dec=10,
        sigma_init=2.0, sigma_prior=25.0, logp_fn=None):
    """Residuo estimado r_hat en TODAS las filas post-PS (identico a model.hmm_refine)."""
    cut, n_out = w["cut"], len(w["md"]) - w["cut"]
    cal = calib(w)
    if cal is None:
        return np.zeros(n_out)
    grid = np.arange(-span, span + step, step)
    rows = np.arange(cut, len(w["md"]), dec)
    r0 = float(w["tvt_in"][cut - 1] - w["prior"][cut - 1])
    logp = (logp_fn or emis_logp)(w, cal, grid, rows, sigma_gr)
    if sigma_prior is not None and np.isfinite(sigma_prior):
        logp = logp - 0.5 * ((grid - r0) / sigma_prior) ** 2
    emis = np.exp(logp - logp.max(1, keepdims=True))
    dmd = np.diff(w["md"][rows], prepend=w["md"][rows[0]])
    sig_states = np.maximum(sigma_r * np.abs(dmd) / step, 1e-3)
    r_hat = fb(emis, sig_states, grid, r0, sigma_init)
    return np.interp(np.arange(cut, len(w["md"])), rows, r_hat)


def pooled(sses, ns):
    return float(np.sqrt(np.sum(sses) / np.sum(ns)))
