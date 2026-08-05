"""ROGII wellbore v11 — blend ADAPTATIVO por varianza inversa + GBM refit + capas.

Cambio de fondo vs v6: el peso superficie/PF deja de ser fijo (0.45/0.55) y se
decide POR PUNTO por varianza inversa, con dos curvas de error ajustadas en 200
pozos DISJUNTOS de los subconjuntos k60/k150 (research/v7_wadapt.py; curvas
embebidas abajo como literales):

    ss = sigma_s(nn_aniso)    error esperado de la superficie segun la
                              distancia anisotropa al vecino mas cercano
                              (la MISMA metrica que devuelve Field.query)
    sp = sigma_p(md_since)    error esperado del PF segun el MD avanzado
                              desde el punto PS
    w  = clip(sp^2 / (ss^2 + sp^2), 0.05, 0.9)
    pred = w * superficie + (1 - w) * PF64

Motivacion (transferencia): los pozos del test oculto parecen mas aislados que
en LOWO (el factor local->LB crece con el peso de superficie); w(nn_dist)
decide por punto en runtime cuanto fiarse de la superficie.
Medido pareado con el mismo cache PF S=64 (research/v7_fit.log):
    k150  fijo 0.45 -> 9.978   adaptativo -> 9.145  (delta -0.83)
    k60   fijo 0.45 -> 10.147  adaptativo -> 9.828
Y sobre la base EXACTA de este kernel (superficie p=1.5): 9.999 -> 9.163 k150.

Stack (todo medido en cv.py con LOWO):
  1. superficie: TVT = S(X,Y) - Z + C_well. Nube BUDA de train (subsample 10),
     metrica anisotropa aniso=16 con THETA FIJO 2.278489 rad (v4_supmax),
     IDW k=24 p=1.5, C_well = mediana de la cola 500 del prefijo.
  2. pf_ancc multiseed: 64 semillas media (ban_lib._pf_ancc_seeds), N=600.
  3. blend adaptativo (arriba); fijo 0.45 solo si la superficie fallo.
  4. GBM residual REFIT sobre la base adaptativa (research/v8_gbm_adapt.log):
     LightGBM 5 arboles, 43 features, embebido en este fichero; corrige el
     blend adaptativo por punto (entrenado en 380 pozos ajenos a k60/k150,
     superficie p=2 y PF S=16 para las features — el builder replica ESO).
     cv LIVE k=150 = 9.034 (post alpha .95 tau 50 w_pf .1 savgol), vs 9.145
     del blend adaptativo solo.
  5. capas finales (v4_capas): IRLS deg2 mix .6 sobre U=TVT+Z con la cola del
     prefijo + rampa tau=1200 anclada al PS + contact override con doble guard
     (geometria <15 ft y prefijo reproducido <1 ft RMSE: clava las copias de
     test y nunca empeora).

Failsafe en cascada por pozo: flat -> geometrico(dip700) -> superficie ->
blend PF adaptativo -> GBM -> capas; cada etapa con try/except, submission
escrita antes de calcular y al final con fillna(mediana).
"""
import base64
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

# ─────────────────────────────── parametros ──────────────────────────────────
W_SURF = 0.45            # blend de RESPALDO (solo si la superficie fallo)
THETA = 2.278489         # rad, FIJO (PCA sur01 + 11 grados; v4_supmax)
ANISO = 16.0
SUBSAMPLE = 10
K_NEIGH = 24
P_BASE = 1.5             # potencia IDW del blend base (v4_supmax fino2)
P_GBM = 2.0              # potencia IDW con la que se entreno el GBM (model.py)
CAL_TAIL = 500
N_SEEDS_BASE = 64        # PF del blend
N_SEEDS_GBM = 16         # PF de las features del GBM (como se entreno)
DIP_WIN = 700
BT_FRAC = 0.65
POST = dict(alpha=0.95, tau=50.0, w_pf=0.1, sg=True)     # v4_gbm_post_resid_adapt
CAPAS = dict(deg=2, mix=0.6, tau=1200.0, pre_rows=1500)  # v4_capas

# curvas de error del blend adaptativo (research/v7_curvas.npz, EMBEBIDAS).
# Ajustadas con model._prior (superficie p=2) y PF S=8 sobre 200 pozos ajenos
# a k60/k150; validadas pareadas tambien sobre la base p=1.5 de este kernel.
NN_BINS = np.array([0., 50., 100., 200., 300., 450., 600., 800., 1000.,
                    1500., 2500., 1e9])
MD_BINS = np.array([0., 250., 500., 1000., 1500., 2000., 3000., 4000.,
                    5000., 7000., 1e9])
SIG_S = np.array([8.89919702, 9.37233619, 8.43532995, 7.87490051, 8.99296893,
                  11.0666257, 11.19150053, 13.82662369, 13.9339494,
                  27.2295968, 50.79014789])
SIG_P = np.array([2.09432202, 4.53292334, 6.81320383, 8.70951537, 10.5740661,
                  12.63210315, 15.30935841, 15.98209675, 17.81102521,
                  12.38879848])
W_CLIP = (0.05, 0.9)

# contact override (v4_capas)
GRID = 100.0
OV_COVER = 0.50
OV_DIST = 15.0
OV_PRERMSE = 1.0
OV_MINPRE = 50

# PF ANCC (pf_publico / ban_lib, port literal)
ANCC_N = 600
ANCC_ALPHA, ANCC_RN, ANCC_PN = 0.998, 0.002, 0.005
ANCC_IS, ANCC_RP, ANCC_RR = 0.3, 0.1, 0.001
ANCC_IRS = 0.01
PF_RESAMP = 0.5
# v10: clip de sigma_GR recalibrado contra NUESTRO CV (barrido v9f).
# Los valores 10/60 venian del notebook publico ajustado a otro leaderboard;
# subirlos hace que el PF confie menos en el GR, coherente con el diagnostico
# de que 1 sigma de ruido equivale a 8.3 ft de TVT. Medido: -0.173 ft (k=150).
PF_GR_SIG_MIN, PF_GR_SIG_MAX, PF_GR_SIG_DEF = 20.0, 120.0, 30.0
# PF-Z (pf_publico, para la feature Z del GBM)
PF_N = 600
PF_MOM, PF_VN, PF_PN = 0.993, 0.005, 0.01
PF_ROUGH_P, PF_ROUGH_V = 0.2, 0.003
PF_GR_WIN, PF_GR_WT = 5, 0.3
# beam cfg0 "cons" (feature B del GBM)
BEAM0 = (10, 20.0, 144.0, 2)

FEATS = ["dS", "dP", "dZ", "dB", "dG", "dPS", "dPZ", "dSG", "dSB", "cstd",
         "cmean", "nn_real", "nn_aniso", "pstd", "zstd", "md_since",
         "hd_since", "frac", "dz", "dzdmd", "gr", "grm21", "grs21", "grm101",
         "grs101", "gres_S", "twgr_S", "twgr_P", "gr_anchor", "lk", "btS",
         "btP", "btZ", "btB", "btG", "btBlend", "btPS_ratio", "pfx_gr",
         "cutn", "npred", "slp700", "slp50", "tw_span"]

# ───────────────────── kernels numba (ban_lib / pf_publico) ──────────────────
@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1. - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N): cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0., 1. / N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u: ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _pf_ancc_seeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
                   ALPHA, RN, PN, IS, RP, RR, RESAMP, IRS, seeds):
    """PF ANCC multi-semilla con loglik (ban_lib, port literal)."""
    S = len(seeds); n = len(md_v)
    pts_all = np.empty((S, n)); std_all = np.empty((S, n)); ll_all = np.zeros(S)
    for s in range(S):
        np.random.seed(seeds[s])
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
        for j in range(N):
            pos[j] = ls + IS * np.random.randn()
            rate[j] = ir + IRS * np.random.randn()
        pm = md_v[0] - 1.; ll = 0.
        for i in range(n):
            dm = md_v[i] - pm; dm = max(dm, 1.)
            for j in range(N):
                rate[j] = ALPHA * rate[j] + RN * np.random.randn()
                pos[j] += rate[j] * dm + PN * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                tvt_j = max(tvt_j, vmin - 50.); tvt_j = min(tvt_j, vmin + len(gg) * step + 50.)
                pos[j] = tvt_j + z_v[i]
            if not np.isnan(gr_v[i]):
                ws = 0.
                for j in range(N):
                    eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                    d = (gr_v[i] - eg) / gs
                    lk = max(np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300)
                    w[j] *= lk; ws += w[j]
                if ws > 0.:
                    ll += np.log(ws)
                    for j in range(N): w[j] /= ws
                else:
                    ll += np.log(1e-300)
                    for j in range(N): w[j] = 1. / N
            ne = 0.
            for j in range(N): ne += w[j] * w[j]
            if 1. / ne < RESAMP * N:
                pos, rate = _resamp(pos, rate, w, N, RP, RR)
                for j in range(N): w[j] = 1. / N
            tv = 0.
            for j in range(N): tv += w[j] * (pos[j] - z_v[i])
            pts_all[s, i] = tv
            va = 0.
            for j in range(N): va += w[j] * (pos[j] - z_v[i] - tv) ** 2
            std_all[s, i] = va ** 0.5
            pm = md_v[i]
        ll_all[s] = ll
    return pts_all, std_all, ll_all


@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
          gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.; pz = z_v[0] - 1.
    for i in range(len(md_v)):
        dm = md_v[i] - pm; dm = max(dm, 1.)
        dzd = (z_v[i] - pz) / dm; ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.); pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600. else 0., 1e-300)
                    lk = (1. - GR_WT) * lp + GR_WT * ls
                else: lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2., 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600. else 0., 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1. / N
        ne = 0.
        for j in range(N): ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1. / N
        wm = 0.
        for j in range(N): wm += w[j] * pos[j]
        pts[i] = wm; va = 0.
        for j in range(N): va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search ±2 delta (pf_publico, port literal)."""
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30);     bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path


# ─────────────────────────── wrappers de señal ───────────────────────────────
def _grid_tw(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def _tw(tw):
    t = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    return t.TVT.values.astype(np.float64), t.GR.values.astype(np.float64)


def _nn_idx(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr): return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v): return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn_idx(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def run_pf_ancc_multi(hw, tw_tvt, tw_gr, seeds, N=ANCC_N):
    """S pasadas del PF ANCC -> pts (S, n_pred) float32, loglik (S,)."""
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.zeros((len(seeds), 0), np.float32), np.zeros(len(seeds))
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid_tw(tw_tvt, tw_gr)
    pts, _std, ll = _pf_ancc_seeds(
        ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
        ev["GR"].values.astype(np.float64), gg, gmin, gst, float(gs), ls, ir, int(N),
        ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR,
        PF_RESAMP, ANCC_IRS, np.asarray(seeds, np.int64))
    return pts.astype(np.float32), ll


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values); dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]; vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)]); c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else: beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2["TVT_input"].values); dmd2 = np.diff(t2["MD"].values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid_tw(tw_tvt, tw_gr)
    gs2, _, _ = _grid_tw(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
                     ev["GR"].values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64),
                     gg, gs2, gmin, gst, gs, float(kna["TVT_input"].iloc[-1]), iv,
                     beta, icpt, zsig, N,
                     PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)


# ───────────────────────────── superficie ────────────────────────────────────
class Field:
    """Nube (X,Y)->BUDA con metrica anisotropa de THETA FIJO + arbol raw."""

    def __init__(self, xy_raw, sval, wids, ids):
        self.xy_raw, self.sval, self.wids, self.ids = xy_raw, sval, wids, ids
        c, s = np.cos(THETA), np.sin(THETA)
        self.rot = np.array([[c, s], [-s, c]])
        self.scale = np.array([ANISO, 1.0])
        self.xy = (xy_raw @ self.rot.T) * self.scale
        self.tree = cKDTree(self.xy)
        self.raw_tree = cKDTree(xy_raw)

    def query(self, X, Y, k=K_NEIGH):
        q = (np.column_stack([X, Y]) @ self.rot.T) * self.scale
        return self.tree.query(q, k=k, workers=-1)

    def interp(self, dist, ind, p):
        w = 1.0 / np.maximum(dist, 1e-3) ** p
        w /= w.sum(1, keepdims=True)
        return (w * self.sval[ind]).sum(1)

    def real_nn(self, X, Y):
        d, _ = self.raw_tree.query(np.column_stack([X, Y]), k=1, workers=-1)
        return d


def _icurve(x, bins, vals):
    """Interpolacion lineal sobre los centros de bin (v7_wadapt._interp_curve).
    El ultimo bin (semiabierto) usa como centro bins[-2]*2; np.interp satura
    en los extremos, que es el comportamiento deseado."""
    c = 0.5 * (bins[:-1] + np.minimum(bins[1:], bins[-2] * 2))
    return np.interp(x, c, vals)


def w_adapt(nn_aniso, md_since):
    """Peso de la superficie por punto: varianza inversa con curvas embebidas."""
    ss = _icurve(np.asarray(nn_aniso, float), NN_BINS, SIG_S)
    sp = _icurve(np.asarray(md_since, float), MD_BINS, SIG_P)
    return np.clip(sp ** 2 / (ss ** 2 + sp ** 2), *W_CLIP)


def prior_from(S, df_h, cut):
    """TVT prior = S - Z + C_well (C = mediana de la cola CAL_TAIL del prefijo)."""
    base = S - df_h.Z.values
    lo = max(0, cut - CAL_TAIL)
    c = np.median(df_h.TVT_input.values[lo:cut] - base[lo:cut])
    return base + c


def hdist(df):
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=DIP_WIN):
    """Recta de dip anclada en el PS (afinar_dip anchor700)."""
    z = df_h.Z.values
    tvt_in = df_h.TVT_input.values
    h = hd - hd[cut - 1]
    flat = tvt_in[cut - 1] + (z[cut - 1] - z)
    lo = max(0, cut - win)
    base = tvt_in[lo] + (z[lo] - z[lo:cut])
    r = tvt_in[lo:cut] - base
    hh = h[lo:cut]
    slope = 0.0
    if len(hh) > 10 and hh[-1] != hh[0]:
        dr, dh = r - r[-1], hh - hh[-1]
        ss = float(np.sum(dh * dh))
        slope = float(np.sum(dr * dh) / ss) if ss > 0 else 0.0
    return flat + slope * h, slope


# ─────────────────── GBM residual: features (v4_gbm.build_well) ──────────────
def candidates(df_h, tw_t, tw_g, cut, field, hd, dist, ind):
    """S,P,Z,B,G post-cut + extras. S con p=2 (como se entreno el GBM)."""
    s_full = prior_from(field.interp(dist, ind, P_GBM), df_h, cut)
    nn_aniso = dist[:, 0]
    pts, _ll = run_pf_ancc_multi(df_h, tw_t, tw_g,
                                 seeds=np.arange(1, N_SEEDS_GBM + 1, dtype=np.int64))
    P, Pstd = pts.mean(0).astype(float), pts.std(0).astype(float)
    Zp, Zstd = run_pf_z(df_h, tw_t, tw_g)
    bs, mc, es, r = BEAM0
    B = beam_search(df_h.GR.values[cut:], tw_t, tw_g,
                    float(df_h.TVT_input.values[cut - 1]), bs, mc, es, r)
    Gfull, slope = geom_pred(df_h, cut, hd)
    return dict(S=s_full[cut:], P=np.asarray(P, float), Z=np.asarray(Zp, float),
                B=np.asarray(B, float), G=Gfull[cut:],
                Pstd=np.asarray(Pstd, float), Zstd=np.asarray(Zstd, float),
                nn_aniso=nn_aniso[cut:], slope=slope)


def backtest_rmse(df_h, tw_t, tw_g, cut, field, hd, dist, ind):
    """RMSE leak-free de cada candidato re-prediciendo [0.65*cut : cut]."""
    c2 = int(round(BT_FRAC * cut))
    out = {f"bt{n}": np.nan for n in "SPZBG"}
    out["btBlend"] = np.nan
    if c2 < 60 or cut - c2 < 30:
        return out
    d2 = df_h.iloc[:cut].copy()
    d2.loc[d2.index[c2:], "TVT_input"] = np.nan
    y = df_h.TVT_input.values[c2:cut]
    try:
        c = candidates(d2, tw_t, tw_g, c2, field, hd[:cut], dist[:cut], ind[:cut])
    except Exception:
        return out
    for n in "SPZBG":
        out[f"bt{n}"] = float(np.sqrt(np.mean((y - c[n]) ** 2)))
    blend = 0.25 * c["S"] + 0.75 * c["P"]
    out["btBlend"] = float(np.sqrt(np.mean((y - blend) ** 2)))
    return out


def build_features(df_h, tw_t, tw_g, cut, field, hd, dist, ind):
    """DataFrame (n_pred, 43) en el ORDEN EXACTO del entrenamiento + aux."""
    c = candidates(df_h, tw_t, tw_g, cut, field, hd, dist, ind)
    bt = backtest_rmse(df_h, tw_t, tw_g, cut, field, hd, dist, ind)
    n = len(df_h) - cut
    lk = float(df_h.TVT_input.values[cut - 1])

    gr = df_h.GR.values
    g_pref = np.interp(df_h.TVT_input.values[:cut], tw_t, tw_g)
    ok = np.isfinite(gr[:cut]) & np.isfinite(g_pref)
    a, b, pfx_gr = 1.0, 0.0, np.nan
    if ok.sum() >= 40:
        A = np.column_stack([g_pref[ok], np.ones(ok.sum())])
        a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
        pfx_gr = float(np.std(gr[:cut][ok] - (a * g_pref[ok] + b)))

    slp50 = 0.0
    hh, tt = hd[max(0, cut - 50):cut], df_h.TVT_input.values[max(0, cut - 50):cut]
    if len(hh) > 10 and hh[-1] != hh[0]:
        slp50 = float(np.polyfit(hh - hh[0], tt - tt[0], 1)[0])

    S, P, Zc, B, G = (c[k] for k in "SPZBG")
    stack = np.stack([S, P, Zc, B, G])
    grs = pd.Series(gr)
    r21 = grs.rolling(21, center=True, min_periods=1)
    r101 = grs.rolling(101, center=True, min_periods=1)
    mdv = df_h.MD.values
    zv = df_h.Z.values
    dz_dmd = pd.Series(np.gradient(zv, np.maximum(mdv, 1e-9))).rolling(
        21, center=True, min_periods=1).mean().values
    twgr_S = a * np.interp(S, tw_t, tw_g) + b
    twgr_P = a * np.interp(P, tw_t, tw_g) + b
    grpost = gr[cut:]
    nn_real = field.real_nn(df_h.X.values[cut:], df_h.Y.values[cut:])
    sc = lambda v: np.full(n, np.float32(v))                       # noqa: E731

    X = pd.DataFrame({
        "dS": S - lk, "dP": P - lk, "dZ": Zc - lk, "dB": B - lk, "dG": G - lk,
        "dPS": P - S, "dPZ": P - Zc, "dSG": S - G, "dSB": S - B,
        "cstd": stack.std(0), "cmean": stack.mean(0) - lk,
        "nn_real": nn_real, "nn_aniso": c["nn_aniso"],
        "pstd": c["Pstd"], "zstd": c["Zstd"],
        "md_since": mdv[cut:] - mdv[cut - 1],
        "hd_since": hd[cut:] - hd[cut - 1],
        "frac": np.arange(n) / max(n - 1, 1),
        "dz": zv[cut:] - zv[cut - 1], "dzdmd": dz_dmd[cut:],
        "gr": grpost,
        "grm21": r21.mean().values[cut:], "grs21": r21.std().values[cut:],
        "grm101": r101.mean().values[cut:], "grs101": r101.std().values[cut:],
        "gres_S": grpost - twgr_S, "twgr_S": twgr_S, "twgr_P": twgr_P,
        "gr_anchor": grpost - (a * np.interp(lk, tw_t, tw_g) + b),
        "lk": sc(lk),
        "btS": sc(bt["btS"]), "btP": sc(bt["btP"]), "btZ": sc(bt["btZ"]),
        "btB": sc(bt["btB"]), "btG": sc(bt["btG"]), "btBlend": sc(bt["btBlend"]),
        "btPS_ratio": sc(bt["btP"] / (bt["btS"] + 1e-6)),
        "pfx_gr": sc(pfx_gr), "cutn": sc(cut), "npred": sc(n),
        "slp700": sc(c["slope"]), "slp50": sc(slp50),
        "tw_span": sc(float(tw_t.max() - tw_t.min())),
    }).astype(np.float32)
    return X, dict(lk=lk, cut=cut)


def postprocess_well(d, dP, md_since, alpha, tau, w_pf, sg):
    """Postproceso del GBM sobre el delta total (v4_gbm.postprocess, 1 pozo)."""
    out = d * (1.0 - w_pf) + dP * w_pf
    if tau:
        out = out * (1.0 - np.exp(-np.maximum(md_since, 0.0) / tau))
    out = out * alpha
    if sg:
        wl = min(17, len(out) if len(out) % 2 else len(out) - 1)
        if wl >= 5:
            out = savgol_filter(out, wl, 3)
    return out


# ──────────────────────── capas finales (v4_capas) ───────────────────────────
class Sigs:
    """Firmas geometricas (X,Y en rejilla MD de 100 ft) de los pozos de train."""

    def __init__(self, ids, g0, x, y):
        self.ids, self.g0, self.x, self.y = ids, np.asarray(g0), x, y
        self.n = np.array([len(v) for v in x])
        self.g1 = self.g0 + (self.n - 1) * GRID


def match_geom(md, x, y, S, exclude=None, cover_min=OV_COVER, dist_max=OV_DIST):
    """Mejor pozo de train con la misma geometria: (id, dist, cover) o None."""
    g0 = np.ceil(md[0] / GRID) * GRID
    g = np.arange(g0, md[-1] + 1e-9, GRID)
    if len(g) < 10:
        return None
    xq = np.interp(g, md, x)
    yq = np.interp(g, md, y)
    span = g[-1] - g[0]
    best = None
    for i, w in enumerate(S.ids):
        if w == exclude:
            continue
        lo = max(g0, S.g0[i])
        hi = min(g[-1], S.g1[i])
        if hi - lo < cover_min * span:
            continue
        a = int(round((lo - g0) / GRID))
        b = int(round((lo - S.g0[i]) / GRID))
        m = int(round((hi - lo) / GRID)) + 1
        if abs(float(xq[a]) - float(S.x[i][b])) + abs(float(yq[a]) - float(S.y[i][b])) > 20 * dist_max:
            continue
        dist = float(np.mean(np.abs(xq[a:a + m] - S.x[i][b:b + m]) +
                             np.abs(yq[a:a + m] - S.y[i][b:b + m])))
        if dist < dist_max and (best is None or dist < best[1]):
            best = (w, dist, (hi - lo) / span)
    return best


def override_pred(df_h, cut, mid, train_dir):
    """TVT desde la copia de train, interpolada POR MD, con guard de prefijo."""
    tr = pd.read_csv(train_dir / f"{mid}__horizontal_well.csv",
                     usecols=["MD", "BUDA"])
    md = df_h.MD.values
    S = np.interp(md, tr.MD.values, tr.BUDA.values)
    cov = (md >= tr.MD.values[0] - 0.5) & (md <= tr.MD.values[-1] + 0.5)
    base = S - df_h.Z.values
    tvt_in = df_h.TVT_input.values
    pre = np.where(cov[:cut] & np.isfinite(tvt_in[:cut]))[0]
    if len(pre) < OV_MINPRE:
        return None
    C = float(np.median(tvt_in[pre] - base[pre]))
    r = tvt_in[pre] - (base[pre] + C)
    return base + C, cov, float(np.sqrt(np.mean(r ** 2))), len(pre)


def contact_override(df_h, cut, sigs, train_dir, exclude=None):
    """Doble guard: match geometrico <15 ft Y prefijo reproducido <1 ft RMSE."""
    mm = match_geom(df_h.MD.values, df_h.X.values, df_h.Y.values, sigs,
                    exclude=exclude)
    if mm is None:
        return None, None
    op = override_pred(df_h, cut, mm[0], train_dir)
    if op is None:
        return None, mm[0]
    pred, cov, prermse, npre = op
    if prermse >= OV_PRERMSE:
        return None, mm[0]
    return np.where(cov, pred, np.nan)[cut:], mm[0]


def irls_poly_well(md_pre, z_pre, tvt_pre, md_post, z_post, p_post,
                   deg=3, iters=4, cval=4.0, mix=0.5):
    """Proyeccion robusta (pesos Cauchy) de U=TVT+Z, dominio prefijo+post."""
    p_post = np.asarray(p_post, float)
    x = np.concatenate([md_pre, md_post])
    u = np.concatenate([tvt_pre + z_pre, p_post + z_post])
    if len(x) < deg + 5 or x[-1] == x[0]:
        return p_post
    t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
    V = np.vander(t, deg + 1)
    w = np.ones(len(t))
    for _ in range(iters):
        beta, *_ = np.linalg.lstsq(V * w[:, None], u * w, rcond=None)
        r = u - V @ beta
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / (1.0 + (r / (cval * sc)) ** 2)
    fit = (V @ beta)[-len(p_post):] - z_post
    return mix * fit + (1 - mix) * p_post


def aplicar_capas(df_h, cut, pred_post, sigs, train_dir, exclude=None):
    """IRLS deg2 + rampa + contact override; cada capa con su try/except."""
    p = np.asarray(pred_post, float)
    md, z = df_h.MD.values, df_h.Z.values
    tvt_in = df_h.TVT_input.values
    hit = ""
    try:
        lo = max(0, cut - CAPAS["pre_rows"])
        q = irls_poly_well(md[lo:cut], z[lo:cut], tvt_in[lo:cut],
                           md[cut:], z[cut:], p, deg=CAPAS["deg"], mix=CAPAS["mix"])
        if np.isfinite(q).all():
            p = q
    except Exception:
        pass
    try:
        mds = md[cut:] - md[cut - 1]
        q = p + (tvt_in[cut - 1] - p[0]) * np.exp(-mds / CAPAS["tau"])
        if np.isfinite(q).all():
            p = q
    except Exception:
        pass
    try:
        ovp, mid = contact_override(df_h, cut, sigs, train_dir, exclude=exclude)
        if ovp is not None:
            q = np.where(np.isfinite(ovp), ovp, p)
            if np.isfinite(q).all():
                p = q
                hit = mid
    except Exception:
        pass
    return p, hit


# ───────────────────────── carga de datos y estaticos ────────────────────────
def discover_base():
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    base = next((p for p in candidates if (p / "sample_submission.csv").exists()), None)
    if base is None:
        base = next(p.parent for p in Path("/kaggle/input").rglob("sample_submission.csv"))
    return base


def build_static(base):
    """Una sola pasada por train/: nube de superficie + firmas del override."""
    xs, ys, ss, wds = [], [], [], []
    sig_ids, sig_g0, sig_x, sig_y = [], [], [], []
    ids = []
    for p in sorted((base / "train").glob("*__horizontal_well.csv")):
        wid = p.name.split("__")[0]
        try:
            d = pd.read_csv(p, usecols=["MD", "X", "Y", "BUDA"])
        except ValueError:
            continue
        ids.append(wid)
        dd = d[["X", "Y", "BUDA"]][::SUBSAMPLE].dropna()
        if len(dd):
            xs.append(dd.X.values); ys.append(dd.Y.values); ss.append(dd.BUDA.values)
            wds.append(np.full(len(dd), len(ids) - 1, np.int32))
        md = d.MD.values
        g0 = np.ceil(md[0] / GRID) * GRID
        g = np.arange(g0, md[-1] + 1e-9, GRID)
        if len(g) >= 2:
            sig_ids.append(wid)
            sig_g0.append(g0)
            sig_x.append(np.interp(g, md, d.X.values).astype(np.float32))
            sig_y.append(np.interp(g, md, d.Y.values).astype(np.float32))
    xy_raw = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    field = Field(xy_raw, np.concatenate(ss), np.concatenate(wds), ids)
    sigs = Sigs(sig_ids, sig_g0, sig_x, sig_y)
    return field, sigs


def load_booster():
    import lightgbm as lgb
    txt = zlib.decompress(base64.b64decode(GBM_MODEL_B64)).decode()
    booster = lgb.Booster(model_str=txt)
    names = booster.feature_name()
    assert names == FEATS, f"features del modelo != builder: {names[:5]}..."
    return booster


# ──────────────────────────── proceso por pozo ───────────────────────────────
def process_well(df, tw, cut, field, sigs, booster, train_dir, exclude=None):
    """Cascada failsafe. Devuelve (pred_post, via, override_id)."""
    z = df.Z.values
    tvt_in = df.TVT_input.values
    n_out = len(df) - cut
    pred = tvt_in[cut - 1] + (z[cut - 1] - z[cut:])          # flat, siempre
    via = "flat"
    if cut < 20:
        return pred, via, ""
    hd = hdist(df)
    try:
        G, _ = geom_pred(df, cut, hd)
        if np.isfinite(G[cut:]).all():
            pred, via = G[cut:], "geo"
    except Exception:
        pass
    surf_ok, dist = False, None
    try:
        dist, ind = field.query(df.X.values, df.Y.values)
        prior15 = prior_from(field.interp(dist, ind, P_BASE), df, cut)
        assert np.isfinite(prior15[cut:]).all()
        pred, via, surf_ok = prior15[cut:], "surf", True
    except Exception as e:
        print(f"    superficie fallo ({e})", flush=True)
    tw_ok, P64, wS = False, None, None
    try:
        tw_t, tw_g = _tw(tw)
        assert len(tw_t) >= 20
        tw_ok = True
    except Exception as e:
        print(f"    typewell fallo ({e})", flush=True)
    if tw_ok:
        try:
            pts, _ll = run_pf_ancc_multi(df, tw_t, tw_g,
                                         seeds=np.arange(1, N_SEEDS_BASE + 1, dtype=np.int64))
            P64 = np.asarray(pts, float).mean(0)
            assert len(P64) == n_out and np.isfinite(P64).all()
            if surf_ok and dist is not None:
                md = df.MD.values
                wS = w_adapt(dist[cut:, 0], md[cut:] - md[cut - 1])
                assert np.isfinite(wS).all()
                pred = wS * pred + (1.0 - wS) * P64
                via = "blendA"
            else:
                pred = W_SURF * pred + (1.0 - W_SURF) * P64
                via = "blend"
        except Exception as e:
            P64, wS = None, None
            print(f"    PF fallo ({e})", flush=True)
    if booster is not None and surf_ok and tw_ok and P64 is not None and wS is not None:
        try:
            X, aux = build_features(df, tw_t, tw_g, cut, field, hd, dist, ind)
            lk = aux["lk"]
            d_base = wS * (prior15[cut:] - lk) + (1.0 - wS) * (P64 - lk)
            d = d_base + booster.predict(X[FEATS])
            d = postprocess_well(d, P64 - lk, X.md_since.values.astype(np.float64), **POST)
            cand = lk + d
            assert len(cand) == n_out and np.isfinite(cand).all()
            pred, via = cand, "gbm"
        except Exception as e:
            print(f"    GBM fallo ({e})", flush=True)
    ov = ""
    try:
        pred, ov = aplicar_capas(df, cut, pred, sigs, train_dir, exclude=exclude)
    except Exception as e:
        print(f"    capas fallo ({e})", flush=True)
    return pred, via, ov


# ─────────────────────────────────── main ────────────────────────────────────
def main():
    t0 = time.time()
    base = discover_base()
    print("datos en:", base, flush=True)
    sample = pd.read_csv(base / "sample_submission.csv")
    sample.to_csv("submission.csv", index=False)             # failsafe inmediato

    field, sigs = build_static(base)
    print(f"nube: {len(field.sval)} pts de {len(field.ids)} pozos | "
          f"theta={THETA} aniso={ANISO} | firmas={len(sigs.ids)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    try:
        booster = load_booster()
        print(f"GBM embebido: {booster.num_trees()} arboles, "
              f"{len(booster.feature_name())} features", flush=True)
    except Exception as e:
        booster = None
        print(f"AVISO: GBM no disponible ({e}) -> solo blend", flush=True)

    files = sorted((base / "test").glob("*__horizontal_well.csv"))
    out, vias, ov_hits = {}, Counter(), 0
    t1 = time.time()
    for j, p in enumerate(files):
        wid = p.name.split("__")[0]
        try:
            df = pd.read_csv(p)
            m = df.TVT_input.isna()
            cut = int(m.idxmax()) if m.any() else len(df)
            if cut < 1 or cut >= len(df):
                continue
            try:
                tw = pd.read_csv(p.parent / f"{wid}__typewell.csv")
            except Exception as e:
                tw = pd.DataFrame(columns=["TVT", "GR"])
                print(f"  {wid}: typewell ilegible ({e})", flush=True)
            pred, via, ov = process_well(df, tw, cut, field, sigs, booster,
                                         base / "train")
            vias[via] += 1
            ov_hits += bool(ov)
            for i2, i in enumerate(range(cut, len(df))):
                out[f"{wid}_{i}"] = float(pred[i2])
        except Exception as e:
            vias["fallo"] += 1
            print(f"  {wid}: FALLO TOTAL ({e})", flush=True)
        if (j + 1) % 10 == 0:
            dt = time.time() - t1
            eta = dt / (j + 1) * (len(files) - j - 1)
            print(f"  {j+1}/{len(files)}  {dt:.0f}s  ({dt/(j+1):.1f}s/pozo, "
                  f"ETA {eta/60:.0f}min)", flush=True)

    print(f"vias: {dict(vias)} | override hits: {ov_hits} | "
          f"{(time.time()-t1)/max(len(files),1):.1f}s/pozo", flush=True)
    sample["tvt"] = sample.id.map(out)
    faltan = int(sample.tvt.isna().sum())
    if faltan:
        print(f"AVISO: {faltan} ids sin prediccion -> mediana", flush=True)
        sample["tvt"] = sample.tvt.fillna(sample.tvt.median())
    assert np.isfinite(sample.tvt.values).all()
    sample.to_csv("submission.csv", index=False)
    print(sample.shape, f"submission.csv escrito ({(time.time()-t0)/60:.1f} min)")
    print(sample.head())


# ──────────── modelo GBM embebido (v4_gbm_model_resid_adapt.txt) ─────────────
GBM_MODEL_B64 = (
    "eNrtvUtzHEmSrbnnr/D9CELsbeYU4aY2tS2RnFWXtEBAIEjiFgigATDr8evnfMcsQBBE1c2+PbOY7uyuZAQiPPxhpqamj6NHnx6O"
    "x3e/Hh8er+9uP/xa3t1++3p+eXPx+Pgh+v2Tvj+/Pz6cXz8dHy6eOCq+u7n4eLw5v769Ov7tQ3j39eJv55+OF0/fHo7n11d/+1DS"
    "u7uP/+t4+XT96/HDw/Hzw/GRs787HXN78fX4+OHql+3qT9vVv21Xf9iu/qj3/K2/ftH7X/6wXT4+XW2XX48Xt9vt7fnD8eKG14vb"
    "68e77Z7v/sE/X6/OH69vL4/bl9ObTw8Xl9vVP/S/q69X2+cH/e9rivr30f9+jcF/zJfj4/kv29NfPz+cXv6kD3WRyy93D9vNX7aP"
    "T7/ovz/pv3/Tf3/Qf3/k9eZ4e8Xnv5x7RLb7T38715Uuvz3pZu8fjlfb4819D4GXGnTm88f7i+/Pf3376e7xw5/PYo6HtLccQ0ll"
    "hNL296mPQy559BR6yDHVf9/+fDbiIfde++i55hprS+9rPbSxt1FHijXnFPI6sLdYRko6Q9trju9HP+Q2ckslNU4REwfGcgh9b7po"
    "7GPkHOL7uB+q/txL6Tm11oIPTKkfYgmh1KiPfOmU+yH1HlLOsRbd/+4DWzyUPvKeay1t9Po+Vt1N2Me+7zq25D6fJRxaSzHsmUNy"
    "OKSWW49jfpt6PeSk268x9JLiGO9z0Uc97bXyRdMd+AH0yzBG3VNNu+5BB6a+HzSYbZQw0mnsdFSNMZay11GyHniPetQ4Dhrj1EPV"
    "D0tvjVM2bjc0jcXo+xhpxPc9c8asMerzjjnj+66f6FlTbBrCsT7MY4R2OF0z5DI0QrrZUKKGtXF7h9BKS0PD6GfbfWgMGrKuB6yt"
    "praX/l43F7J+0kYrIWpweGDddAhh/mQfMabYdRtjxLrH4e8kSi/uMfJEOe+HXPVd01Wm3JQUDvpP5y2aQySNAzULkeN037U3yaAk"
    "UbdW52cx7ilokLmN/VBaHjqohl6RK4lDOVTmL2ukLWE6LsXDvtemX40sgZI8vU9pHFJOVVOmMUGU5mC2g8Y31NLLmMLNz9uhdk1x"
    "ySmmsQekLkrii/4YOkNlbXgsqiSjlBH3PWlCah3vmwRe89pj08jspfrAM0mqJC3oTnRLTGZ5r+k7hJQ03bpMi90TknSVrGeNXfOu"
    "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"
    "vv+Jnlh3O6f5haLz0SwgjpsP/u3y6aT1L2+fkBn+77QfPFzcXt19fRZzLqNNX0P9kV1CO9Tj0izro4+64Zv1EaPy9Vpq9PQA/zg+"
    "3J1fPH7/8KUo3aOWrm+e2KY4mL91/qfrpdyQ6r/evVQ3X7R5cPTUGB+P2Lo3377e+qmnk/XyE6+2lx9cf769s6A8f3LJpncn/SS7"
    "+eUQTtlm+n6UbN3jpbae8083d1qDHo11P/M9Q/zp+rN/5B88b9qnEW1h5CUolzcXj4/z2TXE324/XtwgZksodEsaD+0x88HmcY/X"
    "n7/esRd767Da+vSg6ZLEPsgSm5/LOjh//I+HtZIubu6/XJxW4KeLa93jGu87zYqWw0/6+NA99Lrf6+P5rxcyIBD++7u/eqYO37Wz"
    "ZOUvks5vt5e2ByTSvx5v2Fx+PEKD/nUJtCcNpe3BeXHMPQvQm9e15EVP8O1GF/7HxXdZOMoKOb94mhL+7ebp+vz48HCn1TE1H6e/"
    "+HZ5/vXbGrDHZ4Pl68XlF62kNdQ3d0z2jcyG4+05S/u927Bb3K6/Hs/vvvmTudz8Qx/8oxx8PyVydv/t/P7m4snbBZNjLcKny+h6"
    "+zOf9vkErJ2r+xda5P6b7/dkVX+3DbETb680Si+E9/2f//3d/wPUdVPB"
)

if __name__ == "__main__":
    main()
