"""Banco de senales: beam arreglado, multi_scale_ncc portado y PF multi-semilla.

Todo se computa en el re-run (nada de artifacts). Importar SIEMPRE con
sys.path incluyendo research/ para que el cache de numba case:
    sys.path.insert(0, ".../research"); import ban_lib

Piezas:
  make_beam(cfg)            predictor de una config de BEAMS (arregla el bug del
                            wrapper: cv pasa `wid` como 3er posicional).
  multi_scale_ncc(...)      port literal de roman_all.py (correlaciona ventanas
                            del GR post-PS contra el GR del PREFIJO DEL PROPIO POZO).
  run_pf_ancc_multi(...)    S pasadas de run_pf_ancc con semillas distintas,
                            devuelve la matriz (S, n) y el loglik de cada pasada
                            (estimador PF de la verosimilitud marginal).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pf_publico import (ANCC_ALPHA, ANCC_IR, ANCC_IS, ANCC_N, ANCC_PN, ANCC_RN,
                        ANCC_RP, ANCC_RR, BEAMS, PF_GR_SIG_DEF, PF_GR_SIG_MAX,
                        PF_GR_SIG_MIN, PF_RESAMP, _grid, _nn, _smooth, _tw,
                        beam_search, run_pf_ancc, run_pf_z)


def cut_of(df_h):
    m = df_h.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df_h)


# ------------------------------------------------------------------ 1) BEAM
def make_beam(cfg):
    """predict(df_h, tw) para la config `cfg` de BEAMS (2 args: cv NO le pasa wid)."""
    bs, mc, es, r, name = BEAMS[cfg]

    def predict(df_h, tw):
        t, g = _tw(tw)
        cut = cut_of(df_h)
        gr = df_h.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(g)))
        start = float(df_h.TVT_input.values[cut - 1])
        return np.asarray(beam_search(gr.values[cut:], t, g, start, bs, mc, es, r), float)

    predict.__name__ = f"beam_{name}"
    return predict


def beam_all(df_h, tw):
    """Las 7 trayectorias beam apiladas (7, n_pred)."""
    t, g = _tw(tw)
    cut = cut_of(df_h)
    gr = df_h.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(g)))
    start = float(df_h.TVT_input.values[cut - 1])
    return np.stack([np.asarray(beam_search(gr.values[cut:], t, g, start, bs, mc, es, r), float)
                     for (bs, mc, es, r, _n) in BEAMS])


# --------------------------------------------------------- 2) MULTI-SCALE NCC
def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    """Port literal de roman_all.py:369. Devuelve ([(tvt,score) por escala], ensemble).

    kgr/ktvt: GR y TVT del PREFIJO del propio pozo. hgr: GR post-PS.
    Para cada punto post-PS busca la ventana del prefijo cuyo GR mejor correlaciona
    y devuelve el TVT de ese punto del prefijo.
    """
    out = []
    for hw in hws:
        win = 2 * hw + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32); M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


def ncc_signals(df_h, tw, hws=(8, 15, 25), stride=3):
    """dict con sc8/sc15/sc25/sc_ens/sc_cons y sus scores, en el formato de cv."""
    t, g = _tw(tw)
    cut = cut_of(df_h)
    gr_full = df_h.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(g)))
    kgr = gr_full.values[:cut].astype(np.float32)
    hgr = gr_full.values[cut:].astype(np.float32)
    ktvt = df_h.TVT_input.values[:cut].astype(np.float32)
    res, ens = multi_scale_ncc(kgr, ktvt, hgr, hws=hws, stride=stride)
    d = {f"sc{hw}": res[i][0].astype(float) for i, hw in enumerate(hws)}
    d.update({f"score{hw}": res[i][1].astype(float) for i, hw in enumerate(hws)})
    d["sc_ens"] = ens.astype(float)
    d["sc_cons"] = np.mean([res[i][0] for i in range(len(hws))], 0).astype(float)
    return d


# ------------------------------------------------- 3) PF ANCC MULTI-SEMILLA
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
    """Igual que pf_publico._pf_ancc pero con semilla explicita y loglik.

    Devuelve pts (S,n), std (S,n), loglik (S,). loglik = sum_i log( sum_j w_j*p(gr_i|x_j) ),
    el estimador insesgado del PF de la verosimilitud marginal de esa pasada.
    """
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


def _gr_sig(hw, tw_tvt, tw_gr, lo=PF_GR_SIG_MIN, hi=PF_GR_SIG_MAX):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
                         lo, hi))


def run_pf_ancc_multi(hw, tw_tvt, tw_gr, seeds=(0,), N=ANCC_N, alpha=ANCC_ALPHA,
                      rn=ANCC_RN, pn=ANCC_PN, is_spr=ANCC_IS, rp=ANCC_RP, rr=ANCC_RR,
                      gs_lo=PF_GR_SIG_MIN, gs_hi=PF_GR_SIG_MAX, ir_spr=0.01,
                      resamp=PF_RESAMP, gs=None):
    """S pasadas independientes del PF ANCC. -> pts (S,n) float32, loglik (S,)."""
    if gs is None:
        gs = _gr_sig(hw, tw_tvt, tw_gr, gs_lo, gs_hi)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.zeros((len(seeds), 0), np.float32), np.zeros(len(seeds))
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, _std, ll = _pf_ancc_seeds(
        ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
        ev["GR"].values.astype(np.float64), gg, gmin, gst, float(gs), ls, ir, int(N),
        float(alpha), float(rn), float(pn), float(is_spr), float(rp), float(rr),
        float(resamp), float(ir_spr), np.asarray(seeds, np.int64))
    return pts.astype(np.float32), ll


def softmax_w(ll, scale):
    """Pesos softmax(loglik/scale) estabilizados."""
    z = (ll - ll.max()) / scale
    w = np.exp(np.clip(z, -700, 0))
    return w / w.sum()
