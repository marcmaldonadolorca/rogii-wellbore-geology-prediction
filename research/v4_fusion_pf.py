"""v4: fusion bayesiana de la superficie DENTRO del particle filter ANCC.

El PF publico (pf_publico._pf_ancc) hace random-walk del nivel U = TVT + Z
observando solo GR. Aqui se añade una OBSERVACION por punto del prior de
superficie U_prior(i) = S_aniso16(X_i, Y_i) + C_well (via model._prior, LOWO),
con sigma_s constante o adaptativo en nn_dist (fiabilidad medida en este campo:
rmse(nn) ~ 6.5 + 0.007*nn, espacio transformado aniso=16). Opcional:
  - init del rate con la pendiente del prior dU/dMD (init_prior)
  - atraccion en la transicion: pos += kappa*dm*(U_prior - pos)
  - blend a posteriori sobre la salida (para medir si la fusion agota el prior)

Kernel copiado de pf_publico._pf_ancc (NO editado alli) + seed para
reproducibilidad y comparaciones pareadas entre variantes.

Uso:
  .venv/bin/python research/v4_fusion_pf.py fase1 [k]   # barrido sigma_s
  .venv/bin/python research/v4_fusion_pf.py fase2 [k]   # init/kappa/blend sobre el mejor
  .venv/bin/python research/v4_fusion_pf.py confirm <variante> [k]
"""
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # raiz: cv.py, model.py
sys.path.insert(0, str(_HERE))          # research: pf_publico.py

from model import SurfaceField, _prior                       # noqa: E402
from pf_publico import (_interp1, _resamp, _grid, _gr_sig, _tw,   # noqa: E402
                        ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS,
                        ANCC_RP, ANCC_RR, PF_RESAMP, ANCC_N)


# ---------------------------------------------------------------- kernel numba
@njit(cache=True)
def _pf_fused(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, up, sig, kappa, N,
              ALPHA, RN, PN, IS, RP, RR, RESAMP, seed):
    """_pf_ancc + observacion del prior de superficie + atraccion kappa + seed.

    up[i]  = U_prior en el punto i (TVT_prior + Z), ya en nivel absoluto
    sig[i] = sigma de la observacion de superficie (>=1e8 -> desactivada)
    """
    np.random.seed(seed)
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); pm = md_v[0] - 1.
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            if kappa > 0.:
                pos[j] += kappa * dm * (up[i] - pos[j])
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
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
        if sig[i] < 1e8:                       # observacion de superficie
            ws = 0.
            for j in range(N):
                d = (pos[j] - up[i]) / sig[i]
                lk = max(np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
        ne = 0.
        for j in range(N): ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1. / N
        tv = 0.
        for j in range(N): tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        pm = md_v[i]
    return pts


# ---------------------------------------------------------------- wrapper
def run_pf_fused(hw, tw_tvt, tw_gr, up_ev, sig_ev, kappa=0.0, init_prior=False,
                 N=ANCC_N, seed=0):
    """Como pf_publico.run_pf_ancc pero con prior de superficie por punto."""
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return np.array([])
    ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
    md_ev = ev['MD'].values.astype(np.float64)
    ir = None
    if init_prior:                       # rate inicial = pendiente del prior
        m0 = md_ev <= md_ev[0] + 500.
        if m0.sum() >= 5:
            A = np.column_stack([md_ev[m0], np.ones(int(m0.sum()))])
            ir = float(np.linalg.lstsq(A, up_ev[m0], rcond=None)[0][0])
    if ir is None:                       # original: cola del prefijo
        tail = kn.tail(30); dt = np.diff(tail['TVT_input'].values)
        dz = np.diff(tail['Z'].values); dmd = np.diff(tail['MD'].values); m = dmd > 0
        ir = float(np.median((dt + dz)[m] / dmd[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    return _pf_fused(md_ev, ev['Z'].values.astype(np.float64),
                     ev['GR'].values.astype(np.float64), gg, gmin, gst, gs, ls, ir,
                     np.ascontiguousarray(up_ev, dtype=np.float64),
                     np.ascontiguousarray(sig_ev, dtype=np.float64), float(kappa),
                     N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR,
                     PF_RESAMP, seed)


class Fusion:
    """SurfaceField(aniso=16, k=24) + cache de priors por pozo (LOWO exacto)."""

    def __init__(self, field=None):
        self.field = field or SurfaceField(aniso=16.0)
        self._cache = {}

    def prior(self, df_h, wid):
        if wid is None or wid not in self._cache:
            m = df_h.TVT_input.isna()
            cut = int(m.idxmax()) if m.any() else len(df_h)
            p, nn = _prior(df_h, self.field, wid, cut)
            if wid is None:
                return p, nn, cut
            self._cache[wid] = (p, nn, cut)
        return self._cache[wid]

    def make(self, sigma=None, ab=None, kappa=0.0, init_prior=False, blend=None,
             seed=0, N=ANCC_N):
        """predict(df_h, tw, wid) para cv.evaluate.

        sigma: sigma_s constante (ft). ab=(a,b): sigma_s = a + b*nn_dist.
        Ninguno de los dos -> observacion de superficie desactivada (= pf_ancc).
        blend: w -> salida w*prior + (1-w)*pf (a posteriori, para referencia).
        """
        def predict(df_h, tw, wid=None):
            prior, nn, cut = self.prior(df_h, wid)
            up = (prior + df_h.Z.values)[cut:]
            if ab is not None:
                sig = ab[0] + ab[1] * nn[cut:]
            elif sigma is not None:
                sig = np.full(len(up), float(sigma))
            else:
                sig = np.full(len(up), 1e9)
            t, g = _tw(tw)
            out = np.asarray(run_pf_fused(df_h, t, g, up, sig, kappa=kappa,
                                          init_prior=init_prior, N=N, seed=seed),
                             dtype=float)
            if blend is not None:
                out = blend * prior[cut:] + (1. - blend) * out
            return out
        return predict


# ---------------------------------------------------------------- barridos
FASE1 = [
    ("pf_base_seed0", dict()),
    ("blend25_ref", dict(blend=0.25)),
    ("sig5", dict(sigma=5.)),
    ("sig10", dict(sigma=10.)),
    ("sig20", dict(sigma=20.)),
    ("sig40", dict(sigma=40.)),
    ("ad_6_007", dict(ab=(6., 0.007))),
    ("ad_12_014", dict(ab=(12., 0.014))),
    ("ad_24_028", dict(ab=(24., 0.028))),
]

# fase 2: sobre el mejor sigma_s de fase 1 (BASE2 se ajusta tras fase 1)
BASE2 = dict(ab=(24., 0.028))
FASE2 = [
    ("f2_base", dict(BASE2)),
    ("f2_init", dict(BASE2, init_prior=True)),
    ("f2_k001", dict(BASE2, kappa=0.001)),
    ("f2_k01", dict(BASE2, kappa=0.01)),
    ("f2_init_k001", dict(BASE2, init_prior=True, kappa=0.001)),
    ("f2_blend15", dict(BASE2, blend=0.15)),      # ¿post-blend aun ayuda?
    ("f2_kappa_solo001", dict(kappa=0.001)),      # atraccion sin observacion
    ("f2_kappa_solo01", dict(kappa=0.01)),
]

VARIANTS = {n: kw for n, kw in FASE1 + FASE2}


def _run(variants, k):
    from cv import evaluate
    fus = Fusion()
    rows = []
    for name, kw in variants:
        t0 = time.time()
        res = evaluate(fus.make(**kw), k=k, verbose=False)
        dt = time.time() - t0
        rows.append((name, res["rmse"], res["rmse_lb_proxy"], dt / res["n_wells"]))
        print(f"{name:16s} rmse={res['rmse']:7.3f}  proxy={res['rmse_lb_proxy']:7.3f} "
              f" ({dt:.0f}s, {dt/res['n_wells']:.2f}s/pozo)", flush=True)
    print("\nmejor:", min(rows, key=lambda r: r[1]))
    return rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fase1"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if mode == "fase1":
        _run(FASE1, k)
    elif mode == "fase2":
        _run(FASE2, k)
    elif mode == "confirm":
        name = sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else 150
        _run([(name, VARIANTS[name])], k)
