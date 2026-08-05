"""Barrido de los hiperparametros del particle filter contra NUESTRO CV.

Los valores actuales (alpha=0.998, N=600, sigma_GR clip [10,60], ruidos 0.002/0.005)
vienen del notebook publico de Roman, ajustados contra SU leaderboard. Nunca se han
barrido contra nuestra validacion, y el PF pesa 0.55 en el blend: es la ultima
superficie de mejora sin explorar.

Se evalua el BLEND adaptativo 1D completo (lo que decide), con S semillas para que
cada config sea asumible. Control pareado: misma S, mismos pozos, misma superficie.

Uso: python research/v9f_pfparams.py [k] [S]
"""
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import v7_wadapt as v7                                # noqa: E402
from ban_lib import run_pf_ancc_multi                 # noqa: E402
from cv import evaluate                               # noqa: E402
from model import _prior                              # noqa: E402
from pf_publico import _tw                            # noqa: E402

CURV = np.load(R / "v7_curvas.npz")

# (nombre, kwargs del PF). El primero es el control: parametros actuales.
CONFIGS = [
    ("control (Roman)",      {}),
    ("alpha 0.995",          dict(alpha=0.995)),
    ("alpha 0.999",          dict(alpha=0.999)),
    ("alpha 0.9995",         dict(alpha=0.9995)),
    ("N=1500",               dict(N=1500)),
    ("sigGR clip [5,40]",    dict(gs_lo=5.0, gs_hi=40.0)),
    ("sigGR clip [15,90]",   dict(gs_lo=15.0, gs_hi=90.0)),
    ("ruido pos x2",         dict(pn=0.010)),
    ("ruido rate x2",        dict(rn=0.004)),
    ("init spread 1.0",      dict(is_spr=1.0)),
]


def make_pred(S, pfkw):
    seeds = np.arange(1, S + 1, dtype=np.int64)
    ss_b, sp_b = CURV["sig_s"], CURV["sig_p"]
    nnb, mdb = CURV["nn_bins"], CURV["md_bins"]

    def predict(df_h, tw, wid=None):
        cut = v7.cut_of(df_h)
        prior, nn = _prior(df_h, v7.field(), wid, cut)
        t, g = _tw(tw)
        pts, _ = run_pf_ancc_multi(df_h, t, g, seeds=seeds, **pfkw)
        pf = pts.mean(0).astype(float)
        md = df_h.MD.values
        mdp = md[cut:] - md[cut - 1]
        ss = v7._interp_curve(nn[cut:], nnb, ss_b)
        sp = v7._interp_curve(mdp, mdb, sp_b)
        w = np.clip(sp**2 / (ss**2 + sp**2), 0.05, 0.9)
        return w * prior[cut:] + (1 - w) * pf

    return predict


def main(k=150, S=16):
    print(f"== barrido de parametros del PF | k={k} S={S} (blend adaptativo 1D) ==")
    base = None
    for nombre, kw in CONFIGS:
        t0 = time.time()
        r = evaluate(make_pred(S, kw), k=k, verbose=False)
        if base is None:
            base = r["rmse"]
        d = r["rmse"] - base
        marca = "  <<<" if d < -0.15 else ""
        print(f"  {nombre:22s} rmse={r['rmse']:7.3f}  d={d:+.3f}  "
              f"proxy={r['rmse_lb_proxy']:7.3f}  ({time.time()-t0:.0f}s){marca}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150,
         int(sys.argv[2]) if len(sys.argv) > 2 else 16)
