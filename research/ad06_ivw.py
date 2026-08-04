"""Peso por VARIANZA INVERSA con modelos de error entrenables en el re-run.

Idea: el peso optimo entre superficie (S) y particle filter (P) es
    w_S = var_P / (var_S + var_P)
y ambas varianzas SON predecibles con senales disponibles en test:
    var_S ~ f(nn_dist, md_since)        (la superficie falla lejos de vecinos)
    var_P ~ g(spread entre replicas, std posterior, md_since)
Los modelos se ajustan en un conjunto de pozos de train DISJUNTO del de
evaluacion (honesto y reproducible en el re-run) y se aplican al de evaluacion.

Uso: python research/ad06_ivw.py <cache_fit.npz> <cache_eval.npz>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adlib import Cache, report  # noqa: E402

EPS = 1e-3


def feats_S(c):
    return np.column_stack([np.log(np.maximum(c.nn, 1.0)),
                            np.log(np.maximum(c.md_since, 1.0)),
                            np.ones(len(c.y))])


def feats_P(c):
    return np.column_stack([np.log(c.Pspr + 1.0), np.log(c.Pstd + 1.0),
                            np.log(np.maximum(c.md_since, 1.0)),
                            np.ones(len(c.y))])


def fit_var(X, e2):
    """OLS sobre log(e2) + reescalado para que la media coincida con E[e2]."""
    z = np.log(np.maximum(e2, 1e-4))
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    v = np.exp(X @ beta)
    k = float(np.mean(e2) / np.mean(v))
    return beta, k


def pred_var(X, beta, k):
    return k * np.exp(np.clip(X @ beta, -20, 20))


def main():
    cf = Cache(sys.argv[1])
    ce = Cache(sys.argv[2])
    print(f"fit={Path(sys.argv[1]).name} ({cf.NW} pozos)  eval={Path(sys.argv[2]).name} ({ce.NW} pozos)")

    for c, nm in ((cf, "fit"), (ce, "eval")):
        print(f"  [{nm}] S={c.rmse(c.S):.3f} P={c.rmse(c.P):.3f} "
              f"0.25/0.75={c.rmse(0.25*c.S+0.75*c.P):.3f}")

    bS, kS = fit_var(feats_S(cf), (cf.y - cf.S) ** 2)
    bP, kP = fit_var(feats_P(cf), (cf.y - cf.P) ** 2)
    print(f"\n  beta_S (lognn, logmd, 1) = {np.round(bS,3)}  k={kS:.3f}")
    print(f"  beta_P (logspr, logstd, logmd, 1) = {np.round(bP,3)}  k={kP:.3f}")

    for c, nm in ((cf, "fit"), (ce, "eval")):
        vS = pred_var(feats_S(c), bS, kS)
        vP = pred_var(feats_P(c), bP, kP)
        e2S, e2P = (c.y - c.S) ** 2, (c.y - c.P) ** 2
        print(f"\n== {nm} ==")
        print(f"   calibracion: mean vS {vS.mean():8.1f} vs real {e2S.mean():8.1f} | "
              f"mean vP {vP.mean():7.1f} vs real {e2P.mean():7.1f}")
        REF = report(c, "base 0.25S+0.75P", 0.25 * c.S + 0.75 * c.P)
        best = None
        for t in (0.25, 0.5, 0.75, 1.0, 1.5):
            for g in (0.25, 0.5, 1.0, 2.0, 4.0):
                w = (g * vP) ** t / ((g * vP) ** t + vS ** t)
                for wmax in (0.5, 0.7, 1.0):
                    ww = np.clip(w, 0, wmax)
                    r = c.rmse(ww * c.S + (1 - ww) * c.P)
                    if best is None or r < best[0]:
                        best = (r, t, g, wmax)
        print(f"   mejor IVW: t={best[1]} g={best[2]} wmax={best[3]} -> {best[0]:.3f} "
              f"({best[0]-REF:+.3f})   [w medio {np.mean((best[2]*vP)**best[1]/((best[2]*vP)**best[1]+vS**best[1])):.2f}]")
        # variante sin ajuste libre: t=1, g=1
        w = vP / (vP + vS)
        report(c, "   IVW puro t=1 g=1", w * c.S + (1 - w) * c.P, REF)


if __name__ == "__main__":
    main()
