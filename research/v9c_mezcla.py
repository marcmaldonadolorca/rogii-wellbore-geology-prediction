"""Promedio de los dos enfoques que empataron en LB: GBM(base adapt 1D) y blend 2D.

v7 (GBM resid_adapt) LB 9.072 | v8 (blend adaptativo 2D) LB 9.079. Empatan pero
son estructuralmente distintos: uno corrige con arboles, el otro pondera por
varianza. Si sus errores estan decorrelacionados, el promedio gana gratis.
"""
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

from cv import evaluate                                # noqa: E402
import v4_gbm as G                                     # noqa: E402
import v9b_sigp64 as V9                                # noqa: E402


def main(k=150):
    import numpy as np
    tab = V9.TAB
    p_blend = V9.make_pred(tab["sig_p_1d"])          # v8: blend adaptativo 2D
    p_gbm = G.make_cv_predictor("resid_adapt")       # v7: GBM sobre base adapt

    cache = {}

    def pred_blend(df_h, tw, wid=None):
        r = np.asarray(p_blend(df_h, tw, wid), float)
        cache["b"] = r
        return r

    def pred_gbm(df_h, tw, wid=None):
        return np.asarray(p_gbm(df_h, tw, wid), float)

    def mk_mix(w):
        def p(df_h, tw, wid=None):
            a = np.asarray(p_blend(df_h, tw, wid), float)
            b = np.asarray(p_gbm(df_h, tw, wid), float)
            return w * a + (1 - w) * b
        return p

    print(f"== mezcla blend2D x GBM, k={k} (pareado) ==")
    for name, fn in [("blend2D (v8)", pred_blend), ("GBM (v7)", pred_gbm)]:
        r = evaluate(fn, k=k, verbose=False)
        print(f"  {name:14s} rmse={r['rmse']:7.3f} proxy={r['rmse_lb_proxy']:7.3f}")
    for w in (0.3, 0.5, 0.7):
        r = evaluate(mk_mix(w), k=k, verbose=False)
        print(f"  mezcla w={w:<4} rmse={r['rmse']:7.3f} proxy={r['rmse_lb_proxy']:7.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
