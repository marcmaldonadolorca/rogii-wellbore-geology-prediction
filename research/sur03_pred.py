"""SUR-03: predictor de superficie multi-formacion + barrido (submuestreo, k, combinacion).

Predictor:
    S_f(X,Y) interpolada por formacion f (LOWO)   ->  prior_f = S_f - Z + C_f
    C_f = mediana en la cola del prefijo pre-PS de (TVT_input + Z - S_f)
    prediccion = combinacion de los 6 priors (media / mediana / pesos por el
    error de cada formacion EN EL PREFIJO)

Uso:  python sur03_pred.py <bloque>   (bloques: base, sub, k, comb, all)
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii/research")

from cv import evaluate                     # noqa: E402
import sur02_lib as L                       # noqa: E402

CAL_TAIL = 500
SCORE_WIN = 2000


def _cut(df_h):
    m = df_h.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df_h)


def priors(cloud, df_h, wid, cut, interp, cal_tail=CAL_TAIL, kmax=L.KMAX):
    """(P, dr, cut): P (n,6) priors por formacion ya calibrados con el prefijo."""
    X, Y, Z = df_h.X.values, df_h.Y.values, df_h.Z.values
    d, gi, dr = cloud.neighbors(wid, X, Y, kmax=kmax)
    S = interp(cloud, d, gi, X, Y)
    base = S - Z[:, None]
    lo = max(0, cut - cal_tail)
    r = df_h.TVT_input.values[lo:cut, None] - base[lo:cut]
    C = np.nanmedian(r, axis=0)
    bad = ~np.isfinite(C)
    if bad.any():
        C[bad] = np.nanmedian(C[~bad]) if (~bad).any() else 0.0
    return base + C, dr


def combine(P, tvt_in, cut, mode="buda", score_win=SCORE_WIN, temp=1.0,
            topn=None):
    """(n_post,) combinando las 6 columnas de P[cut:]."""
    Q = P[cut:]
    good = np.isfinite(Q).all(0)
    if not good.any():
        good = np.ones(6, bool)
    if mode == "buda":
        return P[cut:, 5]
    if mode == "mean":
        return np.nanmean(Q[:, good], 1)
    if mode == "median":
        return np.nanmedian(Q[:, good], 1)
    # pesos por error en el prefijo
    lo = max(0, cut - score_win)
    e = tvt_in[lo:cut, None] - P[lo:cut]
    mse = np.nanmean(e ** 2, axis=0)
    mse = np.where(np.isfinite(mse) & good, mse, np.inf)
    if mode == "best":
        return Q[:, int(np.argmin(mse))]
    if mode == "topn":
        order = np.argsort(mse)[:topn]
        return np.nanmean(Q[:, order], 1)
    if mode == "winv":                      # w ~ 1/mse
        w = 1.0 / np.maximum(mse, 1e-6)
    elif mode == "winv_sd":                 # w ~ 1/sd
        w = 1.0 / np.maximum(np.sqrt(mse), 1e-3)
    elif mode == "softmax":                 # w ~ exp(-mse/(temp*min))
        w = np.exp(-mse / (temp * max(np.min(mse), 1e-6)))
    else:
        raise ValueError(mode)
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    return np.nansum(Q * w, 1)


def make_pred(cloud, interp=None, mode="buda", cal_tail=CAL_TAIL,
              score_win=SCORE_WIN, kmax=L.KMAX, ret_P=False, **ckw):
    interp = interp or (lambda c, d, gi, X, Y: L.idw(c, d, gi, 16))

    def predict(df_h, tw, wid=None):
        cut = _cut(df_h)
        P, dr = priors(cloud, df_h, wid, cut, interp, cal_tail, kmax)
        out = combine(P, df_h.TVT_input.values, cut, mode, score_win, **ckw)
        return np.where(np.isfinite(out), out, P[cut:, 5])

    return predict


def run(name, pred, k=60, log=None):
    t0 = time.time()
    r = evaluate(pred, k=k, verbose=False)
    dt = time.time() - t0
    line = (f"{name:44s} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}"
            f"  med={r['per_well'].rmse.median():6.2f}  {dt:5.1f}s")
    print(line, flush=True)
    if log is not None:
        log.append({"variante": name, "rmse": r["rmse"],
                    "proxy": r["rmse_lb_proxy"],
                    "med": float(r["per_well"].rmse.median()), "s": dt})
    return r


if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "base"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    log = []

    if block in ("base", "all"):
        from model import make_predictor
        run("REF model.py idw s10 k16 BUDA", make_predictor(use_hmm=False), K, log)
        c = L.Cloud(subsample=10)
        run("mio idw s10 k16 BUDA", make_pred(c, mode="buda"), K, log)

    if block in ("sub", "all"):
        for sub in (10, 3, 1):
            c = L.Cloud(subsample=sub)
            for k in (8, 16, 32, 64):
                run(f"idw s{sub} k{k} BUDA",
                    make_pred(c, interp=lambda cc, d, gi, X, Y, k=k: L.idw(cc, d, gi, k)),
                    K, log)
            del c

    if block in ("comb", "all"):
        c = L.Cloud(subsample=10)
        for k in (16, 32):
            it = lambda cc, d, gi, X, Y, k=k: L.idw(cc, d, gi, k)  # noqa: E731
            for mode in ("buda", "mean", "median", "best", "winv", "winv_sd",
                         "softmax"):
                run(f"idw s10 k{k} {mode}", make_pred(c, interp=it, mode=mode), K, log)
            for tn in (2, 3, 4):
                run(f"idw s10 k{k} top{tn}",
                    make_pred(c, interp=it, mode="topn", topn=tn), K, log)

    if log:
        pd.DataFrame(log).to_csv(
            f"/home/ftpx100/work/active/kaggle-rogii/research/sur03_{block}_k{K}.csv",
            index=False)
