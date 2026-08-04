"""Barrido de estimadores de C_well sobre el cache LOWO (k=150).

Modelo: TVT_hat = S_interp - Z + c_hat(punto). El error es e - c_hat con
e = TVT + Z - S_interp. Todo lo que se ajusta usa SOLO el prefijo pre-PS.

Variantes: ventana, estimador robusto, tendencia (arco / plano X,Y), shrink,
clip de gradiente, amortiguacion de la extrapolacion, pesos por calidad de
interpolacion (distancia al vecino mas cercano).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cal02_diag import load, pooled  # noqa: E402


# ---------- estimadores de nivel (sin tendencia) ----------
def _wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def level(v, w, est):
    if len(v) == 0:
        return 0.0
    if est == "median":
        return _wmedian(v, w)
    if est == "mean":
        return float(np.average(v, weights=w))
    if est.startswith("trim"):
        p = float(est[4:]) / 100.0
        lo, hi = np.quantile(v, [p, 1 - p])
        m = (v >= lo) & (v <= hi)
        return float(np.average(v[m], weights=w[m])) if m.any() else float(np.median(v))
    if est == "huber":
        c = _wmedian(v, w)
        for _ in range(15):
            r = v - c
            s = 1.4826 * np.median(np.abs(r)) + 1e-9
            k = 1.345 * s
            ww = w * np.minimum(1.0, k / np.maximum(np.abs(r), 1e-12))
            c_new = float(np.average(v, weights=ww))
            if abs(c_new - c) < 1e-6:
                break
            c = c_new
        return c
    if est == "last":                      # ultimo valor conocido (ancla dura)
        return float(v[-1])
    raise ValueError(est)


# ---------- regresion robusta lineal (IRLS Huber, pesos externos) ----------
def rfit(A, y, w, robust=True, ridge=0.0):
    ww = w.copy()
    beta = np.zeros(A.shape[1])
    for it in range(20 if robust else 1):
        W = ww[:, None]
        M = A.T @ (W * A)
        if ridge:
            R = np.eye(A.shape[1]) * ridge
            R[0, 0] = 0.0
            M = M + R
        try:
            b = np.linalg.solve(M, A.T @ (ww * y))
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(A, y, rcond=None)[0]
        if not robust:
            return b
        r = y - A @ b
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        k = 1.345 * s
        ww = w * np.minimum(1.0, k / np.maximum(np.abs(r), 1e-12))
        if np.max(np.abs(b - beta)) < 1e-8:
            return b
        beta = b
    return beta


def weights(d, sl, mode, d0=300.0):
    n = sl.stop - sl.start
    if mode == "none":
        return np.ones(n)
    nn = d["nn"][sl].astype(np.float64)
    if mode == "inv2":
        return 1.0 / (1.0 + (nn / d0) ** 2)
    if mode == "inv1":
        return 1.0 / (1.0 + nn / d0)
    if mode == "recent":                   # peso exponencial hacia el PS
        return np.exp(-(n - 1 - np.arange(n)) / 400.0)
    raise ValueError(mode)


# ---------- predictor generico ----------
def chat(d, win=500, est="median", trend="none", shrink=1.0, clip=np.inf,
         damp=np.inf, wmode="none", robust=True, ridge=0.0):
    c = d["cut"]
    lo = max(0, c - win)
    sl = slice(lo, c)
    y = d["ein"][sl].astype(np.float64)
    w = weights(d, sl, wmode)
    post = slice(c, len(d["e"]))
    npost = post.stop - post.start
    if trend == "none":
        return np.full(npost, level(y, w, est))

    u0 = d["arc"][c - 1]
    if trend == "arc":
        up, uq = d["arc"][sl] - u0, d["arc"][post] - u0
        A = np.column_stack([np.ones(len(y)), up])
        Q = np.column_stack([np.ones(npost), uq])
        span = float(np.ptp(up)) if len(up) > 1 else 0.0
    elif trend == "plane":
        xp = (d["x"][sl] - d["x"][c - 1]) / 1000.0
        yp = (d["y"][sl] - d["y"][c - 1]) / 1000.0
        xq = (d["x"][post] - d["x"][c - 1]) / 1000.0
        yq = (d["y"][post] - d["y"][c - 1]) / 1000.0
        A = np.column_stack([np.ones(len(y)), xp, yp])
        Q = np.column_stack([np.ones(npost), xq, yq])
        span = float(np.hypot(np.ptp(xp), np.ptp(yp)))
    else:
        raise ValueError(trend)

    b = rfit(A, y, w, robust=robust, ridge=ridge)
    g = b[1:] * shrink
    gn = np.linalg.norm(g)
    if gn > clip:
        g = g * (clip / gn)
    # amortiguacion: la tendencia solo se extrapola hasta damp*span del prefijo
    if np.isfinite(damp) and span > 0:
        lim = damp * span
        nrm = np.linalg.norm(Q[:, 1:], axis=1)
        sc = np.minimum(1.0, lim / np.maximum(nrm, 1e-9))
        Q = Q.copy()
        Q[:, 1:] = Q[:, 1:] * sc[:, None]
    return b[0] + Q[:, 1:] @ g


def run(W, name, **kw):
    errs = [d["e"][d["cut"]:] - chat(d, **kw) for d in W]
    per = np.array([float(np.sqrt((e ** 2).mean())) for e in errs])
    return {"variante": name, "rmse": pooled(errs), "med": float(np.median(per)),
            "p90": float(np.quantile(per, .9)), "max": float(per.max())}


if __name__ == "__main__":
    W = load()
    res = []

    print("== 1. ventana (mediana, sin tendencia) ==")
    for win in (100, 250, 500, 1000, 2000, 10 ** 9):
        res.append(run(W, f"win={win} median", win=win))
        print(f"  {res[-1]['variante']:26s} rmse={res[-1]['rmse']:7.3f} med={res[-1]['med']:6.2f}")

    print("== 2. estimador (win=500) ==")
    for est in ("median", "mean", "trim10", "trim20", "huber", "last"):
        res.append(run(W, f"win=500 {est}", win=500, est=est))
        print(f"  {res[-1]['variante']:26s} rmse={res[-1]['rmse']:7.3f} med={res[-1]['med']:6.2f}")

    print("== 3. tendencia en arco (huber, shrink/clip/damp) ==")
    for win in (500, 1000, 2000, 10 ** 9):
        for sh in (0.25, 0.5, 0.75, 1.0):
            r = run(W, f"arc win={win} sh={sh}", win=win, est="huber", trend="arc", shrink=sh)
            res.append(r)
            print(f"  {r['variante']:26s} rmse={r['rmse']:7.3f} med={r['med']:6.2f} max={r['max']:7.1f}")

    print("== 4. tendencia plano X,Y ==")
    for win in (500, 1000, 10 ** 9):
        for sh in (0.25, 0.5, 1.0):
            r = run(W, f"plane win={win} sh={sh}", win=win, trend="plane", shrink=sh)
            res.append(r)
            print(f"  {r['variante']:26s} rmse={r['rmse']:7.3f} med={r['med']:6.2f} max={r['max']:7.1f}")

    print("== 5. pesos por calidad de interpolacion (win=500, median) ==")
    for wm in ("none", "inv1", "inv2", "recent"):
        r = run(W, f"w={wm}", win=500, wmode=wm)
        res.append(r)
        print(f"  {r['variante']:26s} rmse={r['rmse']:7.3f} med={r['med']:6.2f}")

    df = pd.DataFrame(res).sort_values("rmse")
    df.to_csv(ROOT / "research" / "cal03_sweep.csv", index=False)
    print("\nTOP 12:")
    print(df.head(12).to_string(index=False))
