"""Blend ADAPTATIVO superficie <-> particle filter. Barridos sobre el cache.

Secciones:
  0) candidatos solos y blends fijos (referencia: 0.25S+0.75P)
  1) oraculos (techo no implementable)
  2) w(nn_dist) sigmoide, por punto y por pozo
  3) w(nn_dist, md_since) y w(spread del PF)
  4) auto-verificacion leak-free: diagnostico + seleccion + pesos por pozo
  5) NNLS por pozo sobre el tramo enmascarado, con shrink al peso global

Uso: python research/ad04_blend.py <cache.npz> [cache_fit.npz]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adlib import FRACS, NAMES, Cache, report  # noqa: E402

np.set_printoptions(suppress=True)


def sig(x, d0, s):
    return 1.0 / (1.0 + np.exp(np.clip((x - d0) / s, -50, 50)))


def main():
    cpath = sys.argv[1]
    c = Cache(cpath)
    print(f"== {Path(cpath).name}: {c.NW} pozos, {len(c.y)} puntos ==\n")

    print("0) candidatos solos")
    for n in NAMES:
        report(c, n, getattr(c, n))
    print("\n0b) blends fijos S/P")
    base = None
    for w in np.arange(0, 0.61, 0.05):
        r = report(c, f"{w:.2f}*S + {1-w:.2f}*P", w * c.S + (1 - w) * c.P)
        if abs(w - 0.25) < 1e-9:
            base = r
    REF = base
    print(f"\n  REFERENCIA (0.25S+0.75P) = {REF:.3f}\n")
    print("0c) otros pares y 3-way")
    for na, nb in (("H", "P"), ("S", "Z"), ("S", "B"), ("P", "Z"), ("P", "B"), ("G", "P")):
        a, b = getattr(c, na), getattr(c, nb)
        tab = [(w, c.rmse(w * a + (1 - w) * b)) for w in np.arange(0, 1.001, 0.05)]
        w0, r0 = min(tab, key=lambda t: t[1])
        print(f"  {na}/{nb}: mejor w={w0:.2f} -> {r0:.3f}")
    best3 = None
    for wa in np.arange(0, 1.001, 0.05):
        for wb in np.arange(0, 1.001 - wa + 1e-9, 0.05):
            for wc in np.arange(0, 1.001 - wa - wb + 1e-9, 0.05):
                wd = 1 - wa - wb - wc
                r = c.rmse(wa * c.S + wb * c.P + wc * c.Z + wd * c.B)
                if best3 is None or r < best3[0]:
                    best3 = (r, wa, wb, wc, wd)
    print(f"  convexo S/P/Z/B optimo: S{best3[1]:.2f} P{best3[2]:.2f} Z{best3[3]:.2f} "
          f"B{best3[4]:.2f} -> {best3[0]:.3f}")

    print("\n1) oraculos (techo)")
    S, P = c.S, c.P
    e = np.where(np.abs(c.y - S) < np.abs(c.y - P), S, P)
    report(c, "min(S,P) por punto", e, REF)
    ss, sp = c.sse_well(S), c.sse_well(P)
    report(c, "min(S,P) por pozo", np.where(c.expand(ss <= sp), S, P), REF)
    wo = c.wopt_well(P, S)
    report(c, "w* por pozo (libre)", P + c.expand(wo) * (S - P), REF)
    report(c, "w* por pozo (clip 0..1)", P + c.expand(np.clip(wo, 0, 1)) * (S - P), REF)
    print(f"   w* por pozo: mediana {np.median(np.clip(wo,0,1)):.2f} "
          f"p10 {np.quantile(np.clip(wo,0,1),.1):.2f} p90 {np.quantile(np.clip(wo,0,1),.9):.2f}")

    print("\n2) w(nn_dist) sigmoide  [w = peso a la superficie]")
    nnw, nnp = c.per_well_med(c.nn)
    res = []
    for modo, x in (("punto", c.nn), ("pozo", nnp)):
        for d0 in (100, 150, 200, 300, 400, 600, 900, 1400, 2000):
            for s in (20, 50, 100, 200, 400, 800):
                for wmax in (0.25, 0.35, 0.5, 0.7, 1.0):
                    for wmin in (0.0, 0.1, 0.15, 0.2):
                        if wmin >= wmax:
                            continue
                        w = wmin + (wmax - wmin) * sig(x, d0, s)
                        res.append((modo, d0, s, wmax, wmin, c.rmse(w * S + (1 - w) * P)))
    rdf = pd.DataFrame(res, columns=["modo", "d0", "s", "wmax", "wmin", "rmse"]).sort_values("rmse")
    print(rdf.head(10).to_string(index=False))
    print("  mejor por modo:")
    print(rdf.groupby("modo").head(1).to_string(index=False))
    rdf.to_csv(Path(cpath).parent / "ad04_sig_nn.csv", index=False)

    print("\n2b) RMSE por bins de nn_dist")
    edges = [0, 100, 200, 300, 600, 1000, 1e9]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c.nn >= lo) & (c.nn < hi)
        if m.sum() < 200:
            continue
        rows.append({"bin": f"{lo:.0f}-{hi:.0f}", "n": int(m.sum()),
                     "S": round(c.rmse(S, m), 2), "P": round(c.rmse(P, m), 2),
                     "w*": round(c.wopt(P, S, m), 2),
                     "0.25/0.75": round(c.rmse(0.25 * S + 0.75 * P, m), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n3) w(nn, md_since): w* por celda")
    nnb = np.digitize(c.nn, [100, 200, 400, 800])
    mdb = np.digitize(c.md_since, [500, 1500, 3000, 5000])
    nnl = ["<100", "100-200", "200-400", "400-800", ">800"]
    mdl = ["<500", "0.5-1.5k", "1.5-3k", "3-5k", ">5k"]
    for lab, fn in (("w* (peso S)", lambda m: c.wopt(P, S, m)),
                    ("RMSE S", lambda m: c.rmse(S, m)),
                    ("RMSE P", lambda m: c.rmse(P, m))):
        rows = []
        for i, ln in enumerate(nnl):
            r = {"nn": ln}
            for j, lm in enumerate(mdl):
                m = (nnb == i) & (mdb == j)
                r[lm] = round(fn(m), 2) if m.sum() > 200 else np.nan
            rows.append(r)
        print(f"-- {lab} --")
        print(pd.DataFrame(rows).to_string(index=False))
    # sigmoide 2D: w = wmin+(wmax-wmin)*sig(nn)*ramp(md)
    res = []
    for d0 in (150, 250, 400, 700):
        for s in (50, 150, 400):
            for tau in (500, 1500, 3000, 6000, 1e9):
                for wmax in (0.3, 0.5, 0.8):
                    ramp = 1 - np.exp(-c.md_since / tau)
                    w = wmax * sig(c.nn, d0, s) * ramp
                    res.append((d0, s, tau, wmax, c.rmse(w * S + (1 - w) * P)))
    r2 = pd.DataFrame(res, columns=["d0", "s", "tau", "wmax", "rmse"]).sort_values("rmse")
    print("\n  sigmoide(nn) x rampa(md_since):")
    print(r2.head(8).to_string(index=False))

    if c.Pspr is not None and c.Pspr.max() > 0:
        print("\n3b) w por la incertidumbre del PF (spread entre replicas / std posterior)")
        for nm, x in (("spread", c.Pspr), ("std_post", c.Pstd)):
            q = np.quantile(x, [0, .2, .4, .6, .8, 1.0])
            rows = []
            for lo, hi in zip(q[:-1], q[1:]):
                m = (x >= lo) & (x <= hi)
                rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                             "S": round(c.rmse(S, m), 2), "P": round(c.rmse(P, m), 2),
                             "w*": round(c.wopt(P, S, m), 2)})
            print(f"-- {nm} --")
            print(pd.DataFrame(rows).to_string(index=False))
            best = None
            for x0 in np.quantile(x, [.3, .5, .7, .9]):
                for sc in (x0 / 4, x0 / 2, x0, 2 * x0):
                    for wmax in (0.3, 0.5, 0.8, 1.0):
                        w = wmax * (1 - sig(x, x0, max(sc, 1e-6)))
                        r = c.rmse(w * S + (1 - w) * P)
                        if best is None or r < best[0]:
                            best = (r, x0, sc, wmax)
            print(f"   mejor w=wmax*(1-sig({nm})): x0={best[1]:.1f} s={best[2]:.1f} "
                  f"wmax={best[3]:.1f} -> {best[0]:.3f}")

    print("\n4) auto-verificacion leak-free en la cola del prefijo")
    if not c.bt:
        print("   (cache sin backtests)")
        return
    # RMSE de backtest por pozo y candidato, pooled sobre los cortes disponibles
    btr = {}
    for n in NAMES:
        sse = np.zeros(c.NW); cnt = np.zeros(c.NW)
        for f, b in c.bt.items():
            if n not in b or len(b["y"]) == 0:
                continue
            sse += np.bincount(b["wid"], (b["y"] - b[n]) ** 2, c.NW)
            cnt += np.bincount(b["wid"], None, c.NW)
        btr[n] = np.sqrt(sse / np.maximum(cnt, 1))
        btr[n][cnt == 0] = np.nan
    realr = {n: np.sqrt(c.sse_well(getattr(c, n)) / c.n_well()) for n in NAMES}
    ok = np.isfinite(btr["S"]) & np.isfinite(btr["P"])
    print(f"   pozos con backtest: {ok.sum()}/{c.NW}")
    for n in NAMES:
        r = np.corrcoef(np.log(btr[n][ok] + 1), np.log(realr[n][ok] + 1))[0, 1]
        print(f"   corr(log bt_{n}, log real_{n}) = {r:+.3f}   "
              f"bt mediana {np.nanmedian(btr[n]):6.2f} vs real {np.median(realr[n]):6.2f}")
    dr_bt = np.log((btr["S"] + 1) / (btr["P"] + 1))
    dr_re = np.log((realr["S"] + 1) / (realr["P"] + 1))
    print(f"   corr(log ratio S/P bt, real) = {np.corrcoef(dr_bt[ok], dr_re[ok])[0,1]:+.3f}")

    # 4a) seleccion dura por pozo
    sel = np.where(btr["S"] < btr["P"], 1.0, 0.0)
    sel[~ok] = 0.25
    report(c, "seleccion dura S/P por backtest", c.expand(sel) * S + (1 - c.expand(sel)) * P, REF)
    # 4b) peso por inversa del MSE de backtest, con temperatura y shrink
    best = None
    for t in (0.5, 1.0, 2.0, 4.0):
        wr = (btr["P"] ** t) / (btr["P"] ** t + btr["S"] ** t)   # peso a S
        wr[~ok] = 0.25
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = np.clip((1 - lam) * wr + lam * 0.25, 0, 1)
            r = c.rmse(c.expand(w) * S + (1 - c.expand(w)) * P)
            if best is None or r < best[0]:
                best = (r, t, lam)
            print(f"   invMSE^t t={t:.1f} shrink={lam:.2f} -> {r:7.3f}")
    print(f"   mejor invMSE: t={best[1]} shrink={best[2]} -> {best[0]:.3f} ({best[0]-REF:+.3f})")

    # 4c) w por minimos cuadrados en el tramo enmascarado + shrink
    num = np.zeros(c.NW); den = np.zeros(c.NW); nb = np.zeros(c.NW)
    for f, b in c.bt.items():
        if "S" not in b:
            continue
        d = b["S"] - b["P"]
        num += np.bincount(b["wid"], (b["y"] - b["P"]) * d, c.NW)
        den += np.bincount(b["wid"], d * d, c.NW)
        nb += np.bincount(b["wid"], None, c.NW)
    wls = np.where(den > 1e-9, num / np.maximum(den, 1e-12), 0.25)
    wls_c = np.clip(wls, 0, 1)
    wtrue = np.clip(c.wopt_well(P, S), 0, 1)
    m2 = np.isfinite(wls_c) & (nb > 0)
    print(f"   corr(w_backtest, w_real) = {np.corrcoef(wls_c[m2], wtrue[m2])[0,1]:+.3f}  "
          f"(w_bt mediana {np.median(wls_c[m2]):.2f}, w_real mediana {np.median(wtrue[m2]):.2f})")
    for lam in (0.0, 0.2, 0.4, 0.6, 0.8):
        w = (1 - lam) * wls_c + lam * 0.25
        w[~m2] = 0.25
        report(c, f"w_LS backtest, shrink lam={lam:.1f}", c.expand(w) * S + (1 - c.expand(w)) * P, REF)
    # shrink bayesiano por n de backtest
    for k0 in (200, 1000, 5000, 20000):
        a = nb / (nb + k0)
        w = np.clip(a * wls_c + (1 - a) * 0.25, 0, 1)
        w[~m2] = 0.25
        report(c, f"w_LS shrink bayes k0={k0}", c.expand(w) * S + (1 - c.expand(w)) * P, REF)


if __name__ == "__main__":
    main()
