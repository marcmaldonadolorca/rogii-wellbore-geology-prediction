"""TECHO del GR: cuanto queda por ganar corrigiendo el residuo del prior.

(a) Verifica que el cache reproduce los numeros de model.py (20.325 / 17.681).
(b) ORACULOS post-PS: mejor constante r*, mejor recta r0+s*md, mejor cuadratica,
    y mejor r(md) suavizado (paso bajo) -> cotas de lo alcanzable.
(c) ¿Recupera el GR el r* oraculo? Correlacion de r* con:
      - r_hmm  = media del r estimado por el HMM actual
      - r_ll   = argmax de la log-verosimilitud ACUMULADA sobre todo el post-PS
                 con un desplazamiento CONSTANTE (sin dinamica, sin prior)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grlib import Wells, calib, hmm, pooled                           # noqa: E402

SPAN, STEP = 80.0, 0.5


def lowpass(r, md, scale):
    """Media movil gaussiana en MD (aprox: en indice, con paso medio)."""
    from scipy.ndimage import gaussian_filter1d
    dmd = np.median(np.diff(md)) if len(md) > 1 else 1.0
    sig = max(scale / max(dmd, 1e-6), 0.5)
    return gaussian_filter1d(r, sig, mode="nearest")


def main():
    W = Wells()
    rows = []
    grid = np.arange(-SPAN, SPAN + STEP, STEP)
    for w in W:
        cut = w["cut"]
        n = len(w["md"]) - cut
        tvt, prior, md = w["tvt"][cut:], w["prior"][cut:], w["md"][cut:]
        r = tvt - prior                       # residuo verdadero post-PS
        mdc = md - md[0]

        rec = {"well": w["wid"], "n": n, "nn": float(np.median(w["nn"][cut:])),
               "len_ft": float(md[-1] - md[0]),
               "sse_base": float((r ** 2).sum()),
               "r_mean": float(r.mean()), "r_std": float(r.std()),
               "r_ps": float(w["tvt_in"][cut - 1] - w["prior"][cut - 1])}

        # --- oraculo constante: r* = media (minimiza SSE) ---
        rec["sse_const"] = float(((r - r.mean()) ** 2).sum())
        rec["r_star"] = float(r.mean())

        # --- oraculo lineal en MD ---
        A = np.column_stack([np.ones(n), mdc])
        coef = np.linalg.lstsq(A, r, rcond=None)[0]
        rec["sse_lin"] = float(((r - A @ coef) ** 2).sum())
        rec["slope"] = float(coef[1])

        # --- oraculo lineal ANCLADO en el PS (r(0) = r_ps conocido) ---
        s_anch = float(np.dot(mdc, r - rec["r_ps"]) / max(np.dot(mdc, mdc), 1e-9))
        rec["sse_lin_anch"] = float(((r - (rec["r_ps"] + s_anch * mdc)) ** 2).sum())

        # --- oraculo cuadratico ---
        A2 = np.column_stack([np.ones(n), mdc, mdc ** 2])
        c2 = np.linalg.lstsq(A2, r, rcond=None)[0]
        rec["sse_quad"] = float(((r - A2 @ c2) ** 2).sum())

        # --- oraculos suavizados (que podria dar un r(md) suave perfecto) ---
        for sc in (2000, 1000, 500, 200):
            rec[f"sse_lp{sc}"] = float(((r - lowpass(r, md, sc)) ** 2).sum())

        # --- estimadores reales ---
        rh = hmm(w, sigma_r=0.01, dec=10)
        rec["sse_hmm"] = float(((r - rh) ** 2).sum())
        rec["r_hmm"] = float(rh.mean())

        # constante del HMM (aplicar su media, no su forma)
        rec["sse_hmm_const"] = float(((r - rh.mean()) ** 2).sum())

        # --- argmax de la verosimilitud acumulada, desplazamiento CONSTANTE ---
        cal = calib(w)
        if cal is None:
            rec["r_ll"] = np.nan
            rec["sse_ll"] = rec["sse_base"]
        else:
            a, b, sg = cal
            idx = np.arange(cut, len(w["md"]), 10)
            gr_i = w["gr"][idx]
            ok = np.isfinite(gr_i)
            ll = np.zeros(len(grid))
            for j, g in zip(np.where(ok)[0], gr_i[ok]):
                i = idx[j]
                pg = a * np.interp(w["prior"][i] + grid, w["tw_tvt"], w["tw_gr"]) + b
                ll += -0.5 * ((g - pg) / sg) ** 2
            r_ll = float(grid[int(np.argmax(ll))])
            rec["r_ll"] = r_ll
            rec["sse_ll"] = float(((r - r_ll) ** 2).sum())
            # el mismo pero restringido a +-15 ft alrededor del r del PS
            m = np.abs(grid - rec["r_ps"]) <= 15
            r_ll15 = float(grid[m][int(np.argmax(ll[m]))])
            rec["r_ll15"] = r_ll15
            rec["sse_ll15"] = float(((r - r_ll15) ** 2).sum())
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(Path(__file__).resolve().parent / "gr02_oraculo.csv", index=False)
    N = d.n.values

    def P(name, col):
        print(f"  {name:<34s} {pooled(d[col].values, N):7.3f}")

    print(f"pozos={len(d)} puntos={N.sum()}\n")
    print("RMSE pooled (k=150):")
    P("superficie sola (prior)", "sse_base")
    P("superficie + HMM actual", "sse_hmm")
    P("  HMM usando solo su media", "sse_hmm_const")
    print("  --- oraculos (techo) ---")
    P("ORACULO constante r*", "sse_const")
    P("ORACULO recta r0+s*md", "sse_lin")
    P("ORACULO recta anclada en PS", "sse_lin_anch")
    P("ORACULO cuadratica", "sse_quad")
    for sc in (2000, 1000, 500, 200):
        P(f"ORACULO suave (escala {sc} ft)", f"sse_lp{sc}")
    print("  --- estimadores del GR sin dinamica ---")
    P("argmax ll acumulada (const)", "sse_ll")
    P("argmax ll acumulada |r-r_ps|<=15", "sse_ll15")

    print("\n¿Recupera el GR el r* oraculo? (correlaciones sobre 150 pozos)")
    sub = d.dropna(subset=["r_ll"])
    for name, col in [("r_hmm  (media del HMM)", "r_hmm"),
                      ("r_ll   (argmax ll global)", "r_ll"),
                      ("r_ll15 (argmax ll acotado)", "r_ll15"),
                      ("r_ps   (residuo en el PS)", "r_ps")]:
        x, y = sub[col].values, sub.r_star.values
        pe = np.corrcoef(x, y)[0, 1]
        sp = pd.Series(x).corr(pd.Series(y), method="spearman")
        rmse_est = np.sqrt(np.mean((x - y) ** 2))
        print(f"  {name:<28s} pearson {pe:+.3f}  spearman {sp:+.3f}"
              f"  |err| med {np.median(np.abs(x-y)):5.2f} ft  rmse {rmse_est:5.2f}")
    print(f"  r* : mediana {np.median(d.r_star):+.2f} | std {d.r_star.std():.2f}"
          f" | p10 {np.percentile(d.r_star,10):+.1f} | p90 {np.percentile(d.r_star,90):+.1f}")
    print(f"  pendiente oraculo: mediana |s| {np.median(np.abs(d.slope))*1000:.2f} ft/1000ft")


if __name__ == "__main__":
    main()
