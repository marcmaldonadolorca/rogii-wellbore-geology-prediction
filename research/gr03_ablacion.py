"""¿La ganancia del HMM es del GR, o del ANCLAJE en el punto PS?

El HMM arranca el estado en r0 = TVT_input[PS] - prior[PS] y ademas lleva un
prior gaussiano centrado en r0 (sigma_prior=25). Eso solo ya desplaza el prior.
Control obligatorio: el mismo HMM con la emision de GR APAGADA.

Mide tambien correlaciones parciales: ¿aporta r_hmm algo sobre r_ps?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grlib import Wells, hmm, pooled                                  # noqa: E402


def zero_logp(w, cal, grid, rows, sigma_gr=None):
    return np.zeros((len(rows), len(grid)))


def main():
    W = Wells()
    rows = []
    for w in W:
        cut = w["cut"]
        r = w["tvt"][cut:] - w["prior"][cut:]
        n = len(r)
        r_ps = float(w["tvt_in"][cut - 1] - w["prior"][cut - 1])
        rec = {"well": w["wid"], "n": n, "r_ps": r_ps, "r_star": float(r.mean()),
               "sse_base": float((r ** 2).sum())}
        for sh in (0.25, 0.5, 0.75, 1.0):
            rec[f"sse_ps{sh}"] = float(((r - sh * r_ps) ** 2).sum())
        rh = hmm(w, sigma_r=0.01, dec=10)
        rec["sse_hmm"] = float(((r - rh) ** 2).sum())
        rec["r_hmm"] = float(rh.mean())
        r0 = hmm(w, sigma_r=0.01, dec=10, logp_fn=zero_logp)
        rec["sse_noGR"] = float(((r - r0) ** 2).sum())
        rec["r_noGR"] = float(r0.mean())
        rg = hmm(w, sigma_r=0.01, dec=10, sigma_prior=np.inf)
        rec["sse_puroGR"] = float(((r - rg) ** 2).sum())
        rec["r_puroGR"] = float(rg.mean())
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(Path(__file__).resolve().parent / "gr03_ablacion.csv", index=False)
    N = d.n.values

    def P(name, col):
        print(f"  {name:<40s} {pooled(d[col].values, N):7.3f}")

    print(f"pozos={len(d)} puntos={N.sum()}\nRMSE pooled:")
    P("prior solo", "sse_base")
    for sh in (0.25, 0.5, 0.75, 1.0):
        P(f"prior + {sh}*r_ps (constante, SIN GR)", f"sse_ps{sh}")
    P("HMM completo (GR + anclaje)", "sse_hmm")
    P("HMM con emision GR APAGADA (control)", "sse_noGR")
    P("HMM sin prior de superficie (GR puro)", "sse_puroGR")

    print("\ncorrelaciones con r* (oraculo constante):")
    for c in ["r_ps", "r_hmm", "r_noGR", "r_puroGR"]:
        print(f"  {c:<10s} pearson {np.corrcoef(d[c], d.r_star)[0,1]:+.3f}")
    # parcial: r_hmm vs r* controlando r_ps
    def partial(x, y, z):
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = y - np.polyval(np.polyfit(z, y, 1), z)
        return np.corrcoef(rx, ry)[0, 1]
    print(f"  parcial r_hmm~r* | r_ps      : {partial(d.r_hmm.values, d.r_star.values, d.r_ps.values):+.3f}")
    print(f"  parcial r_noGR~r* | r_ps     : {partial(d.r_noGR.values, d.r_star.values, d.r_ps.values):+.3f}")
    # cuanto se aleja el HMM del anclaje
    print(f"\n  |r_hmm - r_noGR| mediana {np.median(np.abs(d.r_hmm-d.r_noGR)):.2f} ft"
          f"  (desviacion que introduce el GR)")
    dev = d.r_hmm - d.r_noGR
    err_gain = np.abs(d.r_noGR - d.r_star) - np.abs(d.r_hmm - d.r_star)
    print(f"  el GR acerca a r* en {np.mean(err_gain>0):.1%} de pozos"
          f" (mejora media {err_gain.mean():+.2f} ft)")
    print(f"  corr(desviacion del GR, error que quedaba) = "
          f"{np.corrcoef(dev, d.r_star - d.r_noGR)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
