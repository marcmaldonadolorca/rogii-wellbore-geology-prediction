"""Riesgo de composicion: el LB publico esta sesgado a pozos largos con PS temprano
(proxy medido: n_pred >= 5147). Comprueba si el mejor modelo degrada MAS que el
baseline en ese subconjunto (=> el LB seria peor que el CV).

Reporta, para cada modelo: RMSE(todos), RMSE(proxy LB), ratio proxy/todos.
Si el ratio del modelo nuevo > ratio del baseline, el modelo es mas fragil al
sesgo de composicion del LB.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orclib import Cache, PRED, rmse, subset  # noqa: E402

LB_NPRED = 5147
LB_REAL = {"geom": 46.615, "surf_hmm": 19.829}   # envios reales


def main():
    c = Cache()
    y = c.d["y"]
    npred_pt = np.repeat(c.meta.n_pred.values, c.lens)
    psf_pt = np.repeat(c.meta.ps_frac.values, c.lens)
    mods = {p: c.d[p] for p in PRED}
    mods["mix .25surf+.75ancc"] = c.blend(surf=0.25, ancc=0.75)
    mods["media(ancc,pfz,beam)"] = (c.d["ancc"] + c.d["pfz"] + c.d["beam"]) / 3

    rows = []
    for tag, ids in (("TODOS(770)", None), ("k=60", subset(60)), ("k=150", subset(150))):
        base = c.mask(ids)
        m_lb = base & (npred_pt >= LB_NPRED)
        m_ps = base & (psf_pt <= 0.247)
        m_corto = base & (npred_pt < LB_NPRED)
        print(f"\n=== {tag} ===  pozos_lbproxy="
              f"{int(np.unique(c.wpt[m_lb]).size)}/{int(np.unique(c.wpt[base]).size)}")
        print(f"{'modelo':24s} {'todos':>8s} {'largos':>8s} {'cortos':>8s} "
              f"{'ratio':>7s} {'psfrac<=.247':>12s}")
        for k, p in mods.items():
            a, b, d, e = (rmse(y, p, base), rmse(y, p, m_lb),
                          rmse(y, p, m_corto), rmse(y, p, m_ps))
            print(f"{k:24s} {a:8.3f} {b:8.3f} {d:8.3f} {b/a:7.3f} {e:12.3f}")
            rows.append({"set": tag, "modelo": k, "rmse_todos": a, "rmse_largos": b,
                         "rmse_cortos": d, "ratio": b / a, "rmse_psfrac": e})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "orc04_riesgo.csv", index=False)

    # --- estimacion de LB con la calibracion de los dos envios reales --------
    print("\n=== calibracion local->LB con los dos envios reales ===")
    m_all = c.mask(None)
    r_geom = rmse(y, c.d["geom"], m_all)
    print(f"  geometrico local(cache,770) {r_geom:.3f} -> LB real {LB_REAL['geom']:.3f} "
          f"factor {LB_REAL['geom']/r_geom:.3f}")
    print("  (surf+HMM local 17.681 -> LB 19.829, factor 1.121)")
    f = LB_REAL["surf_hmm"] / 17.681
    for k, p in mods.items():
        print(f"  {k:24s} local770 {rmse(y,p,m_all):8.3f} -> LB estimado {rmse(y,p,m_all)*f:8.3f}")

    # --- degradacion relativa vs el baseline por decil de n_pred -------------
    print("\n=== RMSE por decil de n_pred (770 pozos) ===")
    q = np.quantile(c.meta.n_pred.values, np.linspace(0, 1, 11))
    print(f"{'decil n_pred':>16s} " + " ".join(f"{k[:9]:>9s}" for k in mods))
    for i in range(10):
        sel = m_all & (npred_pt >= q[i]) & (npred_pt <= q[i + 1])
        print(f"{int(q[i]):7d}-{int(q[i+1]):7d} " +
              " ".join(f"{rmse(y,p,sel):9.2f}" for p in mods.values()))


if __name__ == "__main__":
    main()
