"""4) Barrido de hiperparametros del PF ANCC contra NUESTRO CV (no contra el LB publico).

Sus valores (ANCC_ALPHA=0.998, N=600, sigma_GR clip [10,60], IS=0.3) estan tuneados
por otros. Aqui se mueve un factor cada vez desde esa base.

La varianza entre semillas de UNA pasada es ~+-0.7 ft, mayor que las diferencias
que se buscan: por eso cada config se mide como la MEDIA de 4 semillas FIJAS
(1,2,3,4) -> comparacion pareada y estimador con menos ruido.
"""
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod
from ban_lib import run_pf_ancc_multi
from cv import evaluate
from pf_publico import _tw

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cvmod.load_well = lru_cache(maxsize=None)(cvmod.load_well)
SEEDS = np.arange(1, NS + 1, dtype=np.int64)

BASE = dict(N=600, alpha=0.998, rn=0.002, pn=0.005, is_spr=0.3, rp=0.1, rr=0.001,
            gs_lo=10.0, gs_hi=60.0, ir_spr=0.01, resamp=0.5)

VARIANTS = [("base", {})]
VARIANTS += [(f"alpha={a}", {"alpha": a}) for a in (0.98, 0.99, 0.995, 0.999, 0.9995, 1.0)]
VARIANTS += [(f"N={n}", {"N": n}) for n in (150, 300, 1200, 2400)]
VARIANTS += [(f"gs_clip=[{lo},{hi}]", {"gs_lo": lo, "gs_hi": hi})
             for lo, hi in ((5, 60), (10, 30), (15, 45), (20, 80), (30, 30), (10, 120))]
VARIANTS += [(f"IS={s}", {"is_spr": s}) for s in (0.05, 0.1, 1.0, 3.0, 8.0)]
VARIANTS += [(f"PN={p}", {"pn": p}) for p in (0.002, 0.01, 0.02)]
VARIANTS += [(f"RN={p}", {"rn": p}) for p in (0.001, 0.004, 0.008)]
VARIANTS += [(f"RP={p}", {"rp": p}) for p in (0.03, 0.3)]
VARIANTS += [(f"RR={p}", {"rr": p}) for p in (0.0003, 0.003)]
VARIANTS += [(f"resamp={p}", {"resamp": p}) for p in (0.3, 0.8)]
VARIANTS += [(f"ir_spr={p}", {"ir_spr": p}) for p in (0.003, 0.03)]

rows = []
for name, over in VARIANTS:
    kw = {**BASE, **over}
    t0 = time.time()

    def predict(df_h, tw, wid=None, kw=kw):
        t, g = _tw(tw)
        pts, _ = run_pf_ancc_multi(df_h, t, g, seeds=SEEDS, **kw)
        return pts.mean(0)

    res = evaluate(predict, k=K, verbose=False)
    dt = time.time() - t0
    rows.append({"variante": name, "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"],
                 "s_por_pozo_total": dt / res["n_wells"], "n_semillas": NS, **kw})
    print(f"{name:20s} rmse {res['rmse']:8.3f}  lb_proxy {res['rmse_lb_proxy']:7.3f}  "
          f"({dt/res['n_wells']:.2f}s/pozo con {NS} semillas)", flush=True)
    pd.DataFrame(rows).to_csv(R / f"ban04_sweep_k{K}.csv", index=False)

df = pd.DataFrame(rows).sort_values("rmse")
print(df[["variante", "rmse", "lb_proxy", "s_por_pozo_total"]].to_string(index=False))
