"""2) multi_scale_ncc portado: ventanas hw=8/15/25 del GR post-PS contra el GR
del PREFIJO DEL PROPIO POZO (no contra el typewell). Medido con cv k=60.

Extra: variante 'suavizada' (mediana movil del sc_ens) y anclada al ultimo TVT
conocido, por si el problema es el ruido punto a punto y no el sesgo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

from ban_lib import cut_of, ncc_signals
from cv import evaluate

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
CACHE = {}


def sigs(df_h, tw):
    key = (id(df_h), len(df_h))
    if key not in CACHE:
        CACHE.clear()
        CACHE[key] = ncc_signals(df_h, tw)
    return CACHE[key]


def mk(name):
    def predict(df_h, tw):
        return sigs(df_h, tw)[name]
    return predict


def mk_smooth(name, w):
    def predict(df_h, tw):
        v = pd.Series(sigs(df_h, tw)[name])
        return v.rolling(w, center=True, min_periods=1).median().values
    return predict


def mk_anchor(name, w=201):
    """sc suavizado pero re-anclado: el offset se fija con el ultimo TVT conocido."""
    def predict(df_h, tw):
        cut = cut_of(df_h)
        v = pd.Series(sigs(df_h, tw)[name]).rolling(w, center=True, min_periods=1).median().values
        return v - v[0] + float(df_h.TVT_input.values[cut - 1])
    return predict


rows = []
for name in ["sc8", "sc15", "sc25", "sc_cons", "sc_ens"]:
    res = evaluate(mk(name), k=K, verbose=False)
    rows.append({"variante": name, "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
    print(f"{name:12s} -> {res['rmse']:8.3f}", flush=True)

for w in (51, 201):
    res = evaluate(mk_smooth("sc_ens", w), k=K, verbose=False)
    rows.append({"variante": f"sc_ens_med{w}", "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
    print(f"sc_ens_med{w:<4d} -> {res['rmse']:8.3f}", flush=True)

res = evaluate(mk_anchor("sc_ens"), k=K, verbose=False)
rows.append({"variante": "sc_ens_med201_anclado", "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
print(f"sc_ens_anclado -> {res['rmse']:8.3f}", flush=True)

pd.DataFrame(rows).to_csv(R / f"ban02_ncc_k{K}.csv", index=False)
print(pd.DataFrame(rows).sort_values("rmse").to_string(index=False))
