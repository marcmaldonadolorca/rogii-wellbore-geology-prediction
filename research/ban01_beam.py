"""1) predict_beam arreglado + las 7 configs de BEAMS medidas por separado (cv k=60).

BUG del wrapper viejo: `predict_beam(df_h, tw, cfg=0)` tiene 3 parametros
posicionales, asi que cv._takes_wid() lo da por interpolador espacial y le pasa
el ID del pozo como `cfg` -> BEAMS["0390d174"] -> TypeError. La solucion es una
FACTORIA que cierra sobre cfg y expone solo 2 argumentos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

from ban_lib import beam_all, make_beam
from cv import evaluate
from pf_publico import BEAMS

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60

rows = []
for cfg, (bs, mc, es, r, name) in enumerate(BEAMS):
    fn = make_beam(cfg)
    res = evaluate(fn, k=K, verbose=False)
    rows.append({"variante": f"beam_{name}", "bs": bs, "mc": mc, "es": es, "r": r,
                 "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
    print(f"beam_{name:6s} bs={bs:3d} mc={mc:5.1f} es={es:6.1f} r={r} -> {res['rmse']:8.3f}", flush=True)


def beam_mean(df_h, tw):
    return beam_all(df_h, tw).mean(0)


def beam_med(df_h, tw):
    return np.median(beam_all(df_h, tw), 0)


def beam_ref(df_h, tw):
    p = beam_all(df_h, tw)
    return (p[0] + p[3]) / 2.0        # (cons + sm5)/2, el beam_ref del notebook


for name, fn in [("beam_mean7", beam_mean), ("beam_median7", beam_med), ("beam_ref(cons+sm5)", beam_ref)]:
    res = evaluate(fn, k=K, verbose=False)
    rows.append({"variante": name, "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
    print(f"{name:20s} -> {res['rmse']:8.3f}", flush=True)

pd.DataFrame(rows).to_csv(R / f"ban01_beam_k{K}.csv", index=False)
print(pd.DataFrame(rows).sort_values("rmse").to_string(index=False))
