"""Cuanto ruido mete la aleatoriedad del particle filter en el pooled RMSE (k=60).

El PF usa np.random dentro de numba: cada ejecucion da un resultado distinto.
Antes de comparar variantes hay que saber la desviacion tipica de la metrica.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from cv import SEED, TEST_COLS, _select, load_well, well_ids  # noqa: E402
import model as M  # noqa: E402
import pf_publico as PF  # noqa: E402

REP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ids = _select(well_ids(), 60, SEED)
field = M.SurfaceField()

Y, S, Preps = [], [], [[] for _ in range(REP)]
t0 = time.time()
for wid in ids:
    df, tw, cut = load_well(wid)
    if cut < 20 or cut >= len(df):
        continue
    df_h = df[TEST_COLS].copy()
    s, _ = M._prior(df_h, field, wid, cut)
    Y.append(df.TVT.values[cut:])
    S.append(s[cut:])
    t, g = PF._tw(tw)
    for r in range(REP):
        p, _ = PF.run_pf_ancc(df_h, t, g)
        Preps[r].append(np.asarray(p, float))
y = np.concatenate(Y)
s = np.concatenate(S)
P = [np.concatenate(p) for p in Preps]


def rmse(p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


print(f"{len(y)} puntos, {REP} replicas, {time.time()-t0:.0f}s")
rp = [rmse(p) for p in P]
print(f"pf_ancc solo      : {np.round(rp,3)}  media {np.mean(rp):.3f} sd {np.std(rp):.3f}")
for w in (0.15, 0.2, 0.25, 0.3, 0.35):
    rb = [rmse(w * s + (1 - w) * p) for p in P]
    print(f"{w:.2f}S+{1-w:.2f}P      : {np.round(rb,3)}  media {np.mean(rb):.3f} sd {np.std(rb):.3f}")
pm = np.mean(P, 0)
print(f"media de {REP} replicas del PF: {rmse(pm):.3f}   "
      f"0.25S+0.75Pmedia = {rmse(0.25*s+0.75*pm):.3f}")
np.savez_compressed(ROOT / "research/ad02_pf_reps.npz",
                    y=y.astype(np.float32), S=s.astype(np.float32),
                    P=np.array(P, np.float32),
                    lens=np.array([len(v) for v in Y]))
