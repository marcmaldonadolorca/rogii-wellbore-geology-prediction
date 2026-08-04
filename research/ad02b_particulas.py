"""Reducir la varianza del PF: mas particulas (N) vs promediar R replicas de N=600.

Coste ~ N*R. Se compara a coste igual. k=60, pooled RMSE de pf_ancc solo y
blendeado 0.25S+0.75P.
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

CFG = [(600, 1), (600, 3), (600, 6), (1800, 1), (3600, 1), (1800, 3)]
ids = _select(well_ids(), 60, SEED)
field = M.SurfaceField()

Y, S = [], []
acc = {c: [] for c in CFG}
tim = {c: 0.0 for c in CFG}
for wid in ids:
    df, tw, cut = load_well(wid)
    if cut < 20 or cut >= len(df):
        continue
    df_h = df[TEST_COLS].copy()
    s, _ = M._prior(df_h, field, wid, cut)
    Y.append(df.TVT.values[cut:])
    S.append(s[cut:])
    t, g = PF._tw(tw)
    for c in CFG:
        n, r = c
        t0 = time.time()
        ps = [np.asarray(PF.run_pf_ancc(df_h, t, g, N=n)[0], float) for _ in range(r)]
        tim[c] += time.time() - t0
        acc[c].append(np.mean(ps, 0))
y = np.concatenate(Y)
s = np.concatenate(S)


def rmse(p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


print(f"{len(y)} puntos")
print(f"{'N':>6} {'R':>3} {'coste':>7} {'s/pozo':>7} {'P solo':>8} {'0.25S+0.75P':>12} {'w opt':>6} {'rmse w':>8}")
for c in CFG:
    p = np.concatenate(acc[c])
    d = s - p
    w = float(np.clip(((y - p) @ d) / (d @ d), 0, 1))
    print(f"{c[0]:6d} {c[1]:3d} {c[0]*c[1]/600:7.1f} {tim[c]/len(Y):7.2f} {rmse(p):8.3f} "
          f"{rmse(0.25*s+0.75*p):12.3f} {w:6.2f} {rmse(p + w*d):8.3f}")
