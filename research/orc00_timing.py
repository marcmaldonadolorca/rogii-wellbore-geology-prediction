"""Mide el coste por pozo de cada predictor (para el presupuesto del re-run)."""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))

from cv import TEST_COLS, load_well, well_ids, _select, SEED  # noqa: E402
import model as M  # noqa: E402
import pf_publico as P  # noqa: E402

ids = _select(well_ids(), 8, SEED)
t0 = time.time(); field = M.SurfaceField(); t_field = time.time() - t0
print(f"SurfaceField: {t_field:.1f}s  {len(field.s)} pts", flush=True)

acc = {k: 0.0 for k in ("surf", "ancc", "pfz", "beam", "geom")}
npred = 0
for wid in ids:
    df, tw, cut = load_well(wid)
    if cut < 20 or cut >= len(df):
        continue
    d = df[TEST_COLS].copy()
    npred += len(df) - cut
    t = time.time(); M._prior(d, field, wid, cut); acc["surf"] += time.time() - t
    t = time.time(); P.predict_pf_ancc(d, tw); acc["ancc"] += time.time() - t
    t = time.time(); P.predict_pf_z(d, tw); acc["pfz"] += time.time() - t
    t = time.time(); P.predict_beam(d, tw); acc["beam"] += time.time() - t
n = len(ids)
print(f"pozos={n}  n_pred medio={npred/n:.0f}")
for k, v in acc.items():
    print(f"  {k:6s} {v/n:6.3f} s/pozo   -> 770 pozos: {v/n*770/60:6.1f} min")
