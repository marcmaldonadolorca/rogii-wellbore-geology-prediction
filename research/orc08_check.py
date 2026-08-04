"""Comprobacion: el cache reproduce cv.evaluate para el mejor modelo (k=60)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))

from cv import evaluate  # noqa: E402
import model as M  # noqa: E402
import pf_publico as P  # noqa: E402

field = M.SurfaceField()


def mk(w_surf):
    def predict(df_h, tw, wid=None):
        m = df_h.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df_h)
        prior, _ = M._prior(df_h, field, wid, cut)
        t, g = P._tw(tw)
        a, _ = P.run_pf_ancc(df_h, t, g)
        return w_surf * prior[cut:] + (1 - w_surf) * np.asarray(a, float)
    return predict


for k in (60, 150):
    print(f"\n##### k={k} mix 0.25*surf + 0.75*ancc via cv.evaluate #####")
    evaluate(mk(0.25), k=k)
