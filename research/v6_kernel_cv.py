"""V6-kernel: CV honesto (LOWO) del stack EXACTO de kernel/rogii_v6.py.

Importa el kernel como modulo y evalua process_well con cv.evaluate:
  - superficie: nube LOWO (se excluyen los puntos del propio pozo, arbol nuevo
    por pozo, ~0.5 s) — igual que vera un pozo oculto de verdad.
  - override: exclude=wid (no puede matchearse consigo mismo).
  - GBM: entrenado en 380 pozos ajenos a k=60/k=150 -> honesto por construccion.

Uso: python research/v6_kernel_cv.py [k]
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv import evaluate  # noqa: E402

spec = importlib.util.spec_from_file_location("rogii_v6", ROOT / "kernel/rogii_v6.py")
K = importlib.util.module_from_spec(spec)
sys.modules["rogii_v6"] = K   # sin esto numba no puede picklear las funciones ('<dynamic>')
spec.loader.exec_module(K)

BASE = ROOT / "data/raw"

t0 = time.time()
FIELD, SIGS = K.build_static(BASE)
BOOSTER = K.load_booster()
IDX = {w: i for i, w in enumerate(FIELD.ids)}
print(f"estaticos: nube {len(FIELD.sval)} pts, {len(SIGS.ids)} firmas, "
      f"GBM {BOOSTER.num_trees()} arboles ({time.time()-t0:.0f}s)", flush=True)


def predict(df_h, tw, wid):
    keep = FIELD.wids != IDX[wid]
    f = K.Field(FIELD.xy_raw[keep], FIELD.sval[keep], FIELD.wids[keep], FIELD.ids)
    m = df_h.TVT_input.isna()
    cut = int(m.idxmax()) if m.any() else len(df_h)
    pred, via, ov = K.process_well(df_h, tw, cut, f, SIGS, BOOSTER,
                                   BASE / "train", exclude=wid)
    predict.vias[via] = predict.vias.get(via, 0) + 1
    predict.ov += bool(ov)
    return pred


predict.vias, predict.ov = {}, 0

if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"== v6 kernel stack, LOWO k={k} ==", flush=True)
    res = evaluate(predict, k=k)
    print(f"vias: {predict.vias} | override hits: {predict.ov}", flush=True)
