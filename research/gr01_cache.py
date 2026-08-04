"""Cache de los 150 pozos de cv (k=150) con el prior LOWO ya calculado.

Guarda por pozo: MD, GR, TVT real, prior (superficie LOWO + offset del prefijo),
cut, nn_dist en post-PS, y el typewell (TVT, GR). Todo lo demas (oraculo, HMM,
barridos) se hace despues sobre este cache sin reconstruir el KDTree.

Salida: research/gr01_cache.npz
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv import TEST_COLS, _select, load_well, well_ids                # noqa: E402
from model import SurfaceField, _prior                                # noqa: E402

OUT = Path(__file__).resolve().parent / "gr01_cache.npz"


def main(k=150):
    ids = _select(well_ids(), k, 42)
    field = SurfaceField()
    store, t0 = {}, time.time()
    kept = []
    for n, wid in enumerate(ids):
        df, twdf, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        dfh = df[TEST_COLS]
        prior, nn = _prior(dfh, field, wid, cut)
        tw = twdf.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        store[f"{wid}/md"] = df.MD.values.astype(np.float32)
        store[f"{wid}/gr"] = df.GR.values.astype(np.float32)
        store[f"{wid}/tvt"] = df.TVT.values.astype(np.float32)
        store[f"{wid}/tvt_in"] = dfh.TVT_input.values.astype(np.float32)
        store[f"{wid}/prior"] = prior.astype(np.float32)
        store[f"{wid}/nn"] = nn.astype(np.float32)
        store[f"{wid}/cut"] = np.int32(cut)
        store[f"{wid}/tw_tvt"] = tw.TVT.values.astype(np.float32)
        store[f"{wid}/tw_gr"] = tw.GR.values.astype(np.float32)
        kept.append(wid)
        if n % 25 == 0:
            print(f"  {n}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    store["ids"] = np.array(kept)
    np.savez_compressed(OUT, **store)
    print(f"{len(kept)} pozos -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
