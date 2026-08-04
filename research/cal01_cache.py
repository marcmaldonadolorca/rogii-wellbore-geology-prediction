"""Cache LOWO de la superficie para los pozos del split k=150 de cv.py.

Guarda por pozo: MD,X,Y,Z,TVT(real),TVT_input,s_interp(LOWO),nn_dist,cut.
Con esto los barridos de calibracion de C_well no repiten el kd-tree (lo caro).

    python research/cal01_cache.py [k]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv import RAW, _select, well_ids  # noqa: E402
from model import SurfaceField  # noqa: E402

OUT = ROOT / "research" / "cal01_cache_k{}.npz"


def main(k=150):
    ids = _select(well_ids(), k, 42)
    fld = SurfaceField()
    print(f"nube: {len(fld.s)} puntos, {len(fld.names)} pozos")
    store = {}
    for n, wid in enumerate(ids):
        df = pd.read_csv(RAW / "train" / f"{wid}__horizontal_well.csv",
                         usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        m = df.TVT_input.isna()
        cut = int(m.idxmax()) if m.any() else len(df)
        if cut < 20 or cut >= len(df):
            continue
        s, nn = fld.interp(df.X.values, df.Y.values, exclude=wid)
        store[f"{wid}|md"] = df.MD.values.astype(np.float32)
        store[f"{wid}|x"] = df.X.values
        store[f"{wid}|y"] = df.Y.values
        store[f"{wid}|z"] = df.Z.values.astype(np.float32)
        store[f"{wid}|tvt"] = df.TVT.values.astype(np.float32)
        store[f"{wid}|tin"] = df.TVT_input.values.astype(np.float32)
        store[f"{wid}|s"] = s.astype(np.float32)
        store[f"{wid}|nn"] = nn.astype(np.float32)
        store[f"{wid}|cut"] = np.array([cut])
        if n % 25 == 0:
            print(f"  {n}/{len(ids)} {wid}", flush=True)
    np.savez_compressed(str(OUT).format(k), **store)
    print("guardado", str(OUT).format(k), len(store) // 9, "pozos")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
