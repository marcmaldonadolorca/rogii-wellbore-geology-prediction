"""Cuanto del oraculo es REALIZABLE: seleccion/pesos por pozo elegidos con un
backtest leak-free (se enmascara la cola del prefijo conocido y se juzga ahi).

Para cada pozo: cut2 = round(f*cut). Se recalculan los 5 predictores con
TVT_input enmascarado desde cut2 y se puntuan contra el TVT_input REAL en
[cut2, cut) -- que no es leak porque en test tambien se conoce.
Luego se aplica la eleccion al tramo post-PS real (predicciones del cache).

Salida: research/orc07_realizable.npz (scores de backtest por pozo y predictor).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from cv import TEST_COLS, load_well, well_ids  # noqa: E402
import model as M  # noqa: E402
import pf_publico as P  # noqa: E402
from orc01_cache import geom_pred, hdist  # noqa: E402

FRAC = 0.65
PRED = ("geom", "surf", "ancc", "pfz", "beam")


@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)


def main():
    _seed_numba(0)
    np.random.seed(0)
    field = M.SurfaceField()
    print(f"campo listo: {len(field.s)} puntos", flush=True)
    rows = []
    t0 = time.time()
    ids = well_ids()
    for j, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        c2 = int(round(FRAC * cut))
        rec = {"well": wid, "cut": cut, "c2": c2, "nbt": cut - c2}
        if c2 < 60 or cut - c2 < 30:
            for p in PRED:
                rec[f"bt_{p}"] = np.nan
            rec["bt_w"] = ""
            rows.append(rec)
            continue
        d2 = df[TEST_COLS].copy()
        d2.loc[d2.index[c2:], "TVT_input"] = np.nan
        hd = hdist(d2)
        yb = df.TVT_input.values[c2:cut]           # conocido en test: no hay leak
        sl = slice(0, cut - c2)                    # los predictores empiezan en c2

        g = geom_pred(d2, c2, hd)[c2:]
        s, _ = M._prior(d2, field, wid, c2)
        s = s[c2:]
        t, gr = P._tw(tw)
        a, _ = P.run_pf_ancc(d2, t, gr)
        z, _ = P.run_pf_z(d2, t, gr)
        b = P.predict_beam(d2, tw)
        cur = {"geom": g, "surf": s, "ancc": np.asarray(a, float),
               "pfz": np.asarray(z, float), "beam": np.asarray(b, float)}
        for p in PRED:
            rec[f"bt_{p}"] = float(np.sqrt(np.mean((yb - cur[p][sl]) ** 2)))
            rec[f"bias_{p}"] = float(np.mean(yb - cur[p][sl]))
        # pesos LS libres surf+ancc en el backtest
        A = np.column_stack([cur["surf"][sl], cur["ancc"][sl]])
        w = np.linalg.lstsq(A, yb, rcond=None)[0]
        rec["w_surf"], rec["w_ancc"] = float(w[0]), float(w[1])
        rows.append(rec)
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "orc07_backtest.csv", index=False)
    print(f"guardado orc07_backtest.csv  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
