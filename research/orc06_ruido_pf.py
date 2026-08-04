"""Ruido de RNG del particle filter: cuanto se mueve el RMSE del mejor modelo
solo por cambiar la semilla del PF (misma data, mismo codigo).

Es la cota inferior de "diferencia significativa": si dos variantes difieren
menos que este ruido, no se pueden distinguir.
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

from cv import TEST_COLS, load_well  # noqa: E402
import pf_publico as P  # noqa: E402
from orclib import Cache, subset  # noqa: E402


@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    c = Cache()
    ids = sorted(subset(k))
    idx = [c.widx[w] for w in ids if w in c.widx]
    y = np.concatenate([c.d["y"][c.sl(i)] for i in idx])
    surf = np.concatenate([c.d["surf"][c.sl(i)] for i in idx])
    ancc0 = np.concatenate([c.d["ancc"][c.sl(i)] for i in idx])
    print(f"k={k}: {len(idx)} pozos, {len(y)} puntos")
    print(f"  cache seed0: ancc {np.sqrt(np.mean((y-ancc0)**2)):.4f} "
          f"mix {np.sqrt(np.mean((y-(0.25*surf+0.75*ancc0))**2)):.4f}")

    rows = []
    t0 = time.time()
    for s in range(1, nseed + 1):
        _seed_numba(s * 1000 + 7)
        np.random.seed(s)
        preds = []
        for w in ids:
            df, tw, cut = load_well(w)
            if cut < 20 or cut >= len(df):
                continue
            d = df[TEST_COLS].copy()
            t, gr = P._tw(tw)
            a, _ = P.run_pf_ancc(d, t, gr)
            preds.append(np.asarray(a, float))
        a = np.concatenate(preds)
        r_a = float(np.sqrt(np.mean((y - a) ** 2)))
        r_m = float(np.sqrt(np.mean((y - (0.25 * surf + 0.75 * a)) ** 2)))
        rows.append({"seed": s, "rmse_ancc": r_a, "rmse_mix": r_m})
        print(f"  seed {s}: ancc {r_a:.4f}  mix {r_m:.4f}   ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"orc06_ruido_pf_k{k}.csv", index=False)
    print(f"\nancc: media {df.rmse_ancc.mean():.4f} sd {df.rmse_ancc.std(ddof=1):.4f} "
          f"rango {df.rmse_ancc.max()-df.rmse_ancc.min():.4f}")
    print(f"mix : media {df.rmse_mix.mean():.4f} sd {df.rmse_mix.std(ddof=1):.4f} "
          f"rango {df.rmse_mix.max()-df.rmse_mix.min():.4f}")
    # media de PF con varias semillas (ensemble de semillas): cuanto gana
    print("\n(si promediaramos las semillas del PF el RMSE bajaria: ver orc07)")


if __name__ == "__main__":
    main()
