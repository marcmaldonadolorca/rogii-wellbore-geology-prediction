"""¿Localiza mejor una emision por CORRELACION DE FORMA (NCC) que la gaussiana?

La gaussiana sobre valores absolutos falla: sensibilidad 0.93 API/ft vs 7.66 API
de ruido (1 sigma ~ 8.3 ft). La NCC compara el PATRON de una ventana de GR contra
el patron del typewell en el candidato, tras z-normalizar ambos: invariante a
escala y offset locales, que es lo que hace toda la familia publica.

Mide, para ventanas post-PS, el |error| del maximo de NCC vs el residuo real,
para varios tamanos de ventana, y lo compara con no hacer nada.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv import load_well, well_ids                                    # noqa: E402
from model import SurfaceField, _prior                                # noqa: E402

WINS = [50, 150, 400]
SPAN, STEP = 60.0, 0.5


def zn(a, axis=-1):
    a = a - a.mean(axis, keepdims=True)
    s = a.std(axis, keepdims=True)
    return a / np.maximum(s, 1e-9)


def main(nwells=25):
    field = SurfaceField()
    grid = np.arange(-SPAN, SPAN + STEP, STEP)
    err = {w: [] for w in WINS}
    base = {w: [] for w in WINS}

    for wid in well_ids()[:nwells]:
        df, twdf, cut = load_well(wid)
        dfh = df[["MD", "X", "Y", "Z", "GR", "TVT_input"]]
        prior, _ = _prior(dfh, field, wid, cut)
        r_true = df.TVT.values - prior
        gr = df.GR.values
        tw = twdf.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        tvt_tw, gr_tw = tw.TVT.values, tw.GR.values

        for W in WINS:
            for start in range(cut, len(df) - W, 700):
                idx = np.arange(start, start + W)
                idx = idx[np.isfinite(gr[idx])]        # el GR tiene huecos
                if len(idx) < W // 2:
                    continue
                obs = zn(gr[idx])
                # matriz (n_grid, W): typewell evaluado en prior+r para cada r
                cand = np.interp((prior[idx][None, :] + grid[:, None]).ravel(),
                                 tvt_tw, gr_tw).reshape(len(grid), len(idx))
                ncc = (zn(cand, axis=1) * obs[None, :]).mean(1)
                rt = float(np.median(r_true[idx]))
                err[W].append(abs(grid[np.argmax(ncc)] - rt))
                base[W].append(abs(rt))

    print(f"pozos={nwells}   (|error| mediano del maximo de NCC, ft)")
    print(f"{'W':>6} {'n':>5} {'sin GR':>8} {'NCC':>8} {'mejora':>8}")
    for W in WINS:
        if not err[W]:
            continue
        e, e0 = np.median(err[W]), np.median(base[W])
        print(f"{W:>6} {len(err[W]):>5} {e0:>8.2f} {e:>8.2f} {100*(1-e/max(e0,1e-9)):>7.0f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
