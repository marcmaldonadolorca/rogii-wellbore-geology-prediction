"""Diagnostico: ¿la verosimilitud del GR localiza el residuo real?

Para varios pozos y varias ventanas post-PS, evalua log p(GR | r) sobre la rejilla
de residuos y compara el argmin con el residuo VERDADERO r = TVT_real - prior.
Si la emision tiene senal, el minimo debe caer cerca de r_real.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv import load_well, well_ids                                    # noqa: E402
from model import SurfaceField, _prior                                # noqa: E402

WINDOWS = [1, 25, 100, 400]      # puntos promediados en la verosimilitud
SPAN, STEP = 80.0, 0.5


def main(nwells=15):
    field = SurfaceField()
    grid = np.arange(-SPAN, SPAN + STEP, STEP)
    acc = {w: [] for w in WINDOWS}
    accr = {w: [] for w in WINDOWS}
    for wid in well_ids()[:nwells]:
        df, tw, cut = load_well(wid)
        dfh = df[["MD", "X", "Y", "Z", "GR", "TVT_input"]]
        prior, _ = _prior(dfh, field, wid, cut)
        r_true = df.TVT.values - prior

        tw = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        tvt_tw, gr_tw = tw.TVT.values, tw.GR.values
        gr = df.GR.values

        known = dfh.TVT_input.values[:cut]
        gp = np.interp(known, tvt_tw, gr_tw)
        ok = np.isfinite(gr[:cut]) & np.isfinite(gp)
        if ok.sum() < 40:
            continue
        A = np.column_stack([gp[ok], np.ones(ok.sum())])
        a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
        sg = float(np.clip(np.std(gr[:cut][ok] - (a * gp[ok] + b)), 5, 60))

        for start in range(cut, len(df) - max(WINDOWS), 600):
            for W in WINDOWS:
                idx = np.arange(start, start + W)
                idx = idx[np.isfinite(gr[idx])]
                if len(idx) < max(1, W // 2):
                    continue
                # r constante en la ventana: suma de log-verosimilitudes
                ll = np.zeros(len(grid))
                for i in idx:
                    pg = a * np.interp(prior[i] + grid, tvt_tw, gr_tw) + b
                    ll += -0.5 * ((gr[i] - pg) / sg) ** 2
                r_hat = grid[np.argmax(ll)]
                rt = float(np.median(r_true[idx]))
                acc[W].append(abs(r_hat - rt))
                accr[W].append(abs(rt))     # error si no hicieramos nada (r=0 -> |rt|)

    print(f"pozos={nwells}  ventanas post-PS evaluadas por tamano:")
    print(f"{'W':>5} {'n':>5} {'|err| MAP':>11} {'|err| sin GR':>13} {'mejora':>8}")
    for W in WINDOWS:
        if not acc[W]:
            continue
        e, e0 = np.array(acc[W]), np.array(accr[W])
        print(f"{W:>5} {len(e):>5} {np.median(e):>11.2f} {np.median(e0):>13.2f} "
              f"{100*(1-np.median(e)/max(np.median(e0),1e-9)):>7.0f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
