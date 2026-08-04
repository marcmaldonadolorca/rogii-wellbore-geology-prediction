"""¿Que mapa de referencia GR(TVT) localiza mejor el residuo?

Compara tres referencias en la verosimilitud:
  tw    : GR(TVT) del typewell (lo que usaba model.hmm_refine)
  self  : GR(TVT) construido con el PREFIJO del propio pozo (TVT_input, GR)
          -- el enunciado (slide 9) dice que tiene mejor resolucion
  both  : media de ambas, cada una z-normalizada

Tambien informa de la COBERTURA: que fraccion de los TVT candidatos cae dentro
del rango que cada referencia realmente cubre (fuera de rango np.interp aplana
y la verosimilitud engaña).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv import load_well, well_ids                                    # noqa: E402
from model import SurfaceField, _prior                                # noqa: E402

WINDOWS = [25, 100, 400]
SPAN, STEP = 60.0, 0.5


def build_ref(tvt, gr):
    m = np.isfinite(tvt) & np.isfinite(gr)
    if m.sum() < 30:
        return None
    o = np.argsort(tvt[m])
    t, g = tvt[m][o], gr[m][o]
    # promediar duplicados de TVT para que np.interp sea estable
    ut, inv = np.unique(np.round(t, 2), return_inverse=True)
    ug = np.bincount(inv, weights=g) / np.bincount(inv)
    return ut, ug


def main(nwells=15):
    field = SurfaceField()
    grid = np.arange(-SPAN, SPAN + STEP, STEP)
    res = {k: {W: [] for W in WINDOWS} for k in ("tw", "self", "both")}
    base = {W: [] for W in WINDOWS}
    cover = {"tw": [], "self": []}

    for wid in well_ids()[:nwells]:
        df, twdf, cut = load_well(wid)
        dfh = df[["MD", "X", "Y", "Z", "GR", "TVT_input"]]
        prior, _ = _prior(dfh, field, wid, cut)
        r_true = df.TVT.values - prior
        gr = df.GR.values

        refs = {"tw": build_ref(twdf.TVT.values, twdf.GR.values),
                "self": build_ref(dfh.TVT_input.values[:cut], gr[:cut])}
        if refs["tw"] is None or refs["self"] is None:
            continue
        # cobertura: TVT reales post-PS dentro del rango de cada referencia
        for k, r in refs.items():
            t = df.TVT.values[cut:]
            cover[k].append(float(np.mean((t >= r[0][0]) & (t <= r[0][-1]))))

        # calibracion afin de cada referencia contra el GR del prefijo
        cal = {}
        for k, (rt, rg) in refs.items():
            gp = np.interp(dfh.TVT_input.values[:cut], rt, rg)
            ok = np.isfinite(gr[:cut]) & np.isfinite(gp)
            if ok.sum() < 40:
                cal = None
                break
            A = np.column_stack([gp[ok], np.ones(ok.sum())])
            a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
            sg = float(np.clip(np.std(gr[:cut][ok] - (a * gp[ok] + b)), 5, 60))
            cal[k] = (a, b, sg)
        if cal is None:
            continue

        for start in range(cut, len(df) - max(WINDOWS), 600):
            for W in WINDOWS:
                idx = np.arange(start, start + W)
                idx = idx[np.isfinite(gr[idx])]
                if len(idx) < W // 2:
                    continue
                rt_med = float(np.median(r_true[idx]))
                base[W].append(abs(rt_med))
                lls = {}
                for k, (rtab, rgab) in refs.items():
                    a, b, sg = cal[k]
                    ll = np.zeros(len(grid))
                    for i in idx:
                        pg = a * np.interp(prior[i] + grid, rtab, rgab) + b
                        ll += -0.5 * ((gr[i] - pg) / sg) ** 2
                    lls[k] = ll / len(idx)
                    res[k][W].append(abs(grid[np.argmax(lls[k])] - rt_med))
                comb = lls["tw"] + lls["self"]
                res["both"][W].append(abs(grid[np.argmax(comb)] - rt_med))

    print(f"pozos={nwells}")
    print("cobertura del rango TVT post-PS:  " + "  ".join(
        f"{k}={np.mean(v):.1%}" for k, v in cover.items()))
    print(f"\n{'W':>5} {'sin GR':>8} {'tw':>8} {'self':>8} {'both':>8}   (|error| mediano del MAP, ft)")
    for W in WINDOWS:
        if not base[W]:
            continue
        row = f"{W:>5} {np.median(base[W]):>8.2f}"
        for k in ("tw", "self", "both"):
            row += f" {np.median(res[k][W]):>8.2f}"
        print(row)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
