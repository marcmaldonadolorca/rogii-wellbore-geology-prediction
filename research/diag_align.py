"""¿Esta el typewell alineado en TVT con el pozo horizontal?

En el PREFIJO conocemos TVT real, asi que se puede medir directamente: se busca
el desplazamiento D que maximiza la correlacion entre GR del horizontal y
GR_typewell(TVT_real + D). Si el maximo no cae en D=0, la emision del HMM estaba
evaluando el mapa en el sitio equivocado.

Informa ademas la CALIDAD del matching en el prefijo (correlacion en el optimo):
si es baja, el typewell es mal mapa para ese pozo y el GR no puede ayudar.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv import load_well, well_ids                                    # noqa: E402

SHIFTS = np.arange(-60, 60.5, 0.5)


def main(nwells=60):
    best_d, corr0, corrb, gain = [], [], [], []
    for wid in well_ids()[:nwells]:
        df, twdf, cut = load_well(wid)
        tw = twdf.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        if len(tw) < 30 or cut < 100:
            continue
        t_tw, g_tw = tw.TVT.values, tw.GR.values
        tvt = df.TVT_input.values[:cut]
        gr = df.GR.values[:cut]
        m = np.isfinite(tvt) & np.isfinite(gr)
        if m.sum() < 100:
            continue
        tvt, gr = tvt[m], gr[m]
        g = gr - gr.mean()
        cs = []
        for d in SHIFTS:
            p = np.interp(tvt + d, t_tw, g_tw)
            p = p - p.mean()
            den = np.sqrt((g**2).sum() * (p**2).sum())
            cs.append((g * p).sum() / den if den > 0 else 0.0)
        cs = np.array(cs)
        k = int(np.argmax(cs))
        best_d.append(SHIFTS[k])
        corrb.append(cs[k])
        corr0.append(cs[np.argmin(np.abs(SHIFTS))])
        gain.append(cs[k] - cs[np.argmin(np.abs(SHIFTS))])

    best_d, corr0, corrb = np.array(best_d), np.array(corr0), np.array(corrb)
    print(f"pozos analizados: {len(best_d)}")
    print(f"\ndesplazamiento optimo D del typewell (ft):")
    print(f"  |D|<=1 ft en {np.mean(np.abs(best_d)<=1):.1%} de pozos"
          f"   |D|<=5 ft en {np.mean(np.abs(best_d)<=5):.1%}")
    print(f"  mediana {np.median(best_d):+.2f} | p10 {np.percentile(best_d,10):+.2f}"
          f" | p90 {np.percentile(best_d,90):+.2f} | mediana|D| {np.median(np.abs(best_d)):.2f}")
    print(f"\ncalidad del matching en el prefijo (correlacion):")
    print(f"  en D=0    : mediana {np.median(corr0):+.3f}  (p10 {np.percentile(corr0,10):+.3f})")
    print(f"  en D optimo: mediana {np.median(corrb):+.3f}")
    print(f"  ganancia por alinear: {np.median(np.array(gain)):+.3f}")
    print(f"\npozos con correlacion<0.3 en D=0: {np.mean(corr0<0.3):.1%}"
          f"  | <0 : {np.mean(corr0<0):.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
