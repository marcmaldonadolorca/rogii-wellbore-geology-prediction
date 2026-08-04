"""¿Discrimina la emision, o es la busqueda la que falla?

Compara la log-verosimilitud media por punto en r_verdadero vs r=0 vs r aleatorio,
y mide el error del MAP restringiendo el span (para separar 'sin senal' de 'alias
lejanos'). Tambien mide la sensibilidad dGR_tw/dTVT en la zona de trabajo.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv import load_well, well_ids                                    # noqa: E402
from model import SurfaceField, _prior                                # noqa: E402

W = 200
SPANS = [5.0, 15.0, 30.0, 60.0]


def main(nwells=20):
    field = SurfaceField()
    rng = np.random.default_rng(0)
    dll_true, dll_rand, sens, gr_std = [], [], [], []
    maperr = {s: [] for s in SPANS}
    base = []

    for wid in well_ids()[:nwells]:
        df, twdf, cut = load_well(wid)
        dfh = df[["MD", "X", "Y", "Z", "GR", "TVT_input"]]
        prior, _ = _prior(dfh, field, wid, cut)
        r_true = df.TVT.values - prior
        gr = df.GR.values

        tw = twdf.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        tvt_tw, gr_tw = tw.TVT.values, tw.GR.values

        gp = np.interp(dfh.TVT_input.values[:cut], tvt_tw, gr_tw)
        ok = np.isfinite(gr[:cut]) & np.isfinite(gp)
        if ok.sum() < 40:
            continue
        A = np.column_stack([gp[ok], np.ones(ok.sum())])
        a, b = np.linalg.lstsq(A, gr[:cut][ok], rcond=None)[0]
        sg = float(np.clip(np.std(gr[:cut][ok] - (a * gp[ok] + b)), 5, 60))

        def ll_at(idx, r):
            pg = a * np.interp(prior[idx] + r, tvt_tw, gr_tw) + b
            return float(np.mean(-0.5 * ((gr[idx] - pg) / sg) ** 2))

        for start in range(cut, len(df) - W, 700):
            idx = np.arange(start, start + W)
            idx = idx[np.isfinite(gr[idx])]
            if len(idx) < W // 2:
                continue
            rt = float(np.median(r_true[idx]))
            base.append(abs(rt))
            dll_true.append(ll_at(idx, rt) - ll_at(idx, 0.0))
            dll_rand.append(ll_at(idx, float(rng.uniform(-60, 60))) - ll_at(idx, 0.0))
            # sensibilidad local del mapa: cuanto cambia GR_tw por ft de TVT
            t0 = prior[idx].mean()
            sens.append(abs(np.interp(t0 + 5, tvt_tw, gr_tw) - np.interp(t0 - 5, tvt_tw, gr_tw)) / 10)
            gr_std.append(np.std(gr[idx]))
            for S in SPANS:
                grid = np.arange(-S, S + 0.5, 0.5)
                ll = np.array([ll_at(idx, r) for r in grid])
                maperr[S].append(abs(grid[np.argmax(ll)] - rt))

    print(f"pozos={nwells}  ventanas={len(base)}  |r_true| mediano = {np.median(base):.2f} ft")
    print(f"\nlog-verosimilitud media por punto, relativa a r=0:")
    print(f"  en r_VERDADERO : {np.median(dll_true):+8.3f}   (>0 = la emision prefiere el correcto)")
    print(f"  en r aleatorio : {np.median(dll_rand):+8.3f}")
    print(f"  fraccion de ventanas con ll(r_true) > ll(0): {np.mean(np.array(dll_true) > 0):.1%}")
    print(f"\nsensibilidad del mapa |dGR_tw/dTVT| = {np.median(sens):.2f} API/ft"
          f"   vs std(GR) en ventana = {np.median(gr_std):.2f} API"
          f"   -> ft equivalentes a 1 sigma: {np.median(gr_std)/max(np.median(sens),1e-9):.1f}")
    print(f"\n{'span':>6} {'|err| MAP':>10}   (sin GR = {np.median(base):.2f})")
    for S in SPANS:
        print(f"{S:>6} {np.median(maperr[S]):>10.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
