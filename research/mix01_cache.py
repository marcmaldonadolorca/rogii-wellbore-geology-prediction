"""Cachea, para los 150 pozos de cv.evaluate(k=150), las predicciones post-PS de
los tres predictores (A geometrico, B superficie, C superficie+HMM), la nn_dist
por punto y el backtest leak-free en la cola del prefijo.

Todo se calcula con el df en formato test exacto (MD,X,Y,Z,GR,TVT_input) y con
LOWO estricto en la superficie. El pooled RMSE reconstruido desde este cache es
identico al de cv.evaluate (se comprueba en mix02).

Salida: research/mix01_cache.npz  (arrays concatenados + offsets por pozo)
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv import TEST_COLS, load_well, well_ids, _select, SEED  # noqa: E402
import model as M  # noqa: E402

BACKTEST_FRACS = (0.5, 0.65, 0.75)
DIP_WIN = 700
OUT = ROOT / "research/mix01_cache.npz"


def hdist(df):
    """Distancia horizontal acumulada (ft) a lo largo de la trayectoria."""
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=DIP_WIN):
    """A) recta de dip anclada en PS (research/afinar_dip.py: anchor700)."""
    z = df_h.Z.values
    tvt_in = df_h.TVT_input.values
    h = hd - hd[cut - 1]
    flat = tvt_in[cut - 1] + (z[cut - 1] - z)
    lo = max(0, cut - win)
    base = tvt_in[lo] + (z[lo] - z[lo:cut])
    r = tvt_in[lo:cut] - base
    hh = h[lo:cut]
    slope = 0.0
    if len(hh) > 10 and hh[-1] != hh[0]:
        dr, dh = r - r[-1], hh - hh[-1]
        ss = float(np.sum(dh * dh))
        slope = float(np.sum(dr * dh) / ss) if ss > 0 else 0.0
    return flat + slope * h, slope


def three(df_h, tw, wid, cut, field, hd):
    """(A, B, C, nn) sobre TODAS las filas; el caller recorta a [cut:]."""
    a, slope = geom_pred(df_h, cut, hd)
    b, nn = M._prior(df_h, field, wid, cut)
    r = M.hmm_refine(df_h, tw, b, cut, sigma_r=0.01, dec=10)
    c = b.copy()
    c[cut:] = b[cut:] + r
    return a, b, c, nn, slope, float(r[0])


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    mode = sys.argv[2] if len(sys.argv) > 2 else "eval"
    ev = set(_select(well_ids(), 150, SEED))
    if mode == "eval":
        ids = sorted(ev)
        tag = "eval"
    else:                       # pozos DISJUNTOS del set de evaluacion: para ajustar
        rest = [w for w in well_ids() if w not in ev]
        rng = np.random.default_rng(7)
        ids = sorted(rng.choice(rest, size=min(k, len(rest)), replace=False))
        tag = "fit"
    global OUT
    OUT = ROOT / f"research/mix01_cache_{tag}.npz"
    field = M.SurfaceField()
    print(f"campo listo: {len(field.s)} puntos, {len(field.names)} pozos", flush=True)

    store = {n: [] for n in ("y", "A", "B", "C", "nn", "md_since", "hd_since")}
    meta = []
    t0 = time.time()
    for j, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[TEST_COLS].copy()
        hd = hdist(df_h)
        a, b, c, nn = three(df_h, tw, wid, cut, field, hd)
        store["y"].append(df.TVT.values[cut:])
        store["A"].append(a[cut:]); store["B"].append(b[cut:]); store["C"].append(c[cut:])
        store["nn"].append(nn[cut:])
        store["md_since"].append(df_h.MD.values[cut:] - df_h.MD.values[cut - 1])
        store["hd_since"].append(hd[cut:] - hd[cut - 1])

        # --- backtest leak-free: enmascarar la cola del prefijo ---
        bt = {}
        for f in BACKTEST_FRACS:
            c2 = int(round(f * cut))
            if c2 < 60 or cut - c2 < 30:
                bt[f] = (np.nan, np.nan, np.nan, 0)
                continue
            d2 = df_h.copy()
            d2.loc[d2.index[c2:], "TVT_input"] = np.nan
            a2, b2, c2p, _ = three(d2, tw, wid, c2, field, hd)
            yb = df_h.TVT_input.values[c2:cut]          # conocido: no hay leak
            sl = slice(c2, cut)
            bt[f] = (float(np.sqrt(np.mean((yb - a2[sl]) ** 2))),
                     float(np.sqrt(np.mean((yb - b2[sl]) ** 2))),
                     float(np.sqrt(np.mean((yb - c2p[sl]) ** 2))),
                     cut - c2)
        tv0 = float(df_h.TVT_input.values[cut - 1])
        m = {"well": wid, "cut": cut, "n": len(df), "n_pred": len(df) - cut,
             "nn_med": float(np.median(nn[cut:])), "nn_pre": float(np.median(nn[:cut])),
             # salto en el PS: lo que hay que sumar al predictor para que sea continuo
             "d0A": tv0 - float(a[cut - 1]), "d0B": tv0 - float(b[cut - 1]),
             "d0C": tv0 - float(c[cut - 1]),
             "slope": float((a[cut] - a[cut - 1] - (df_h.Z.values[cut - 1] - df_h.Z.values[cut]))
                            / max(hd[cut] - hd[cut - 1], 1e-6))}
        for f in BACKTEST_FRACS:
            m[f"btA{f}"], m[f"btB{f}"], m[f"btC{f}"], m[f"btn{f}"] = bt[f]
        meta.append(m)
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    lens = np.array([len(v) for v in store["y"]])
    out = {n: np.concatenate(v).astype(np.float32) for n, v in store.items()}
    out["lens"] = lens
    import pandas as pd
    md = pd.DataFrame(meta)
    md.to_csv(str(OUT).replace(".npz", "_meta.csv"), index=False)
    np.savez_compressed(OUT, **out)
    print(f"guardado {OUT}  pozos={len(lens)} puntos={lens.sum()}  {time.time()-t0:.0f}s")
    for n in ("A", "B", "C"):
        print(f"  RMSE {n}: {np.sqrt(np.mean((out['y']-out[n])**2)):.3f}")


if __name__ == "__main__":
    main()
