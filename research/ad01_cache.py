"""Cache para el estudio de BLEND ADAPTATIVO superficie <-> particle filter.

Para cada pozo del subconjunto k de cv._select guarda, en formato test exacto
(MD,X,Y,Z,GR,TVT_input) y con LOWO estricto en la superficie:

  post-PS: y (TVT real), S (superficie), H (superficie+HMM), P (pf_ancc),
           Z (pf_z), B (beam cfg0), G (geometrico dip anclado),
           nn (dist al vecino mas cercano de la nube), md_since, hd_since, MDv, Zv
  backtest leak-free: para cada corte f in FRACS se enmascara la COLA del
           prefijo (filas [c2:cut], c2=round(f*cut)) y se re-predice ese tramo
           desde el prefijo acortado con los mismos 7 candidatos. El objetivo
           ahi (TVT_input real) SI se conoce -> es honesto, no hay leak.

Salida: research/ad01_cache_k{K}.npz  (+ ad01_meta_k{K}.csv)
Uso:    python research/ad01_cache.py 60
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from cv import SEED, TEST_COLS, _select, load_well, well_ids  # noqa: E402
import model as M  # noqa: E402
import pf_publico as PF  # noqa: E402

FRACS = (0.5, 0.65, 0.75)
DIP_WIN = 700
NAMES = ("S", "H", "P", "Z", "B", "G")


def hdist(df):
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=DIP_WIN):
    """Recta de dip anclada en el PS (research/afinar_dip.py: anchor700)."""
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
    return flat + slope * h


def candidates(df_h, tw, wid, cut, field, hd):
    """Los 6 candidatos sobre las filas POST-cut + nn de esas filas."""
    n = len(df_h)
    s_full, nn = M._prior(df_h, field, wid, cut)
    S = s_full[cut:]
    H = S + M.hmm_refine(df_h, tw, s_full, cut)
    t, g = PF._tw(tw)
    P, _ = PF.run_pf_ancc(df_h, t, g)
    Zp, _ = PF.run_pf_z(df_h, t, g)
    bs, mc, es, r, _ = PF.BEAMS[0]
    B = PF.beam_search(df_h.GR.values[cut:], t, g, float(df_h.TVT_input.values[cut - 1]),
                       bs, mc, es, r)
    G = geom_pred(df_h, cut, hd)[cut:]
    out = dict(S=S, H=H, P=np.asarray(P, float), Z=np.asarray(Zp, float),
               B=np.asarray(B, float), G=G)
    for k, v in out.items():
        assert len(v) == n - cut, (k, len(v), n - cut)
    return out, nn[cut:]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ids = _select(well_ids(), k, SEED)
    field = M.SurfaceField()
    print(f"campo: {len(field.s)} puntos / {len(field.names)} pozos; {len(ids)} pozos a evaluar",
          flush=True)

    store = {n: [] for n in NAMES}
    store.update({n: [] for n in ("y", "nn", "md_since", "hd_since", "MDv", "Zv")})
    bt = {f: {n: [] for n in NAMES} for f in FRACS}
    for f in FRACS:
        bt[f]["y"] = []
        bt[f]["md_since"] = []
    lens, btlens, meta = [], {f: [] for f in FRACS}, []
    t0 = time.time()
    for j, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[TEST_COLS].copy()
        hd = hdist(df_h)
        cand, nn = candidates(df_h, tw, wid, cut, field, hd)
        for n in NAMES:
            store[n].append(cand[n].astype(np.float32))
        store["y"].append(df.TVT.values[cut:].astype(np.float32))
        store["nn"].append(nn.astype(np.float32))
        store["md_since"].append((df_h.MD.values[cut:] - df_h.MD.values[cut - 1]).astype(np.float32))
        store["hd_since"].append((hd[cut:] - hd[cut - 1]).astype(np.float32))
        store["MDv"].append(df_h.MD.values[cut:].astype(np.float32))
        store["Zv"].append(df_h.Z.values[cut:].astype(np.float32))
        lens.append(len(cand["S"]))

        m = {"well": wid, "cut": cut, "n": len(df), "n_pred": len(df) - cut,
             "nn_med": float(np.median(nn))}
        for f in FRACS:
            c2 = int(round(f * cut))
            if c2 < 60 or cut - c2 < 30:
                btlens[f].append(0)
                for n in list(NAMES) + ["y", "md_since"]:
                    bt[f][n].append(np.zeros(0, np.float32))
                continue
            d2 = df_h.iloc[:cut].copy()
            d2.loc[d2.index[c2:], "TVT_input"] = np.nan
            c2d, _ = candidates(d2, tw, wid, c2, field, hd[:cut])
            yb = df_h.TVT_input.values[c2:cut]
            for n in NAMES:
                bt[f][n].append(c2d[n].astype(np.float32))
                m[f"bt{n}{f}"] = float(np.sqrt(np.mean((yb - c2d[n]) ** 2)))
            bt[f]["y"].append(yb.astype(np.float32))
            bt[f]["md_since"].append((df_h.MD.values[c2:cut] - df_h.MD.values[c2 - 1]).astype(np.float32))
            btlens[f].append(cut - c2)
        meta.append(m)
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    out = {n: np.concatenate(v) for n, v in store.items()}
    out["lens"] = np.array(lens)
    for f in FRACS:
        tag = str(f).replace(".", "")
        out[f"btlens_{tag}"] = np.array(btlens[f])
        for n in list(NAMES) + ["y", "md_since"]:
            out[f"bt{n}_{tag}"] = np.concatenate(bt[f][n]) if bt[f][n] else np.zeros(0, np.float32)
    outf = ROOT / f"research/ad01_cache_k{k}.npz"
    np.savez_compressed(outf, **out)
    pd.DataFrame(meta).to_csv(ROOT / f"research/ad01_meta_k{k}.csv", index=False)
    y = out["y"].astype(np.float64)
    print(f"\nguardado {outf}  pozos={len(lens)} puntos={sum(lens)}  {time.time()-t0:.0f}s")
    for n in NAMES:
        print(f"  RMSE {n}: {np.sqrt(np.mean((y - out[n].astype(np.float64))**2)):7.3f}")
    mix = 0.25 * out["S"].astype(np.float64) + 0.75 * out["P"].astype(np.float64)
    print(f"  RMSE 0.25S+0.75P: {np.sqrt(np.mean((y-mix)**2)):7.3f}   (referencia 12.402)")


if __name__ == "__main__":
    main()
