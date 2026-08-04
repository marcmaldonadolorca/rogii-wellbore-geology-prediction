"""Cache maestro para los oraculos: predicciones post-PS de los 5 predictores
sobre TODOS los pozos evaluables (770), con LOWO estricto en la superficie y el
df en formato test exacto (MD,X,Y,Z,GR,TVT_input).

Predictores cacheados por punto post-PS:
  geom  recta de dip anclada en PS (baseline geometrico, dip_win=700)
  surf  superficie IDW + offset calibrado en la cola del prefijo (model._prior)
  ancc  particle filter ANCC de roman_all (pf_publico.run_pf_ancc)
  pfz   particle filter Z-acoplado (pf_publico.run_pf_z)
  beam  beam search +-2 sobre el typewell (BEAMS[0])
mas covariables: y (TVT real), nn (distancia al vecino de la nube), md_since,
hd_since, y las desviaciones de los PF (ancc_sd, pfz_sd).

Salida: research/orc01_cache.npz + research/orc01_meta.csv
Coste medido: ~0.6 s/pozo => ~8 min los 770.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from cv import TEST_COLS, load_well, well_ids  # noqa: E402
import model as M  # noqa: E402
import pf_publico as P  # noqa: E402

OUT = ROOT / "research/orc01_cache.npz"
DIP_WIN = 700


@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)


def hdist(df):
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=DIP_WIN):
    """Recta de dip anclada en PS (research/afinar_dip.py: anchor700)."""
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


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tag = "" if seed == 0 else f"_s{seed}"
    out = Path(str(OUT).replace(".npz", f"{tag}.npz"))
    _seed_numba(seed)
    np.random.seed(seed)

    ids = well_ids()
    field = M.SurfaceField()
    print(f"campo listo: {len(field.s)} puntos, {len(field.names)} pozos", flush=True)

    cols = ("y", "geom", "surf", "ancc", "pfz", "beam",
            "nn", "md_since", "hd_since", "ancc_sd", "pfz_sd")
    store = {c: [] for c in cols}
    meta = []
    t0 = time.time()
    for j, wid in enumerate(ids):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[TEST_COLS].copy()
        hd = hdist(df_h)
        n = len(df)

        g = geom_pred(df_h, cut, hd)[cut:]
        s, nn = M._prior(df_h, field, wid, cut)
        t, gr = P._tw(tw)
        a, a_sd = P.run_pf_ancc(df_h, t, gr)
        z, z_sd = P.run_pf_z(df_h, t, gr)
        b = P.predict_beam(df_h, tw)

        store["y"].append(df.TVT.values[cut:])
        store["geom"].append(g)
        store["surf"].append(s[cut:])
        store["ancc"].append(np.asarray(a, float))
        store["pfz"].append(np.asarray(z, float))
        store["beam"].append(np.asarray(b, float))
        store["nn"].append(nn[cut:])
        store["md_since"].append(df_h.MD.values[cut:] - df_h.MD.values[cut - 1])
        store["hd_since"].append(hd[cut:] - hd[cut - 1])
        store["ancc_sd"].append(np.asarray(a_sd, float))
        store["pfz_sd"].append(np.asarray(z_sd, float))

        meta.append({"well": wid, "cut": cut, "n": n, "n_pred": n - cut,
                     "ps_frac": cut / n,
                     "nn_med": float(np.median(nn[cut:])),
                     "md_len": float(df_h.MD.values[-1] - df_h.MD.values[cut - 1]),
                     "hd_len": float(hd[-1] - hd[cut - 1])})
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    lens = np.array([len(v) for v in store["y"]])
    outd = {c: np.concatenate(v).astype(np.float32) for c, v in store.items()}
    outd["lens"] = lens
    np.savez_compressed(out, **outd)
    pd.DataFrame(meta).to_csv(str(out).replace(".npz", "_meta.csv"), index=False)
    print(f"guardado {out}  pozos={len(lens)} puntos={lens.sum()}  {time.time()-t0:.0f}s")
    y = outd["y"]
    for c in ("geom", "surf", "ancc", "pfz", "beam"):
        print(f"  RMSE {c:5s}: {np.sqrt(np.mean((y - outd[c])**2)):7.3f}")
    mix = 0.25 * outd["surf"] + 0.75 * outd["ancc"]
    print(f"  RMSE mix  : {np.sqrt(np.mean((y - mix)**2)):7.3f}")


if __name__ == "__main__":
    main()
