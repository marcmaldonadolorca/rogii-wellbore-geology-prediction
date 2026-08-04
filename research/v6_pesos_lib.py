"""v6-pesos: utilidades comunes de la linea PESOS-POR-POZO del blend.

Señales (orden fijo de pesos): (surf, pf, geom)
  surf : SurfaceField(aniso=16, k=24, theta=THETA_OPT) + offset cola 500 (LOWO)
  pf   : PF ANCC multiseed media (S=64 desde v4_multiseed_k{60,150}.npz para los
         pozos de evaluacion; S=8 fresco para los pozos de entrenamiento)
  geom : recta de dip anclada en PS, dip_win=700 (la de orc01)

Backtest enmascarado del prefijo: se corta el prefijo en cut2 = 65% de cut y se
re-predicen las filas [cut2, cut) con cada señal usando SOLO [0, cut2); ahi se
ajusta NNLS. Todo es informacion pre-PS => el kernel lo puede hacer en runtime.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from cv import TEST_COLS, load_well, well_ids, _select, SEED  # noqa: E402
import model as M                                             # noqa: E402
from ban_lib import run_pf_ancc_multi                         # noqa: E402
from pf_publico import _tw                                    # noqa: E402

THETA_OPT = 2.278489      # theta optimo medido (PCA + 11 deg), v4_supmax
ANISO, K_IDW = 16.0, 24
W_GLOBAL = np.array([0.45, 0.55, 0.0])   # (surf, pf, geom) — blend de referencia
BT_FRAC = 0.65            # cut2 = 65% del prefijo -> backtest en el ultimo 35%
BT_MIN_N = 30             # minimo de puntos de backtest para fiarse del NNLS
S_BT = 8                  # semillas del PF de backtest
S_TRAIN = 8               # semillas del PF post-PS en pozos de entrenamiento
N_PART = 600

CACHE_NPZ = HERE / "v6_pesos_cache.npz"
META_CSV = HERE / "v6_pesos_meta.csv"


def subsets():
    ids = well_ids()
    k60 = set(_select(ids, 60, SEED))
    k150 = set(_select(ids, 150, SEED))
    return ids, k60, k150


def train_pool(n=400):
    """~n pozos DISJUNTOS de k60|k150, sistematicos sobre el ranking de n_pred."""
    ids, k60, k150 = subsets()
    st = pd.read_csv(HERE / "ps_stats.csv").set_index("well").reindex(ids)
    resto = [w for w in st.n_pred.sort_values().index if w not in k60 | k150]
    pick = np.round(np.linspace(0, len(resto) - 1, min(n, len(resto)))).astype(int)
    return sorted(resto[i] for i in sorted(set(pick)))


def hdist(df):
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    return np.cumsum(step)


def geom_pred(df_h, cut, hd, win=700):
    """Recta de dip anclada en el punto `cut`-1 (identica a orc01)."""
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


def make_field():
    t0 = time.time()
    f = M.SurfaceField(aniso=ANISO, theta=THETA_OPT)
    print(f"SurfaceField aniso={ANISO} theta={THETA_OPT}: "
          f"{len(f.s)} pts en {time.time()-t0:.0f}s", flush=True)
    return f


def surf_with_offset(s_all, df_h, lo_cut, tail=M.CAL_TAIL):
    """TVT prior = s - Z + C, con C calibrada en la cola [lo_cut-tail, lo_cut)."""
    lo = max(0, lo_cut - tail)
    z = df_h.Z.values
    c = np.median(df_h.TVT_input.values[lo:lo_cut] + z[lo:lo_cut] - s_all[lo:lo_cut])
    return s_all - z + c


def nnls_w(A, y):
    """NNLS + normalizacion a suma 1. Devuelve (w(3,), rmse_del_ajuste)."""
    from scipy.optimize import nnls
    w, _ = nnls(A, y)
    s = w.sum()
    if s <= 1e-9:
        return None, np.inf
    w = w / s
    r = float(np.sqrt(np.mean((A @ w - y) ** 2)))
    return w, r


PF_NPZ = None


def pf64_lookup(wid):
    """(pred_mean64, sstd, ll) desde los npz v4 si el pozo esta; si no None."""
    global PF_NPZ
    if PF_NPZ is None:
        PF_NPZ = [np.load(HERE / "v4_multiseed_k150.npz"),
                  np.load(HERE / "v4_multiseed_k60.npz")]
    for z in PF_NPZ:
        if f"{wid}_pred" in z.files:
            return (z[f"{wid}_pred"].astype(np.float64),
                    z[f"{wid}_sstd"].astype(np.float64),
                    z[f"{wid}_ll"].astype(np.float64))
    return None


def process_well(wid, field, pf_source):
    """Computa señales post-PS + backtest de un pozo.

    pf_source: "npz64" (evaluacion) o "run8" (entrenamiento).
    Devuelve (dict_arrays, meta_row) o None si el pozo no es evaluable.
    """
    df, tw, cut = load_well(wid)
    if cut < 20 or cut >= len(df):
        return None
    df_h = df[TEST_COLS].copy()
    n = len(df)
    hd = hdist(df_h)
    y = df.TVT.values[cut:]

    # superficie: UNA consulta para todo el pozo, sirve al post y al backtest
    s_all, nn_all = field.interp(df_h.X.values, df_h.Y.values, exclude=wid, k=K_IDW)
    surf_post = surf_with_offset(s_all, df_h, cut)[cut:]
    geom_post = geom_pred(df_h, cut, hd)[cut:]
    t_tw, g_tw = _tw(tw)

    if pf_source == "npz64":
        hit = pf64_lookup(wid)
        assert hit is not None, f"{wid}: sin pred S64 en npz"
        pf_post, sstd, ll = hit
    else:
        pts, ll = run_pf_ancc_multi(df_h, t_tw, g_tw,
                                    seeds=np.arange(1, S_TRAIN + 1, dtype=np.int64),
                                    N=N_PART)
        pts = pts.astype(np.float64)
        pf_post, sstd = pts.mean(0), pts.std(0)
    assert len(pf_post) == n - cut

    md_since = df_h.MD.values[cut:] - df_h.MD.values[cut - 1]
    nn_post = nn_all[cut:]

    # ---------------- backtest enmascarado del prefijo ----------------
    cut2 = int(round(BT_FRAC * cut))
    bt = {}
    w_bt, r_bt = None, np.inf
    if cut2 >= 20 and cut - cut2 >= BT_MIN_N:
        y_bt = df_h.TVT_input.values[cut2:cut].astype(np.float64)
        surf_bt = surf_with_offset(s_all, df_h, cut2)[cut2:cut]
        geom_bt = geom_pred(df_h, cut2, hd)[cut2:cut]
        df_bt = df_h.iloc[:cut].copy()
        df_bt.loc[df_bt.index[cut2:], "TVT_input"] = np.nan
        pts_bt, ll_bt = run_pf_ancc_multi(df_bt, t_tw, g_tw,
                                          seeds=np.arange(1, S_BT + 1, dtype=np.int64),
                                          N=N_PART)
        pf_bt = pts_bt.astype(np.float64).mean(0)
        assert len(pf_bt) == cut - cut2
        A = np.column_stack([surf_bt, pf_bt, geom_bt])
        w_bt, r_bt = nnls_w(A, y_bt)
        bt = {"y_bt": y_bt, "surf_bt": surf_bt, "pf_bt": pf_bt, "geom_bt": geom_bt,
              "md_bt": df_h.MD.values[cut2:cut] - df_h.MD.values[cut2 - 1],
              "nn_bt": nn_all[cut2:cut]}

    # ---------------- oraculo NNLS post (target del meta-modelo) ------
    A_post = np.column_stack([surf_post, pf_post, geom_post])
    w_or, r_or = nnls_w(A_post, y)

    def _r(p):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    meta = {
        "well": wid, "cut": cut, "n_pred": n - cut, "ps_frac": cut / n,
        "pf_source": pf_source,
        "md_len": float(md_since[-1]),
        "nn_med": float(np.median(nn_post)), "nn_p90": float(np.quantile(nn_post, .9)),
        "sstd_mean": float(np.mean(sstd)), "sstd_med": float(np.median(sstd)),
        "ll_sd": float(np.std(ll)),
        "dis_sp": float(np.mean(np.abs(surf_post - pf_post))),
        "dis_sg": float(np.mean(np.abs(surf_post - geom_post))),
        "dis_pg": float(np.mean(np.abs(pf_post - geom_post))),
        "rmse_surf": _r(surf_post), "rmse_pf": _r(pf_post), "rmse_geom": _r(geom_post),
        "n_bt": len(bt.get("y_bt", ())),
        "w_bt_s": np.nan if w_bt is None else w_bt[0],
        "w_bt_p": np.nan if w_bt is None else w_bt[1],
        "w_bt_g": np.nan if w_bt is None else w_bt[2],
        "rmse_nnls_bt": r_bt,
        "w_or_s": np.nan if w_or is None else w_or[0],
        "w_or_p": np.nan if w_or is None else w_or[1],
        "w_or_g": np.nan if w_or is None else w_or[2],
        "rmse_nnls_or": r_or,
    }
    if bt:
        for k in ("surf", "pf", "geom"):
            meta[f"rmse_{k}_bt"] = float(np.sqrt(np.mean((bt["y_bt"] - bt[f"{k}_bt"]) ** 2)))
    else:
        meta.update(rmse_surf_bt=np.nan, rmse_pf_bt=np.nan, rmse_geom_bt=np.nan)

    arrays = {"y": y, "surf": surf_post, "pf": pf_post, "geom": geom_post,
              "nn": nn_post, "md": md_since, "sstd": sstd}
    arrays.update(bt)
    return arrays, meta


class PesosCache:
    """Vista comoda del cache v6_pesos: arrays por pozo + meta con features."""

    def __init__(self, npz=CACHE_NPZ, csv=META_CSV):
        self.z = np.load(npz)
        self.meta = pd.read_csv(csv)
        self.meta_ix = self.meta.set_index("well")

    def arr(self, wid, name):
        return self.z[f"{wid}_{name}"].astype(np.float64)

    def wells(self, pf_source=None):
        m = self.meta
        if pf_source:
            m = m[m.pf_source == pf_source]
        return m.well.tolist()


def pooled_rmse(sse_n):
    """sse_n: iterable de (sse, n) por pozo -> RMSE pooled."""
    sse = sum(s for s, _ in sse_n)
    n = sum(m for _, m in sse_n)
    return float(np.sqrt(sse / n))
