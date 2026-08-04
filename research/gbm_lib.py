"""Features del GBM arbitro: una sola definicion, usada por el cache y por el
predictor de cv.py (garantiza que train y test ven exactamente lo mismo).

features(r, m) -> DataFrame de n filas post-PS
  r: dict de arrays por punto (salida de gbm01_cache.one_well)
  m: dict de escalares por pozo (idem)
El target es  dTVT = TVT - last_known_tvt  (=  r['y'] - m['lk'] ).
"""
import numpy as np
import pandas as pd

NBEAM = 7
BM = [f"dbm{c}" for c in range(NBEAM)]
CANDS = ["dsurf", "dsurfa", "dhmm", "dancc", "dpfz", "dgeom", "dflat"] + BM
# candidatos con backtest en el prefijo
BTC = ["surf", "surfa", "hmm", "ancc", "pfz", "geom", "flat", "bm0", "bm1", "bm3"]
MSCALAR = ["cut", "n_pred", "nn_pre", "sstd_pre", "gr_sig", "tw_span", "z_pre_s", "slope"]
D0 = ["d0surf", "d0hmm", "d0geom", "d0bm0"]


def blend(f):
    """El mejor blend actual medido (12.402): 0.25*superficie + 0.75*pf_ancc."""
    return 0.25 * f["dsurf"] + 0.75 * f["dancc"]


def features(r, m):
    f = {}
    for c in CANDS:
        f[c] = np.asarray(r[c], dtype=np.float64)
    bm = np.column_stack([f[c] for c in BM])
    f["dbm_mean"] = bm.mean(1)
    f["dbm_std"] = bm.std(1)
    stack = np.column_stack([f["dsurf"], f["dancc"], f["dpfz"], f["dbm_mean"], f["dgeom"]])
    f["cand_mean"] = stack.mean(1)
    f["cand_std"] = stack.std(1)
    f["cand_med"] = np.median(stack, axis=1)
    f["dis_ancc_surf"] = f["dancc"] - f["dsurf"]
    f["dis_ancc_pfz"] = f["dancc"] - f["dpfz"]
    f["dis_ancc_bm"] = f["dancc"] - f["dbm_mean"]
    f["dis_surf_geom"] = f["dsurf"] - f["dgeom"]
    f["dis_surf_flat"] = f["dsurf"] - f["dflat"]
    f["dis_surf_surfa"] = f["dsurf"] - f["dsurfa"]
    f["dis_ancc_flat"] = f["dancc"] - f["dflat"]
    f["blend"] = blend(f)

    md = np.asarray(r["md_since"], dtype=np.float64)
    f["md_since"] = md
    f["hd_since"] = np.asarray(r["hd_since"], dtype=np.float64)
    f["frac"] = md / max(float(md[-1]), 1.0)
    for k in ("sancc", "spfz", "nn", "sstd", "gr", "gr5", "gr21", "gr51", "gr101",
              "dzdm", "dxydm", "grtw_ancc", "grtw_surf", "grtw_pfz"):
        f[k] = np.asarray(r[k], dtype=np.float64)
    mdc = np.maximum(md, 1.0)
    f["sancc_rate"] = f["sancc"] / mdc
    f["spfz_rate"] = f["spfz"] / mdc
    f["s_ratio"] = f["sancc"] / np.maximum(f["spfz"], 1e-3)
    f["gr_z"] = (f["gr21"] - m["gr_pre_m"]) / max(m["gr_pre_s"], 1e-3)
    f["gr_mis_ancc"] = f["gr21"] - f["grtw_ancc"]
    f["gr_mis_surf"] = f["gr21"] - f["grtw_surf"]
    f["gr_mis_ancc_n"] = f["gr_mis_ancc"] / max(m["gr_sig"], 1e-3)
    f["gr_mis_surf_n"] = f["gr_mis_surf"] / max(m["gr_sig"], 1e-3)
    f["gr_rough"] = f["gr"] - f["gr21"]

    n = len(md)
    for k in MSCALAR + D0:
        f[k] = np.full(n, float(m.get(k, np.nan)))
    f["ps_frac"] = np.full(n, float(m["cut"]) / max(float(m["n"]), 1.0))
    for tag in ("bt50", "bt75"):
        for c in BTC:
            f[f"{tag}_{c}"] = np.full(n, float(m.get(f"{tag}_{c}", np.nan)))
        mdbt = max(float(m.get(tag + "_md", np.nan) or np.nan), 1.0)
        f[f"{tag}_rate_surf"] = np.full(n, float(m.get(f"{tag}_surf", np.nan)) / mdbt)
        f[f"{tag}_rate_ancc"] = np.full(n, float(m.get(f"{tag}_ancc", np.nan)) / mdbt)
        f[f"{tag}_ancc_m_surf"] = f[f"{tag}_ancc"] - f[f"{tag}_surf"]
        f[f"{tag}_ancc_m_flat"] = f[f"{tag}_ancc"] - f[f"{tag}_flat"]
        f[f"{tag}_ancc_m_pfz"] = f[f"{tag}_ancc"] - f[f"{tag}_pfz"]
        f[f"{tag}_best"] = np.full(n, np.nanmin([float(m.get(f"{tag}_{c}", np.nan)) for c in BTC]))
    # extrapolacion: cuanto mas lejos que el backtest estamos prediciendo
    f["bt_reach"] = md / max(float(m.get("bt75_md", 1.0) or 1.0), 1.0)
    return pd.DataFrame(f)


FEATS = None


def feature_names(sample_r, sample_m):
    return list(features(sample_r, sample_m).columns)
