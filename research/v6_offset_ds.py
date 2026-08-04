"""v6_offset: dataset por POZO para aprender el OFFSET del blend16.

Blend de referencia (el enviado): 0.45*superficie(theta=2.278489, aniso=16,
k=24 IDW) + 0.55*PF multiseed media. Para los 196 pozos de los subconjuntos
k60/k150 el PF es la media de 64 semillas (npz de v4, identico al enviado);
para los pozos de ENTRENAMIENTO del estimador (disjuntos) el PF es media de
S=16 semillas (coste), y queda registrado en la columna pf_src.

target: off_true = media(TVT - blend) en el tramo post-PS.
features: LEAK-FREE, todas computables en runtime para un pozo oculto:
  - backtest del prefijo: se enmascara el ultimo 35% del prefijo, se re-predice
    con el MISMO blend (superficie calibrada solo con el 65% inicial + PF desde
    ahi) y se mide el residuo contra TVT_input conocido -> off_bt*
  - spread entre semillas del PF post-PS (sstd), loglik stats (normalizados)
  - nn_dist (espacio anisotropo y espacio crudo, LOWO)
  - desacuerdo superficie-PF post-PS (media, |media|, pendiente, std)
  - dip pre-PS, longitudes (n_pred, ps_frac, md_post), residuo GR del prefijo

Uso:
  nice -n 10 .venv/bin/python research/v6_offset_ds.py eval    # 196 pozos k60|k150
  nice -n 10 .venv/bin/python research/v6_offset_ds.py train   # 400 disjuntos
Salida incremental research/v6_offset_ds.csv (restartable: se saltan hechos).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod                                   # noqa: E402
from ban_lib import run_pf_ancc_multi                # noqa: E402  (solo lectura)
from cv import load_well, well_ids, _select          # noqa: E402
from model import SurfaceField                       # noqa: E402
from pf_publico import _tw                           # noqa: E402

THETA_OPT = 2.278489     # PCA + 11 grados (v4_supmax, confirmado k150)
ANISO, W_SURF = 16.0, 0.45
CAL_TAIL = 500           # como model._prior
BT_FRAC = 0.35           # fraccion final del prefijo que se enmascara
S_FEAT = 16              # semillas para FEATURES (y para el PF de train)
N_PART = 600
OUT = R / "v6_offset_ds.csv"
CACHE = R / "v6_offset_cache"; CACHE.mkdir(exist_ok=True)
MS_CACHE = R / "v4_multiseed_cache"

COLS = ["well", "split", "pf_src", "n_pred", "cut_len", "ps_frac", "md_post",
        "off_bt", "off_bt_tail", "off_bt_end", "bt_slope", "bt_rmse", "bt_len", "has_bt",
        "sstd_mean", "sstd_p90", "ll_mean_n", "ll_std_n", "ll_rng_n",
        "nn_mean", "nn_p90", "nnr_mean", "nnr_p90",
        "dis_mean", "dis_abs", "dis_slope", "dis_std",
        "dip30", "dip200", "gr_res_std", "gr_corr",
        "off_true", "off_med", "rmse_blend"]


def pf16(wid, df_h, tw):
    """(pts (16,n) float32, ll (16,)) — cache S64 recortado, cache propio, o fresco."""
    f64 = MS_CACHE / f"pf_{wid}_S64_N{N_PART}.npz"
    if f64.exists():
        z = np.load(f64)
        return z["pts"][:S_FEAT].astype(np.float32), z["ll"][:S_FEAT].astype(float)
    f16 = CACHE / f"pf_{wid}_S16_N{N_PART}.npz"
    if f16.exists():
        z = np.load(f16)
        return z["pts"].astype(np.float32), z["ll"].astype(float)
    t, g = _tw(tw)
    pts, ll = run_pf_ancc_multi(df_h, t, g, seeds=np.arange(1, S_FEAT + 1), N=N_PART)
    np.savez_compressed(f16, pts=pts.astype(np.float32), ll=ll)
    return pts.astype(np.float32), ll


def slope_per_kft(x_md, y):
    """Pendiente robusta (ft por 1000 ft de MD) via polyfit."""
    if len(y) < 5 or np.ptp(x_md) < 1e-6:
        return 0.0
    return float(np.polyfit((x_md - x_md[0]) / 1000.0, y, 1)[0])


def dip_tail(tvti, z, md, cut, win):
    """Mediana de d(TVT+Z)/dMD en las ultimas `win` filas pre-PS."""
    lo = max(0, cut - win)
    s = tvti[lo:cut] + z[lo:cut]
    dm = np.diff(md[lo:cut]); ds = np.diff(s); m = dm > 0
    return float(np.median(ds[m] / dm[m])) if m.sum() >= 3 else 0.0


def build_row(wid, split, field, raw_tree_of, pf64_pred):
    df, tw, cut = load_well(wid)
    if cut < 20 or cut >= len(df):
        return None
    df_h = df[cvmod.TEST_COLS].copy()
    z, tvti, md = df.Z.values, df.TVT_input.values, df.MD.values
    n_pred = len(df) - cut

    # ---- superficie (LOWO) sobre TODO el pozo, una sola query
    s_all, nn = field.interp(df.X.values, df.Y.values, exclude=wid)
    lo = max(0, cut - CAL_TAIL)
    c = np.median(tvti[lo:cut] + z[lo:cut] - s_all[lo:cut])
    prior = s_all - z + c

    # ---- PF features (S=16) + componente PF del blend
    pts, ll = pf16(wid, df_h, tw)
    if wid in pf64_pred:
        pf, pf_src = pf64_pred[wid].astype(float), 64
    else:
        pf, pf_src = pts.mean(0).astype(float), 16
    assert len(pf) == n_pred, f"{wid}: pf {len(pf)} != n_pred {n_pred}"

    blend = W_SURF * prior[cut:] + (1 - W_SURF) * pf
    resid = df.TVT.values[cut:] - blend
    off_true = float(resid.mean()); off_med = float(np.median(resid))
    rmse_blend = float(np.sqrt((resid ** 2).mean()))

    # ---- backtest del prefijo (enmascara el ultimo 35%)
    bt_cut = max(20, int(round((1 - BT_FRAC) * cut)))
    nb = cut - bt_cut
    if nb >= 30:
        hw_bt = df_h.iloc[:cut].copy()
        hw_bt.loc[bt_cut:, "TVT_input"] = np.nan
        t, g = _tw(tw)
        pts_bt, _llbt = run_pf_ancc_multi(hw_bt, t, g,
                                          seeds=np.arange(1, S_FEAT + 1), N=N_PART)
        pf_bt = pts_bt.mean(0).astype(float)
        lo_bt = max(0, bt_cut - CAL_TAIL)
        c_bt = np.median(tvti[lo_bt:bt_cut] + z[lo_bt:bt_cut] - s_all[lo_bt:bt_cut])
        prior_bt = s_all[bt_cut:cut] - z[bt_cut:cut] + c_bt
        resid_bt = tvti[bt_cut:cut] - (W_SURF * prior_bt + (1 - W_SURF) * pf_bt)
        md_bt = md[bt_cut:cut]
        n_end = max(nb // 10, 20)
        bt = dict(off_bt=float(resid_bt.mean()),
                  off_bt_tail=float(resid_bt[nb // 2:].mean()),
                  off_bt_end=float(resid_bt[-n_end:].mean()),
                  bt_slope=slope_per_kft(md_bt, resid_bt),
                  bt_rmse=float(np.sqrt((resid_bt ** 2).mean())),
                  bt_len=nb, has_bt=1)
        np.savez_compressed(CACHE / f"bt_{wid}.npz",
                            resid=resid_bt.astype(np.float32), md=md_bt.astype(np.float32))
    else:
        bt = dict(off_bt=0.0, off_bt_tail=0.0, off_bt_end=0.0, bt_slope=0.0,
                  bt_rmse=0.0, bt_len=nb, has_bt=0)

    # ---- spread / loglik (normalizado por longitud del PF)
    sstd = pts.std(0)
    lln = ll / max(n_pred, 1)

    # ---- nn crudo (LOWO): arbol sin el propio pozo, decimado x10
    tree_r, ok = raw_tree_of(wid)
    if ok:
        q = np.column_stack([df.X.values[cut::10], df.Y.values[cut::10]])
        dr, _ = tree_r.query(q, k=1, workers=-1)
    else:
        dr = np.array([np.nan])

    # ---- desacuerdo superficie-PF post-PS
    dis = prior[cut:] - pf

    # ---- GR del prefijo contra typewell (ajuste afin)
    gr = df.GR.values
    twc = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    gr_res_std, gr_corr = np.nan, np.nan
    if len(twc) >= 20:
        g_pref = np.interp(tvti[:cut], twc.TVT.values, twc.GR.values)
        okg = np.isfinite(gr[:cut]) & np.isfinite(g_pref)
        if okg.sum() >= 40:
            A = np.column_stack([g_pref[okg], np.ones(okg.sum())])
            a, b = np.linalg.lstsq(A, gr[:cut][okg], rcond=None)[0]
            fit = a * g_pref[okg] + b
            gr_res_std = float(np.std(gr[:cut][okg] - fit))
            if np.std(fit) > 0 and np.std(gr[:cut][okg]) > 0:
                gr_corr = float(np.corrcoef(fit, gr[:cut][okg])[0, 1])

    return dict(
        well=wid, split=split, pf_src=pf_src, n_pred=n_pred, cut_len=cut,
        ps_frac=cut / len(df), md_post=float(md[-1] - md[cut]),
        **bt,
        sstd_mean=float(sstd.mean()), sstd_p90=float(np.percentile(sstd, 90)),
        ll_mean_n=float(lln.mean()), ll_std_n=float(lln.std()),
        ll_rng_n=float(lln.max() - lln.min()),
        nn_mean=float(nn[cut:].mean()), nn_p90=float(np.percentile(nn[cut:], 90)),
        nnr_mean=float(np.nanmean(dr)), nnr_p90=float(np.nanpercentile(dr, 90)),
        dis_mean=float(dis.mean()), dis_abs=float(abs(dis.mean())),
        dis_slope=slope_per_kft(md[cut:], dis), dis_std=float(dis.std()),
        dip30=dip_tail(tvti, z, md, cut, 30), dip200=dip_tail(tvti, z, md, cut, 200),
        gr_res_std=gr_res_std, gr_corr=gr_corr,
        off_true=off_true, off_med=off_med, rmse_blend=rmse_blend)


def main():
    block = sys.argv[1] if len(sys.argv) > 1 else "eval"
    ids = well_ids()
    k60, k150 = set(_select(ids, 60, 42)), set(_select(ids, 150, 42))
    ev = sorted(k60 | k150)
    rest = [w for w in ids if w not in k60 | k150]
    tr = _select(rest, 400, 42)          # 400 disjuntos, estratificados por n_pred
    todo = [(w, "eval") for w in ev] if block == "eval" else \
           [(w, "train") for w in tr] if block == "train" else \
           [(w, "eval") for w in ev] + [(w, "train") for w in tr]

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, usecols=["well"]).well)
    todo = [(w, sp) for w, sp in todo if w not in done]
    print(f"bloque={block}: {len(todo)} pozos por hacer ({len(done)} ya en CSV)", flush=True)
    if not todo:
        return

    # PF S=64 medio de los npz (la componente PF del blend enviado)
    pf64_pred = {}
    for f in ("v4_multiseed_k60.npz", "v4_multiseed_k150.npz"):
        z = np.load(R / f)
        for kk in z.files:
            if kk.endswith("_pred"):
                pf64_pred[kk[:-5]] = z[kk]
    print(f"pf64 en npz: {len(pf64_pred)} pozos", flush=True)

    t0 = time.time()
    field = SurfaceField(aniso=ANISO, theta=THETA_OPT)
    print(f"SurfaceField theta_opt: {time.time()-t0:.0f}s", flush=True)

    _rt = {}
    def raw_tree_of(wid):
        """KDTree en coordenadas CRUDAS sin el propio pozo (cachea el ultimo)."""
        if _rt.get("wid") != wid:
            if wid in field.idx:
                keep = field.wid != field.idx[wid]
                _rt.update(wid=wid, tree=cKDTree(field.xy_raw[keep]), ok=True)
            else:
                _rt.update(wid=wid, tree=None, ok=False)
        return _rt["tree"], _rt["ok"]

    hdr = not OUT.exists()
    tp = time.time()
    for i, (wid, split) in enumerate(todo):
        try:
            row = build_row(wid, split, field, raw_tree_of, pf64_pred)
        except Exception as e:  # noqa: BLE001 — un pozo malo no tumba la pasada
            print(f"  ERROR {wid}: {type(e).__name__}: {e}", flush=True)
            continue
        if row is None:
            continue
        pd.DataFrame([row])[COLS].to_csv(OUT, mode="a", header=hdr, index=False)
        hdr = False
        if (i + 1) % 20 == 0:
            el = time.time() - tp
            print(f"  {i+1}/{len(todo)}  ({el/(i+1):.1f}s/pozo, quedan "
                  f"~{el/(i+1)*(len(todo)-i-1)/60:.0f} min)", flush=True)
    print(f"HECHO bloque={block} en {(time.time()-tp)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
