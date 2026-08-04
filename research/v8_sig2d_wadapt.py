"""v8: sigma de la superficie (y del PF) en 2D — nn_dist x md_since.

Hipotesis: el offset C_well calibrado pre-PS envejece a lo largo del tramo,
asi que sig_s deberia crecer tambien con md_since (hoy solo depende de nn).
Simetricamente se prueba sig_p(md_since, nn): el init del PF depende del espacio.

Construccion de cada rejilla (nn_bins x md_bins):
  - celda con >= MIN_PTS puntos: RMS crudo del residuo.
  - celda con menos: modelo multiplicativo sig_1d(eje primario) * g(eje secundario),
    con g = ratio marginal sqrt(sum e2 / sum sig_1d^2) por bin secundario
    (controla la composicion del eje primario dentro del bin).

Evaluacion PAREADA con el mismo cache PF S=64 que v7 (variante "1d" = control,
debe reproducir 9.828 k60 / 9.145 k150).

Uso:
  python research/v8_sig2d_wadapt.py fit      # residuos crudos en 200 fit wells (~6 min)
  python research/v8_sig2d_wadapt.py build    # construye y muestra matrices
  python research/v8_sig2d_wadapt.py eval 60  # evalua variantes (pareado)
  python research/v8_sig2d_wadapt.py all      # fit (si falta) + build + eval 60 + eval 150
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod                 # noqa: E402
import v7_wadapt as v7             # noqa: E402
from cv import evaluate, load_well # noqa: E402
from model import _prior           # noqa: E402

FITDATA = R / "v8_sig2d_fitdata.npz"
NN_BINS, MD_BINS = v7.NN_BINS, v7.MD_BINS
MIN_PTS = 300
W_CAP = (0.05, 0.9)


def fit():
    """Como v7.fit() pero guarda los residuos CRUDOS por punto (no solo binned)."""
    fw = v7.fit_wells()
    es2, nns, ep2, mds = [], [], [], []
    t0 = time.time()
    for i, wid in enumerate(fw):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[cvmod.TEST_COLS].copy()
        prior, nn = _prior(df_h, v7.field(), wid, cut)
        pfp = v7.pf_of(df_h, tw, wid, v7.S_FIT)
        y = df.TVT.values[cut:]
        md = df.MD.values
        es2.append((y - prior[cut:]) ** 2)
        nns.append(nn[cut:])
        ep2.append((y - pfp) ** 2)
        mds.append(md[cut:] - md[cut - 1])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fw)}  {time.time()-t0:.0f}s", flush=True)
    np.savez(FITDATA,
             es2=np.concatenate(es2), nns=np.concatenate(nns),
             ep2=np.concatenate(ep2), mds=np.concatenate(mds))
    print(f"guardado {FITDATA}")


def _grid2d(x_r, x_c, e2, bins_r, bins_c, sig1d_r):
    """Rejilla (filas=bins_r, cols=bins_c): RMS crudo o relleno multiplicativo."""
    ir = np.clip(np.digitize(x_r, bins_r) - 1, 0, len(bins_r) - 2)
    ic = np.clip(np.digitize(x_c, bins_c) - 1, 0, len(bins_c) - 2)
    nr, nc = len(bins_r) - 1, len(bins_c) - 1
    cnt = np.zeros((nr, nc), int)
    s = np.zeros((nr, nc))
    np.add.at(cnt, (ir, ic), 1)
    np.add.at(s, (ir, ic), e2)
    with np.errstate(invalid="ignore"):
        raw = np.sqrt(np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan))
    # ratio marginal g por columna, controlando la composicion de filas
    pred2 = sig1d_r[ir] ** 2
    g = np.ones(nc)
    for j in range(nc):
        m = ic == j
        if m.sum() >= MIN_PTS:
            g[j] = np.sqrt(e2[m].sum() / pred2[m].sum())
    mult = sig1d_r[:, None] * g[None, :]
    grid = np.where(cnt >= MIN_PTS, raw, mult)
    return grid, mult, cnt, g


def build(verbose=True):
    z = np.load(FITDATA)
    c = np.load(v7.CURVAS)
    sig_s_1d, sig_p_1d = c["sig_s"], c["sig_p"]
    # sig_s: filas=nn, cols=md_since
    s_grid, s_mult, s_cnt, g_md = _grid2d(z["nns"], z["mds"], z["es2"],
                                          NN_BINS, MD_BINS, sig_s_1d)
    # sig_p: filas=md_since, cols=nn
    p_grid, p_mult, p_cnt, h_nn = _grid2d(z["mds"], z["nns"], z["ep2"],
                                          MD_BINS, NN_BINS, sig_p_1d)
    if verbose:
        np.set_printoptions(linewidth=200, suppress=True)
        print("g(md) para sig_s (ratio marginal):", np.round(g_md, 2))
        print("h(nn) para sig_p (ratio marginal):", np.round(h_nn, 2))
        print("\nsig_s 2D (filas=nn_bins, cols=md_bins):")
        print(np.round(s_grid, 1))
        print("celdas crudas (>=%d pts): %d/%d" % (MIN_PTS, (s_cnt >= MIN_PTS).sum(), s_cnt.size))
        print("\ncounts sig_s:\n", s_cnt)
        print("\nsig_p 2D (filas=md_bins, cols=nn_bins):")
        print(np.round(p_grid, 1))
        print("celdas crudas: %d/%d" % ((p_cnt >= MIN_PTS).sum(), p_cnt.size))
    return dict(s_grid=s_grid, s_mult=s_mult, p_grid=p_grid, p_mult=p_mult,
                sig_s_1d=sig_s_1d, sig_p_1d=sig_p_1d,
                s_cnt=s_cnt, p_cnt=p_cnt, g_md=g_md, h_nn=h_nn)


def _centros(bins):
    return 0.5 * (bins[:-1] + np.minimum(bins[1:], bins[-2] * 2))


def _interp2d(bins_r, bins_c, grid):
    cr, cc = _centros(bins_r), _centros(bins_c)
    f = RegularGridInterpolator((cr, cc), grid, method="linear",
                                bounds_error=False, fill_value=None)

    def q(x_r, x_c):
        xr = np.clip(x_r, cr[0], cr[-1])
        xc = np.clip(x_c, cc[0], cc[-1])
        return f(np.column_stack([xr, xc]))

    return q


# ---- evaluacion pareada: memoiza (prior, nn, md_since, pf) por pozo ----
_memo = {}


def _base(df_h, tw, wid):
    if wid not in _memo:
        cut = v7.cut_of(df_h)
        prior, nn = _prior(df_h, v7.field(), wid, cut)
        pfp = v7.pf_of(df_h, tw, wid, v7.S_EVAL)
        md = df_h.MD.values
        _memo[wid] = (prior[cut:], nn[cut:], md[cut:] - md[cut - 1], pfp)
    return _memo[wid]


def make_pred(ss_fn, sp_fn):
    def predict(df_h, tw, wid=None):
        s_post, nn_post, md_post, pfp = _base(df_h, tw, wid)
        ss = ss_fn(nn_post, md_post)
        sp = sp_fn(md_post, nn_post)
        w = np.clip(sp ** 2 / (ss ** 2 + sp ** 2), *W_CAP)
        return w * s_post + (1 - w) * pfp
    return predict


def variantes():
    b = build(verbose=False)
    ss1 = lambda nn, md: v7._interp_curve(nn, NN_BINS, b["sig_s_1d"])   # noqa: E731
    sp1 = lambda md, nn: v7._interp_curve(md, MD_BINS, b["sig_p_1d"])   # noqa: E731
    ss2 = _interp2d(NN_BINS, MD_BINS, b["s_grid"])
    ss2m = _interp2d(NN_BINS, MD_BINS, b["s_mult"])
    sp2 = _interp2d(MD_BINS, NN_BINS, b["p_grid"])
    return [
        ("1d (control v7)", make_pred(ss1, sp1)),
        ("s2d crudo+fill ", make_pred(ss2, sp1)),
        ("s2d multiplic. ", make_pred(ss2m, sp1)),
        ("p2d, s 1d      ", make_pred(ss1, sp2)),
        ("s2d + p2d      ", make_pred(ss2, sp2)),
    ]


def evalk(k):
    print(f"== eval k={k} (pareado: mismo cache PF S=64) ==", flush=True)
    for name, p in variantes():
        r = evaluate(p, k=k, verbose=False)
        print(f"  {name} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "fit":
        fit()
    elif cmd == "build":
        build()
    elif cmd == "eval":
        evalk(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    else:
        if not FITDATA.exists():
            fit()
        build()
        evalk(60)
        evalk(150)
        print("DONE_V8_SIG2D")
