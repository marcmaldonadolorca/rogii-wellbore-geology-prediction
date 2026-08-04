"""v7: blend ADAPTATIVO por punto via varianza inversa.

w_s(punto) = sigma_p^2(md_since) / (sigma_s^2(nn) + sigma_p^2(md_since))
pred = w_s * superficie + (1 - w_s) * PF_multiseed

Las dos curvas de error se ajustan en ~200 pozos DISJUNTOS de los subconjuntos
k=60/k=150 de cv._select (sin contaminacion). La motivacion es la transferencia:
los pozos del test oculto parecen mas aislados que en LOWO; con w dependiente de
nn_dist el kernel decide por pozo/punto en runtime cuanto fiarse de la superficie.

Uso:
  python research/v7_wadapt.py fit      # ajusta y guarda las curvas (v7_curvas.npz)
  python research/v7_wadapt.py eval 60  # evalua adaptativo vs fijo 0.45 (pareado)
"""
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod                                     # noqa: E402
from ban_lib import run_pf_ancc_multi                  # noqa: E402
from cv import evaluate, load_well, well_ids, _select  # noqa: E402
from model import SurfaceField, _prior                 # noqa: E402
from pf_publico import _tw                             # noqa: E402

THETA_OPT = 2.278489
N_FIT = 200
S_FIT = 8            # semillas del PF para ajustar curvas (barato)
S_EVAL = 64          # semillas en evaluacion (cache de la suite si existe)
CACHE = R / "v4_multiseed_cache"
CURVAS = R / "v7_curvas.npz"

NN_BINS = np.array([0, 50, 100, 200, 300, 450, 600, 800, 1000, 1500, 2500, 1e9])
MD_BINS = np.array([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 1e9])

_field = None


def field():
    global _field
    if _field is None:
        _field = SurfaceField(aniso=16.0, theta=THETA_OPT)
    return _field


def fit_wells():
    ids = well_ids()
    excl = set(_select(ids, 60, 42)) | set(_select(ids, 150, 42))
    rest = [w for w in ids if w not in excl]
    rng = np.random.default_rng(7)
    return sorted(rng.choice(rest, size=N_FIT, replace=False))


def pf_of(df_h, tw, wid, S):
    f = CACHE / f"pf_{wid}_S64_N600.npz"
    if f.exists():
        z = np.load(f)
        return z["pts"][:S].mean(0).astype(float)
    t, g = _tw(tw)
    pts, _ = run_pf_ancc_multi(df_h, t, g, seeds=np.arange(1, S + 1, dtype=np.int64))
    return pts.mean(0).astype(float)


def cut_of(df_h):
    m = df_h.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df_h)


def _binned_rms(x, e2, bins):
    idx = np.digitize(x, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() > 200:
            out[b] = np.sqrt(np.mean(e2[m]))
    # rellenar huecos con el vecino valido mas cercano
    v = np.where(np.isfinite(out))[0]
    for b in range(len(out)):
        if not np.isfinite(out[b]):
            out[b] = out[v[np.argmin(np.abs(v - b))]]
    return out


def fit():
    fw = fit_wells()
    es2, nns, ep2, mds = [], [], [], []
    t0 = time.time()
    for i, wid in enumerate(fw):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[cvmod.TEST_COLS].copy()
        prior, nn = _prior(df_h, field(), wid, cut)
        pfp = pf_of(df_h, tw, wid, S_FIT)
        y = df.TVT.values[cut:]
        md = df.MD.values
        es2.append((y - prior[cut:]) ** 2)
        nns.append(nn[cut:])
        ep2.append((y - pfp) ** 2)
        mds.append(md[cut:] - md[cut - 1])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fw)}  {time.time()-t0:.0f}s", flush=True)
    es2, nns = np.concatenate(es2), np.concatenate(nns)
    ep2, mds = np.concatenate(ep2), np.concatenate(mds)
    sig_s = _binned_rms(nns, es2, NN_BINS)
    sig_p = _binned_rms(mds, ep2, MD_BINS)
    np.savez(CURVAS, sig_s=sig_s, sig_p=sig_p, nn_bins=NN_BINS, md_bins=MD_BINS)
    print("sigma_s(nn):     ", np.round(sig_s, 1))
    print("sigma_p(md_since):", np.round(sig_p, 1))
    print(f"guardado {CURVAS}")


def _interp_curve(x, bins, vals):
    centros = 0.5 * (bins[:-1] + np.minimum(bins[1:], bins[-2] * 2))
    return np.interp(x, centros, vals)


def make_predictor(mode="adapt", w_fixed=0.45, S=S_EVAL, w_cap=(0.05, 0.9)):
    z = np.load(CURVAS)
    sig_s_b, sig_p_b = z["sig_s"], z["sig_p"]

    def predict(df_h, tw, wid=None):
        cut = cut_of(df_h)
        prior, nn = _prior(df_h, field(), wid, cut)
        pfp = pf_of(df_h, tw, wid, S)
        md = df_h.MD.values
        s_post, nn_post = prior[cut:], nn[cut:]
        if mode == "fixed":
            w = w_fixed
        else:
            ss = _interp_curve(nn_post, NN_BINS, sig_s_b)
            sp = _interp_curve(md[cut:] - md[cut - 1], MD_BINS, sig_p_b)
            w = np.clip(sp ** 2 / (ss ** 2 + sp ** 2), *w_cap)
        return w * s_post + (1 - w) * pfp

    return predict


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        fit()
    else:
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        print(f"== eval k={k} (pareado: mismo cache PF) ==")
        for name, p in [("fijo 0.45", make_predictor("fixed")),
                        ("adaptativo", make_predictor("adapt"))]:
            r = evaluate(p, k=k, verbose=False)
            print(f"  {name:12s} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}")
