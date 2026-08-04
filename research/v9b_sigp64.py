"""Recalibrar sig_p con el PF REAL del kernel (media S=64), no con S=8.

Sesgo señalado por el verificador: sig_p(md_since) se ajusto con PF S=8; el
kernel corre S=64 (mejor) => sp sobreestimada => w inclinada de mas a superficie.
Reajusta sig_p sobre los mismos 200 fit_wells con S=64 y evalua pareado el blend
adaptativo 2D (S_MULT igual; solo cambia sig_p).

Uso: python research/v9b_sigp64.py fit | eval 60 | eval 150
"""
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod                                    # noqa: E402
import v7_wadapt as v7                                # noqa: E402
from ban_lib import run_pf_ancc_multi                 # noqa: E402
from cv import evaluate, load_well                    # noqa: E402
from model import _prior                              # noqa: E402
from pf_publico import _tw                            # noqa: E402

TAB = np.load(R / "v8_s2d_tablas.npz")
OUT = R / "v9b_sigp64.npz"
S = 64


def pf64_fresh(df_h, tw, wid):
    f = R / "v4_multiseed_cache" / f"pf_{wid}_S64_N600.npz"
    if f.exists():
        return np.load(f)["pts"][:S].mean(0).astype(float)
    t, g = _tw(tw)
    pts, _ = run_pf_ancc_multi(df_h, t, g, seeds=np.arange(1, S + 1, dtype=np.int64))
    np.savez_compressed(f, pts=pts.astype(np.float32), ll=np.zeros(S))
    return pts.mean(0).astype(float)


def fit():
    fw = v7.fit_wells()
    ep2, mds = [], []
    t0 = time.time()
    for i, wid in enumerate(fw):
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):
            continue
        df_h = df[cvmod.TEST_COLS].copy()
        p = pf64_fresh(df_h, tw, wid)
        y = df.TVT.values[cut:]
        md = df.MD.values
        ep2.append((y - p) ** 2)
        mds.append(md[cut:] - md[cut - 1])
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(fw)}  {time.time()-t0:.0f}s", flush=True)
    ep2, mds = np.concatenate(ep2), np.concatenate(mds)
    sig_p64 = v7._binned_rms(mds, ep2, v7.MD_BINS)
    np.savez(OUT, sig_p64=sig_p64)
    print("sig_p S=8 :", np.round(TAB["sig_p_1d"], 2))
    print("sig_p S=64:", np.round(sig_p64, 2))


def _interp2d_smult(nn, md):
    g = TAB["s_mult"]; nnb, mdb = TAB["nn_bins"], TAB["md_bins"]
    def cent(b): return 0.5 * (b[:-1] + np.minimum(b[1:], b[-2] * 2))
    cr, cc = cent(nnb), cent(mdb)
    ri = np.clip(np.interp(nn, cr, np.arange(len(cr))), 0, len(cr) - 1)
    ci = np.clip(np.interp(md, cc, np.arange(len(cc))), 0, len(cc) - 1)
    r0 = np.clip(np.floor(ri).astype(int), 0, len(cr) - 2)
    c0 = np.clip(np.floor(ci).astype(int), 0, len(cc) - 2)
    fr, fc = ri - r0, ci - c0
    return ((1-fr)*(1-fc)*g[r0, c0] + fr*(1-fc)*g[r0+1, c0]
            + (1-fr)*fc*g[r0, c0+1] + fr*fc*g[r0+1, c0+1])


def make_pred(sig_p):
    mdb = TAB["md_bins"]
    def cent(b): return 0.5 * (b[:-1] + np.minimum(b[1:], b[-2] * 2))
    cc = cent(mdb)

    def predict(df_h, tw, wid=None):
        cut = v7.cut_of(df_h)
        prior, nn = _prior(df_h, v7.field(), wid, cut)
        p = v7.pf_of(df_h, tw, wid, S)
        md = df_h.MD.values
        mdp = md[cut:] - md[cut - 1]
        ss = _interp2d_smult(nn[cut:], mdp)
        sp = np.interp(mdp, cc, sig_p)
        w = np.clip(sp**2 / (ss**2 + sp**2), 0.05, 0.9)
        return w * prior[cut:] + (1 - w) * p

    return predict


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        fit()
    else:
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        z = np.load(OUT)
        print(f"== pareado k={k}: sig_p S=8 (v8) vs S=64 recalibrada ==")
        for name, sp in [("sig_p S=8 (v8)", TAB["sig_p_1d"]), ("sig_p S=64", z["sig_p64"])]:
            r = evaluate(make_pred(sp), k=k, verbose=False)
            print(f"  {name:16s} rmse={r['rmse']:7.3f} proxy={r['rmse_lb_proxy']:7.3f}")
