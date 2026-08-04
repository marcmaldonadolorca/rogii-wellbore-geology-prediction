"""SUR-06: barrido unificado de superficies -> RMSE solo, blend con pf_ancc y
curva error-vs-distancia-al-vecino.

python sur06_sweep.py <bloque> [k]
bloques: ref | ksub | aniso | interp | perwell | prefix | final
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/ftpx100/work/active/kaggle-rogii")
sys.path.insert(0, HERE)

import cv as CV            # noqa: E402
import sur02_lib as L      # noqa: E402
import sur05_blend as B    # noqa: E402
from sur03_pred import combine  # noqa: E402

BINS = [0, 100, 300, 600, 1000, 1e9]
BINL = ["<100", "100-300", "300-600", "600-1000", ">1000"]
CAL_TAIL = 500
SCORE_WIN = 2000


class Runner:
    """Carga los pozos una vez y evalua variantes de superficie sobre ellos."""

    def __init__(self, k=60):
        self.ids = B.wells(k)
        self.W = {}
        for w in self.ids:
            df, tw, cut = CV.load_well(w)
            self.W[w] = (df[CV.TEST_COLS].copy(), cut, df.TVT.values[cut:])
        self.tru = {w: self.W[w][2] for w in self.ids}
        self.pf = B.pf_cache(self.ids)
        self.rows = []

    def run(self, name, surf_fn, alpha=0.25, verbose=True, keep=False):
        """surf_fn(df_h, cut, wid) -> (pred_post, dr_post)."""
        t0 = time.time()
        pred, dr = {}, {}
        for w in self.ids:
            df_h, cut, _ = self.W[w]
            p, d = surf_fn(df_h, cut, w)
            pred[w] = np.asarray(p, float)
            dr[w] = np.asarray(d, float)
        dt = time.time() - t0
        r_s = B.pooled(pred, self.tru)
        bl = {w: alpha * pred[w] + (1 - alpha) * self.pf[w] for w in self.ids}
        r_b = B.pooled(bl, self.tru)
        e = np.concatenate([self.tru[w] - pred[w] for w in self.ids])
        dd = np.concatenate([dr[w] for w in self.ids])
        b = np.digitize(dd, BINS) - 1
        curve = [float(np.sqrt((e[b == i] ** 2).mean())) if (b == i).any() else np.nan
                 for i in range(5)]
        share = [float((e[b == i] ** 2).sum() / (e ** 2).sum()) for i in range(5)]
        pw = B.per_well(pred, self.tru)
        row = {"variante": name, "rmse": r_s, "blend": r_b,
               "med_pozo": float(pw.rmse.median()), "s_pozo": dt / len(self.ids)}
        row.update({f"d{l}": c for l, c in zip(BINL, curve)})
        row.update({f"w{l}": s for l, s in zip(BINL, share)})
        self.rows.append(row)
        if verbose:
            print(f"{name:46s} sup={r_s:7.3f} blend={r_b:7.3f} med={row['med_pozo']:6.2f}"
                  f" | " + " ".join(f"{c:6.1f}" for c in curve) + f"  {dt/len(self.ids):.2f}s/pozo",
                  flush=True)
        return (pred, dr) if keep else None

    def save(self, tag):
        pd.DataFrame(self.rows).to_csv(os.path.join(HERE, f"sur06_{tag}.csv"), index=False)


# ───────────────────── constructores de surf_fn ──────────────────────────────
def surf_cloud(cloud, interp, mode="buda", cal_tail=CAL_TAIL, score_win=SCORE_WIN,
               kmax=L.KMAX, **ckw):
    def fn(df_h, cut, wid):
        X, Y, Z = df_h.X.values, df_h.Y.values, df_h.Z.values
        d, gi, dr = cloud.neighbors(wid, X, Y, kmax=kmax)
        S = interp(cloud, d, gi, X, Y)
        base = S - Z[:, None]
        lo = max(0, cut - cal_tail)
        C = np.nanmedian(df_h.TVT_input.values[lo:cut, None] - base[lo:cut], axis=0)
        bad = ~np.isfinite(C)
        if bad.any():
            C[bad] = np.nanmedian(C[~bad]) if (~bad).any() else 0.0
        P = base + C
        out = combine(P, df_h.TVT_input.values, cut, mode, score_win, **ckw)
        out = np.where(np.isfinite(out), out, P[cut:, 5])
        return out, dr[cut:]
    return fn


def surf_perwell(F, interp, mwell=12, mode="buda", cal_tail=CAL_TAIL,
                 score_win=SCORE_WIN, **ckw):
    def fn(df_h, cut, wid):
        X, Y, Z = df_h.X.values, df_h.Y.values, df_h.Z.values
        D, V, _ = F.sample(wid, X, Y, mwell=mwell)
        S = interp(D, V, np.column_stack([X, Y]))
        base = S - Z[:, None]
        lo = max(0, cut - cal_tail)
        C = np.nanmedian(df_h.TVT_input.values[lo:cut, None] - base[lo:cut], axis=0)
        bad = ~np.isfinite(C)
        if bad.any():
            C[bad] = np.nanmedian(C[~bad]) if (~bad).any() else 0.0
        P = base + C
        out = combine(P, df_h.TVT_input.values, cut, mode, score_win, **ckw)
        out = np.where(np.isfinite(out), out, P[cut:, 5])
        return out, D[cut:, 0]
    return fn


if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "ref"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    R = Runner(K)
    print(f"{len(R.ids)} pozos | pf_ancc solo = {B.pooled(R.pf, R.tru):.3f}", flush=True)

    if block in ("ref",):
        c = L.Cloud(subsample=10)
        R.run("IDW s10 k16 BUDA (referencia)",
              surf_cloud(c, lambda cc, d, gi, X, Y: L.idw(cc, d, gi, 16)))
        for mode in ("mean", "median", "winv", "softmax"):
            R.run(f"IDW s10 k16 {mode}",
                  surf_cloud(c, lambda cc, d, gi, X, Y: L.idw(cc, d, gi, 16), mode=mode))

    if block == "aniso":
        for a in (1, 2, 3, 5, 8, 15, 30):
            c = L.Cloud(subsample=10, aniso=a)
            for k in (16, 32):
                R.run(f"IDW s10 k{k} aniso={a}",
                      surf_cloud(c, lambda cc, d, gi, X, Y, k=k: L.idw(cc, d, gi, k)))
            del c

    if block == "ksub":
        for sub in (10, 3):
            c = L.Cloud(subsample=sub)
            for k in (8, 16, 32, 64):
                R.run(f"IDW s{sub} k{k} BUDA",
                      surf_cloud(c, lambda cc, d, gi, X, Y, k=k: L.idw(cc, d, gi, k)))
            del c

    if block == "interp":
        c = L.Cloud(subsample=10)
        for k, lam in [(16, 1e-2), (32, 1e-2), (32, 1e-1), (32, 1.0), (32, 10.0), (64, 1.0)]:
            R.run(f"ridge k{k} lam{lam:g}",
                  surf_cloud(c, lambda cc, d, gi, X, Y, k=k, lam=lam:
                             L.linridge(cc, d, gi, X, Y, k, lam)))
        for k, rg in [(24, 600.0), (24, 1500.0), (24, 4000.0)]:
            R.run(f"krige k{k} rango{rg:g}",
                  surf_cloud(c, lambda cc, d, gi, X, Y, k=k, rg=rg:
                             L.krige(cc, d, gi, X, Y, k, 1.0, 1000.0, rg)))

    R.save(block)
