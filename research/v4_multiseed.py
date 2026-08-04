"""v4: ensemble MULTI-SEMILLA del PF ANCC ponderado por verosimilitud + blend
con la superficie anisotropa. La familia publica (9.9 -> 7.3) usa 128 semillas x
500 particulas con pesos softmax(loglik/scale), scales {3,5,8,12}, promediando.

Uso:
    python research/v4_multiseed.py suite [K=60] [S=64]   # suite completa
    python research/v4_multiseed.py k150 [S] [W]          # confirmacion k=150

Reutiliza SOLO LECTURA ban_lib.run_pf_ancc_multi (kernel numba _pf_ancc_seeds:
identico a pf_publico._pf_ancc + semilla explicita + loglik por pasada, con
loglik = sum_i log(sum_j w_j * lk_j) = estimador PF de la verosimilitud
marginal, acumulado ANTES de normalizar). Todo lo demas vive aqui.

Salidas: v4_multiseed_k{K}.csv (variantes), v4_multiseed_k{K}.npz (pred, seed
spread por punto y loglik por pozo, para el GBM), y el log con diagnosticos.
"""
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod                                    # noqa: E402
from ban_lib import run_pf_ancc_multi, softmax_w      # noqa: E402  (solo lectura)
from cv import evaluate, load_well                    # noqa: E402
from model import SurfaceField, _prior                # noqa: E402
from pf_publico import BEAMS, _tw, beam_search        # noqa: E402

cvmod.load_well = lru_cache(maxsize=None)(cvmod.load_well)  # evita releer CSVs

N_PART = 600
ANISO, K_IDW = 16.0, 24        # superficie de referencia: 12.88 @ k=60

# ------------------------------------------------------------------ caches
PF_CACHE, SURF_CACHE, BEAM_CACHE = {}, {}, {}
PF_T, SURF_T, BEAM_T = [], [], []
FIELD = None
SEEDS = None
CACHE_DIR = R / "v4_multiseed_cache"   # PF por pozo en disco (pasadas siguientes gratis)
CACHE_DIR.mkdir(exist_ok=True)


def _pf_disk(wid):
    """Busca en disco un cache con >= S semillas (seeds siempre 1..S) y recorta."""
    S = len(SEEDS)
    for st in (128, 96, 64, 48, 32, 16, 8):
        if st < S:
            break
        f = CACHE_DIR / f"pf_{wid}_S{st}_N{N_PART}.npz"
        if f.exists():
            z = np.load(f)
            return z["pts"][:S].astype(np.float32), z["ll"][:S].astype(np.float64)
    return None


def cut_of(df_h):
    m = df_h.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df_h)


def pf_of(df_h, tw, wid):
    """(pts (S,n) float32, loglik (S,)) del PF ANCC con SEEDS semillas."""
    if wid not in PF_CACHE:
        hit = _pf_disk(wid)
        if hit is None:
            t, g = _tw(tw)
            t0 = time.time()
            pts, ll = run_pf_ancc_multi(df_h, t, g, seeds=SEEDS, N=N_PART)
            PF_T.append(time.time() - t0)
            hit = (pts.astype(np.float32), ll)
            np.savez_compressed(CACHE_DIR / f"pf_{wid}_S{len(SEEDS)}_N{N_PART}.npz",
                                pts=hit[0], ll=ll)
        PF_CACHE[wid] = hit
    return PF_CACHE[wid]


def surf_of(df_h, wid):
    """Prior de superficie aniso (post-PS), LOWO via exclude=wid."""
    if wid not in SURF_CACHE:
        cut = cut_of(df_h)
        t0 = time.time()
        prior, nn = _prior(df_h, FIELD, wid, cut)
        SURF_T.append(time.time() - t0)
        SURF_CACHE[wid] = (prior[cut:].astype(np.float64), float(np.median(nn[cut:])))
    return SURF_CACHE[wid][0]


def beam7_of(df_h, tw, wid):
    """Las 7 trayectorias beam (7,n). Wrapper ARREGLADO: el predict_beam de
    pf_publico fallaba porque cv pasa wid como 3er posicional y aterrizaba en
    cfg -> BEAMS[wid] TypeError. Aqui cfg va en closure y wid es keyword."""
    if wid not in BEAM_CACHE:
        t, g = _tw(tw)
        cut = cut_of(df_h)
        gr = df_h.GR.astype(float).interpolate(limit_direction="both") \
                 .fillna(float(np.nanmean(g))).values[cut:]
        start = float(df_h.TVT_input.values[cut - 1])
        t0 = time.time()
        BEAM_CACHE[wid] = np.stack([
            np.asarray(beam_search(gr, t, g, start, bs, mc, es, r), float)
            for (bs, mc, es, r, _n) in BEAMS])
        BEAM_T.append(time.time() - t0)
    return BEAM_CACHE[wid]


def make_beam(cfg):
    """predict de UNA config de BEAMS, interfaz cv (arreglo del TypeError)."""
    def predict(df_h, tw, wid=None):
        return beam7_of(df_h, tw, wid)[cfg]
    return predict


# ------------------------------------------------------------- variantes PF
def mk_seed(i):
    def predict(df_h, tw, wid=None):
        return pf_of(df_h, tw, wid)[0][i]
    return predict


def mk_mean(s):
    def predict(df_h, tw, wid=None):
        return pf_of(df_h, tw, wid)[0][:s].mean(0)
    return predict


def mk_median(s):
    def predict(df_h, tw, wid=None):
        return np.median(pf_of(df_h, tw, wid)[0][:s], 0)
    return predict


def mk_soft(scale, s):
    """softmax(loglik_total/scale) sobre s semillas."""
    def predict(df_h, tw, wid=None):
        pts, ll = pf_of(df_h, tw, wid)
        w = softmax_w(ll[:s], scale)
        return (pts[:s] * w[:, None]).sum(0)
    return predict


def mk_public(s, scales=(3., 5., 8., 12.)):
    """La receta publica: media sobre scales de la media softmax(ll/scale)."""
    def predict(df_h, tw, wid=None):
        pts, ll = pf_of(df_h, tw, wid)
        acc = np.zeros(pts.shape[1])
        for sc in scales:
            w = softmax_w(ll[:s], sc)
            acc += (pts[:s] * w[:, None]).sum(0)
        return acc / len(scales)
    return predict


def mk_soft_adapt(beta, s):
    """softmax con scale ADAPTATIVO = beta*std(loglik entre semillas del pozo).
    Corrige que el rango del loglik dependa de la longitud del pozo."""
    def predict(df_h, tw, wid=None):
        pts, ll = pf_of(df_h, tw, wid)
        sd = float(np.std(ll[:s]))
        if sd <= 0:
            return pts[:s].mean(0)
        w = softmax_w(ll[:s], beta * sd)
        return (pts[:s] * w[:, None]).sum(0)
    return predict


def mk_argmax(s):
    def predict(df_h, tw, wid=None):
        pts, ll = pf_of(df_h, tw, wid)
        return pts[int(np.argmax(ll[:s]))]
    return predict


def mk_topk(kk, s):
    """Media SIMPLE de las kk semillas con mayor loglik (descarta las malas
    sin concentrar el peso en una sola, que es lo que rompe al softmax)."""
    def predict(df_h, tw, wid=None):
        pts, ll = pf_of(df_h, tw, wid)
        idx = np.argsort(ll[:s])[-kk:]
        return pts[idx].mean(0)
    return predict


# ------------------------------------------------------------------ blends
def mk_blend(pf_fn, w):
    def predict(df_h, tw, wid=None):
        return w * surf_of(df_h, wid) + (1. - w) * pf_fn(df_h, tw, wid)
    return predict


def mk_blend3(pf_fn, ws, wb):
    def predict(df_h, tw, wid=None):
        bm = beam7_of(df_h, tw, wid).mean(0)
        return ws * surf_of(df_h, wid) + wb * bm + (1. - ws - wb) * pf_fn(df_h, tw, wid)
    return predict


def mk_surf(df_h, tw, wid=None):
    return surf_of(df_h, wid)


def mk_beam_m7(df_h, tw, wid=None):
    return beam7_of(df_h, tw, wid).mean(0)


def mk_beam_med7(df_h, tw, wid=None):
    return np.median(beam7_of(df_h, tw, wid), 0)


# ------------------------------------------------------------------ runner
ROWS = []


def run(name, fn, K):
    res = evaluate(fn, k=K, verbose=False)
    ROWS.append({"variante": name, "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
    print(f"{name:34s} -> {res['rmse']:8.3f}  (lb_proxy {res['rmse_lb_proxy']:7.3f})", flush=True)
    return res["rmse"]


def dump(K, tag=""):
    pd.DataFrame(ROWS).to_csv(R / f"v4_multiseed_k{K}{tag}.csv", index=False)


def save_npz(K, best_fn):
    """pred de la mejor variante + spread entre semillas por punto + loglik."""
    out = {}
    for wid, (pts, ll) in PF_CACHE.items():
        df, tw, cut = load_well(wid)
        df_h = df[cvmod.TEST_COLS].copy()
        out[f"{wid}_pred"] = np.asarray(best_fn(df_h, tw, wid), np.float32)
        out[f"{wid}_sstd"] = pts.std(0).astype(np.float32)   # spread entre semillas
        out[f"{wid}_ll"] = ll.astype(np.float32)
    np.savez_compressed(R / f"v4_multiseed_k{K}.npz", **out)
    print(f"npz: {len(PF_CACHE)} pozos -> v4_multiseed_k{K}.npz", flush=True)


def diagnostics():
    """(a) el loglik sabe que semilla es buena?  (b) el spread predice el error?"""
    corr_ll, corr_sp = [], []
    for wid, (pts, ll) in PF_CACHE.items():
        df, tw, cut = load_well(wid)
        y = df.TVT.values[cut:]
        err_seed = np.sqrt(((pts - y[None, :]) ** 2).mean(1))
        if np.std(ll) > 0 and np.std(err_seed) > 0:
            corr_ll.append(float(np.corrcoef(ll, err_seed)[0, 1]))
        sp = pts.std(0); err = np.abs(pts.mean(0) - y)
        if np.std(sp) > 0 and np.std(err) > 0:
            corr_sp.append(float(np.corrcoef(sp, err)[0, 1]))
    lls = np.array([ll for _, ll in PF_CACHE.values()], dtype=object)
    rng = np.mean([ll.max() - ll.min() for ll in lls])
    sd = np.mean([ll.std() for ll in lls])
    print(f"\nDIAG loglik: rango medio entre semillas = {rng:.1f} | std media = {sd:.1f}")
    print(f"DIAG corr(loglik, rmse_semilla) media   = {np.mean(corr_ll):+.3f} "
          f"(negativo = el loglik SI identifica la semilla buena)")
    print(f"DIAG corr(spread_semillas, |err|) media = {np.mean(corr_sp):+.3f} "
          f"(positivo = el spread ES señal de confianza)", flush=True)


def costs(S):
    if PF_T:
        pf = np.mean(PF_T)
        print(f"\nCOSTE pf S={S}: {pf:.2f}s/pozo ({pf/S:.3f}s/pozo/semilla) "
              f"| re-run 770 pozos ~ {pf*770/3600:.1f}h")
    if SURF_T:
        print(f"COSTE superficie: {np.mean(SURF_T):.2f}s/pozo (LOWO; sin LOWO es menor)")
    if BEAM_T:
        print(f"COSTE beam7: {np.mean(BEAM_T):.2f}s/pozo", flush=True)


# ==================================================================== main
if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "suite"

    if block == "suite":
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        S = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        SEEDS = np.arange(1, S + 1, dtype=np.int64)
        print(f"== v4 multiseed suite: k={K} S={S} N={N_PART} ==", flush=True)

        t0 = time.time()
        run("pf 1 semilla (#1)", mk_seed(0), K)        # llena el cache PF
        print(f"  cache PF listo en {time.time()-t0:.0f}s", flush=True)

        for s in (8, 16, 32, 64):
            if s <= S:
                run(f"pf media {s}", mk_mean(s), K)
        for s in (8, 16, 32, 64):
            if s <= S:
                run(f"pf mediana {s}", mk_median(s), K)
        run(f"pf argmax ll ({S})", mk_argmax(S), K)
        for sc in (3, 5, 8, 12):
            run(f"pf softmax ll/{sc} ({S})", mk_soft(sc, S), K)
        for sc in (100, 300, 1000, 3000):
            run(f"pf softmax ll/{sc} ({S})", mk_soft(sc, S), K)
        for s in (8, 16, 32, 64):
            if s <= S:
                run(f"pf publica scales(3,5,8,12) {s}", mk_public(s), K)
        for b in (1.0, 2.0, 4.0):
            run(f"pf softmax adapt b={b} ({S})", mk_soft_adapt(b, S), K)
        for kk in (S // 4, S // 2, 3 * S // 4):
            if kk >= 2:
                run(f"pf top{kk}/{S} por ll", mk_topk(kk, S), K)
        diagnostics()
        dump(K)

        # superficie + blends
        print("\ncargando SurfaceField aniso=16 ...", flush=True)
        t0 = time.time()
        FIELD = SurfaceField(aniso=ANISO)
        print(f"  {time.time()-t0:.0f}s", flush=True)
        run("superficie aniso16 k24", mk_surf, K)
        run("beam media7", mk_beam_m7, K)
        run("beam mediana7", mk_beam_med7, K)
        for cfg in (5,):                       # beam_mid, la mejor individual
            run(f"beam cfg{cfg} ({BEAMS[cfg][4]})", make_beam(cfg), K)

        best_pf = mk_mean(S)                   # se refina tras ver la tabla
        for w in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            run(f"blend {w:.2f}*surf+{1-w:.2f}*pfmean{S}", mk_blend(best_pf, w), K)
        pub = mk_public(S)
        for w in (0.20, 0.25, 0.30):
            run(f"blend {w:.2f}*surf+{1-w:.2f}*pfpub{S}", mk_blend(pub, w), K)
        for nm, fn2 in (("soft1000", mk_soft(1000, S)), (f"top{S//2}", mk_topk(S // 2, S)),
                        (f"median{S}", mk_median(S))):
            for w in (0.25, 0.30, 0.35):
                run(f"blend {w:.2f}*surf+{1-w:.2f}*{nm}", mk_blend(fn2, w), K)
        for ws in (0.20, 0.25, 0.30, 0.35):
            for wb in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
                run(f"blend3 s{ws:.2f} b{wb:.2f} pf{1-ws-wb:.2f}",
                    mk_blend3(best_pf, ws, wb), K)
        dump(K)
        best = min(ROWS, key=lambda r: r["rmse"])
        print(f"\nMEJOR: {best['variante']} -> {best['rmse']:.3f}")
        save_npz(K, mk_mean(S))
        costs(S)

    elif block == "k150":
        K = 150
        S = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
        WB = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0
        SEEDS = np.arange(1, S + 1, dtype=np.int64)
        print(f"== v4 confirmacion k=150: S={S} w={W} wb={WB} ==", flush=True)
        t0 = time.time()
        FIELD = SurfaceField(aniso=ANISO)
        print(f"SurfaceField: {time.time()-t0:.0f}s", flush=True)

        run(f"pf media {S}", mk_mean(S), K)
        run("superficie aniso16 k24", mk_surf, K)
        best_pf = mk_mean(S)
        for w in (W - 0.05, W, W + 0.05):
            run(f"blend {w:.2f}*surf+{1-w:.2f}*pfmean{S}", mk_blend(best_pf, w), K)
        if WB >= 0:
            for wb in (WB - 0.05, WB, WB + 0.05):
                if wb >= 0:
                    run(f"blend3 s{W:.2f} b{wb:.2f} pf{1-W-wb:.2f}",
                        mk_blend3(best_pf, W, wb), K)
        diagnostics()
        dump(K)
        save_npz(K, mk_mean(S))
        costs(S)
