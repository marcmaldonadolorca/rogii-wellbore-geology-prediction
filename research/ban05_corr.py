"""5) Matriz de correlacion de ERRORES entre todas las senales + blends.

Senales (todas LOWO estricto, formato test exacto):
  geom      baseline geometrico (dip pre-PS)
  surf      superficie BUDA IDW k=16 + offset calibrado (model._prior)
  surf_hmm  surf + hmm_refine
  pf_ancc1  run_pf_ancc, una pasada (semilla 1)
  pf_ancc16 media de 16 semillas
  pf_z      run_pf_z
  beam_mid  la mejor config individual de BEAMS
  beam_m7   media de las 7 configs
  ncc       multi_scale_ncc, ensemble score-weighted

Salidas: ban05_rmse_k*.csv, ban05_corr_k*.csv (Pearson de errores pooled),
ban05_blend_k*.csv, y ban05_signals_k*.npz (errores por punto, para el GBM).
"""
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))

import cv as cvmod
from ban_lib import beam_all, cut_of, ncc_signals, run_pf_ancc_multi
from baseline import predict as geom_predict
from cv import _select, evaluate, load_well, well_ids
from model import SurfaceField, _prior, hmm_refine
from pf_publico import _tw, run_pf_z

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
S = int(sys.argv[2]) if len(sys.argv) > 2 else 16
cvmod.load_well = lru_cache(maxsize=None)(cvmod.load_well)
SEEDS = np.arange(1, S + 1, dtype=np.int64)
NAMES = ["geom", "surf", "surf_hmm", "pf_ancc1", "pf_ancc16", "pf_z", "beam_mid", "beam_m7", "ncc"]

print("cargando SurfaceField...", flush=True)
t0 = time.time()
FIELD = SurfaceField()
print(f"  {time.time()-t0:.0f}s", flush=True)

CACHE = {}
COST = []


def signals(df_h, tw, wid):
    if wid in CACHE:
        return CACHE[wid]
    cut = cut_of(df_h)
    t, g = _tw(tw)
    tt = {}
    t0 = time.time(); prior, _ = _prior(df_h, FIELD, wid, cut); tt["surf"] = time.time() - t0
    t0 = time.time(); hmm = hmm_refine(df_h, tw, prior, cut); tt["hmm"] = time.time() - t0
    t0 = time.time(); pts, _ll = run_pf_ancc_multi(df_h, t, g, seeds=SEEDS); tt["pf_ancc"] = time.time() - t0
    t0 = time.time(); pz, _ = run_pf_z(df_h, t, g); tt["pf_z"] = time.time() - t0
    t0 = time.time(); bm = beam_all(df_h, tw); tt["beam7"] = time.time() - t0
    t0 = time.time(); nc = ncc_signals(df_h, tw)["sc_ens"]; tt["ncc"] = time.time() - t0
    d = {"geom": geom_predict(df_h, cut, 500),
         "surf": prior[cut:],
         "surf_hmm": prior[cut:] + hmm,
         "pf_ancc1": pts[0].astype(float),
         "pf_ancc16": pts.mean(0).astype(float),
         "pf_z": np.asarray(pz, float),
         "beam_mid": bm[5],
         "beam_m7": bm.mean(0),
         "ncc": np.asarray(nc, float)}
    COST.append(tt)
    CACHE[wid] = d
    return d


def mk(name):
    def predict(df_h, tw, wid=None):
        return signals(df_h, tw, wid)[name]
    return predict


rows = []
for n in NAMES:
    res = evaluate(mk(n), k=K, verbose=False)
    rows.append({"senal": n, "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
    print(f"{n:10s} -> {res['rmse']:9.3f}", flush=True)
pd.DataFrame(rows).to_csv(R / f"ban05_rmse_k{K}.csv", index=False)

c = pd.DataFrame(COST)
print("\ncoste medio por pozo (s):")
print(c.mean().to_string(), flush=True)
print(f"  total {c.sum(1).mean():.2f} s/pozo (pf_ancc con {S} semillas)")

# ---- errores por punto -> correlacion
ids = _select(well_ids(), K, 42)
E = {n: [] for n in NAMES}; Y = []; WELL = []
for wid in ids:
    df, tw, cut = load_well(wid)
    if cut < 20 or cut >= len(df):
        continue
    d = CACHE[wid]
    y = df.TVT.values[cut:]
    Y.append(y); WELL.append(np.full(len(y), wid))
    for n in NAMES:
        E[n].append(np.asarray(d[n], float) - y)
Emat = np.column_stack([np.concatenate(E[n]) for n in NAMES])
Y = np.concatenate(Y); WELL = np.concatenate(WELL)
np.savez_compressed(R / f"ban05_signals_k{K}.npz", err=Emat.astype(np.float32),
                    names=np.array(NAMES), y=Y.astype(np.float32), well=WELL)

corr = pd.DataFrame(np.corrcoef(Emat.T), index=NAMES, columns=NAMES)
corr.to_csv(R / f"ban05_corr_k{K}.csv")
print("\ncorrelacion de errores (Pearson, pooled):")
print(corr.round(3).to_string(), flush=True)
print("\nRMSE individual:", {n: round(float(np.sqrt((Emat[:, i]**2).mean())), 3) for i, n in enumerate(NAMES)})

# ---- blends
def blend_rmse(w):
    return float(np.sqrt((((Emat * w).sum(1))**2).mean()))


br = []
gi = {n: i for i, n in enumerate(NAMES)}
for a in np.arange(0, 1.01, 0.05):
    w = np.zeros(len(NAMES)); w[gi["surf"]] = a; w[gi["pf_ancc16"]] = 1 - a
    br.append({"blend": f"surf {a:.2f} + pf_ancc16 {1-a:.2f}", "rmse": blend_rmse(w)})
for a in np.arange(0, 1.01, 0.05):
    w = np.zeros(len(NAMES)); w[gi["surf"]] = a; w[gi["pf_ancc1"]] = 1 - a
    br.append({"blend": f"surf {a:.2f} + pf_ancc1 {1-a:.2f}", "rmse": blend_rmse(w)})
for combo in [("surf", "pf_ancc16", "beam_m7"), ("surf", "pf_ancc16", "pf_z"),
              ("surf", "pf_ancc16", "beam_m7", "pf_z"),
              ("surf_hmm", "pf_ancc16", "beam_m7"),
              ("surf", "pf_ancc16", "beam_m7", "pf_z", "surf_hmm"),
              tuple(NAMES)]:
    idx = [gi[n] for n in combo]
    A = Emat[:, idx]
    # pesos que suman 1: min ||A w||^2 con 1'w = 1  (solucion cerrada)
    G = A.T @ A / len(A)
    Gi = np.linalg.pinv(G)
    w = Gi @ np.ones(len(idx)); w /= w.sum()
    full = np.zeros(len(NAMES)); full[idx] = w
    # honesto: pesos ajustados en pozos pares, evaluados en impares
    uw = np.array(sorted(set(WELL))); tr = np.isin(WELL, uw[::2]); te = ~tr
    Gt = A[tr].T @ A[tr] / tr.sum(); wt = np.linalg.pinv(Gt) @ np.ones(len(idx)); wt /= wt.sum()
    oos = float(np.sqrt((((A[te] * wt).sum(1))**2).mean()))
    br.append({"blend": "+".join(combo), "rmse": blend_rmse(full),
               "rmse_oos_mitad": oos, "pesos": np.round(w, 3).tolist()})

bdf = pd.DataFrame(br).sort_values("rmse")
bdf.to_csv(R / f"ban05_blend_k{K}.csv", index=False)
print("\nmejores blends:")
print(bdf.head(20).to_string(index=False))
