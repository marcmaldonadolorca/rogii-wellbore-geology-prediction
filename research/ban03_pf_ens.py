"""3) Ensemble multi-semilla de run_pf_ancc ponderado por verosimilitud.

Una sola pasada del PF da 13.766 (k=60). La familia publica usa 128 semillas x
500 particulas con pesos softmax(loglik/scale), scales {3,5,8,12}. Aqui se mide
la TENDENCIA con S semillas (por defecto 16) y se compara:
  - semilla unica (cada una por separado -> dispersion del estimador)
  - media simple de 2/4/8/16 semillas
  - softmax(loglik/scale) con scale in {3,5,8,12,50,200}
  - argmax loglik (la "mejor" pasada segun el propio PF)
Coste por pozo medido y reportado.
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
from ban_lib import run_pf_ancc_multi, softmax_w
from cv import evaluate
from pf_publico import _tw

K = int(sys.argv[1]) if len(sys.argv) > 1 else 60
S = int(sys.argv[2]) if len(sys.argv) > 2 else 16
N = int(sys.argv[3]) if len(sys.argv) > 3 else 600
cvmod.load_well = lru_cache(maxsize=None)(cvmod.load_well)   # solo evita releer CSV

SEEDS = np.arange(1, S + 1, dtype=np.int64)
CACHE = {}
TIMES = []


def get(df_h, tw, wid):
    if wid not in CACHE:
        t, g = _tw(tw)
        t0 = time.time()
        pts, ll = run_pf_ancc_multi(df_h, t, g, seeds=SEEDS, N=N)
        TIMES.append((time.time() - t0, len(pts[0])))
        CACHE[wid] = (pts, ll)
    return CACHE[wid]


def mk_mean(s):
    def predict(df_h, tw, wid=None):
        pts, _ = get(df_h, tw, wid)
        return pts[:s].mean(0)
    return predict


def mk_soft(scale, s=None):
    def predict(df_h, tw, wid=None):
        pts, ll = get(df_h, tw, wid)
        n = s or len(ll)
        w = softmax_w(ll[:n], scale)
        return (pts[:n] * w[:, None]).sum(0)
    return predict


def mk_seed(i):
    def predict(df_h, tw, wid=None):
        pts, _ = get(df_h, tw, wid)
        return pts[i]
    return predict


def mk_best(df_h, tw, wid=None):
    pts, ll = get(df_h, tw, wid)
    return pts[int(np.argmax(ll))]


def mk_median(df_h, tw, wid=None):
    pts, _ = get(df_h, tw, wid)
    return np.median(pts, 0)


rows = []


def run(name, fn):
    res = evaluate(fn, k=K, verbose=False)
    rows.append({"variante": name, "rmse": res["rmse"], "lb_proxy": res["rmse_lb_proxy"]})
    print(f"{name:24s} -> {res['rmse']:8.3f}  (lb_proxy {res['rmse_lb_proxy']:7.3f})", flush=True)
    return res


run("1 semilla (#1)", mk_seed(0))
print(f"  coste 1a pasada: {sum(t for t, _ in TIMES):.1f}s totales para {len(TIMES)} pozos "
      f"con {S} semillas => {sum(t for t, _ in TIMES)/max(len(TIMES),1):.2f}s/pozo "
      f"({sum(t for t, _ in TIMES)/max(len(TIMES),1)/S:.3f}s/pozo/semilla, N={N})", flush=True)

for i in (1, 2, 3):
    run(f"1 semilla (#{i+1})", mk_seed(i))
for s in (2, 4, 8, 16, 32, 64):
    if s <= S:
        run(f"media {s} semillas", mk_mean(s))
run(f"mediana {S} semillas", mk_median)
run(f"argmax loglik ({S})", mk_best)
for sc in (3, 5, 8, 12, 30, 100, 1000):
    run(f"softmax ll/{sc} ({S})", mk_soft(sc))

df = pd.DataFrame(rows)
df.to_csv(R / f"ban03_pf_ens_k{K}_S{S}_N{N}.csv", index=False)
print(df.sort_values("rmse").to_string(index=False))

# dispersion del loglik entre semillas (diagnostico del peso softmax)
lls = np.array([ll for _, ll in CACHE.values()])
print(f"\nloglik: rango medio entre semillas del mismo pozo = {np.mean(lls.max(1)-lls.min(1)):.1f} "
      f"| std medio = {np.mean(lls.std(1)):.1f}")
