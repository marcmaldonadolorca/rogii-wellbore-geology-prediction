"""Fase 3 de la fusion superficie-en-PF: cierre de la linea.

Pendiente tras fase1 (sig10=11.050 mejor fusion, blend25_ref=11.492) y fase2
(sig10+blend15=10.918 mejor global; init_prior y kappa EMPEORAN; k=150 de
sig10+blend15=12.017 pero el log se corto sin las referencias):

  1) k=60: sigma fino (8/12/14), peso del blend fino (10/20), adaptativo suave
     (8+0.004*nn) y MEDIA MULTI-SEED del PF fusionado (S=8) — reduce el ruido
     de filtro sin tocar la matematica (la linea multiseed del hermano pondera
     el PF SIN fusion; esto es distinto: promedia el PF fusionado).
  2) k=150: confirmacion pareada del mejor puro, mejor+blend, ms8 si ayuda,
     y las referencias blend25_ref y pf_base_seed0 que faltan.

Todo con seed=0 (mismas semillas que fase1/2: comparaciones pareadas).

Uso:  .venv/bin/python research/v4_fusion_fase3.py
"""
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import cv as cvmod                                   # noqa: E402
from cv import evaluate                              # noqa: E402
from pf_publico import _tw                           # noqa: E402
from v4_fusion_pf import Fusion, run_pf_fused        # noqa: E402

cvmod.load_well = lru_cache(maxsize=None)(cvmod.load_well)   # menos IO bajo carga

FUS = Fusion()          # SurfaceField(aniso=16, k=24) + cache de priors LOWO


def make_pred(sigma=None, ab=None, blend=None, S=1):
    """predict(df_h, tw, wid): fusion con sigma fijo o adaptativo, media de S
    semillas (0..S-1) y blend a posteriori opcional. S=1,seed=0 == fase1/2."""
    def predict(df_h, tw, wid=None):
        prior, nn, cut = FUS.prior(df_h, wid)
        up = (prior + df_h.Z.values)[cut:]
        if ab is not None:
            sig = ab[0] + ab[1] * nn[cut:]
        elif sigma is not None:
            sig = np.full(len(up), float(sigma))
        else:
            sig = np.full(len(up), 1e9)
        t, g = _tw(tw)
        acc = np.zeros(len(up))
        for s in range(S):
            acc += np.asarray(run_pf_fused(df_h, t, g, up, sig, seed=s), float)
        out = acc / S
        if blend is not None:
            out = blend * prior[cut:] + (1. - blend) * out
        return out
    return predict


def run(name, kw, k, results):
    t0 = time.time()
    r = evaluate(make_pred(**kw), k=k, verbose=False)
    dt = time.time() - t0
    print(f"[k={k}] {name:24s} rmse={r['rmse']:7.3f}  proxy={r['rmse_lb_proxy']:7.3f}"
          f"  ({dt:.0f}s, {dt/r['n_wells']:.2f}s/pozo)", flush=True)
    results[name] = r["rmse"]
    return r["rmse"]


def parse_prev():
    """Numeros k=60 de fase1/fase2 (mismo seed y mismo subconjunto: comparables)."""
    out = {}
    for f in ("v4_fase1_k60.log", "v4_fase2.log"):
        for ln in (_HERE / f).read_text().splitlines():
            m = re.match(r"(?:\[k=60\] )?(\S+)\s+rmse=\s*([\d.]+)", ln)
            if m and "CONFIRM" not in ln:
                out[m.group(1)] = float(m.group(2))
    return out


def main():
    prev = parse_prev()
    print("previos k=60:", {n: round(v, 3) for n, v in prev.items()
                            if n in ("pf_base_seed0", "blend25_ref", "sig10",
                                     "ad_6_007", "sig10+blend15")}, flush=True)
    res60 = {"sig10": prev["sig10"], "sig10+blend15": prev["sig10+blend15"]}

    print("\n== fase 3a: k=60 fino ==", flush=True)
    run("sig8", dict(sigma=8.), 60, res60)
    run("sig12", dict(sigma=12.), 60, res60)
    run("sig14", dict(sigma=14.), 60, res60)
    run("ad_8_004", dict(ab=(8., 0.004)), 60, res60)
    run("sig10+blend10", dict(sigma=10., blend=0.10), 60, res60)
    run("sig10+blend20", dict(sigma=10., blend=0.20), 60, res60)
    run("sig10_ms8", dict(sigma=10., S=8), 60, res60)
    run("sig10_ms8+blend15", dict(sigma=10., S=8, blend=0.15), 60, res60)

    sigmas = {"sig8": 8., "sig10": 10., "sig12": 12., "sig14": 14.}
    best_sig = min(sigmas, key=lambda n: res60[n])
    blends = {n: v for n, v in res60.items() if "+blend" in n and "ms8" not in n}
    best_bl = min(blends, key=blends.get)
    w_bl = float(best_bl.split("blend")[1]) / 100.
    print(f"\nmejor sigma puro: {best_sig} ({res60[best_sig]:.3f}) | "
          f"mejor blend: {best_bl} ({res60[best_bl]:.3f})", flush=True)

    # si el sigma ganador no es 10 por margen real, mide su blend tambien
    if best_sig != "sig10" and res60["sig10"] - res60[best_sig] > 0.15:
        run(f"{best_sig}+blend15", dict(sigma=sigmas[best_sig], blend=0.15), 60, res60)

    ms_gain = res60["sig10+blend15"] - res60.get("sig10_ms8+blend15", 99.)
    use_ms = ms_gain > 0.15
    print(f"ms8 aporta {ms_gain:+.3f} -> {'SI' if use_ms else 'NO (ruido)'}", flush=True)

    print("\n== fase 3b: confirmacion k=150 (pareada) ==", flush=True)
    res150 = {}
    run("pf_base_seed0", dict(), 150, res150)
    run("blend25_ref", dict(blend=0.25), 150, res150)
    s_win = sigmas[best_sig]
    run(best_sig, dict(sigma=s_win), 150, res150)
    run(f"{best_sig}+blend{int(w_bl*100)}", dict(sigma=s_win, blend=w_bl), 150, res150)
    if use_ms:
        run(f"{best_sig}_ms8+blend{int(w_bl*100)}",
            dict(sigma=s_win, S=8, blend=w_bl), 150, res150)

    print("\nRESUMEN k=150:", {n: round(v, 3) for n, v in res150.items()}, flush=True)
    print("(fase2 ya midio sig10+blend15 @150 = 12.017)", flush=True)


if __name__ == "__main__":
    main()
