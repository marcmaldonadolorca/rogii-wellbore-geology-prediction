"""Postproceso sobre la prediccion base: cada tecnica se barre y se reporta su
ganancia POR SEPARADO, y luego encadenada.

  1) rampa de continuidad: p + d0*exp(-md_since/tau), d0 = TVT(PS) - p(primer punto)
  2) rampa hacia el geometrico: (1-r)*G + r*p,  r = 1-exp(-md_since/tau)
  3) shrink alpha hacia la capa plana: flat + alpha*(p-flat), flat = tv0 + (z0-Z)
  4) Savitzky-Golay por pozo
  5) proyeccion robusta IRLS de U = TVT+Z con polinomio grado 3-4 en MD normalizado

Uso: python research/ad05_post.py <cache.npz> [w_sup]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adlib import Cache, report  # noqa: E402


def sig(x, d0, s):
    return 1.0 / (1.0 + np.exp(np.clip((x - d0) / s, -50, 50)))


def per_well(c):
    off = np.concatenate([[0], np.cumsum(c.lens)])
    return [slice(off[i], off[i + 1]) for i in range(c.NW)]


def savgol_wells(p, sl, win, order):
    out = p.copy()
    for s in sl:
        n = s.stop - s.start
        w = min(win if win % 2 else win + 1, n if n % 2 else n - 1)
        if w > order + 1:
            out[s] = savgol_filter(p[s], w, order)
    return out


def irls_poly(p, Z, MD, sl, deg=3, iters=4, cval=4.0):
    """Ajusta U = p+Z con polinomio robusto (IRLS Huber) en MD normalizado."""
    out = p.copy()
    for s in sl:
        u = p[s] + Z[s]
        x = MD[s]
        if len(x) < deg + 5 or x[-1] == x[0]:
            continue
        t = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
        V = np.vander(t, deg + 1)
        w = np.ones(len(t))
        for _ in range(iters):
            W = w[:, None]
            beta, *_ = np.linalg.lstsq(V * W, u * w, rcond=None)
            r = u - V @ beta
            sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            w = 1.0 / np.maximum(1.0, np.abs(r) / (cval * sc))
        out[s] = V @ beta - Z[s]
    return out


def main():
    cpath = sys.argv[1]
    c = Cache(cpath)
    sl = per_well(c)
    w_sup = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    base = w_sup * c.S + (1 - w_sup) * c.P
    print(f"== {Path(cpath).name} ==")
    REF = report(c, f"base fija {w_sup:.2f}S+{1-w_sup:.2f}P", base)

    tv0 = c.meta.tv0.values
    z0 = c.meta.z0.values
    first = np.array([s.start for s in sl])
    flat = c.expand(tv0) + (c.expand(z0) - c.Zv)

    print("\n1) rampa de continuidad  p + d0*exp(-md/tau)")
    d0 = tv0 - base[first]
    print(f"   |d0| mediana {np.median(np.abs(d0)):.2f} ft  p90 {np.quantile(np.abs(d0),.9):.2f}")
    for tau in (100, 250, 500, 1000, 2000, 4000):
        report(c, f"   tau={tau}", base + c.expand(d0) * np.exp(-c.md_since / tau), REF)

    print("\n2) rampa hacia el geometrico  (1-r)G + r*base")
    for tau in (100, 250, 500, 1000, 2000, 4000):
        r = 1 - np.exp(-c.md_since / tau)
        report(c, f"   tau={tau}", (1 - r) * c.G + r * base, REF)

    print("\n3) shrink alpha hacia la capa plana")
    for a in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1):
        report(c, f"   alpha={a}", flat + a * (base - flat), REF)

    print("\n4) Savitzky-Golay por pozo")
    for win in (51, 151, 401, 1001):
        for order in (2, 3):
            report(c, f"   win={win} order={order}", savgol_wells(base, sl, win, order), REF)

    print("\n5) proyeccion IRLS de U=TVT+Z, polinomio en MD")
    for deg in (2, 3, 4, 5):
        pr = irls_poly(base, c.Zv, c.MDv, sl, deg=deg)
        report(c, f"   grado {deg} (sustituye)", pr, REF)
        for lam in (0.3, 0.5, 0.7):
            report(c, f"   grado {deg} mezcla lam={lam}", (1 - lam) * base + lam * pr, REF)


if __name__ == "__main__":
    main()
