"""Estructura del residuo e = TVT + Z - S_interp (lo que C_well tiene que estimar).

Error del modelo = e_i - c_hat_i en las filas post-PS. Este script mide:
  - reproduccion de la referencia (mediana ultimas 500) desde el cache
  - oraculos: mejor constante por pozo, mejor recta en arco, mejor plano X,Y
  - geometria: extension lateral del prefijo vs del tramo post-PS
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "research" / "cal01_cache_k150.npz"


def load(k=150):
    z = np.load(str(CACHE))
    wids = sorted({key.split("|")[0] for key in z.files})
    out = []
    for w in wids:
        d = {f: z[f"{w}|{f}"] for f in ("md", "x", "y", "z", "tvt", "tin", "s", "nn")}
        d["cut"] = int(z[f"{w}|cut"][0])
        d["wid"] = w
        d["e"] = d["tvt"].astype(np.float64) + d["z"] - d["s"]          # residuo real
        d["ein"] = d["tin"].astype(np.float64) + d["z"] - d["s"]        # conocido pre-PS
        dx = np.diff(d["x"], prepend=d["x"][0]); dy = np.diff(d["y"], prepend=d["y"][0])
        d["arc"] = np.cumsum(np.hypot(dx, dy))
        out.append(d)
    return out


def pooled(errs):
    sse = sum(float((e ** 2).sum()) for e in errs)
    n = sum(len(e) for e in errs)
    return np.sqrt(sse / n)


if __name__ == "__main__":
    W = load()
    print(f"{len(W)} pozos")

    ref, orc_c, orc_lin, orc_plane, zero = [], [], [], [], []
    rows = []
    for d in W:
        c, e = d["cut"], d["e"]
        post = slice(c, len(e))
        lo = max(0, c - 500)
        cref = np.median(d["ein"][lo:c])
        ref.append(e[post] - cref)
        zero.append(e[post] - 0.0)
        orc_c.append(e[post] - e[post].mean())
        # oraculo recta en arco (post-PS)
        u = d["arc"][post] - d["arc"][c - 1]
        A = np.column_stack([np.ones(len(u)), u])
        orc_lin.append(e[post] - A @ np.linalg.lstsq(A, e[post], rcond=None)[0])
        Ap = np.column_stack([np.ones(len(u)), d["x"][post] - d["x"][c - 1],
                              d["y"][post] - d["y"][c - 1]])
        orc_plane.append(e[post] - Ap @ np.linalg.lstsq(Ap, e[post], rcond=None)[0])
        rows.append((d["wid"], c, len(e) - c,
                     float(np.ptp(d["arc"][:c])), float(np.ptp(d["arc"][post])),
                     float(np.std(d["ein"][lo:c])), float(np.std(e[post])),
                     float(e[post].mean() - cref), float(np.median(d["nn"][post]))))

    print(f"\nRMSE pooled post-PS")
    print(f"  c=0 (superficie cruda, sin calibrar) : {pooled(zero):7.3f}")
    print(f"  REFERENCIA mediana ultimas 500       : {pooled(ref):7.3f}")
    print(f"  oraculo mejor constante por pozo     : {pooled(orc_c):7.3f}")
    print(f"  oraculo mejor recta(arco) por pozo   : {pooled(orc_lin):7.3f}")
    print(f"  oraculo mejor plano(X,Y) por pozo    : {pooled(orc_plane):7.3f}")

    import pandas as pd
    df = pd.DataFrame(rows, columns=["well", "cut", "npred", "arc_pre", "arc_post",
                                     "sd_e_pre500", "sd_e_post", "bias", "nn_post"])
    df["rmse_ref"] = [float(np.sqrt((r ** 2).mean())) for r in ref]
    df.to_csv(ROOT / "research" / "cal02_perwell.csv", index=False)
    print("\ngeometria (mediana):")
    print(df[["arc_pre", "arc_post", "sd_e_pre500", "sd_e_post", "nn_post"]].median())
    print("\n|bias| = |media(e post) - c_ref|:")
    print(df.bias.abs().describe([.5, .75, .9]))
    print("\ndescomposicion pooled: sesgo vs dispersion")
    b2 = np.average(df.bias ** 2, weights=df.npred)
    v2 = np.average(df.sd_e_post ** 2, weights=df.npred)
    print(f"  sqrt(E[bias^2]) = {np.sqrt(b2):.3f}   sqrt(E[var_post]) = {np.sqrt(v2):.3f}"
          f"   suma = {np.sqrt(b2 + v2):.3f}")
