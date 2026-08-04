"""V4-TECHO extra: bootstrap pareado a k=60/150/770 (umbral de senal) y
volcado del multiseed150 al CSV de resultados.

Uso:  .venv/bin/python research/v4_techo_extra.py boot
      .venv/bin/python research/v4_techo_extra.py append_ms150 <rmse_1seed_media> <sd> <rmse_media5>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from orclib import Cache  # noqa: E402

OUT_CSV = HERE / "v4_techo_resultados.csv"
W = 0.25


def boot():
    c = Cache()
    z = np.load(HERE / "v4_techo_surf16.npz")
    assert (z["lens"] == c.lens).all()
    c.d["surf16"] = z["surf16"].astype(np.float64)
    c.d["surf5"] = z["surf5"].astype(np.float64)
    c.d["blend16"] = W * c.d["surf16"] + (1 - W) * c.d["ancc"]
    c.d["blend5"] = W * c.d["surf5"] + (1 - W) * c.d["ancc"]

    y = c.d["y"]
    n = c.lens.astype(float)
    pares = [("blend16", "blend5"), ("blend16", "ancc"), ("blend16", "surf16")]
    sses = {}
    for v in {p for par in pares for p in par}:
        e2 = (y - c.d[v]) ** 2
        sses[v] = np.bincount(c.wpt, weights=e2, minlength=len(c.lens))

    rng = np.random.default_rng(0)
    B = 2000
    rows = []
    for a, b in pares:
        for kk in (60, 150, 770):
            sel = rng.integers(0, len(c.lens), size=(B, kk))
            ra = np.sqrt(sses[a][sel].sum(1) / n[sel].sum(1))
            rb = np.sqrt(sses[b][sel].sum(1) / n[sel].sum(1))
            d = ra - rb
            rows.append({"bloque": "varianza", "set": f"boot k={kk}",
                         "variante": f"{a}-{b}", "rmse": float(np.mean(d)),
                         "nota": f"sd_pareada={np.std(d):.3f} => senal si |d|>{2*np.std(d):.2f}"})
            print(f"boot k={kk:3d}  {a}-{b}: media={np.mean(d):+7.3f}  "
                  f"sd={np.std(d):.3f}  (senal 2sd: {2*np.std(d):.2f} ft)")
    df = pd.read_csv(OUT_CSV)
    nuevos = pd.DataFrame(rows)
    clave = ["bloque", "set", "variante"]
    df = df[~df.set_index(clave).index.isin(nuevos.set_index(clave).index)]
    pd.concat([df, nuevos]).to_csv(OUT_CSV, index=False)
    print(f"CSV actualizado: {OUT_CSV}")


def append_ms150(media1, sd1, media5):
    df = pd.read_csv(OUT_CSV)
    fila = {"bloque": "multiseed", "set": "k=150", "variante": "blend16 media5seeds",
            "rmse": float(media5), "nota": f"1seed media={float(media1):.3f} sd={float(sd1):.3f}"}
    df = df[~((df.bloque == "multiseed") & (df["set"] == "k=150"))]
    pd.concat([df, pd.DataFrame([fila])]).to_csv(OUT_CSV, index=False)
    print("CSV actualizado con multiseed k=150")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "boot"
    if modo == "boot":
        boot()
    else:
        append_ms150(*sys.argv[2:5])
