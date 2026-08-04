"""¿Que subconjunto de train reproduce el gap local 39.4 -> LB publico 46.6?

Corre el baseline (dip_win=500) pozo a pozo, guarda error por pozo + features
(longitud, ps_frac, distancia a vecino mas cercano, dip, etc.), y luego:
  1. RMSE global pooled por estratos (cuartiles de cada feature).
  2. Bootstrap de subconjuntos de K pozos -> ¿que prob. hay de ver 46.6 por azar?
Salida: research/perwell_baseline.csv + resumen stdout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baseline import RAW, TEST_IDS, cut_of, load, predict, wells  # noqa: E402

DIP_WIN = 500


def main():
    rows = []
    for wid in wells("train"):
        if wid in TEST_IDS:
            continue
        df = load("train", wid)
        cut = cut_of(df)
        if cut < 20 or cut >= len(df):
            continue
        d = df.TVT.values[cut:] - predict(df, cut, DIP_WIN)
        z, tvt = df.Z.values, df.TVT.values
        # dip real post-PS: cuanto se desvia la geologia del modelo plano
        flat = df.TVT_input.values[cut - 1] + (z[cut - 1] - z[cut:])
        rows.append({
            "well": wid, "n": len(df), "cut": cut, "ps_frac": cut / len(df),
            "n_pred": len(d), "sse": float((d**2).sum()),
            "rmse": float(np.sqrt((d**2).mean())),
            "md_total": df.MD.iloc[-1] - df.MD.iloc[0],
            "x": float(df.X.median()), "y": float(df.Y.median()),
            "gr_mean": float(df.GR.mean()),
            "dev_flat_max": float(np.abs(tvt[cut:] - flat).max()),
            "z_range_post": float(z[cut:].max() - z[cut:].min()),
        })
    per = pd.DataFrame(rows)

    # distancia al vecino (pozo) mas cercano en el plano XY
    xy = per[["x", "y"]].values
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    per["nn_dist"] = np.sqrt(d2.min(1))
    per.to_csv(Path(__file__).parent / "perwell_baseline.csv", index=False)

    def pooled(m):
        return float(np.sqrt(per.sse[m].sum() / per.n_pred[m].sum()))

    print(f"pozos={len(per)} | RMSE global pooled = {pooled(per.index >= 0):.3f} ft (objetivo LB: 46.615)")
    print(f"rmse por pozo: mediana {per.rmse.median():.2f} | p90 {per.rmse.quantile(.9):.2f} | max {per.rmse.max():.2f}")
    top = per.nlargest(10, "sse")
    print(f"top-10 pozos por SSE concentran {top.sse.sum()/per.sse.sum():.1%} del error total\n")

    print("=== RMSE pooled por cuartil de cada feature ===")
    for feat in ["n", "ps_frac", "n_pred", "md_total", "nn_dist", "dev_flat_max", "z_range_post", "gr_mean"]:
        qs = per[feat].quantile([.25, .5, .75]).values
        labels, vals = [], []
        edges = [-np.inf, *qs, np.inf]
        for i in range(4):
            m = (per[feat] > edges[i]) & (per[feat] <= edges[i + 1])
            labels.append(f"Q{i+1}")
            vals.append(pooled(m))
        print(f"{feat:>14}: " + "  ".join(f"{l}={v:6.2f}" for l, v in zip(labels, vals)))

    print("\n=== Bootstrap: RMSE pooled de subconjuntos aleatorios de K pozos ===")
    rng = np.random.default_rng(42)
    for k in [50, 100, 200, 400]:
        sims = []
        for _ in range(4000):
            idx = rng.choice(len(per), k, replace=False)
            sims.append(np.sqrt(per.sse.values[idx].sum() / per.n_pred.values[idx].sum()))
        sims = np.array(sims)
        print(f"K={k:>3}: media {sims.mean():6.2f} | std {sims.std():5.2f} | "
              f"p5 {np.percentile(sims,5):6.2f} | p95 {np.percentile(sims,95):6.2f} | "
              f"P(>=46.6) = {(sims>=46.615).mean():.3f}")


if __name__ == "__main__":
    main()
