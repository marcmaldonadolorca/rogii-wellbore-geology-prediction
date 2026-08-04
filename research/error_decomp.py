"""Descomposicion del error del baseline geometrico (dip_win=500).

Sobre 200 pozos held-out (muestra determinista, sin los 3 de test):
  1. RMSE por pozo vs longitud tramo / dip / curvatura / NaN GR (Spearman).
  2. Curva error-vs-distancia al PS (bins de distancia horizontal).
  3. Saltos bruscos de TVT real (faults) >20 ft y su aporte al RMSE.
  4. Senal GR: corr(GR_horiz post-PS, GR_typewell(TVT_real)).
Pozo a pozo, RAM minima.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TEST_IDS = {"000d7d20", "00bbac68", "00e12e8b"}
DIP_WIN = 500
N_SAMPLE = 200
SEED = 0


def wells():
    return sorted(p.name.split("__")[0] for p in (RAW / "train").glob("*__horizontal_well.csv"))


def cut_of(df):
    m = df.TVT_input.isna()
    return int(m.idxmax()) if m.any() else len(df)


def predict(df, cut, dip_win=DIP_WIN):
    z, tvt_in = df.Z.values, df.TVT_input.values
    flat = tvt_in[cut - 1] + (z[cut - 1] - z)
    step = np.hypot(np.diff(df.X.values, prepend=df.X.values[0]),
                    np.diff(df.Y.values, prepend=df.Y.values[0]))
    hd = np.cumsum(step) - np.cumsum(step)[cut - 1]
    pred = flat.copy()
    slope = 0.0
    if dip_win:
        lo = max(0, cut - dip_win)
        base = tvt_in[lo] + (z[lo] - z[lo:cut])
        resid = tvt_in[lo:cut] - base
        h = hd[lo:cut] - hd[lo]
        if len(h) > 10 and h[-1] != 0:
            slope = np.polyfit(h, resid, 1)[0]
            pred = flat + slope * hd
    return pred[cut:], hd, slope


def azimuth_curv(x, y, cut):
    """Cambio medio absoluto de azimut (grados por 100 ft) en el tramo predicho."""
    dx, dy = np.diff(x[cut - 1:]), np.diff(y[cut - 1:])
    step = np.hypot(dx, dy)
    ok = step > 0.1
    if ok.sum() < 5:
        return 0.0
    az = np.unwrap(np.arctan2(dy[ok], dx[ok]))
    dist = np.cumsum(step[ok])
    if dist[-1] < 100:
        return 0.0
    return float(np.sum(np.abs(np.diff(az))) * 180 / np.pi / (dist[-1] / 100))


def faults_of(tvt, z, thr=20.0, win=5):
    """Saltos de TVT no explicados por Z en ventana de `win` muestras."""
    r = tvt + z  # residuo flat: constante salvo dip suave o fault
    jump = np.abs(r[win:] - r[:-win])
    idx = np.where(jump > thr)[0]
    if len(idx) == 0:
        return []
    # agrupar indices contiguos en eventos
    events, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > win:
            events.append(start)
            start = b
    events.append(start)
    return events


def main():
    rng = np.random.default_rng(SEED)
    all_wells = [w for w in wells() if w not in TEST_IDS]
    sample = sorted(rng.choice(all_wells, size=N_SAMPLE, replace=False))

    rows, pt_hd, pt_err = [], [], []
    fault_pts_sq, fault_pts_n = 0.0, 0
    nofault_pts_sq, nofault_pts_n = 0.0, 0
    gr_corrs = []

    for i, wid in enumerate(sample):
        df = pd.read_csv(RAW / "train" / f"{wid}__horizontal_well.csv")
        cut = cut_of(df)
        if cut < 20 or cut >= len(df):
            continue
        pred, hd, slope = predict(df, cut)
        tvt = df.TVT.values
        err = tvt[cut:] - pred
        hpost = hd[cut:]
        rmse = float(np.sqrt(np.mean(err ** 2)))

        # features
        length = float(hpost[-1]) if len(hpost) else 0.0
        gr_post = df.GR.values[cut:]
        nan_gr = float(np.mean(~np.isfinite(gr_post)))
        curv = azimuth_curv(df.X.values, df.Y.values, cut)

        # faults (en todo el pozo y en el tramo predicho)
        ev = faults_of(tvt, df.Z.values)
        ev_post = [e for e in ev if e >= cut]
        n_faults = len(ev_post)
        if ev_post:
            first = min(ev_post) - cut
            fault_pts_sq += float(np.sum(err[first:] ** 2)); fault_pts_n += len(err) - first
            nofault_pts_sq += float(np.sum(err[:first] ** 2)); nofault_pts_n += first
        else:
            nofault_pts_sq += float(np.sum(err ** 2)); nofault_pts_n += len(err)

        # senal GR (subconjunto: 1 de cada 3 pozos para acotar coste)
        if i % 3 == 0:
            tw = pd.read_csv(RAW / "train" / f"{wid}__typewell.csv")
            twv = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
            g = gr_post.copy()
            t = tvt[cut:]
            ok = np.isfinite(g) & np.isfinite(t) & (t >= twv.TVT.min()) & (t <= twv.TVT.max())
            if ok.sum() > 100 and np.nanstd(g[ok]) > 0:
                gtw = np.interp(t[ok], twv.TVT.values, twv.GR.values)
                if np.std(gtw) > 0:
                    gr_corrs.append((wid, float(np.corrcoef(g[ok], gtw)[0, 1]), int(ok.sum())))

        rows.append((wid, len(err), rmse, length, abs(slope), curv, nan_gr, n_faults))
        pt_hd.append(hpost); pt_err.append(err)

    per = pd.DataFrame(rows, columns=["well", "n", "rmse", "length_ft", "dip_abs",
                                      "curv_deg100ft", "nan_gr", "n_faults"])
    per.to_csv(ROOT / "research/per_well_decomp.csv", index=False)
    hd_all = np.concatenate(pt_hd); err_all = np.concatenate(pt_err)

    print(f"pozos={len(per)} puntos={len(err_all)}")
    print(f"RMSE global muestra: {np.sqrt(np.mean(err_all**2)):.2f} ft | "
          f"mediana por pozo {per.rmse.median():.2f} | p90 {per.rmse.quantile(.9):.2f}")

    print("\n== 1. Spearman rmse-pozo vs features ==")
    for c in ["length_ft", "dip_abs", "curv_deg100ft", "nan_gr", "n_faults"]:
        r, p = stats.spearmanr(per.rmse, per[c])
        print(f"  {c:16s} rho={r:+.3f} p={p:.1e}")

    print("\n== 2. error vs distancia al PS (bins de 1000 ft) ==")
    bins = np.arange(0, min(hd_all.max(), 12000) + 1000, 1000)
    for a, b in zip(bins[:-1], bins[1:]):
        m = (hd_all >= a) & (hd_all < b)
        if m.sum() > 50:
            e = err_all[m]
            print(f"  {a:5.0f}-{b:5.0f} ft: n={m.sum():7d} RMSE={np.sqrt(np.mean(e**2)):7.2f} "
                  f"|err| p50={np.median(np.abs(e)):6.2f} p90={np.quantile(np.abs(e),.9):7.2f} "
                  f"bias={np.mean(e):+7.2f}")

    print("\n== 3. faults (salto TVT+Z > 20 ft en 5 muestras) ==")
    frac = (per.n_faults > 0).mean()
    print(f"  pozos con fault post-PS: {frac*100:.1f}% ({(per.n_faults>0).sum()}/{len(per)})")
    tot_sq = fault_pts_sq + nofault_pts_sq
    print(f"  RMSE puntos tras 1er fault: {np.sqrt(fault_pts_sq/max(fault_pts_n,1)):.2f} ft (n={fault_pts_n})")
    print(f"  RMSE puntos sin fault previo: {np.sqrt(nofault_pts_sq/max(nofault_pts_n,1)):.2f} ft (n={nofault_pts_n})")
    print(f"  aporte al MSE global: faults {fault_pts_sq/tot_sq*100:.1f}% de la suma de cuadrados "
          f"con {fault_pts_n/(fault_pts_n+nofault_pts_n)*100:.1f}% de los puntos")
    print(f"  RMSE global sin pozos con fault: "
          f"{np.sqrt(np.mean(np.concatenate([e for e, r in zip(pt_err, rows) if r[7]==0])**2)):.2f} ft")

    print("\n== 4. senal GR: corr(GR_horiz, GR_typewell(TVT_real)) post-PS ==")
    gc = pd.DataFrame(gr_corrs, columns=["well", "corr", "n"])
    gc.to_csv(ROOT / "research/gr_corr.csv", index=False)
    print(f"  pozos medidos: {len(gc)} | corr mediana {gc['corr'].median():+.3f} | "
          f"p25 {gc['corr'].quantile(.25):+.3f} p75 {gc['corr'].quantile(.75):+.3f}")
    print(f"  corr>0.5: {(gc['corr']>0.5).mean()*100:.0f}% | corr>0.3: {(gc['corr']>0.3).mean()*100:.0f}% | "
          f"corr<0: {(gc['corr']<0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
