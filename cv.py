"""Harness de validacion local para ROGII Wellbore Geology Prediction.

Evalua una funcion de prediccion sobre pozos de train held-out simulando
EXACTAMENTE el formato test: el horizontal solo trae MD,X,Y,Z,GR,TVT_input
(TVT_input = NaN desde el punto PS), mas el typewell (TVT,GR,Geology).
Metrica identica a la oficial: RMSE pooled de dTVT (pies) sobre todos los
puntos predichos de todos los pozos.

Uso:
    from cv import evaluate

    def predict(df_h, df_t):
        # df_h: DataFrame con MD,X,Y,Z,GR,TVT_input (NaN tras PS)
        # df_t: typewell con TVT,GR,Geology
        # devolver array de TVT predicho para las filas post-PS
        # (se acepta tambien el array de longitud completa; se recorta)
        ...

    res = evaluate(predict)          # los 770 pozos (~1-2 min)
    res = evaluate(predict, k=150)   # subconjunto fijo, estratificado por n_pred
    res["rmse"]           # metrica oficial local, pooled
    res["rmse_lb_proxy"]  # pooled solo en pozos con n_pred >= 5147
    res["lb_estimate"]    # rmse * factor de calibracion del envio real
    res["per_well"]       # DataFrame well,n_pred,rmse ordenado por rmse desc

CLI (smoke test con el baseline):  python cv.py [k]

Como leer los numeros (research/gap_analysis.py, 2026-07-30):
  - El baseline dio 39.43 local (770 pozos) y 46.615 en LB publico. El gap NO
    es ruido de muestreo: bootstrap de 200 pozos da P(RMSE>=46.6) = 0.003.
  - El LB lo reproducen los pozos con segmento de prediccion largo:
    n_pred>=5147 (308 pozos) -> 46.24 ; ps_frac<=0.247 (308) -> 46.49.
  - Por eso `rmse_lb_proxy` es el numero a vigilar para anticipar el LB, y
    `rmse` (todos los pozos) el criterio para comparar modelos entre si.
  - `lb_estimate` usa el factor 46.615/39.429 medido con UN envio; es solo
    orientativo y hay que recalibrarlo cuando haya un segundo envio real.
Regla para los 2 envios finales: elegir por `rmse` con `rmse_lb_proxy` como
desempate; no perseguir el LB publico (composicion desconocida).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "data/raw"
TEST_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
SAMPLE_TEST_IDS = {"000d7d20", "00bbac68", "00e12e8b"}  # copias de train en test/
LB_PROXY_NPRED = 5147          # p60 de n_pred; pooled baseline = 46.24 ~ LB 46.615
LB_CALIBRATION = 46.615 / 39.429  # envio 2026-07-30 (baseline dip_win=500)
SEED = 42


def well_ids():
    """Los 770 pozos evaluables (train menos las 3 copias del test de muestra)."""
    ids = sorted(p.name.split("__")[0] for p in (RAW / "train").glob("*__horizontal_well.csv"))
    return [w for w in ids if w not in SAMPLE_TEST_IDS]


def load_well(wid):
    """(df_horizontal_completo, df_typewell, cut). cut = indice del punto PS."""
    df = pd.read_csv(RAW / "train" / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(RAW / "train" / f"{wid}__typewell.csv")
    m = df.TVT_input.isna()
    cut = int(m.idxmax()) if m.any() else len(df)
    return df, tw, cut


def _select(ids, k, seed):
    """Subconjunto de k pozos representativo: sistematico sobre el ranking de
    n_pred (usa research/ps_stats.csv si existe; si no, aleatorio con seed)."""
    stats_f = Path(__file__).resolve().parent / "research/ps_stats.csv"
    if stats_f.exists():
        st = pd.read_csv(stats_f).set_index("well").reindex(ids)
        order = st.n_pred.sort_values().index.to_list()
        pick = np.round(np.linspace(0, len(order) - 1, min(k, len(order)))).astype(int)
        return sorted(order[i] for i in pick)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ids, size=min(k, len(ids)), replace=False))


def _takes_wid(fn):
    """True si fn admite un tercer parametro posicional (el id del pozo)."""
    import inspect
    try:
        p = [q for q in inspect.signature(fn).parameters.values()
             if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
        return len(p) >= 3
    except (TypeError, ValueError):
        return False


def evaluate(predict_fn, k=None, seed=SEED, verbose=True):
    """Evalua predict_fn(df_h, df_t) -> array sobre k pozos held-out (None = todos).

    La seleccion con k fijo es determinista: dos modelos evaluados con el
    mismo k son comparables pozo a pozo (muestras pareadas).
    """
    ids = well_ids()
    if k is not None:
        ids = _select(ids, k, seed)

    rows, sse_all, n_all = [], 0.0, 0
    for wid in ids:
        df, tw, cut = load_well(wid)
        if cut < 20 or cut >= len(df):  # sin historia pre-PS o nada que predecir
            continue
        df_h = df[TEST_COLS].copy()  # formato test exacto, sin columnas de train
        # si predict_fn acepta un tercer argumento, se le pasa el id del pozo
        # (necesario para excluirse a si mismo en interpoladores espaciales)
        args = (df_h, tw, wid) if _takes_wid(predict_fn) else (df_h, tw)
        pred = np.asarray(predict_fn(*args), dtype=float)
        if len(pred) == len(df):
            pred = pred[cut:]
        n_pred = len(df) - cut
        assert len(pred) == n_pred, f"{wid}: predict devolvio {len(pred)} valores, esperaba {n_pred}"
        d = df.TVT.values[cut:] - pred
        assert np.isfinite(d).all(), f"{wid}: prediccion con NaN/inf"
        sse = float((d**2).sum())
        rows.append({"well": wid, "n_pred": n_pred, "sse": sse,
                     "rmse": np.sqrt(sse / n_pred)})
        sse_all += sse
        n_all += n_pred

    per = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    rmse = float(np.sqrt(sse_all / n_all))
    proxy = per[per.n_pred >= LB_PROXY_NPRED]
    rmse_proxy = float(np.sqrt(proxy.sse.sum() / proxy.n_pred.sum())) if len(proxy) else float("nan")
    res = {"rmse": rmse, "rmse_lb_proxy": rmse_proxy, "lb_estimate": rmse * LB_CALIBRATION,
           "n_wells": len(per), "n_points": n_all, "per_well": per}
    if verbose:
        print(f"pozos={res['n_wells']} puntos={n_all}")
        print(f"RMSE local (metrica oficial):  {rmse:7.3f} ft")
        print(f"RMSE lb-proxy (n_pred>=p60):   {rmse_proxy:7.3f} ft  [{len(proxy)} pozos]")
        print(f"LB estimado (calibracion x{LB_CALIBRATION:.3f}): {res['lb_estimate']:7.3f} ft")
        print(f"por pozo: mediana {per.rmse.median():.2f} | p90 {per.rmse.quantile(.9):.2f} "
              f"| max {per.rmse.max():.2f} ({per.well.iloc[0]})")
    return res


if __name__ == "__main__":
    import sys

    from baseline import cut_of, predict as _bpred

    def _baseline(df_h, df_t, dip_win=500):
        return _bpred(df_h, cut_of(df_h), dip_win)

    k = int(sys.argv[1]) if len(sys.argv) > 1 else None
    evaluate(_baseline, k=k)
