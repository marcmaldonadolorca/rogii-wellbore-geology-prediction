"""v6-pesos etapa 1: cache de señales post-PS + backtest enmascarado.

Pozos:
  - evaluacion: k60 | k150 (196 pozos), PF post = media S=64 de los npz v4.
  - entrenamiento: ~400 pozos disjuntos, PF post = media S=8 fresca.
Todos llevan backtest enmascarado (PF S=8 sobre el 35% final del prefijo).

Salida: research/v6_pesos_cache.npz + research/v6_pesos_meta.csv
Al final imprime la reproduccion del baseline 0.45/0.55 en k=150 (debe dar
~10.200) como control de alineamiento.
"""
import sys
import time

import numpy as np
import pandas as pd

from v6_pesos_lib import (CACHE_NPZ, META_CSV, W_GLOBAL, make_field,
                          process_well, subsets, train_pool)


def main(n_train=400):
    ids, k60, k150 = subsets()
    eval_ids = sorted(k60 | k150)
    train_ids = train_pool(n_train)
    assert not set(train_ids) & set(eval_ids)
    field = make_field()

    store, meta = {}, []
    t0 = time.time()
    for grupo, wids, src in (("eval", eval_ids, "npz64"), ("train", train_ids, "run8")):
        for j, wid in enumerate(wids):
            out = process_well(wid, field, src)
            if out is None:
                continue
            arrays, m = out
            for k, v in arrays.items():
                store[f"{wid}_{k}"] = np.asarray(v, np.float32)
            meta.append(m)
            if (j + 1) % 25 == 0:
                print(f"  {grupo} {j+1}/{len(wids)}  {time.time()-t0:.0f}s", flush=True)
        print(f"{grupo}: {len(wids)} pozos listos  {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(CACHE_NPZ, **store)
    dfm = pd.DataFrame(meta)
    dfm.to_csv(META_CSV, index=False)
    print(f"guardado {CACHE_NPZ} ({len(dfm)} pozos)  {time.time()-t0:.0f}s", flush=True)

    # control: baseline 0.45*surf + 0.55*pf en k150 con las señales cacheadas
    sse, n = 0.0, 0
    for wid in sorted(k150):
        y = store[f"{wid}_y"].astype(float)
        p = (W_GLOBAL[0] * store[f"{wid}_surf"] + W_GLOBAL[1] * store[f"{wid}_pf"]
             + W_GLOBAL[2] * store[f"{wid}_geom"]).astype(float)
        sse += float(((y - p) ** 2).sum())
        n += len(y)
    print(f"CONTROL k150 blend 0.45/0.55: {np.sqrt(sse/n):.3f} (esperado ~10.200)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    main(n)
