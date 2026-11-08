import numpy as np
import pandas as pd
from workbench import features as F


def synth_prices(n=30, years=8, seed=1, drifts=None, idio=0.008, with_bench=True, end="2026-10-30"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=int(252 * years))
    T = len(idx)
    mkt = rng.normal(0.0003, 0.008, T)
    cols = {}
    for i in range(n):
        mu = (drifts[i] if drifts is not None else 0.0)
        r = rng.uniform(.6, 1.2) * mkt + rng.normal(mu, idio, T)
        cols[f"E{i:02d}"] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    if with_bench:
        cols["BENCH"] = pd.Series(100 * np.exp(np.cumsum(mkt)), index=idx)
    return pd.DataFrame(cols)


def panel_of(prices):
    return F.build_panel(prices)
