"""The vectorised engine must reproduce the ORIGINAL v7.8 numbers (same definitions)."""
import sys, types, os, pathlib
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

ORIG = pathlib.Path(os.environ.get("ORIG_V78", ""))   # path to etf_momentum_dashboard_v7_8.py
pytestmark = pytest.mark.skipif(not ORIG.exists(), reason="set ORIG_V78 to the v7.8 file to run parity tests")

from workbench import features as F


@pytest.fixture(scope="module")
def orig():
    for m in ["streamlit", "plotly", "plotly.express", "plotly.graph_objects", "plotly.subplots", "yfinance"]:
        sys.modules.setdefault(m, MagicMock())
    src = ORIG.read_text().split("# ----------------------------- UI")[0]
    ns = {"__file__": str(pathlib.Path(os.environ["TMPDIR_FAKE"]) / "app.py"), "__name__": "orig"}
    exec(compile(src, "orig_engine", "exec"), ns)
    return types.SimpleNamespace(**ns)


def make_prices(tickers, years=8, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-11-06", periods=int(252 * years))
    T = len(idx)
    mkt = rng.normal(0.0004, 0.009, T)
    cols = {}
    for i, t in enumerate(tickers):
        r = rng.uniform(.3, 1.3) * mkt + rng.normal(rng.normal(0.0001, 0.0002), rng.uniform(.003, .012), T)
        s = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        if i % 9 == 0:
            s.iloc[: int(T * rng.uniform(.2, .6))] = np.nan
        cols[t] = s
    return pd.DataFrame(cols)


def test_snapshot_matches_original(orig):
    import pandas as pd
    master = orig.load_master(pathlib.Path(os.environ["UNIVERSE_XLSX"]))
    bench = "SSAC.L"
    tickers = list(dict.fromkeys(master.Yahoo_Ticker.tolist() + [bench]))
    prices = make_prices(tickers)
    panel = F.build_panel(prices)
    assert panel.prices.shape == prices.shape                      # complete calendar: nothing dropped
    for ma, mh in [(200, 252), (160, 189)]:
        dates = [prices.index[-1], prices.index[-400], prices.index[-900], prices.index[-1500]]
        feats = F.compute_features(panel, dates, F.Params(ma_days=ma, min_history_days=mh, benchmark=bench))
        for d in dates:
            o = orig.score_snapshot(master, prices, bench, d, ma, mh).set_index("Yahoo_Ticker")
            n = feats.snapshot(d)
            assert set(o.index) == set(n.index), (d, set(o.index) ^ set(n.index))
            pairs = [("Excess_6M", "Excess_6M"), ("Excess_12M", "Excess_12M"), ("Excess_6M_Skip1M", "Excess_6M_Skip1M"),
                     ("Excess_12M_Skip1M", "Excess_12M_Skip1M"), ("RelReg_AnnSlope_125D", "RelReg_AnnSlope"),
                     ("RelReg_R2_125D", "RelReg_R2"), ("RelReg_Score_125D", "RelReg_Score"), ("Price", "Price"), ("MA", "MA"),
                     ("Vol_63D", "Vol_63D"), ("DownsideVol_63D", "DownsideVol_63D"), ("ATRpct_20D_proxy", "ATRpct_20D_proxy"),
                     ("MomentumScore", "MomentumScore"), ("MomentumPercentile", "MomentumPercentile")]
            for oc, nc in pairs:
                a, b = o[oc].astype(float), n[nc].astype(float).reindex(o.index)
                both = a.notna() & b.notna()
                assert (a.notna() == b.notna()).all(), (d, oc, a.index[a.notna() != b.notna()].tolist()[:5])
                assert np.allclose(a[both], b[both], rtol=1e-9, atol=1e-12), (d, oc)
            assert (o["TrendGate"].astype(bool) == n["TrendGate"].reindex(o.index)).all()
            assert (o["EligibleSignal"].astype(bool) == n["EligibleSignal"].reindex(o.index)).all()
