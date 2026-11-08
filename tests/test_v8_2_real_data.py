"""Regression tests built from faults found in the user's real Yahoo download (Nov 2026)."""
import numpy as np
import pandas as pd

from workbench import data_macro as M, quality as Q


def _series(vals, start="2023-01-02"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


def test_permanent_unit_switch_is_rescaled_to_latest_scale():
    """SMEA.L: ~60 (pounds) for years, then ~6,885 (pence).  Old history is multiplied by 100."""
    rng = np.random.default_rng(1)
    old = 60 * np.cumprod(1 + rng.normal(0, 0.005, 200))
    new = old[-1] * 100 * np.cumprod(1 + rng.normal(0, 0.005, 100))
    s = _series(np.r_[old, new])
    fixed, notes = Q.repair_unit_errors(s)
    assert notes and fixed.pct_change().abs().max() < 0.05
    assert abs(fixed.iloc[-1] - s.iloc[-1]) < 1e-9           # newest scale untouched


def test_short_glitch_is_rescaled_back():
    """WOOD.L / INDU.L: a few days quoted ÷100 in the middle of normal prices."""
    base = 1900 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.004, 120))
    v = base.copy(); v[60:64] = v[60:64] / 100
    fixed, notes = Q.repair_unit_errors(_series(v))
    assert len(notes) == 2
    assert np.allclose(fixed.to_numpy(), base)


def test_unclear_jump_is_not_guessed():
    """A ×100 step followed by a real 80% crash is not repaired – it stays a FAIL for the user to see."""
    v = np.r_[np.full(50, 10.0), np.full(50, 1000.0) * 0.2]
    fixed, notes = Q.repair_unit_errors(_series(v))
    assert not notes and (fixed.to_numpy() == v).all()
    assert Q.assess_series(fixed, fixed.index[-1])[0][0] == "FAIL"


def test_repaired_series_reported_and_not_failing():
    base = 1900 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.004, 100))
    v = base.copy(); v[50:52] /= 100
    df = pd.DataFrame({"A.L": v}, index=pd.bdate_range("2023-01-02", periods=100))
    fx, log = Q.repair_unit_frame(df)
    rep = Q.quality_report(fx, repaired=log)
    assert rep.loc[0, "Severity"] == "WARN" and "UNIT_REPAIRED" in rep.loc[0, "Issues"]
    assert not Q.failing_tickers(rep)


def test_flat_history_is_flagged():
    """COFF.L: three years of stale placeholder prices."""
    rng = np.random.default_rng(4)
    flat = np.full(600, 23.0)
    live = 23 * np.cumprod(1 + rng.normal(0, 0.01, 400))
    rep = Q.assess_series(_series(np.r_[flat, live], "2015-01-01"), pd.Timestamp("2018-01-01") + pd.Timedelta(days=1500))
    assert any(c == "FLAT_PERIOD" for _, c, _ in rep)


def test_unfinished_month_is_not_published():
    """A week of VIX in November must not become a 'November' row."""
    vix = pd.Series(np.linspace(15, 30, 470), index=pd.bdate_range("2025-04-01", periods=470), name="VIXCLS")
    vix = vix.loc[:"2026-11-06"]
    f, Z = M.build_monthly_table({"VIXCLS": vix})
    assert f.index.max() == pd.Timestamp("2026-10-31")


def test_longer_history_is_fetched_once_for_cached_tickers(tmp_path):
    """XLK first saved with 12 years must be extended when 25 years are requested – and only once."""
    from workbench import data_prices as DP
    today = pd.Timestamp("2026-11-06")
    calls = []

    def fetcher(tickers, start, end):
        calls.append((tuple(tickers), start))
        idx = pd.bdate_range(max(pd.Timestamp(start), pd.Timestamp("1999-01-04")), min(pd.Timestamp(end), today))
        return pd.DataFrame({t: 10 + 0.002 * (idx - pd.Timestamp('1999-01-01')).days for t in tickers}, index=idx)

    store = DP.PriceStore(tmp_path)
    store.update(["XLK"], history_years=12, fetcher=fetcher, today=today)
    assert store.load()["XLK"].dropna().index.min().year >= 2014
    store.update(["XLK"], history_years=25, fetcher=fetcher, today=today)
    assert store.load()["XLK"].dropna().index.min().year <= 2001          # extended back
    n = len(calls)
    store.update(["XLK"], history_years=25, fetcher=fetcher, today=today)
    assert all(c[1] > "2020" for c in calls[n:])                           # no second full download
    # a genuinely young fund (starts 2018 even when asked for 25 years) is not re-downloaded forever
    def young(tickers, start, end):
        idx = pd.bdate_range("2018-06-01", today)
        return pd.DataFrame({t: 10 + 0.002 * (idx - pd.Timestamp('1999-01-01')).days for t in tickers}, index=idx)
    s2 = DP.PriceStore(tmp_path / "y")
    s2.update(["XLC"], history_years=25, fetcher=young, today=today)
    calls2 = []
    s2.update(["XLC"], history_years=25, fetcher=lambda t, s, e: (calls2.append(s), young(t, s, e))[1], today=today)
    assert all(c > "2020" for c in calls2)


def test_trend_persistence_when_asset_never_crosses_its_average():
    import numpy as np, pandas as pd
    from workbench import cycles as CY
    idx = pd.bdate_range("2018-01-01", periods=600)
    p = pd.Series(np.linspace(100, 20, 600), index=idx)  # always below its 200-day average
    t = CY.trend_persistence(p, 200)
    assert len(t) == 2 and t.loc[t.State.str.startswith("Above"), "Longest spell (days)"].iloc[0] == 0
