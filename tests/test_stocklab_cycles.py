import numpy as np
import pandas as pd
import pytest
from workbench import stocklab as SL, cycles as CY, features as F


def idx(n=800):
    return pd.bdate_range(end="2026-10-30", periods=n)


def test_breadth_known_values():
    i = idx(300)
    up = pd.DataFrame({f"U{k}": 100 * 1.001 ** np.arange(300) for k in range(6)}, index=i)    # always rising
    dn = pd.DataFrame({f"D{k}": 100 * 0.999 ** np.arange(300) for k in range(4)}, index=i)    # always falling
    b = SL.breadth_series(pd.concat([up, dn], axis=1))
    assert b["% above 200-day average"].iloc[-1] == pytest.approx(60.0)
    assert b["% with positive 6M return"].iloc[-1] == pytest.approx(60.0)
    assert b["% at 52-week high"].iloc[-1] == pytest.approx(60.0) and b["% at 52-week low"].iloc[-1] == pytest.approx(40.0)


def test_avg_pairwise_corr_extremes():
    i = idx(400)
    rng = np.random.default_rng(0)
    common = rng.normal(0, .01, 400)
    same = pd.DataFrame({f"S{k}": 100 * np.cumprod(1 + common) for k in range(8)}, index=i)
    assert SL.avg_pairwise_corr(same).iloc[-1] == pytest.approx(1.0, abs=1e-9)
    ind = pd.DataFrame({f"I{k}": 100 * np.cumprod(1 + rng.normal(0, .01, 400)) for k in range(30)}, index=i)
    assert abs(SL.avg_pairwise_corr(ind, window=252).mean()) < 0.06


def test_contribution_sums_to_group_return():
    i = idx(100)
    rng = np.random.default_rng(1)
    p = pd.DataFrame({f"S{k}": 100 * np.cumprod(1 + rng.normal(0.001, .01, 100)) for k in range(7)}, index=i)
    c = SL.contribution(p, i[0], i[-1])
    assert c.Contribution.sum() == pytest.approx((p.iloc[-1] / p.iloc[0] - 1).mean())


def test_factor_regression_recovers_betas_and_attribution():
    rng = np.random.default_rng(2)
    n = 1000
    i = idx(n)
    f = pd.DataFrame(rng.normal(0, .008, (n, 3)), index=i, columns=["Mkt-RF", "SMB", "HML"])
    f["RF"] = 0.0001
    y = 0.0002 + 1.2 * f["Mkt-RF"] - 0.5 * f["SMB"] + 0.3 * f["HML"] + rng.normal(0, .004, n) + f["RF"]
    res = SL.factor_regression(y, f, window=n)
    b = res["betas"]["coef"]
    assert b["Mkt-RF"] == pytest.approx(1.2, abs=0.05) and b["SMB"] == pytest.approx(-0.5, abs=0.07)
    assert res["alpha_annual"] == pytest.approx(0.0002 * 252, abs=0.04)
    # attribution identity: factor contributions + alpha + residual == total excess return
    assert res["contrib"].sum() + res["residual"] == pytest.approx(res["total_excess"], rel=1e-9)
    assert res["r2"] > 0.8


def test_market_regression_beta_alpha():
    rng = np.random.default_rng(3)
    i = idx(900)
    m = pd.Series(rng.normal(0.0004, .01, 900), index=i)
    r = 0.0001 + 0.8 * m + rng.normal(0, .005, 900)
    res = SL.market_regression(r, m)
    assert res["beta"] == pytest.approx(0.8, abs=0.05) and 0.5 < res["r2"] < 0.9
    assert res["idio_vol"] < res["total_vol"]


def test_group_snapshot_ranks_planted_leaders():
    rng = np.random.default_rng(4)
    i = idx(900)
    mkt = rng.normal(0.0003, .008, 900)
    cols = {"IDX": 100 * np.cumprod(1 + mkt)}
    for k in range(12):
        mu = 0.0008 if k < 3 else 0.0
        cols[f"S{k}"] = 100 * np.cumprod(1 + mkt + rng.normal(mu, .006, 900))
    prices = pd.DataFrame(cols, index=i)
    panel = F.build_panel(prices)
    members = pd.DataFrame({"Ticker": [f"S{k}" for k in range(12)], "Name": [f"Stock {k}" for k in range(12)],
                            "Sector": ["A"] * 6 + ["B"] * 6})
    snap = SL.group_snapshot(panel, members, "IDX", i[-1])
    assert set(snap.index[:3]) == {"S0", "S1", "S2"}
    assert snap.Beta_1Y.between(0.3, 1.8).all()
    ss = SL.sector_summary(snap)
    assert ss.loc["A", "Avg 12M"] > ss.loc["B", "Avg 12M"] and ss.Members.sum() == 12


def test_composite_vs_index_normalised():
    i = idx(200)
    p = pd.DataFrame({"A": 100 * 1.001 ** np.arange(200), "B": 100 * 1.0005 ** np.arange(200)}, index=i)
    out = SL.composite_vs_index(p, p["A"])
    assert out.iloc[0].tolist() == [1.0, 1.0, 1.0]


def test_seasonality_detects_planted_january_effect_and_flags_nothing_else():
    rng = np.random.default_rng(5)
    ix = pd.date_range("1980-01-31", periods=45 * 12, freq="ME")
    r = rng.normal(0.005, 0.03, len(ix))
    r[ix.month == 1] += 0.04
    p = pd.Series(100 * np.cumprod(1 + r), index=ix)
    s = CY.seasonality(p).set_index("Month")
    assert s.loc["Jan", "Stands out after multiple-testing?"]
    assert s["Stands out after multiple-testing?"].sum() <= 2


def test_trend_persistence_and_profile_and_years():
    i = idx(1500)
    rng = np.random.default_rng(6)
    p = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, .01, 1500)), index=i)
    tp = CY.trend_persistence(p)
    assert tp["% of time"].sum() == pytest.approx(1.0)
    prof = CY.asset_profile(p)
    assert prof["Worst drawdown"] < 0 and "Return, last 5y (annualised)" in prof
    yr = CY.calendar_year_returns(p)
    assert yr.index.is_monotonic_increasing and np.prod(1 + yr.values) == pytest.approx(p.iloc[-1] / p.iloc[0], rel=1e-9)
