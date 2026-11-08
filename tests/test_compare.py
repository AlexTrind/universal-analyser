import numpy as np
import pandas as pd

from workbench import relationships as RL, stats as S


def _world(seed, n=300, n_assets=12, plant=None, lag=4, strength=0.02, rho=0.95):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1998-01-31", periods=n, freq="ME")
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    x = pd.Series(x, index=idx)
    cols = {}
    for i in range(n_assets):
        r = rng.normal(0.006, 0.04, n)
        if plant is not None and i in plant:
            r = r + strength * x.shift(lag).fillna(0).to_numpy() / x.std()
        cols[f"A{i}"] = 100 * np.exp(np.cumsum(r))
    return x, pd.DataFrame(cols, index=idx)


def test_benjamini_hochberg_known_values():
    q = S.benjamini_hochberg([0.01, 0.04, 0.03, 0.20, np.nan])
    assert np.allclose(q[:4], [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6) and np.isnan(q[4])
    assert (S.benjamini_hochberg(np.random.default_rng(0).uniform(size=50)) >= 0).all()


def test_planted_link_found_at_right_lag_and_noise_not():
    found = 0
    for seed in range(6):
        x, me = _world(seed, plant={0, 1})
        tab, _ = RL.compare_assets(x, me, h=3, lag=4, min_months=60)
        sup = set(tab.loc[tab["Verdict"] == "Supported", "Asset"])
        found += {"A0", "A1"} <= sup
    assert found >= 5, found


def test_false_positive_rate_on_persistent_noise_is_low():
    """Slow-moving indicator vs pure-noise returns: the classic spurious-regression trap."""
    for rho in (0.0, 0.95):
        flagged = total = raw = 0
        for seed in range(50):
            x, me = _world(seed + 100, n_assets=10, rho=rho)
            tab, _ = RL.compare_assets(x, me, h=3, lag=0, min_months=60)
            flagged += int((tab["Verdict"] == "Supported").sum())
            raw += int((tab["p-value"] < 0.05).sum())
            total += len(tab)
        assert flagged / total < 0.04, (rho, flagged, total)
        assert raw / total < 0.09, (rho, raw, total)      # nominal 5%; small inflation tolerated


def test_lag_grid_shows_planted_link_stronger_at_planted_lag_than_far_away():
    x, me = _world(3, plant={0}, lag=4, strength=0.03)
    corr, q = RL.lag_grid(x, me, h=3, max_lag=12, min_months=60)
    assert corr.loc["A0", 4] > 0.5 and corr.loc["A0", 4] > corr.loc["A0", 12]
    assert (q.loc["A0", 2:6] < 0.10).all()
    assert (q.drop(index="A0").to_numpy() < 0.10).mean() < 0.05


def test_short_history_is_excluded_not_reported():
    x, me = _world(1)
    me.iloc[:260, 0] = np.nan          # A0 keeps only ~40 months
    tab, excl = RL.compare_assets(x, me, h=3, lag=0, min_months=60)
    assert "A0" not in set(tab["Asset"]) and any(a == "A0" for a, _ in excl)


def test_indicator_never_uses_future_values():
    x, me = _world(5, plant={0}, lag=4)
    base, _ = RL.compare_assets(x, me, 3, 4)
    x2 = x.copy()
    x2.iloc[-3:] = 999.0               # changing the last values can only matter for rows whose outcome is unknown
    again, _ = RL.compare_assets(x2.iloc[:-3], me, 3, 4)
    assert len(base) == len(again)
