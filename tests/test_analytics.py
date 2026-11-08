import numpy as np
import pandas as pd
import pytest
from workbench import forecast as FC, regimes as RG, relationships as RL


def monthly_prices(n=360, seed=0, mu=0.006, sd=0.04):
    idx = pd.date_range("1995-01-31", periods=n, freq="ME")
    r = np.random.default_rng(seed).normal(mu, sd, n)
    return pd.Series(100 * np.cumprod(1 + r), index=idx), idx


def persistent_labels(idx, rng, stay=0.9, states=("A", "B")):
    lab, cur = [], 0
    for _ in idx:
        if rng.random() > stay:
            cur = 1 - cur
        lab.append(states[cur])
    return pd.Series(lab, index=idx)


# ------------------------------------------------------------------ regimes
def test_quadrants_and_levels():
    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    comp = pd.DataFrame({"Growth": [1, 1, -1, -1.0], "Inflation": [-1, 1, 1, -1.0],
                         "Financial stress": [0, 0, 0, 0.0], "Liquidity": [0, 0, 0, 0.0]}, index=idx)
    lab = RG.classify(comp, pd.DataFrame(index=idx))["Growth/Inflation"]
    assert list(lab) == [RG.QUADRANTS[(True, False)], RG.QUADRANTS[(True, True)], RG.QUADRANTS[(False, True)], RG.QUADRANTS[(False, False)]]


def test_curve_and_policy_labels():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    feats = pd.DataFrame({"T10Y3M": [1.0] * 10 + [-0.5] * 4 + [1.0] * 26, "FEDFUNDS": np.r_[np.full(12, 1.0), np.linspace(1, 4, 28)]}, index=idx)
    comp = pd.DataFrame(0.0, index=idx, columns=["Growth", "Inflation", "Financial stress", "Liquidity"])
    lab = RG.classify(comp, feats)
    assert lab["Yield curve"].iloc[5] == "Normal" and lab["Yield curve"].iloc[11] == "Inverted now"
    assert lab["Yield curve"].iloc[16] == "Was inverted in last 12m" and lab["Yield curve"].iloc[-1] == "Normal"
    assert lab["Policy stance"].iloc[-1] == "Tightening"


def test_composite_min_components():
    Z = pd.DataFrame({"INDPRO": [1.0, np.nan], "PAYEMS": [3.0, np.nan], "ICSA": [1.0, 2.0]})
    g = RG.composite(Z, RG.GROWTH_SIGNS)
    assert g.iloc[0] == pytest.approx((1 + 3 - 1) / 3)         # ICSA enters with a negative sign
    assert np.isnan(g.iloc[1])                                   # only one component -> not reported


def test_current_summary_run_length():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    labels = pd.DataFrame({"X": ["a", "a", "b", "b", "b", "b"]}, index=idx)
    r = RG.current_summary(labels, {}).iloc[0]
    assert r["Current"] == "b" and r["Months in this state"] == 4 and r["Since"] == idx[2]


# ------------------------------------------------------------------ relationships
def test_regime_table_detects_planted_effect():
    rng = np.random.default_rng(1)
    idx = pd.date_range("1990-01-31", periods=480, freq="ME")
    lab = persistent_labels(idx, rng, 0.92)
    r = np.where(lab == "A", 0.015, -0.004) + rng.normal(0, 0.03, len(idx))
    p = pd.Series(100 * np.cumprod(1 + r), index=idx)
    t = RL.regime_table(p, lab.shift(0), h=1, n_perm=500).set_index("Regime")
    assert t.at["A", "Avg 1M forward return"] > t.at["B", "Avg 1M forward return"]
    assert t.at["A", "p-value (luck?)"] < 0.05
    assert t.at["A", "Confidence"] in ("Moderate", "Reasonable")


def test_regime_table_null_false_positive_rate():
    rej, reps = 0, 40
    for sd in range(reps):
        rng = np.random.default_rng(1000 + sd)
        p, idx = monthly_prices(360, seed=sd)
        lab = persistent_labels(idx, rng, 0.93)
        t = RL.regime_table(p, lab, h=3, n_perm=200, seed=sd)
        rej += (t["p-value (luck?)"] < 0.05).any()               # chance either regime looks "special" (2 looks => nominal ~10%)
    assert rej / reps < 0.25


def test_predictive_regression_planted_vs_null():
    rng = np.random.default_rng(5)
    n = 400
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    x = pd.Series(0.0, index=idx)
    for t in range(1, n):
        x.iloc[t] = 0.8 * x.iloc[t - 1] + rng.normal()
    r = 0.004 * x.shift(1).fillna(0).to_numpy() + rng.normal(0.005, 0.03, n)
    p = pd.Series(100 * np.cumprod(1 + r), index=idx)
    res = RL.predictive_regression(x.to_frame("x"), p, h=1)
    assert res["ok"] and res["oos_r2"] > 0.01 and "beat" in res["verdict"]
    assert res["in_sample"].t[1] > 3
    # null: no relation
    bad = 0
    for sd in range(15):
        p0, idx0 = monthly_prices(400, seed=sd)
        x0 = pd.Series(np.random.default_rng(900 + sd).normal(size=400), index=idx0)
        rr = RL.predictive_regression(x0.to_frame("x"), p0, h=3)
        bad += "beat" in rr["verdict"]
    assert bad <= 2


def test_lead_lag_finds_planted_lag():
    rng = np.random.default_rng(6)
    n = 420
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    x = pd.Series(rng.normal(size=n), index=idx)
    r = 0.01 * x.shift(4).fillna(0).to_numpy() + rng.normal(0, 0.02, n)       # indicator leads by 4 months
    p = pd.Series(100 * np.cumprod(1 + r), index=idx)
    ll = RL.lead_lag(x, p, h=1, max_lag=8).set_index("Indicator lead (months)")
    # forward return from t reflects x_{t+1-4}=x_{t-3}: strongest when x is lagged by 3 relative to t
    assert ll["t-stat (HAC)"].abs().idxmax() == 3


def test_macro_betas_recovers_loading():
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    d = pd.DataFrame({"d_yield": rng.normal(0, .3, n), "oil": rng.normal(0, .08, n)}, index=idx)
    r = -0.02 * d["d_yield"] + 0.1 * d["oil"] + rng.normal(0.004, .01, n)
    p = pd.Series(100 * np.cumprod(1 + r), index=idx)
    # first return is NaN in pct_change; betas should be recovered within noise
    b = RL.macro_betas(p, d)
    assert b.at["d_yield", "coef"] == pytest.approx(-0.02, abs=0.006) and b.at["oil", "coef"] == pytest.approx(0.1, abs=0.03)
    assert b.at["oil", "Significant (|t|>2)"]


# ------------------------------------------------------------------ forecasts
def planted_world(seed, n=420, strength=0.012):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.85 * z[t - 1] + rng.normal()
    r = np.r_[0, strength * z[:-1]] + rng.normal(0.004, 0.03, n)
    return pd.Series(100 * np.cumprod(1 + r), index=idx), pd.DataFrame({"signal": z}, index=idx)


def test_label_construction():
    p, idx = monthly_prices(120, seed=1)
    fwd = RL.forward_returns(p, 1)
    lab = FC.make_labels(fwd, p, 1, "fixed", 0.02)
    assert set(lab.dropna().unique()) <= {0.0, 1.0, 2.0}
    assert (lab[fwd > 0.02] == 2).all() and (lab[fwd < -0.02] == 0).all() and (lab[fwd.abs() <= 0.02] == 1).all()
    assert np.isnan(lab.iloc[-1])                                   # future unknown -> no label


def test_forecast_finds_planted_skill():
    p, X = planted_world(3)
    res = FC.run_forecast(p, X, None, h=1, use_price=False, min_train=72)
    assert res.ok and res.scores["brier_skill"] > 0.0
    assert res.scores["gain_ci"][0] > 0 and "beat the naive" in res.verdict
    assert res.model_probs.sum() == pytest.approx(1.0) and res.model_probs["Up"] > 0.34 or res.model_probs["Down"] > 0.34


def test_forecast_noise_gets_no_skill_verdict():
    wrong = 0
    reps = 14
    for sd in range(reps):
        p, idx = monthly_prices(420, seed=300 + sd)
        X = pd.DataFrame({"noise": np.random.default_rng(sd).normal(size=420)}, index=idx)
        res = FC.run_forecast(p, X, None, h=3, use_price=False, min_train=72)
        assert res.ok
        wrong += "beat the naive" in res.verdict
    assert wrong <= 2


def test_forecast_has_no_leakage():
    """Probabilities for months <= T must not change when the data AFTER T are changed."""
    p, X = planted_world(4, n=300)
    cut = p.index[200]
    p2, X2 = p.copy(), X.copy()
    rng = np.random.default_rng(1)
    fut = p2.index > cut
    p2[fut] = p2[fut] * np.exp(np.cumsum(rng.normal(0, 0.1, fut.sum())))
    X2.loc[fut, "signal"] = rng.normal(size=fut.sum()) * 5
    h = 3
    a = FC.run_forecast(p, X, None, h=h, use_price=True, min_train=72)
    b = FC.run_forecast(p2, X2, None, h=h, use_price=True, min_train=72)
    oa, ob = a.oos.loc[:cut], b.oos.loc[:cut]
    common = oa.index.intersection(ob.index)
    assert len(common) > 50
    assert np.allclose(oa.loc[common, ["p_Down", "p_Sideways", "p_Up"]], ob.loc[common, ["p_Down", "p_Sideways", "p_Up"]], atol=1e-9)


def test_forecast_regime_table_and_short_history():
    p, X = planted_world(5)
    lab = pd.Series(np.where(X["signal"] > 0, "pos", "neg"), index=X.index)
    res = FC.run_forecast(p, X, lab, h=3)
    rt = res.regime_rate
    assert rt is not None and rt["Probability"].sum() == pytest.approx(1.0) and (rt["Low"] <= rt["Probability"] + 1e-9).all()
    short = FC.run_forecast(p.iloc[:60], X.iloc[:60], None, h=3)
    assert not short.ok and "months" in short.reason
