import dataclasses
import numpy as np
import pandas as pd
import pytest
from workbench import features as F
from workbench import portfolio as PF
from workbench.backtest import BacktestConfig, run_backtest, run_grid
from workbench.portfolio import PortfolioParams, build_portfolio, cap_weights
from helpers import synth_prices, panel_of


# ----------------------------------------------------------------------------- caps
def test_cap_weights_properties_random():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 12))
        raw = rng.uniform(0.01, 5, n)
        cl = rng.integers(1, 4, n)
        sc = float(rng.uniform(0.1, 1.0)); cc = float(rng.uniform(0.2, 1.0))
        w = cap_weights(raw, cl, sc, cc)
        assert (w >= -1e-12).all() and w.sum() <= 1 + 1e-9
        assert (w <= sc + 1e-9).all()
        for c in np.unique(cl):
            assert w[cl == c].sum() <= cc + 1e-9
        # if feasible (caps allow 100%) the sum must be 1
        feasible = min(n * sc, sum(min(cc, sc * (cl == c).sum()) for c in np.unique(cl)))
        if feasible >= 1 + 1e-9:
            assert w.sum() == pytest.approx(1.0, abs=1e-8)


def test_cap_weights_known_case():
    w = cap_weights(np.array([6., 2., 1., 1.]), np.array([1, 2, 3, 4]), 0.4, 1.0)
    # .6/.2/.1/.1 -> cap first at .4 -> residual .2 shared in proportion to remaining room (0,.2,.3,.3)
    assert np.allclose(w, [0.4, 0.25, 0.175, 0.175])


def test_cap_infeasible_leaves_residual_unallocated():
    w = cap_weights(np.array([1., 1.]), np.array([1, 2]), 0.3, 1.0)
    assert np.allclose(w, [0.3, 0.3])


# ----------------------------------------------------------------------------- selection
def _snap(rows):
    df = pd.DataFrame(rows).set_index("t")
    for c, d in [("Vol_63D", .15), ("DownsideVol_63D", .1), ("ATRpct_20D_proxy", .01)]:
        if c not in df: df[c] = d
    return df


def test_hysteresis_keeps_held_name_that_fails_entry_filters():
    snap = _snap([
        dict(t="A", MomentumScore=90, MomentumPercentile=95, EligibleSignal=True, TrendGate=True),
        dict(t="B", MomentumScore=70, MomentumPercentile=70, EligibleSignal=False, TrendGate=True),   # held, weak but above exit
        dict(t="C", MomentumScore=40, MomentumPercentile=30, EligibleSignal=False, TrendGate=True),   # held, below exit -> dropped
        dict(t="D", MomentumScore=30, MomentumPercentile=20, EligibleSignal=False, TrendGate=False),
    ])
    idx = pd.bdate_range("2025-01-01", periods=200)
    rets = pd.DataFrame(np.random.default_rng(0).normal(0, .01, (200, 4)), index=idx, columns=list("ABCD"))
    p = PortfolioParams(active_fraction=0.5, n_active=3, entry_pct=80, exit_pct=60, sizing="Equal weight", max_single=1, max_cluster=1)
    port, _ = build_portfolio(snap, rets, "BENCH", p, held=["B", "C"])
    got = set(port.Yahoo_Ticker) - {"BENCH"}
    assert got == {"A", "B"}                         # B retained (v7.8 silently dropped it), C exited
    assert port.PortfolioWeight.sum() == pytest.approx(1.0)
    assert port.set_index("Yahoo_Ticker").at["B", "Role"].startswith("Held")
    # fresh start: B is not eligible, so only A
    port2, _ = build_portfolio(snap, rets, "BENCH", p, held=[])
    assert set(port2.Yahoo_Ticker) - {"BENCH"} == {"A"}


def test_no_candidates_means_all_benchmark_or_cash():
    snap = _snap([dict(t="A", MomentumScore=1, MomentumPercentile=10, EligibleSignal=False, TrendGate=False)])
    rets = pd.DataFrame({"A": [0.0] * 30}, index=pd.bdate_range("2025-01-01", periods=30))
    p = PortfolioParams()
    port, _ = build_portfolio(snap, rets, "BENCH", p)
    assert list(port.Yahoo_Ticker) == ["BENCH"] and port.PortfolioWeight.iloc[0] == 1.0
    port, _ = build_portfolio(snap, rets, "BENCH", dataclasses.replace(p, residual_to="Cash"))
    assert list(port.Yahoo_Ticker) == [PF.CASH]


def test_cluster_diversification_prefers_new_cluster():
    idx = pd.bdate_range("2025-01-01", periods=150)
    base = np.random.default_rng(1).normal(0, .01, 150)
    rets = pd.DataFrame({"A": base, "B": base + np.random.default_rng(2).normal(0, .0005, 150),
                         "C": np.random.default_rng(3).normal(0, .01, 150)}, index=idx)
    snap = _snap([dict(t="A", MomentumScore=90, MomentumPercentile=99, EligibleSignal=True, TrendGate=True),
                  dict(t="B", MomentumScore=89, MomentumPercentile=98, EligibleSignal=True, TrendGate=True),
                  dict(t="C", MomentumScore=50, MomentumPercentile=85, EligibleSignal=True, TrendGate=True)])
    p = PortfolioParams(active_fraction=1.0, n_active=2, sizing="Equal weight", max_single=1, max_cluster=1)
    port, corr = build_portfolio(snap, rets, "BENCH", p)
    assert set(port.Yahoo_Ticker) - {"BENCH"} == {"A", "C"}      # B is a near-clone of A, so C takes the slot
    assert corr.loc["A", "B"] > 0.95


# ----------------------------------------------------------------------------- backtest
def cfg_for(prices, **kw):
    base = dict(start=prices.index[300], end=prices.index[-1], benchmark="BENCH", freq="Monthly", lag_days=1, cost_bps=10.0,
                ma_days=100, min_history_days=126,
                portfolio=PortfolioParams(active_fraction=0.5, n_active=4, entry_pct=70, exit_pct=50, corr_days=63))
    base.update(kw)
    return BacktestConfig(**base)


@pytest.fixture(scope="module")
def trending():
    # first 10 names have real positive drift -> a working momentum strategy should find them
    d = [0.0006] * 10 + [0.0] * 20
    return synth_prices(30, 8, seed=11, drifts=d)


def test_backtest_reconciles_exactly(trending):
    res = run_backtest(panel_of(trending), cfg_for(trending))
    per, al, w = res.periods, res.alloc, res.wealth
    # 1) contributions add up to gross period return
    contrib = al.groupby("Decision").Contribution.sum()
    assert np.allclose(contrib.values, per.set_index("Decision").GrossReturn.values, atol=1e-12)
    # 2) weights sum to 1
    assert np.allclose(al.groupby("Decision").Weight.sum().values, 1.0)
    # 3) net = (1-cost)(1+gross)-1 and compounding the periods reproduces the daily wealth curve
    assert np.allclose(per.NetReturn, (1 - per.Cost) * (1 + per.GrossReturn) - 1)
    assert w.Strategy.iloc[-1] == pytest.approx(np.prod(1 + per.NetReturn), rel=1e-12)
    # 4) wealth on each period end equals running product
    run = np.cumprod(1 + per.NetReturn.values)
    for k, d in enumerate(per.End):
        # the rebalance day shows the value AFTER that day's trading cost (next period's cost)
        nxt_cost = per.Cost.iloc[k + 1] if k + 1 < len(per) else 0.0
        assert w.Strategy.loc[d] == pytest.approx(run[k] * (1 - nxt_cost), rel=1e-12)
    # 5) benchmark curve matches raw prices
    b = trending["BENCH"]
    assert w.Benchmark.iloc[-1] == pytest.approx(b.loc[w.index[-1]] / b.loc[w.index[1] if False else per.Start.iloc[0]], rel=1e-12)


def test_cost_arithmetic_is_both_legs():
    pr = synth_prices(6, 4, seed=3, drifts=[0.002] + [0.0] * 5)
    pp = PortfolioParams(active_fraction=1.0, n_active=1, entry_pct=0, exit_pct=0, sizing="Equal weight", max_single=1, max_cluster=1)
    res = run_backtest(panel_of(pr), cfg_for(pr, ma_days=60, min_history_days=60, portfolio=pp, cost_bps=10.0))
    first = res.periods.iloc[0]
    held = res.alloc[res.alloc.Decision == first.Decision]
    if "BENCH" not in set(held.Ticker):                      # swap 100% BENCH -> one ETF
        assert first.Traded == pytest.approx(2.0)            # sell 100% + buy 100%
        assert first.Cost == pytest.approx(2.0 * 10 / 1e4)   # 20 bp, not the 10 bp v7.8 charged


def test_zero_active_equals_benchmark_exactly():
    pr = synth_prices(10, 4, seed=4)
    pp = PortfolioParams(active_fraction=0.0)
    res = run_backtest(panel_of(pr), cfg_for(pr, ma_days=60, min_history_days=60, portfolio=pp))
    assert np.allclose(res.wealth.Strategy.values[1:], res.wealth.Benchmark.values[1:], rtol=1e-12)
    assert res.periods.Cost.sum() == 0


def test_execution_lag_shifts_start_date():
    pr = synth_prices(12, 4, seed=5)
    p = panel_of(pr)
    r0 = run_backtest(p, cfg_for(pr, lag_days=0, ma_days=60, min_history_days=60))
    r1 = run_backtest(p, cfg_for(pr, lag_days=1, ma_days=60, min_history_days=60))
    pos = {d: i for i, d in enumerate(p.index)}
    assert all(pos[b] - pos[a] == 1 for a, b in zip(r0.periods.Start, r1.periods.Start))
    assert (r0.periods.Decision == r1.periods.Decision).all()


def test_no_lookahead_future_scramble_does_not_change_past_decisions(trending):
    """Scramble every price after a cut-off date. Decisions made on or before the cut-off must not change."""
    cut = trending.index[-300]
    pr2 = trending.copy()
    rng = np.random.default_rng(99)
    fut = pr2.index > cut
    pr2.loc[fut] = pr2.loc[fut] * np.exp(rng.normal(0, 0.05, (fut.sum(), pr2.shape[1])))
    cfg = cfg_for(trending)
    a = run_backtest(panel_of(trending), cfg).alloc
    b = run_backtest(panel_of(pr2), cfg).alloc
    early = a[a.Decision <= cut]
    early2 = b[b.Decision <= cut]
    m = early.merge(early2, on=["Decision", "Ticker"], suffixes=("_a", "_b"), how="outer")
    assert len(early) > 20
    assert m.Weight_a.notna().all() and m.Weight_b.notna().all()
    assert np.allclose(m.Weight_a, m.Weight_b, atol=1e-12)


def test_strategy_finds_planted_momentum(trending):
    res = run_backtest(panel_of(trending), cfg_for(trending, cost_bps=5))
    m = res.metrics()
    assert m.loc["CAGR", "Strategy"] > m.loc["CAGR", "Benchmark"]
    sig = res.significance()
    assert sig["mean_monthly_active"] > 0


def test_pure_noise_is_not_declared_significant():
    """Across independent noise universes the 'is it luck?' test must rarely cry wolf."""
    wins = 0
    reps = 12
    for sd in range(reps):
        pr = synth_prices(30, 8, seed=100 + sd)
        res = run_backtest(panel_of(pr), cfg_for(pr, cost_bps=10))
        sig = res.significance(seed=sd)
        wins += sig["p_not_better"] < 0.05
    assert wins <= 2                                   # nominal 5%: allow slack for 12 trials


def test_missing_prices_are_flagged_not_hidden():
    pr = synth_prices(20, 6, seed=8, drifts=[0.0008] * 5 + [0] * 15)
    # kill the data of the strongest names for the last year -> they were selected earlier and then vanish
    pr.iloc[-250:, :5] = np.nan
    res = run_backtest(panel_of(pr), cfg_for(pr))
    assert not res.flags.empty or res.alloc.Ticker.isin(["E00","E01","E02","E03","E04"]).sum() == 0


def test_grid_runs_and_deflates():
    pr = synth_prices(20, 6, seed=9, drifts=[0.0005] * 6 + [0] * 14)
    cfg = cfg_for(pr)
    g = run_grid(panel_of(pr), cfg, [60, 100], [70, 80], ["Equal weight", "Realized volatility"])
    assert len(g) == 8 and g.CAGR.notna().all()
    assert g.DeflatedSharpeProb.between(0, 1).all()
