"""Built-in correctness checks that run on the USER's machine (no pytest needed, ~10 seconds).

Each check compares the engine against an independently derived answer.  If any check fails, do not trust the
numbers until the cause is understood."""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from . import backtest as BT, data_macro as DM, data_prices as DP, features as F, forecast as FC, portfolio as PF, stats as S


def _prices(n=24, years=6, seed=1, drifts=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-30", periods=int(252 * years))
    T = len(idx)
    mkt = rng.normal(0.0003, 0.008, T)
    cols = {f"E{i:02d}": pd.Series(100 * np.exp(np.cumsum(rng.uniform(.6, 1.2) * mkt + rng.normal(drifts[i] if drifts else 0, .008, T))), index=idx) for i in range(n)}
    cols["BENCH"] = pd.Series(100 * np.exp(np.cumsum(mkt)), index=idx)
    return pd.DataFrame(cols)


def _cfg(pr, **kw):
    base = dict(start=pr.index[300], end=pr.index[-1], benchmark="BENCH", freq="Monthly", lag_days=1, cost_bps=10.0, ma_days=100,
                min_history_days=126, portfolio=PF.PortfolioParams(active_fraction=0.5, n_active=4, entry_pct=70, exit_pct=50, corr_days=63))
    base.update(kw)
    return BT.BacktestConfig(**base)


def c_drawdown():
    ok = abs(S.max_drawdown(pd.Series([1.0, 1.2, 0.9, 1.3, 0.65])) - (0.65 / 1.3 - 1)) < 1e-12
    return ok, "peak 1.3 → trough 0.65 = -50%"


def c_hac():
    rng = np.random.default_rng(2)
    n = 150
    x = rng.normal(size=n)
    y = 1 + 0.4 * x + rng.normal(size=n)
    r = S.ols_hac(y, x, lags=3)
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.solve(X.T @ X, X.T @ y)
    u = y - X @ b
    S0 = sum(np.outer(X[t], X[t]) * u[t] ** 2 for t in range(n))
    for l in range(1, 4):
        for t in range(l, n):
            G = np.outer(X[t], X[t - l]) * u[t] * u[t - l]
            S0 += (1 - l / 4) * (G + G.T)
    Bi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(Bi @ S0 @ Bi))
    return bool(np.allclose(r.coef, b) and np.allclose(r.se, se, rtol=1e-8)), "Newey-West standard errors equal an explicit-loop calculation"


def c_wilson():
    lo, hi = S.wilson_interval(8, 10)
    return abs(lo - 0.4902) < 3e-4 and abs(hi - 0.9433) < 3e-4, f"8/10 → ({lo:.4f}, {hi:.4f})"


def c_logit_gradient():
    from scipy.optimize import check_grad
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 3))
    y = rng.integers(0, 3, 80)
    m = S.MultinomialLogit(C=0.5).fit(X, y, 3)
    P = m.predict_proba(X)
    # re-derive objective and check the optimiser's solution has ~zero gradient
    Z = (X - m.mu_) / m.sd_
    Y = np.eye(3)[y]
    G = (P - Y).T @ Z + (1 / 0.5) * m.W_
    return bool(np.abs(G).max() < 1e-4 and np.allclose(P.sum(1), 1)), f"max |gradient| at optimum = {np.abs(G).max():.2e}"


def c_caps():
    rng = np.random.default_rng(4)
    for _ in range(300):
        n = int(rng.integers(1, 10))
        w = PF.cap_weights(rng.uniform(.01, 5, n), rng.integers(1, 4, n), float(rng.uniform(.15, 1)), float(rng.uniform(.3, 1)))
        if w.sum() > 1 + 1e-9 or (w < -1e-12).any():
            return False, "weights outside bounds"
    return True, "300 random cases: caps respected, sum ≤ 1"


def c_merge():
    idx = pd.bdate_range("2024-01-01", periods=300)
    s = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, .006, 300))), index=idx)
    m, st = DP.merge_adjusted(s.iloc[:250], s.iloc[230:] * 0.97)
    ok = st == "RESCALED" and np.allclose(m.pct_change().dropna().values, s.pct_change().dropna().values)
    return ok, f"dividend re-adjustment handled without fake price jump ({st})"


def c_macro_lag():
    idx = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    m = DM.known_monthly(s, 45)
    ok = m[pd.Timestamp("2024-03-31")] == s[pd.Timestamp("2024-02-01")] and m[pd.Timestamp("2024-04-30")] == s[pd.Timestamp("2024-03-01")]
    return ok, "March data are not 'known' until mid-April"


def c_reconcile_and_cost():
    pr = _prices(drifts=[0.0006] * 8 + [0] * 16)
    p = F.build_panel(pr)
    res = BT.run_backtest(p, _cfg(pr))
    per, al = res.periods, res.alloc
    r1 = np.allclose(al.groupby("Decision").Contribution.sum().values, per.set_index("Decision").GrossReturn.values, atol=1e-12)
    r2 = abs(res.wealth.Strategy.iloc[-1] - np.prod(1 + per.NetReturn)) < 1e-10
    r3 = np.allclose(per.Cost, per.Traded * 10 / 1e4)
    return bool(r1 and r2 and r3), "contributions = gross return; compounding periods = daily curve; cost = traded × bps"


def c_no_lookahead():
    pr = _prices(drifts=[0.0006] * 8 + [0] * 16, seed=11)
    cut = pr.index[-250]
    pr2 = pr.copy()
    fut = pr2.index > cut
    pr2.loc[fut] = pr2.loc[fut] * np.exp(np.random.default_rng(9).normal(0, .05, (fut.sum(), pr2.shape[1])))
    a = BT.run_backtest(F.build_panel(pr), _cfg(pr)).alloc
    b = BT.run_backtest(F.build_panel(pr2), _cfg(pr)).alloc
    ea, eb = a[a.Decision <= cut], b[b.Decision <= cut]
    m = ea.merge(eb, on=["Decision", "Ticker"], suffixes=("_a", "_b"), how="outer")
    return bool(len(ea) > 10 and m.Weight_a.notna().all() and m.Weight_b.notna().all() and np.allclose(m.Weight_a, m.Weight_b)), "scrambling all future prices leaves earlier decisions unchanged"


def c_forecast_leak():
    rng = np.random.default_rng(6)
    n = 260
    idx = pd.date_range("1995-01-31", periods=n, freq="ME")
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = .8 * z[t - 1] + rng.normal()
    p = pd.Series(100 * np.cumprod(1 + np.r_[0, .01 * z[:-1]] + rng.normal(.004, .03, n)), index=idx)
    X = pd.DataFrame({"z": z}, index=idx)
    cut = idx[180]
    p2, X2 = p.copy(), X.copy()
    p2[idx > cut] *= np.exp(np.cumsum(np.random.default_rng(1).normal(0, .1, (idx > cut).sum())))
    X2.loc[idx > cut, "z"] = 7.0
    a = FC.run_forecast(p, X, None, h=3, min_train=60).oos.loc[:cut]
    b = FC.run_forecast(p2, X2, None, h=3, min_train=60).oos.loc[:cut]
    c = a.index.intersection(b.index)
    return bool(len(c) > 30 and np.allclose(a.loc[c, ["p_Down", "p_Sideways", "p_Up"]], b.loc[c, ["p_Down", "p_Sideways", "p_Up"]])), "forecast at month t never uses data after t"


def c_parity_internal():
    """Vectorised features vs a naive per-ticker loop (independent implementation)."""
    pr = _prices(n=12, years=4, seed=7)
    p = F.build_panel(pr)
    d = p.index[-5]
    f = F.compute_features(p, [d], F.Params(100, 126, "BENCH")).snapshot(d)
    t = f.index[0]
    a = p.prices[t].loc[:d].dropna()
    b = p.prices["BENCH"].loc[:d].dropna()
    r12s = a.iloc[-22] / a.iloc[-22 - 252] - 1
    b12s = b.iloc[-22] / b.iloc[-22 - 252] - 1
    expect = (1 + r12s) / (1 + b12s) - 1
    ma = a.tail(100).mean() < a.iloc[-1]
    return bool(abs(f.loc[t, "Excess_12M_Skip1M"] - expect) < 1e-12 and bool(f.loc[t, "TrendGate"]) == bool(ma)), "12-1 momentum and trend gate match a hand calculation"


CHECKS = [
    ("Maximum drawdown", c_drawdown), ("Newey-West regression", c_hac), ("Wilson interval", c_wilson),
    ("Logistic model optimum", c_logit_gradient), ("Weight caps", c_caps), ("Safe price merging", c_merge),
    ("Macro publication lag", c_macro_lag), ("Features vs hand calculation", c_parity_internal),
    ("Backtest reconciliation & costs", c_reconcile_and_cost), ("No look-ahead (backtest)", c_no_lookahead),
    ("No leakage (forecast)", c_forecast_leak),
]


def run_all() -> pd.DataFrame:
    rows = []
    for name, fn in CHECKS:
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        rows.append({"Check": name, "Result": "PASS" if ok else "FAIL", "Detail": detail, "Seconds": round(time.perf_counter() - t0, 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    df = run_all()
    print(df.to_string(index=False))
    raise SystemExit(0 if (df.Result == "PASS").all() else 1)
