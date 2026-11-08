"""Walk-forward backtest with daily mark-to-market, realistic timing and honest costs.

What is different from v7.8 (and why)
  * Trading cost is charged on the TOTAL traded amount (buys + sells) at the stated one-way cost; v7.8 charged
    half of that, understating costs by 2×.  Turnover is measured against weights *after price drift*.
  * Optional execution lag (default 1 trading day): the signal is computed at the close of day d, trades
    happen at the close of day d+1.  v7.8 assumed you could trade at the very close that produced the signal.
  * Strategy value is marked every trading day, so drawdown and volatility are real, not sampled monthly.
  * Hysteresis is applied inside portfolio construction, so names you hold are not dropped by the entry filters.
  * Missing prices are flagged in a table, never silently treated as 0% return.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features as F
from . import stats as S
from .portfolio import CASH, PortfolioParams, build_portfolio


@dataclass
class BacktestConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    benchmark: str
    freq: str = "Monthly"
    lag_days: int = 1
    cost_bps: float = 10.0            # one-way cost per unit traded (each buy and each sell)
    ma_days: int = 200
    min_history_days: int = 252
    portfolio: PortfolioParams = field(default_factory=PortfolioParams)


@dataclass
class BacktestResult:
    config: BacktestConfig
    wealth: pd.DataFrame        # daily: Strategy, Benchmark, EqualWeightUniverse (start = 1.0)
    periods: pd.DataFrame
    alloc: pd.DataFrame
    flags: pd.DataFrame

    @property
    def daily_returns(self) -> pd.DataFrame:
        return self.wealth.pct_change().dropna()

    def metrics(self) -> pd.DataFrame:
        r = self.daily_returns
        a = S.performance_metrics(r["Strategy"], r["Benchmark"], ppy=252)
        b = S.performance_metrics(r["Benchmark"], ppy=252)
        e = S.performance_metrics(r["EqualWeightUniverse"], r["Benchmark"], ppy=252)
        keys = ["Total return", "CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Hit rate", "Best period", "Worst period"]
        df = pd.DataFrame({"Strategy": {k: a.get(k) for k in keys}, "Benchmark": {k: b.get(k) for k in keys},
                           "Equal-weight universe": {k: e.get(k) for k in keys}})
        extra = {k: a.get(k) for k in ["CAGR minus benchmark", "Tracking error", "Information ratio", "Beta", "Upside capture", "Downside capture"]}
        df2 = pd.DataFrame({"Strategy": extra})
        return pd.concat([df, df2])

    def monthly_returns(self) -> pd.DataFrame:
        w = self.wealth.resample("ME").last()
        first = self.wealth.iloc[[0]]
        w = pd.concat([first, w]).drop_duplicates()
        return w.pct_change().dropna()

    def significance(self, seed: int = 0) -> dict:
        """Is the monthly outperformance distinguishable from luck?  Block-bootstrap + probabilistic Sharpe."""
        m = self.monthly_returns()
        act = (m["Strategy"] - m["Benchmark"])
        ci = S.bootstrap_ci(act, np.mean, n_boot=3000, mean_block=4, seed=seed)
        r = self.daily_returns["Strategy"] - self.daily_returns["Benchmark"]
        sr = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan")
        psr = S.probabilistic_sharpe(sr, len(r), float(r.skew()), float(r.kurt() + 3.0))
        return {
            "months": ci["n"],
            "mean_monthly_active": ci["estimate"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
            "p_not_better": ci["p_le_0"],
            "annualised_active": (1 + ci["estimate"]) ** 12 - 1 if np.isfinite(ci["estimate"]) else float("nan"),
            "prob_true_active_sharpe_positive": psr,
        }


def _ffill_cols(M: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down each column (a price that stops updating is frozen, not dropped)."""
    out = M.copy()
    for j in range(out.shape[1]):
        c = out[:, j]
        m = np.isnan(c)
        if m.any() and not m.all():
            idx = np.where(~m, np.arange(len(c)), 0)
            np.maximum.accumulate(idx, out=idx)
            out[:, j] = c[idx]
            lead = np.isnan(out[:, j])
            out[lead, j] = np.nan
    return out


def run_backtest(
    panel: F.Panel,
    cfg: BacktestConfig,
    names: dict[str, str] | None = None,
    feats: F.Features | None = None,
    dec_dates: list[pd.Timestamp] | None = None,
) -> BacktestResult:
    names = names or {}
    P = panel.prices
    A = P.to_numpy(dtype=float)
    cols = list(P.columns)
    cidx = {t: i for i, t in enumerate(cols)}
    bench = cfg.benchmark
    if bench not in cidx:
        raise KeyError(f"Benchmark {bench} missing from price panel")
    dec = dec_dates or F.rebalance_dates(P.index, cfg.start, cfg.end, cfg.freq)
    if feats is None:
        feats = F.compute_features(panel, dec, F.Params(cfg.ma_days, cfg.min_history_days, bench))
    dec = list(feats.dates)
    if len(dec) < 2:
        raise ValueError("Need at least two decision dates; widen the date range")
    ret_all = P.pct_change(fill_method=None)
    pos_dec = feats.pos
    exec_pos = np.minimum(pos_dec + cfg.lag_days, len(P) - 1)

    nav = 1.0
    pos_list: list[int] = []          # panel positions of the daily wealth series
    w_s: list[float] = []             # strategy wealth
    w_e: list[float] = []             # equal-weight-universe wealth
    periods, alloc, flags = [], [], []
    prev_drift: dict[str, float] = {bench: 1.0}
    prev_active: list[str] = []
    bcol = A[:, cidx[bench]]

    for i in range(len(dec) - 1):
        s, e = int(exec_pos[i]), int(exec_pos[i + 1])
        if e <= s:
            flags.append({"Decision": dec[i], "Ticker": "-", "Issue": "ZERO_LENGTH_PERIOD", "Weight": np.nan})
            continue
        snap = feats.snapshot(dec[i])
        port, _ = build_portfolio(snap, ret_all.iloc[: pos_dec[i] + 1], bench, cfg.portfolio, held=prev_active, names=names)
        w = {t: v for t, v in zip(port["Yahoo_Ticker"], port["PortfolioWeight"]) if abs(v) > 1e-12}

        # ---- cost: everything bought plus everything sold, versus the DRIFTED previous weights ----
        union = set(w) | set(prev_drift)
        traded = sum(abs(w.get(t, 0.0) - prev_drift.get(t, 0.0)) for t in union)
        cost = traded * cfg.cost_bps / 1e4

        # ---- holding-period path (buy & hold between rebalances) ----
        priced = [t for t in w if t != CASH and t in cidx]
        for t in w:
            if t != CASH and t not in cidx:
                flags.append({"Decision": dec[i], "Ticker": t, "Issue": "NOT_IN_PRICE_PANEL (held as cash)", "Weight": w[t]})
        ii = [cidx[t] for t in priced]
        seg = A[s : e + 1][:, ii] if ii else np.zeros((e - s + 1, 0))
        unpriced_w = 0.0
        for k, t in enumerate(priced):
            if np.isnan(seg[0, k]):
                unpriced_w += w[t]
                flags.append({"Decision": dec[i], "Ticker": t, "Issue": "NO_PRICE_AT_START (held as cash)", "Weight": w[t]})
            elif np.isnan(seg[-1, k]):
                flags.append({"Decision": dec[i], "Ticker": t, "Issue": "PRICE_STALE_AT_END (last price held)", "Weight": w[t]})
        seg = _ffill_cols(seg)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = seg / seg[0]
        rel = np.where(np.isfinite(rel), rel, 1.0)
        wv = np.array([w[t] for t in priced])
        w_cash = sum(v for t, v in w.items() if t == CASH or t not in cidx)
        V = (rel @ wv if ii else np.zeros(e - s + 1)) + w_cash
        gross = float(V[-1] - 1.0)
        path = nav * (1.0 - cost) * V            # boundary day s carries the post-cost value
        npos = np.arange(s, e + 1)

        # ---- equal-weight comparator over everything rankable on the decision date (no costs) ----
        u = feats.universe.iloc[i]
        uidx = [cidx[t] for t in u.index[u.to_numpy()] if t in cidx and np.isfinite(A[s, cidx[t]])]
        if uidx:
            su = _ffill_cols(A[s : e + 1][:, uidx])
            with np.errstate(invalid="ignore", divide="ignore"):
                ru = su / su[0]
            Vu = np.where(np.isfinite(ru), ru, 1.0).mean(1)
        else:
            Vu = np.ones(e - s + 1)

        if not pos_list:
            pos_list.append(s); w_s.append(path[0]); w_e.append(1.0)
        else:
            w_s[-1] = path[0]
        base_e = w_e[-1]
        for k in range(1, len(npos)):
            pos_list.append(int(npos[k])); w_s.append(path[k]); w_e.append(base_e * Vu[k])

        # ---- drift weights to period end (turnover base for next rebalance) ----
        end_vals = {t: w[t] * rel[-1, k] for k, t in enumerate(priced)}
        if w_cash > 0:
            end_vals[CASH] = w_cash
        tot = sum(end_vals.values())
        prev_drift = {t: v / tot for t, v in end_vals.items()} if tot > 0 else {bench: 1.0}
        prev_active = [t for t in w if t not in (bench, CASH)]
        nav = float(path[-1])

        periods.append({
            "Decision": dec[i], "Start": P.index[s], "End": P.index[e],
            "GrossReturn": gross, "Cost": cost, "NetReturn": (1 - cost) * (1 + gross) - 1,
            "BenchmarkReturn": float(bcol[e] / bcol[s] - 1.0), "Traded": traded, "OneWayTurnover": 0.5 * traded,
            "ActiveNames": len(prev_active), "CoreWeight": w.get(bench, 0.0) + w.get(CASH, 0.0),
            "UnpricedWeight": unpriced_w,
        })
        for t, wt in w.items():
            r_t = float(rel[-1, priced.index(t)] - 1.0) if t in priced else 0.0
            alloc.append({"Date": P.index[s], "Decision": dec[i], "HoldingEnd": P.index[e], "Ticker": t,
                          "Fund_Name": names.get(t, "Cash" if t == CASH else t), "Weight": wt,
                          "AssetReturn": r_t, "Contribution": wt * r_t})

    if not pos_list:
        raise ValueError("No valid holding periods; check dates and data coverage")
    wealth = pd.DataFrame({"Strategy": w_s, "EqualWeightUniverse": w_e}, index=P.index[pos_list])
    wealth["Benchmark"] = bcol[pos_list] / bcol[pos_list[0]]
    wealth = wealth[["Strategy", "Benchmark", "EqualWeightUniverse"]]
    if pos_list[0] > 0:                       # day-0 row = 1.0 so the very first trading cost is inside the return series
        wealth.loc[P.index[pos_list[0] - 1]] = 1.0
        wealth = wealth.sort_index()
    return BacktestResult(cfg, wealth, pd.DataFrame(periods), pd.DataFrame(alloc),
                          pd.DataFrame(flags, columns=["Decision", "Ticker", "Issue", "Weight"]))


def run_grid(panel: F.Panel, base: BacktestConfig, ma_list, entry_list, sizing_list, names=None, progress=None) -> pd.DataFrame:
    """Parameter-robustness grid. Features are computed once per MA length and reused."""
    import dataclasses
    rows = []
    dec = F.rebalance_dates(panel.index, base.start, base.end, base.freq)
    total = len(ma_list) * len(entry_list) * len(sizing_list)
    n = 0
    sharpes = []
    for ma in ma_list:
        feats = F.compute_features(panel, dec, F.Params(ma, base.min_history_days, base.benchmark))
        for ent in entry_list:
            for sz in sizing_list:
                n += 1
                pp = dataclasses.replace(base.portfolio, entry_pct=ent, exit_pct=min(base.portfolio.exit_pct, ent), sizing=sz)
                cfg = dataclasses.replace(base, ma_days=ma, portfolio=pp)
                try:
                    res = run_backtest(panel, cfg, names, feats=feats)
                    m = res.metrics()["Strategy"]
                    r = res.daily_returns["Strategy"]
                    rows.append({"MA": ma, "EntryPct": ent, "Sizing": sz, "CAGR": m["CAGR"], "BenchmarkCAGR": res.metrics().loc["CAGR", "Benchmark"],
                                 "MaxDD": m["Max drawdown"], "Sharpe": m["Sharpe"], "InfoRatio": m["Information ratio"],
                                 "Turnover": res.periods["OneWayTurnover"].mean(), "_r": r})
                except Exception as e:  # noqa: BLE001 - report, never hide
                    rows.append({"MA": ma, "EntryPct": ent, "Sizing": sz, "Error": f"{type(e).__name__}: {e}"})
                if progress:
                    progress(n, total)
    df = pd.DataFrame(rows)
    ok = df[df.get("Error", pd.Series(index=df.index, dtype=object)).isna()] if "Error" in df else df
    if len(ok) > 1:
        per_period_sr = [r.mean() / r.std(ddof=1) for r in ok["_r"] if r.std(ddof=1) > 0]
        var_sr = float(np.var(per_period_sr, ddof=1)) if len(per_period_sr) > 1 else 0.0
        dsr = {}
        for idx, r in ok["_r"].items():
            dsr[idx] = S.deflated_sharpe(r, len(ok), var_sr)["dsr"]
        df["DeflatedSharpeProb"] = pd.Series(dsr)
    return df.drop(columns=["_r"], errors="ignore")
