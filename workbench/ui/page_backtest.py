from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from .. import backtest as BT, content, features as F, stats as S_
from . import charts, controls, data as D, kit


def _signature(S: dict, extra: tuple, price_stamp: float) -> str:
    p = S["params"]
    raw = repr((S["universe"], S["base"], S["benchmark"], dataclasses.astuple(p), S["ma_days"], S["min_history"], S["cost_bps"], S["lag_days"], extra, price_stamp))
    return hashlib.md5(raw.encode()).hexdigest()


def render():
    S = controls.strategy_sidebar()
    kit.page_header("Backtest Lab", "Replay history month by month using only the information available at each date – then test whether the result could be luck.")
    uni, panel, qrep, ccy, names = controls.load_context(S)
    idx = panel.index
    c1, c2, c3 = st.columns(3)
    d_start = max(idx[0], idx[-1] - pd.DateOffset(years=5))
    with c1:
        start = kit.s_date("Start", "bt_start", "bt_start", d_start.date(), idx[0].date(), idx[-1].date())
    with c2:
        end = kit.s_date("End", "bt_end", "bt_end", idx[-1].date(), idx[0].date(), idx[-1].date())
    with c3:
        freq = kit.s_select("Decision frequency", "bt_freq", "bt_freq", ["Monthly", "Weekly", "Quarterly"], "Monthly")
    cfg = BT.BacktestConfig(pd.Timestamp(start), pd.Timestamp(end), S["benchmark"], freq, int(S["lag_days"]), float(S["cost_bps"]),
                            S["ma_days"], S["min_history"], S["params"])
    sig = _signature(S, (str(start), str(end), freq), D.mtime(D.etf_store().f_prices))
    if st.session_state.get("bt_sig") != sig:
        for k in ("bt_res", "grid"):
            st.session_state.pop(k, None)
        st.session_state["bt_sig"] = sig

    if kit.wide(st.button, "Run backtest", type="primary"):
        with st.spinner("Replaying history…"):
            try:
                st.session_state["bt_res"] = BT.run_backtest(panel, cfg, names)
            except Exception as e:  # noqa: BLE001
                st.error(f"Backtest could not run: {e}")
    res: BT.BacktestResult | None = st.session_state.get("bt_res")
    if res is None:
        kit.note("Choose a period and click <b>Run backtest</b>. Monthly decisions over 5 years take about a second.")
        kit.explain("What will I see?", "Growth of 1 versus the benchmark and an equal-weight basket of the whole universe, daily drawdowns, risk statistics, "
                    "holdings through time, a monthly return calendar, and a statistical check on whether the out-performance could be luck.")
        st.stop()

    m = res.metrics()
    sg = res.significance()
    A, B = m["Strategy"], m["Benchmark"]
    if A.get("CAGR") is None or B.get("CAGR") is None or not np.isfinite(float(A["CAGR"])) or not np.isfinite(float(B["CAGR"])):
        st.warning("The backtest produced too little history to calculate returns (fewer than about 20 trading days of usable prices). "
                   "Pick an earlier start date, a shorter 'minimum history' or a benchmark that has prices over the whole period, then run it again.")
        st.stop()
    kit.kpi_row([
        ("CAGR", kit.fmt_pct(A["CAGR"]), content.KPI["CAGR"], f"{A['CAGR'] - B['CAGR']:+.1%} vs benchmark"),
        ("Volatility", kit.fmt_pct(A["Volatility"]), content.KPI["Volatility"], None),
        ("Sharpe", kit.fmt_num(A["Sharpe"]), content.KPI["Sharpe"], f"benchmark {B['Sharpe']:.2f}"),
        ("Max drawdown", kit.fmt_pct(A["Max drawdown"]), content.KPI["Max drawdown"], f"benchmark {B['Max drawdown']:.1%}"),
        ("Info ratio", kit.fmt_num(m.loc["Information ratio", "Strategy"]), content.KPI["Information ratio"], None),
        ("Costs paid", kit.fmt_pct(float(res.periods['Cost'].sum()), 2), "Sum of all trading costs over the period (share of portfolio value).", None),
    ])
    # honest headline: is it luck?
    tone = "good" if (sg["ci_lo"] > 0) else "neutral"
    ann = sg["annualised_active"]
    msg = (f"<b>Is it luck?</b> Average monthly out-performance vs the benchmark: <b>{sg['mean_monthly_active']:+.2%}</b> "
           f"(95% range {sg['ci_lo']:+.2%} to {sg['ci_hi']:+.2%}, {sg['months']} months). ")
    msg += ("The range excludes zero – but this is still one historical path with survivorship-biased funds and many researcher choices. "
            if sg["ci_lo"] > 0 else "The range includes zero, so luck cannot be ruled out. ")
    msg += f"Probability the true active Sharpe is positive: {sg['prob_true_active_sharpe_positive']:.0%}."
    kit.verdict(msg, tone)
    ew = m["Equal-weight universe"]
    if ew["CAGR"] > A["CAGR"]:
        kit.note(f"Note: simply holding an equal-weight basket of the whole universe returned {ew['CAGR']:.1%} a year versus {A['CAGR']:.1%} for the momentum strategy. "
                 "Momentum selection should add something beyond 'being in the universe' to be worth the effort.", "caution")
    if len(res.flags):
        kit.note(f"{len(res.flags)} data issue(s) occurred during the replay (see the <b>Data issues</b> tab). They are never silently ignored.", "caution")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Performance", "Holdings", "Statistics", "Periods & trades", "Robustness", "Data issues"])
    with t1:
        w = res.wealth.rename(columns={"EqualWeightUniverse": "Equal-weight universe"})
        dr = res.daily_returns
        jumps = dr[(dr.abs() > 0.12).any(axis=1)]
        if len(jumps):
            ratios = (panel.prices / panel.prices.shift(1)).stack()
            culprits = ratios[(ratios > 2.5) | (ratios < 0.4)]
            who = ", ".join(f"{t} on {d.date()}" for d, t in list(culprits.index)[:6]) if len(culprits) else "none found in the price panel – inspect the Data Center quality table"
            kit.note(f"<b>Check the data:</b> a line moved more than 12% in a single day ({', '.join(str(d.date()) for d in jumps.index[:4])}). Real diversified funds almost never do that, so it is usually a price error. Suspect tickers: {who}.", "caution")
        kit.chart(charts.line(w, "Growth of 1", yfmt=".2f", hover_fmt=".3f", colors={"Strategy": charts.COL[0], "Benchmark": "#8896A5", "Equal-weight universe": charts.COL[2]}))
        dd = w / w.cummax() - 1
        kit.chart(charts.drawdown(dd[["Strategy", "Benchmark"]]))
        act = (res.daily_returns["Strategy"] - res.daily_returns["Benchmark"])
        roll = act.rolling(252).sum().dropna()
        if len(roll):
            kit.chart(charts.line(roll.to_frame("Rolling 12-month out-performance (sum of daily differences)"), "Rolling 12-month out-performance vs benchmark", yfmt=".0%", hline=0.0, hover_fmt=".1%", height=300))
        kit.chart(charts.monthly_heatmap(res.monthly_returns()["Strategy"], "Strategy monthly returns"))
        kit.explain("What is the ‘Equal-weight universe’ line?", "A no-skill yardstick: at every rebalance, put the same amount into every ETF that was rankable on that date (no costs), hold until the next rebalance, repeat. "
                    "If your strategy cannot beat this, picking ETFs by momentum added nothing over simply owning everything. It can rise faster than the benchmark because it holds many small, volatile areas the benchmark barely touches. A sudden vertical jump is not real performance – it means one ETF had a bad price.")
        kit.explain("How to read these charts", "The first chart shows what £1 would have become. The shaded drawdown chart shows how far below its previous peak each line was on every day – "
                    "that is the pain you would have lived through. The rolling out-performance line shows whether the edge is steady or comes from one lucky stretch.")
    with t2:
        a = res.alloc
        wide_w = a.pivot_table(index="Date", columns="Ticker", values="Weight", aggfunc="sum").fillna(0.0)
        last = res.wealth.index[-1]
        wide_w.loc[last] = wide_w.iloc[-1]
        lab = {t: f"{t}" for t in wide_w.columns}
        kit.chart(charts.stacked_area(wide_w.rename(columns=lab), "Allocation through time (weights decided at each rebalance)"))
        held = a[~a["Ticker"].isin([S["benchmark"], "CASH"])]
        if not held.empty:
            tot = held.groupby(["Ticker", "Fund_Name"]).agg(Rebalances=("Date", "nunique"), AvgWeight=("Weight", "mean"), Contribution=("Contribution", "sum")).reset_index()
            kit.dataframe(tot.sort_values("Contribution", ascending=False), {
                "Ticker": dict(label="Ticker"), "Fund_Name": dict(label="ETF"), "Rebalances": dict(label="Times held", kind="int", help="Number of rebalance periods the ETF was held."),
                "AvgWeight": dict(label="Avg weight when held", kind="pct"),
                "Contribution": dict(label="Total contribution (sum of period contributions)", kind="pct_signed", help="Sum over periods of weight × ETF return. Simple sum, not compounded.")})
    with t3:
        disp = m.copy()
        spec = {c: dict(label=c, kind="num2") for c in disp.columns}
        pct_rows = ["Total return", "CAGR", "Volatility", "Max drawdown", "Hit rate", "Best period", "Worst period", "CAGR minus benchmark", "Tracking error", "Upside capture", "Downside capture"]
        out = disp.copy().astype(object)
        for r in out.index:
            for c in out.columns:
                v = disp.loc[r, c]
                out.loc[r, c] = "–" if pd.isna(v) else (f"{v:.1%}" if r in pct_rows else f"{v:.2f}")
        out.insert(0, "Metric", out.index)
        out["What it means"] = [content.KPI.get(r, "") for r in out["Metric"]]
        kit.dataframe(out.reset_index(drop=True), {"Metric": dict(label="Metric"), "What it means": dict(label="What it means")})
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Worst drawdown episodes (strategy)**")
            dt = S_.drawdown_table(res.wealth["Strategy"])
            if len(dt):
                kit.dataframe(dt, {"Peak date": dict(label="Peak", kind="date"), "Trough date": dict(label="Trough", kind="date"), "Recovery date": dict(label="Recovered", kind="date"),
                                   "Depth": dict(label="Depth", kind="pct"), "Days peak→trough": dict(label="Days down", kind="int"), "Days to recover": dict(label="Days to recover", kind="int")})
        with r2:
            st.markdown("**Calendar-year returns**")
            ye = res.wealth.resample("YE").last()
            prev = ye.shift(1)
            prev.iloc[0] = res.wealth.iloc[0].to_numpy()
            yr = ye / prev - 1
            if len(yr):
                yr.index = yr.index.year
                kit.dataframe(yr.reset_index().rename(columns={"index": "Year", "EqualWeightUniverse": "Equal-weight"}), {
                    "Year": dict(label="Year"), "Strategy": dict(label="Strategy", kind="pct_signed"), "Benchmark": dict(label="Benchmark", kind="pct_signed"),
                    "Equal-weight": dict(label="Equal-weight universe", kind="pct_signed")})
                st.caption("First and last years may be partial.")
    with t4:
        per = res.periods.copy()
        kit.dataframe(per, {
            "Decision": dict(label="Decision date", kind="date", help="Day the signal was computed (close)."), "Start": dict(label="Trade date", kind="date", help="Day the portfolio was bought."),
            "End": dict(label="Next trade date", kind="date"), "GrossReturn": dict(label="Return before costs", kind="pct_signed"), "Cost": dict(label="Trading cost", kind="pct"),
            "NetReturn": dict(label="Return after costs", kind="pct_signed"), "BenchmarkReturn": dict(label="Benchmark", kind="pct_signed"),
            "Traded": dict(label="Amount traded", kind="pct", help="Everything bought plus everything sold, as share of portfolio."),
            "OneWayTurnover": dict(label="One-way turnover", kind="pct", help="Half of the amount traded."), "ActiveNames": dict(label="ETFs held", kind="int"),
            "CoreWeight": dict(label="Core weight", kind="pct"), "UnpricedWeight": dict(label="Weight without price", kind="pct")}, height=420)
        st.caption(f"Average one-way turnover per rebalance: {per['OneWayTurnover'].mean():.1%}. Total cost drag over the period: {per['Cost'].sum():.2%}.")
        st.markdown("**Recorded holdings per period**")
        kit.dataframe(res.alloc.sort_values(["Decision", "Weight"], ascending=[False, False]), {
            "Date": dict(label="Trade date", kind="date"), "Decision": dict(label="Decision", kind="date"), "HoldingEnd": dict(label="Held until", kind="date"),
            "Ticker": dict(label="Ticker"), "Fund_Name": dict(label="ETF"), "Weight": dict(label="Weight", kind="pct"),
            "AssetReturn": dict(label="ETF return in period", kind="pct_signed"), "Contribution": dict(label="Contribution", kind="pct_signed")}, height=360)
    with t5:
        st.markdown("Compare nearby, reasonable settings. A real effect should survive small changes; a result that exists at only one setting is probably noise.")
        g1, g2, g3 = st.columns(3)
        ma_l = g1.multiselect("Trend filter lengths", [120, 160, 180, 200, 220, 240, 280], default=[180, 200, 220], help=content.HELP["grid_ma"])
        en_l = g2.multiselect("Entry ranks", [70, 75, 80, 85, 90], default=[75, 80, 85], help=content.HELP["grid_entry"])
        sz_l = g3.multiselect("Sizing methods", ["Realized volatility", "Downside volatility", "ATR% proxy", "Equal weight"], default=["Realized volatility", "Equal weight"], help=content.HELP["grid_sizing"])
        total = len(ma_l) * len(en_l) * len(sz_l)
        st.caption(f"{total} combinations (about {total * 1.0:.0f}–{total * 1.5:.0f} seconds).")
        if kit.wide(st.button, "Run robustness grid", disabled=total == 0):
            bar = st.progress(0.0)
            st.session_state["grid"] = BT.run_grid(panel, cfg, ma_l, en_l, sz_l, names, progress=lambda n, t: bar.progress(n / t))
            bar.empty()
        g = st.session_state.get("grid")
        if g is not None and len(g):
            if "Error" in g and g["Error"].notna().any():
                st.error("Some combinations failed:")
                st.dataframe(g[g["Error"].notna()])
            ok = g[g.get("Error", pd.Series(index=g.index, dtype=object)).isna()] if "Error" in g else g
            if len(ok):
                beat = float((ok["CAGR"] > ok["BenchmarkCAGR"]).mean())
                kit.verdict(f"<b>{beat:.0%}</b> of tested settings beat the benchmark's CAGR. Median CAGR {ok['CAGR'].median():.1%} vs benchmark {ok['BenchmarkCAGR'].median():.1%}. "
                            "Robust ideas win across most neighbouring settings; the ‘Deflated Sharpe’ column discounts for the number of settings tried (≥95% would be strong evidence).",
                            "good" if beat >= 0.8 else "neutral")
                kit.dataframe(ok.sort_values("Sharpe", ascending=False), {
                    "MA": dict(label="Trend (days)", kind="int"), "EntryPct": dict(label="Entry rank", kind="int"), "Sizing": dict(label="Sizing"),
                    "CAGR": dict(label="CAGR", kind="pct"), "BenchmarkCAGR": dict(label="Benchmark CAGR", kind="pct"), "MaxDD": dict(label="Max drawdown", kind="pct"),
                    "Sharpe": dict(label="Sharpe", kind="num2", help=content.KPI["Sharpe"]), "InfoRatio": dict(label="Info ratio", kind="num2", help=content.KPI["Information ratio"]),
                    "Turnover": dict(label="Turnover/rebalance", kind="pct"),
                    "DeflatedSharpeProb": dict(label="Deflated Sharpe prob.", kind="pct", help=content.GLOSSARY["Deflated Sharpe ratio"])})
                kit.chart(charts.scatter(ok.assign(**{"CAGR %": ok["CAGR"] * 100, "Max drawdown %": ok["MaxDD"] * 100}), "Max drawdown %", "CAGR %", hover=["MA", "EntryPct"], color="Sizing", title="Return versus worst drawdown for each setting"))
    with t6:
        if res.flags.empty:
            st.success("No missing-price or timing problems occurred during the replay.")
        else:
            kit.dataframe(res.flags, {"Decision": dict(label="Decision date", kind="date"), "Ticker": dict(label="Ticker"), "Issue": dict(label="Issue"), "Weight": dict(label="Weight affected", kind="pct")})
    kit.disclaimer_footer()
