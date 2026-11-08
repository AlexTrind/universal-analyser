"""Strategy settings panel shared by the ETF Screener, Backtest Lab, Market Map and Audit pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config, content
from ..portfolio import PortfolioParams, SIZING_METHODS
from . import data as D
from . import kit


def strategy_sidebar() -> dict:
    with st.sidebar:
        st.header("Strategy settings")
        S: dict = {}
        S["universe"] = kit.s_text("Universe workbook", "universe", "universe_file", config.DEFAULT_UNIVERSE)
        S["base"] = kit.s_select("Base currency", "base", "base_currency", config.BASE_CURRENCIES, config.DEFAULT_BASE)
        bname = kit.s_select("Benchmark", "benchmark", "benchmark", list(config.BENCHMARKS), list(config.BENCHMARKS)[0])
        S["benchmark"] = config.BENCHMARKS[bname]
        S["value"] = kit.s_number("Portfolio value", "value", "portfolio_value", 1.0, 1e9, 10000.0, 1000.0)
        st.subheader("Portfolio")
        S["active_fraction"] = kit.s_slider("Momentum sleeve (share of portfolio)", "active_fraction", "active_fraction", 0.0, 1.0, 0.20, 0.05)
        S["n_active"] = kit.s_slider("Maximum ETFs held", "n_active", "n_active", 1, 20, 6)
        st.subheader("Selection rules")
        S["entry_pct"] = kit.s_slider("Entry rank (percentile)", "entry_pct", "entry_pct", 50, 99, 80)
        S["exit_pct"] = kit.s_slider("Exit rank (hysteresis)", "exit_pct", "exit_pct", 1, 99, 60)
        S["exit_pct"] = min(S["exit_pct"], S["entry_pct"])
        S["ma_days"] = kit.s_slider("Trend filter (days)", "ma_days", "ma_days", 120, 300, 200, 10)
        S["min_history"] = kit.s_slider("Minimum history (days)", "min_history", "min_history", 126, 504, 252, 21)
        st.subheader("Risk & diversification")
        S["sizing"] = kit.s_select("Position sizing", "sizing", "sizing", SIZING_METHODS, "Realized volatility")
        S["corr_days"] = kit.s_slider("Correlation look-back (days)", "corr_days", "corr_days", 63, 252, 126, 21)
        S["corr_threshold"] = kit.s_slider("Cluster threshold", "corr_threshold", "corr_threshold", 0.50, 0.95, 0.80, 0.05)
        S["max_cluster"] = kit.s_slider("Max share per cluster", "max_cluster", "max_cluster", 0.20, 1.00, 0.40, 0.05)
        S["max_single"] = kit.s_slider("Max share per ETF", "max_single", "max_single", 0.10, 1.00, 0.30, 0.05)
        S["residual_to"] = kit.s_select("Unused weight goes to", "residual_to", "residual_to", ["Benchmark", "Cash"], "Benchmark")
        st.subheader("Trading realism")
        S["cost_bps"] = kit.s_number("Cost per trade (bps)", "cost_bps", "cost_bps", 0.0, 200.0, 10.0, 1.0)
        S["lag_days"] = kit.s_select("Execution delay (days)", "lag_days", "lag_days", [0, 1, 2], 1)
    S["params"] = PortfolioParams(
        active_fraction=S["active_fraction"], n_active=S["n_active"], entry_pct=float(S["entry_pct"]), exit_pct=float(S["exit_pct"]),
        sizing=S["sizing"], corr_days=S["corr_days"], corr_threshold=S["corr_threshold"], max_cluster=S["max_cluster"],
        max_single=S["max_single"], residual_to=S["residual_to"])
    return S


def load_context(S: dict):
    """Load universe + price panel or stop the page with a friendly message."""
    try:
        uni = D.load_universe(S["universe"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot read the universe workbook '{S['universe']}': {e}")
        st.stop()
    store = D.etf_store()
    panel, qrep, ccy = D.etf_panel(S["universe"], S["base"], S["benchmark"], D.mtime(store.f_prices), D.mtime(store.f_meta))
    if panel is None or S["benchmark"] not in panel.prices.columns:
        kit.note("No price data yet. Open <b>Data Center</b> and click <b>Update ETF prices</b> (first run takes a few minutes).", "caution")
        st.stop()
    names = dict(zip(uni["Yahoo_Ticker"], uni["Fund_Name"]))
    return uni, panel, qrep, ccy, names
