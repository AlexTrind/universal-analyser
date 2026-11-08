"""Shared asset picker: any ETF from the universe or any stock in the stock cache."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import controls, data as D, kit


def pick_asset(S: dict, key: str, help_key: str = "stock_pick"):
    """Return (price Series, label, ticker). Stops the page with a friendly message if nothing is available."""
    src = st.radio("Asset type", ["ETF from my universe", "Stock from the stock cache"], horizontal=True, key=f"{key}_src",
                   help="ETFs come from the Screener's workbook (converted to your base currency). Stocks come from whatever you downloaded in Group Explorer (local currency).")
    if src.startswith("ETF"):
        uni, panel, qrep, ccy, names = controls.load_context(S)
        opts = [t for t in uni["Yahoo_Ticker"] if t in panel.prices.columns]
        t = kit.s_select("Asset", f"{key}_etf", help_key, opts, opts[0], format_func=lambda x: f"{x} – {names.get(x, '')[:45]}")
        return panel.prices[t].dropna(), f"{t} – {names.get(t, '')}", t
    native = D.stock_store().load()
    if native.empty:
        kit.note("The stock cache is empty. Open <b>Group Explorer</b> and download a group first.", "caution")
        st.stop()
    opts = list(native.columns)
    t = kit.s_select("Asset", f"{key}_stk", help_key, opts, opts[0])
    return native[t].dropna(), t, t
