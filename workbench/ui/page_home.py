from __future__ import annotations

import streamlit as st

from .. import APP_NAME, __version__, config, content
from . import data as D, kit


def card(col, title: str, body: str):
    col.markdown(f'<div class="wb-card"><h4>{title}</h4><div>{body}</div></div>', unsafe_allow_html=True)


def render():
    kit.page_header(APP_NAME, f"Version {__version__} · open-source data only · built for learning to make your own informed decisions")
    store = D.etf_store()
    age = store.age_hours()
    if age is None:
        kit.note("<b>Welcome.</b> No data yet. Start with <b>Data Center → Update ETF prices</b>, then <b>Update economic data</b>.", "caution")
    else:
        st.markdown(f'ETF data last updated {age:.0f} hours ago. ' + (kit.pill("fresh", "ok") if age < 36 else kit.pill("consider updating", "warn")), unsafe_allow_html=True)
    st.subheader("What would you like to do?")
    a, b, c = st.columns(3)
    card(a, "🔎 Find what is strong now", "Rank ~140 ETFs by momentum, apply trend and risk rules, and get a diversified model portfolio.")
    card(b, "🧪 Test an idea honestly", "Replay history with realistic costs and delays, then see whether the result could be luck.")
    card(c, "🏢 Study stocks inside an index", "Rank members, measure breadth and concentration, and dissect why a stock moved.")
    d, e, f = st.columns(3)
    card(d, "🌍 Understand the economic cycle", "Growth/inflation regimes, yield curve, stress – and how assets behaved in each.")
    card(e, "🎲 Get honest odds", "Down / sideways / up probabilities, scored against a naive baseline so you know whether to trust them.")
    card(f, "📚 Learn as you go", "Plain-English tutorials, a glossary and hover-help on every setting.")
    st.markdown("&nbsp;")
    st.subheader("Quick start")
    st.markdown("1. **Data Center** → *Update ETF prices*, *Update economic data*, (optionally) *Update style factors*.\n"
                "2. **ETF Screener** → see today's ranking and portfolio.\n3. **Backtest Lab** → *Run backtest*; read the *Is it luck?* line first.\n"
                "4. **Macro & Cycles** and **Probabilities** → connect assets to the economy.\n5. **Learn** → tutorials whenever a term is unfamiliar.")
    kit.note("Use the sidebar to move between pages. Hover over the <b>ⓘ</b> next to any setting for a plain-English explanation.")
    st.caption(content.DISCLAIMER)
