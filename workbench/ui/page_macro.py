from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import data_macro as DM, regimes as RG
from . import charts, data as D, kit
from .page_asset import _spans


def render():
    kit.page_header("Macro & Cycles", "Where are we in the economic cycle? Growth, inflation, financial stress, the yield curve and policy – built only from free FRED data, using values known at each date.")
    c1, c2 = st.columns(2)
    with c1:
        scheme = kit.s_select("Regime definition", "regime_scheme", "regime_scheme", ["level", "direction"], "level")
    with c2:
        q = float(kit.s_slider("Stress percentile", "stress_q", "stress_q", 0.5, 0.95, 0.80, 0.05))
    T = D.macro_tables(D.macro_stamp(), scheme, q)
    if T is None:
        kit.note("No economic data yet. Open <b>Data Center → Update economic data</b>.", "caution")
        st.stop()
    cur = RG.current_summary(T["labels"], T["raw"])
    st.subheader("Where we are now")
    cols = st.columns(len(cur))
    for col, (_, r) in zip(cols, cur.iterrows()):
        col.metric(r["Dimension"], str(r["Current"]), help=f"In this state for {r['Months in this state']:.0f} months." if pd.notna(r["Months in this state"]) else None)
    kit.note("Regimes are simplified descriptions, not forecasts. Quadrants: <b>Expansion</b> (growth up, inflation low), <b>Overheating</b> (growth and inflation high), <b>Stagflation</b> (growth weak, inflation high), <b>Slowdown/Disinflation</b> (growth weak, inflation low).")
    tabs = st.tabs(["Composites", "Regime history", "Single indicators", "Data freshness"])
    shade = _spans(T["recession"]) if len(T["recession"]) else None
    with tabs[0]:
        comp = T["comp"].dropna(how="all")
        kit.chart(charts.line(comp, "Composite indicators (z-scores: 0 = historical average)", shade=shade, hline=0.0, hover_fmt=".2f", height=420))
        st.caption("Grey bands are NBER recessions (shown with hindsight only – never used as an input). Growth/Inflation/Stress/Liquidity are averages of standardised indicators, signed so higher always means ‘more’ of that thing.")
    with tabs[1]:
        lab = T["labels"].dropna(how="all")
        kit.dataframe(lab.tail(36).iloc[::-1].reset_index().rename(columns={"index": "Month"}), {"Month": dict(label="Month", kind="date")} | {c: dict(label=c) for c in lab.columns}, height=420)
        share = pd.concat({c: lab[c].value_counts(normalize=True) for c in lab.columns}, axis=1)
        kit.dataframe(share.reset_index().rename(columns={"index": "State"}), {"State": dict(label="State")} | {c: dict(label=c, kind="pct") for c in lab.columns})
        st.caption("Second table: share of all months spent in each state.")
    with tabs[2]:
        cols_ = list(T["feats"].columns)
        pick = kit.s_select("Indicator", "macro_pick", "macro_pick", cols_, cols_[0], format_func=lambda c: f"{DM.BY_ID[c].name if c in DM.BY_ID else c} ({c})")
        s = T["feats"][pick].dropna()
        meta = DM.BY_ID.get(pick)
        if meta:
            st.write(f"**Why it matters:** {meta.why}")
        kit.chart(charts.line(s.to_frame(pick), None, shade=shade, hover_fmt=".3g", height=340))
    with tabs[3]:
        fr = RG.data_freshness(T["raw"])
        kit.dataframe(fr, {"Series": dict(label="FRED ID"), "Name": dict(label="Indicator"), "Last observation": dict(label="Last observation", kind="date"),
                           "Days old": dict(label="Days old", kind="int"), "Typical delay (days)": dict(label="Normal publication delay", kind="int", help="If 'Days old' is far above this, the series may have stopped updating.")}, height=520)
    kit.disclaimer_footer()
