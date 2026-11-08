from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import content, cycles as CY, data_macro as DM, relationships as RL, stats as S_
from . import assets, charts, controls, data as D, kit


def _tables():
    scheme = kit.s_select("Regime definition", "regime_scheme", "regime_scheme", ["level", "direction"], "level")
    q = float(kit.s_slider("Stress percentile", "stress_q", "stress_q", 0.5, 0.95, 0.80, 0.05))
    return D.macro_tables(D.macro_stamp(), scheme, q)


def render():
    S = controls.strategy_sidebar()
    kit.page_header("Asset Lab", "Understand how one ETF or stock has behaved over the long run – and how that behaviour links to the economy.")
    p, label, tick = assets.pick_asset(S, "al")
    if len(p) < 300:
        kit.note("Fewer than ~1.2 years of data – most long-run statistics need more.", "caution")
        st.stop()
    st.subheader(label)
    prof = CY.asset_profile(p)
    kit.kpi_row([("Years of data", f"{prof['Years of data']:.1f}", None, None),
                 ("CAGR", kit.fmt_pct(prof["Annualised return (CAGR)"], 1, True), "Compound annual growth rate.", None),
                 ("Volatility", kit.fmt_pct(prof["Annualised volatility"], 1), content.KPI.get("Volatility"), None),
                 ("Worst drawdown", kit.fmt_pct(prof["Worst drawdown"], 0), content.KPI.get("Max drawdown"), None),
                 ("Months positive", kit.fmt_pct(prof.get("% of months positive"), 0), None, None)])
    tabs = st.tabs(["History", "Drawdowns", "Seasonality", "Trend behaviour", "Economy & regimes", "Leads & lags"])
    h = int(kit.s_slider("Look-ahead for economy links (months)", "horizon", "horizon", 1, 12, 3))
    with tabs[0]:
        kit.chart(charts.line((p / p.iloc[0]).to_frame(tick), "Growth of 1 (log scale)", log=True, height=380))
        cy = CY.calendar_year_returns(p)
        kit.chart(charts.bars([str(y) for y in cy.index], cy.to_numpy(), "Return by calendar year (first/last years may be partial)", horizontal=False))
        kit.dataframe(pd.DataFrame({"Statistic": list(prof), "Value": [str(v) if not isinstance(v, float) else f"{v:.4g}" for v in prof.values()]}),
                      {"Statistic": dict(label="Statistic"), "Value": dict(label="Value")})
    with tabs[1]:
        w = p / p.iloc[0]
        kit.chart(charts.drawdown(S_.drawdown_series(w).to_frame(tick)))
        dt = S_.drawdown_table(w, 8)
        kit.dataframe(dt, {c: dict(label=c) for c in dt.columns} | {"Depth": dict(label="Depth", kind="pct"), "depth": dict(label="Depth", kind="pct")})
        kit.note("Drawdowns show what holding this asset <i>felt like</i>. Ask yourself whether you would have kept holding through the deepest one.")
    with tabs[2]:
        se = CY.seasonality(p)
        if se.empty:
            st.info("Not enough years of data.")
        else:
            kit.chart(charts.bars(se["Month"].tolist(), se["Average"].to_numpy(), "Average return by calendar month", horizontal=False))
            kit.dataframe(se.drop(columns=["k"]), {"Month": dict(label="Month"), "Average": dict(label="Average", kind="pct_signed"), "Median": dict(label="Median", kind="pct_signed"),
                                                    "% positive": dict(label="% of years positive", kind="pct"), "Years": dict(label="Years", kind="int"),
                                                    "t-stat": dict(label="t-stat", kind="num2", help="|t| above ~3 is needed here because 12 months are examined at once."),
                                                    "Stands out after multiple-testing?": dict(label="Stands out?", kind="bool")})
            kit.note("‘Sell in May’-type patterns usually fail this test: with 12 months examined, some will look special by chance. Only trust months that stand out <i>and</i> have an economic reason.")
    with tabs[3]:
        ma = int(kit.s_slider("Moving-average length", "al_ma", "ma_days", 120, 300, 200, 10))
        tp = CY.trend_persistence(p, ma)
        kit.dataframe(tp, {"State": dict(label="State"), "% of time": dict(label="% of time", kind="pct"), "Typical spell (days, median)": dict(label="Typical spell (days)", kind="num"),
                           "Longest spell (days)": dict(label="Longest spell (days)", kind="int"), "Avg next-3M return": dict(label="Avg next 3M", kind="pct_signed"),
                           "% of next-3M returns positive": dict(label="% next 3M positive", kind="pct")})
        kit.note("Trend-following works only if being above the average has historically been followed by better returns than being below it. Compare the two rows.")
    T = None
    with tabs[4]:
        T = _tables()
        if T is None:
            kit.note("No economic data yet: Data Center → Update economic data.", "caution")
        else:
            me = RL.month_end(p)
            dim = st.selectbox("Regime dimension", list(T["labels"].columns), help="Which description of the economy to group history by.")
            cur = T["labels"][dim].dropna()
            if len(cur):
                st.write(f"Current state of **{dim}**: **{cur.iloc[-1]}**")
            rt = RL.regime_table(me, T["labels"][dim], h)
            if rt.empty:
                st.info("Not enough overlap between asset history and economic data.")
            else:
                kit.dataframe(rt, {"Regime": dict(label="Regime"), "Months observed": dict(label="Months", kind="int"),
                                   "Independent obs (≈)": dict(label="Independent obs", kind="num", help="Overlapping windows are not independent: roughly months ÷ look-ahead."),
                                   f"Avg {h}M forward return": dict(label=f"Avg next {h}M", kind="pct_signed"), "Median": dict(label="Median", kind="pct_signed"),
                                   "% of periods positive": dict(label="% positive", kind="pct"), "Worst": dict(label="Worst", kind="pct_signed"), "Best": dict(label="Best", kind="pct_signed"),
                                   "vs all months": dict(label="vs average", kind="pct_signed"),
                                   "p-value (luck?)": dict(label="p-value", kind="num2", help="Chance of seeing a difference this big by luck. Below 0.05 is suggestive, not proof."),
                                   "Confidence": dict(label="Sample size", help="Few independent observations = anecdote.")})
                kit.note("Read the <b>Sample size</b> and <b>p-value</b> columns before the averages. A striking average from 4 independent observations is a story, not evidence.")
            rec = T["recession"]
            if len(rec):
                shade = _spans(rec)
                kit.chart(charts.line((me / me.iloc[0]).to_frame(tick), "Asset (log) with NBER recessions shaded", log=True, shade=shade, height=340))
    with tabs[5]:
        if T is None:
            kit.note("Load economic data and open the ‘Economy & regimes’ tab first.", "caution")
        else:
            me = RL.month_end(p)
            lagmax = int(kit.s_slider("Max indicator lead (months)", "max_lag", "max_lag", 0, 24, 12))
            cols = list(T["feats"].columns)
            pick = kit.s_select("Indicator", "macro_pick", "macro_pick", cols, cols[0], format_func=lambda c: f"{DM.BY_ID[c].name if c in DM.BY_ID else c} ({c})")
            ll = RL.lead_lag(T["feats"][pick], me, h, lagmax)
            if ll.empty:
                st.info("Not enough overlapping months.")
            else:
                kit.chart(charts.bars([str(x) for x in ll["Indicator lead (months)"]], ll["Correlation"].to_numpy(), "Correlation of indicator (L months earlier) with the next-period return", horizontal=False, fmt="+.2f"))
                kit.dataframe(ll, {"Indicator lead (months)": dict(label="Lead (months)", kind="int"), "Correlation": dict(label="Correlation", kind="num2"), "Slope": dict(label="Slope", kind="num2"),
                                   "t-stat (HAC)": dict(label="t-stat", kind="num2", help="Robust to overlapping windows. |t| above 2 ≈ interesting."), "p-value": dict(label="p-value", kind="num2", help="Judged against the same two series slid out of alignment, so slow-moving indicators are not over-credited."), "Months": dict(label="Months", kind="int")})
                kit.note(f"With {len(ll)} lags examined, a couple will look significant by chance. A real link should be smooth across neighbouring lags and make economic sense.")
            st.subheader("Sensitivity to several drivers at once")
            drivers = st.multiselect("Drivers", cols, default=cols[:3], format_func=lambda c: DM.BY_ID[c].name if c in DM.BY_ID else c, help="Monthly changes in these indicators are regressed against the asset's monthly return.")
            if drivers:
                mb = RL.macro_betas(me, T["feats"][drivers])
                if not mb.empty:
                    kit.dataframe(mb.reset_index(), {c: dict(label=c) for c in mb.reset_index().columns})
    kit.disclaimer_footer()


def _spans(rec: pd.Series):
    r = rec.fillna(0).astype(int)
    out, start = [], None
    for d, v in r.items():
        if v == 1 and start is None:
            start = d
        if v == 0 and start is not None:
            out.append((start, d)); start = None
    if start is not None:
        out.append((start, r.index[-1]))
    return out
