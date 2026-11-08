from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import content, portfolio as PF
from . import charts, controls, kit

GROUPS = {"Asset class": "Asset_Class", "Sector": "Mandate_Sector_Exposure", "Country / region": "Mandate_Country_Exposure",
          "Factor": "Mandate_Factor_Exposure", "Theme": "Mandate_Theme_Exposure"}


def trailing_returns(panel, tickers, asof, months, bench) -> tuple[pd.Series, float]:
    start = pd.Timestamp(asof) - pd.DateOffset(months=months)
    ps = panel.pos([start, asof])
    A = panel.prices
    if ps[0] < 0:
        return pd.Series(np.nan, index=tickers), np.nan
    r = (A.iloc[ps[1]] / A.iloc[ps[0]] - 1).reindex(tickers)
    return r, float(r.get(bench, np.nan))


def render():
    import plotly.express as px

    S = controls.strategy_sidebar()
    kit.page_header("Market Map & Attribution", "See where returns were strongest in any month, then (after running a backtest) see exactly which holdings explain the strategy's gain or loss.")
    uni, panel, qrep, ccy, names = controls.load_context(S)
    dates = panel.index.to_series()
    months = dates.groupby(dates.dt.to_period("M")).max().tolist()
    labels = [d.strftime("%d %b %Y") for d in months]
    if "map_month" not in st.session_state or st.session_state["map_month"] not in labels:
        st.session_state["map_month"] = labels[-1]

    def shift(step):
        i = labels.index(st.session_state["map_month"])
        st.session_state["map_month"] = labels[max(0, min(len(labels) - 1, i + step))]

    b1, b2, b3 = st.columns([1, 6, 1])
    b1.button("← Month", on_click=shift, args=(-1,), help="Go back one month.")
    b3.button("Month →", on_click=shift, args=(1,), help="Go forward one month.")
    chosen = b2.select_slider("Month explorer", options=labels, key="map_month", help="Drag to travel through time. Dates are the last trading day of each month; the latest month may be incomplete.")
    asof = months[labels.index(chosen)]
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.selectbox("Return window (months)", [1, 3, 6, 12], index=0, help="Trailing period ending on the selected date.")
    colour = c2.selectbox("Colour shows", ["Total return", "Return vs benchmark"], help="Total return, or how much better/worse than the benchmark over the same window.")
    group_by = c3.selectbox("Group tiles by", list(GROUPS) + ["Correlation cluster"], help="How tiles are grouped. Groups come from each fund's mandate label, not its holdings. Correlation clusters are recomputed for every date.")
    size = c4.selectbox("Tile size", ["Equal", "Strategy weight"], help="'Strategy weight' sizes tiles by the weight held in the most recent completed backtest period (needs a backtest).")
    scale = st.slider("Colour range (± %)", 1, 60, 15, help="Colours saturate beyond this return, so months are visually comparable. Hover for exact figures.")

    tick = list(uni["Yahoo_Ticker"])
    ret, bret = trailing_returns(panel, tick, asof, horizon, S["benchmark"])
    frame = uni.copy()
    frame["Return"] = frame["Yahoo_Ticker"].map(ret)
    frame["Metric"] = frame["Return"] if colour == "Total return" else (1 + frame["Return"]) / (1 + bret) - 1
    res = st.session_state.get("bt_res")
    start = end = None
    weights = {}
    if res is not None and not res.periods.empty:
        past = res.periods[res.periods["End"] <= asof]
        if not past.empty:
            row = past.iloc[-1]
            start, end, dec = row["Start"], row["End"], row["Decision"]
            weights = res.alloc[res.alloc["Decision"] == dec].set_index("Ticker")["Weight"].to_dict()
    frame["Weight"] = frame["Yahoo_Ticker"].map(weights).fillna(0.0) * 100
    if group_by == "Correlation cluster":
        rr = panel.prices.pct_change(fill_method=None).loc[:asof].tail(126)
        corr = rr.corr(min_periods=40)
        cid = PF.greedy_clusters(sorted(frame["Yahoo_Ticker"]), corr, 0.8)
        frame["Group"] = frame["Yahoo_Ticker"].map(lambda t: f"Cluster {cid.get(t, 0)}")
        st.caption("Clusters are re-computed for each date (greedy, correlation ≥ 0.80 over 126 days); numbers are not permanent labels.")
    else:
        col = GROUPS[group_by]
        frame["Group"] = frame[col].fillna("Unclassified").replace("", "Unclassified") if col in frame else "Unclassified"
    valid = frame[frame["Metric"].notna()].copy()
    if size == "Strategy weight":
        valid = valid[valid["Weight"] > 0].copy()
        valid["Area"] = valid["Weight"]
    else:
        valid["Area"] = 1.0
    if valid.empty:
        st.info("Nothing to show for these settings. For ‘Strategy weight’, run a backtest first and pick a later month.")
    else:
        valid["Performance (%)"] = 100 * valid["Metric"]
        valid["Label"] = valid["Yahoo_Ticker"] + "<br>" + valid["Performance (%)"].map(lambda x: f"{x:+.1f}%")
        fig = px.treemap(valid, path=[px.Constant("ETF universe"), "Group", "Label"], values="Area", color="Performance (%)",
                         color_continuous_scale=["#C23B31", "#F2F2EE", "#0F7B6C"], range_color=[-scale, scale],
                         custom_data=["Fund_Name", "Yahoo_Ticker", "Performance (%)", "Weight"])
        fig.update_traces(root_color="#17232b", hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>Performance: %{customdata[2]:+.2f}%<br>Strategy weight: %{customdata[3]:.2f}%<extra></extra>")
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=35, b=15), title=f"{horizon}-month {colour.lower()} · {chosen}")
        kit.chart(fig)
        up, dn = int((valid["Performance (%)"] > 1).sum()), int((valid["Performance (%)"] < -1).sum())
        st.markdown(f"**{up} rising, {dn} falling, {len(valid) - up - dn} flat** (±1%). Benchmark over the window: **{bret:+.1%}**." if pd.notna(bret) else "Benchmark return unavailable.")
        summ = valid.groupby("Group")["Performance (%)"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        st.caption(f"Strongest group: {summ.index[0]} ({summ['mean'].iloc[0]:+.1f}%, {int(summ['count'].iloc[0])} ETFs); weakest: {summ.index[-1]} ({summ['mean'].iloc[-1]:+.1f}%). "
                   "Group figures are simple averages of overlapping funds, not investable indices.")
        kit.dataframe(valid[["Fund_Name", "Yahoo_Ticker", "Group", "Performance (%)", "Weight"]].sort_values("Performance (%)", ascending=False), {
            "Fund_Name": dict(label="ETF"), "Yahoo_Ticker": dict(label="Ticker"), "Group": dict(label="Group"),
            "Performance (%)": dict(label="Performance (%)", kind="num2"), "Weight": dict(label="Strategy weight (%)", kind="num2")}, height=300)
    st.caption(f"{len(frame) - len(valid)} ETFs have no usable price at the window start/end and are omitted. No fund-size data is available for tile area.")

    st.divider()
    st.subheader("Strategy timeline & what drove it")
    if res is None or start is None:
        kit.note("Run the <b>Backtest Lab</b>, then return here and select a month after its first completed holding period.")
        kit.disclaimer_footer()
        return
    w = res.wealth[["Strategy", "Benchmark"]]
    dd = w / w.cummax() - 1
    wide_w = res.alloc.pivot_table(index="Date", columns="Ticker", values="Weight", aggfunc="sum").fillna(0.0)
    wide_w.loc[res.wealth.index[-1]] = wide_w.iloc[-1]
    kit.chart(charts.timeline(w, dd, wide_w, start, end))
    st.markdown(f"**Highlighted holding period: {start:%d %b %Y} → {end:%d %b %Y}.** These holdings were decided on {pd.Timestamp(dec):%d %b %Y}, before the returns below happened.")
    attr = res.alloc[res.alloc["Decision"] == dec].copy().sort_values("Contribution")
    if not attr.empty:
        kit.chart(charts.bars(attr["Ticker"] + " – " + attr["Fund_Name"].str.slice(0, 38), attr["Contribution"], "Contribution to portfolio return", fmt="+.2%"))
        kit.dataframe(attr[["Ticker", "Fund_Name", "Weight", "AssetReturn", "Contribution"]], {
            "Ticker": dict(label="Ticker"), "Fund_Name": dict(label="ETF"), "Weight": dict(label="Weight at start", kind="pct"),
            "AssetReturn": dict(label="ETF return", kind="pct_signed"), "Contribution": dict(label="Contribution", kind="pct_signed", help="Weight × return.")})
        pr = past.iloc[-1]
        st.write(f"Contributions {attr['Contribution'].sum():+.2%} − trading cost {pr['Cost']:.2%} → **{pr['NetReturn']:+.2%}** strategy return, versus benchmark {pr['BenchmarkReturn']:+.2%}.")
        lo = attr[attr["Contribution"] < 0].head(3)
        hi = attr[attr["Contribution"] > 0].tail(3).iloc[::-1]
        st.write("Largest detractors: " + ("; ".join(f"{r.Fund_Name} ({r.Contribution:+.2%})" for r in lo.itertuples()) or "none") + ".")
        st.write("Largest contributors: " + ("; ".join(f"{r.Fund_Name} ({r.Contribution:+.2%})" for r in hi.itertuples()) or "none") + ".")
        kit.note("This is accounting, not causation: it says <i>which holdings</i> moved the portfolio, not <i>why</i> they moved. For the ‘why’, use the Asset Lab and Macro & Cycles pages.")
    kit.disclaimer_footer()
