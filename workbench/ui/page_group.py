from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import config, content, cycles as CY, data_macro as DM, data_ref as DR, features as F, stats as S, stocklab as SL
from . import charts, data as D, kit

US_GROUPS = {"S&P 500", "Nasdaq-100", "Dow Jones 30"}
SAMPLE = "Sample: sector ETFs + large caps"


@st.cache_data(show_spinner="Reading index list…", ttl=6 * 3600)
def _members(source: str, custom: str) -> tuple[pd.DataFrame, str]:
    if source == "Custom list":
        return DR.parse_custom_list(custom), "CUSTOM"
    if source == SAMPLE:
        return pd.read_csv(config.UNIVERSE_DIR / "sample_stock_groups.csv").assign(SubIndustry=""), "SAMPLE"
    return DR.fetch_constituents(source, DM.default_http_get)


def render():
    kit.page_header("Group Explorer", "Pick an index, a sector or your own list of stocks. See which members lead, how broad the move is, and why a single stock behaved as it did.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        source = kit.s_select("Group of stocks", "group_source", "group_source", list(DR.INDEXES) + [SAMPLE, "Custom list"], SAMPLE)
    custom = ""
    if source == "Custom list":
        custom = st.text_area("Tickers (one per line, or TICKER,Name,Group)", "AAPL\nMSFT\nNVDA\nJPM\nXOM", height=120,
                              help="Yahoo Finance symbols. London: VOD.L, Germany: SAP.DE. Keep one market per group so currencies match.")
    try:
        members, src_state = _members(source, custom)
    except Exception as e:  # noqa: BLE001
        kit.note(f"Could not read the list for {source}: {type(e).__name__}: {e}. Try the sample group or a custom list.", "caution")
        st.stop()
    try:
        from .. import sectors as _SC
        _SC.SectorBook().remember(members)
    except Exception:  # noqa: BLE001
        pass
    default_ix = DR.INDEXES.get(source, {}).get("benchmark", "^GSPC")
    with c2:
        index_t = kit.s_text("Comparison index", f"gx_{source}", "group_index", default_ix)
    with c3:
        max_n = int(kit.s_slider("Max stocks", "group_max", "group_max", 5, 520, min(60, max(5, len(members)))))
    with c4:
        min_hist = int(kit.s_slider("Min history (days)", "group_min_hist", "group_min_hist", 126, 756, 252, 21))
    members = members.head(max_n)
    if src_state.startswith("CACHED"):
        kit.note(f"Showing the last saved copy of the {source} list ({src_state}).", "caution")
    st.caption(f"{len(members)} stocks selected from **{source}**.")
    kit.note("<b>Survivorship bias:</b> free sources list the index's <i>current</i> members only. Stocks that were dropped or went bust are missing, so historic group performance looks better than it really was. Treat long-history results as optimistic.", "caution")
    tickers = list(dict.fromkeys(members["Ticker"].tolist() + [index_t]))
    store = D.stock_store()
    if kit.wide(st.button, "Download / update prices for this group", type="primary", help="First download of ~500 stocks takes several minutes; later updates fetch only new days."):
        bar = st.progress(0.0, text="Downloading…")
        res = store.update(tickers, history_years=12, progress=lambda m: bar.progress(0.5, text=m))
        bar.progress(1.0, text="Done")
        st.cache_data.clear()
        if res.errors:
            st.warning("Problems:\n- " + "\n- ".join(res.errors[:15]))
    panel, q = D.stock_panel(tuple(tickers), D.mtime(store.f_prices))
    if panel is None or index_t not in panel.prices.columns:
        kit.note(f"No prices for the comparison index <b>{index_t}</b> or the group yet. Click the download button above.", "caution")
        st.stop()
    have = [t for t in members["Ticker"] if t in panel.prices.columns]
    miss = sorted(set(members["Ticker"]) - set(have))
    if miss:
        st.caption(f"{len(miss)} without usable prices (excluded): {', '.join(miss[:30])}{' …' if len(miss) > 30 else ''}")
    if len(have) < 5:
        kit.note("Fewer than 5 stocks have data – group statistics need more.", "caution")
        st.stop()
    prices = panel.prices[have]
    index_px = panel.prices[index_t]
    asof = panel.index[-1]
    snap = SL.group_snapshot(panel, members, index_t, asof, 200, min_hist)
    tabs = st.tabs(["Ranking", "Breadth & health", "Group vs index", "Who drove it", "Stock deep-dive"])
    with tabs[0]:
        ranked = snap.reset_index()
        kit.kpi_row([
            ("Stocks ranked", f"{len(ranked)}", None, None),
            ("Above 200-day trend", kit.fmt_pct(ranked["TrendGate"].mean(), 0), "Share of stocks in an uptrend.", None),
            ("Pass full entry test", f"{int(ranked['EligibleSignal'].sum())}", "Uptrend AND outperforming the index over 6 and 12 months excluding the latest month.", None),
            ("Median 12M return", kit.fmt_pct(ranked["Ret_12M"].median(), 1, True), None, None),
        ])
        spec = {"Ticker": dict(label="Ticker"), "Name": dict(label="Company"), "Sector": dict(label="Sector"),
                "MomentumScore": dict(label="Momentum score", kind="num2", help=content.COL["MomentumScore"]),
                "MomentumPercentile": dict(label="Rank percentile", kind="num", help=content.COL["MomentumPercentile"]),
                "TrendGate": dict(label="In uptrend", kind="bool", help=content.COL["TrendGate"]),
                "EligibleSignal": dict(label="Passes entry test", kind="bool"),
                "Ret_1M": dict(label="1M", kind="pct_signed"), "Ret_3M": dict(label="3M", kind="pct_signed"), "Ret_12M": dict(label="12M", kind="pct_signed"),
                "Excess_6M": dict(label="6M vs index", kind="pct_signed"), "Excess_12M": dict(label="12M vs index", kind="pct_signed"),
                "Vol_1Y": dict(label="Volatility (1Y)", kind="pct", help="Annualised ups-and-downs over the past year."),
                "Beta_1Y": dict(label="Beta", kind="num2", help="1.5 = tends to move 1.5× the index. 0.5 = half."),
                "MaxDD_3Y": dict(label="Worst fall (3Y)", kind="pct")}
        kit.dataframe(ranked[[c for c in spec if c in ranked.columns]], spec, height=520)
        if members["Sector"].nunique() > 1 and "Sector" in snap:
            st.subheader("By sector")
            ss = SL.sector_summary(snap.dropna(subset=["Sector"])).reset_index()
            kit.dataframe(ss, {"Sector": dict(label="Sector"), "Members": dict(label="Stocks", kind="int"), "Avg 1M": dict(label="Avg 1M", kind="pct_signed"),
                               "Avg 3M": dict(label="Avg 3M", kind="pct_signed"), "Avg 12M": dict(label="Avg 12M", kind="pct_signed"),
                               "% above 200d trend": dict(label="% in uptrend", kind="num"), "Avg momentum percentile": dict(label="Avg rank percentile", kind="num")})
    with tabs[1]:
        br = SL.breadth_series(prices).dropna(how="all").tail(1260)
        kit.explain("How to read breadth", "Breadth asks: is the move shared by most members or carried by a few? An index can hit new highs while fewer and fewer stocks follow – a classic sign of a narrowing, more fragile rally.")
        kit.chart(charts.line(br[[c for c in br.columns if "52-week" not in c]], "Share of stocks (%)", hover_fmt=".0f", height=380))
        kit.chart(charts.line(br[[c for c in br.columns if "52-week" in c]], "New highs vs new lows (%)", hover_fmt=".0f", height=300))
        ac = SL.avg_pairwise_corr(prices.tail(1260))
        ds = SL.dispersion_series(prices).dropna().tail(1260)
        a, b = st.columns(2)
        with a:
            if len(ac):
                kit.chart(charts.line(ac.to_frame(), "Average pairwise correlation", hover_fmt=".2f", height=300))
            st.caption("High correlation = stocks move together (little diversification, market-driven). Low = stock-picking matters.")
        with b:
            kit.chart(charts.line(ds.to_frame(), "Dispersion of 3-month returns", yfmt=".0%", hover_fmt=".1%", height=300))
            st.caption("How far apart the best and worst members are.")
    with tabs[2]:
        cv = SL.composite_vs_index(prices, index_px)
        kit.chart(charts.line(cv[["Equal-weight members", "Index"]], "Growth of 1 (log scale)", log=True, height=380))
        kit.chart(charts.line(cv[["Equal ÷ Index"]], "Equal-weight ÷ index (rising = average stock beating the index)", height=280, hline=1.0))
        kit.note("When the ratio falls, a few heavyweight stocks are doing the work and the average member is lagging (concentration). When it rises the rally is broad. Note: your list may differ from the index's real membership/weights.")
    with tabs[3]:
        per = st.selectbox("Period", ["1 month", "3 months", "6 months", "12 months", "3 years"], index=2, help="Which window to attribute.")
        mon = {"1 month": 1, "3 months": 3, "6 months": 6, "12 months": 12, "3 years": 36}[per]
        ct = SL.contribution(prices, asof - pd.DateOffset(months=mon), asof)
        st.metric("Equal-weight group return", kit.fmt_pct(ct["Contribution"].sum(), 1, True), help="Buy-and-hold return of the group in the window.")
        top = pd.concat([ct.head(10), ct.tail(10)]).drop_duplicates()
        kit.chart(charts.bars(top.index.tolist(), top["Contribution"].to_numpy(), "Largest positive and negative contributions", fmt="+.2%"))
        share = ct["Contribution"].clip(lower=0)
        if share.sum() > 0:
            st.caption(f"The top 5 contributors supplied {share.nlargest(5).sum() / share.sum():.0%} of the total positive contribution.")
    with tabs[4]:
        pick = kit.s_select("Stock", "stock_pick", "stock_pick", have, have[0])
        win = int(kit.s_slider("Estimation window (days)", "factor_window", "factor_window", 252, 1260, 756, 126))
        p = prices[pick].dropna()
        prof = CY.asset_profile(p)
        kit.kpi_row([("Years of data", f"{prof['Years of data']:.1f}", None, None), ("CAGR", kit.fmt_pct(prof["Annualised return (CAGR)"], 1, True), "Compound annual growth rate over the available history.", None),
                     ("Volatility", kit.fmt_pct(prof["Annualised volatility"], 1), None, None), ("Worst fall", kit.fmt_pct(prof["Worst drawdown"], 0), None, None)])
        both = pd.concat([p / p.iloc[0], index_px.reindex(p.index).ffill()], axis=1).dropna()
        both = both / both.iloc[0]
        both.columns = [pick, index_t]
        kit.chart(charts.line(both, "Growth of 1 (log scale)", log=True, height=360))
        dd = S.drawdown_series(p / p.iloc[0]).to_frame(pick)
        kit.chart(charts.drawdown(dd))
        r = p.pct_change(fill_method=None)
        mr = SL.market_regression(r, index_px.pct_change(fill_method=None), win)
        if mr:
            st.subheader("How much is the market, how much is the company?")
            kit.kpi_row([("Beta", kit.fmt_num(mr["beta"], 2), "Sensitivity to the comparison index.", None),
                         ("R² (market-explained)", kit.fmt_pct(mr["r2"], 0), "Share of daily ups and downs explained by the index.", None),
                         ("Stock-specific volatility", kit.fmt_pct(mr["idio_vol"], 0), "Annual volatility NOT explained by the index.", None),
                         ("Downside beta", kit.fmt_num(mr["down_beta"], 2), "Sensitivity on days the index falls. Higher than upside beta = hurts more in falls.", None)])
            st.write(f"Over the last {mr['n']} days, **{pick}** moved {mr['beta']:.2f}× the index; about **{mr['r2']:.0%}** of its day-to-day movement came from the market and the rest from company-specific news. Annualised alpha {mr['alpha_annual']:+.1%} (t = {mr['alpha_t']:.1f}; |t| below 2 means it is indistinguishable from luck).")
            rb = SL.rolling_beta(r, index_px.pct_change(fill_method=None), 252)
            if len(rb):
                kit.chart(charts.line(rb.to_frame(), "Rolling 1-year beta", height=280, hline=1.0))
        fac = D.french_factors(D.factor_stamp())
        if fac is None:
            kit.note("Style-factor analysis (value, size, quality, momentum…) needs the factor download: Data Center → Style factors.")
        elif source not in US_GROUPS and source not in (SAMPLE, "Custom list"):
            kit.note("Style factors are US-based. For non-US stocks the factor analysis below is only a rough lens.", "caution")
        if fac is not None:
            fr = SL.factor_regression(r, fac, window=win)
            if fr:
                st.subheader("Which investing styles explain the move?")
                tab = fr["betas"].reset_index()
                kit.dataframe(tab, {c: dict(label=c if c != "index" else "Factor") for c in tab.columns} | {"coef": dict(label="Exposure", kind="num2", help="How strongly the stock tracks this factor."), "t": dict(label="t-stat", kind="num2", help="|t| above 2 = unlikely to be chance.")})
                ct2 = fr["contrib"].sort_values()
                kit.chart(charts.bars(ct2.index.tolist(), ct2.to_numpy(), f"Approx. contribution to excess return, {fr['start'].date()} → {fr['end'].date()}", fmt="+.1%"))
                st.caption(f"Factors explain {fr['r2']:.0%} of daily variation. Contributions are an additive approximation; the unexplained part is shown as Alpha.")
    kit.disclaimer_footer()
