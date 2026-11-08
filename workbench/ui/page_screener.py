from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import config, content, features as F, portfolio as PF
from . import charts, controls, data as D, kit


def jaccard(tags: dict[str, set[str]]) -> pd.DataFrame:
    t = list(tags)
    m = pd.DataFrame(0.0, index=t, columns=t)
    for a in t:
        for b in t:
            u = tags[a] | tags[b]
            m.loc[a, b] = len(tags[a] & tags[b]) / len(u) if u else 0.0
    return m


def render():
    S = controls.strategy_sidebar()
    kit.page_header("ETF Screener", "Rank every ETF by momentum, then build a diversified, risk-sized model portfolio – all from prices available on the chosen date.")
    uni, panel, qrep, ccy, names = controls.load_context(S)
    idx = panel.index
    c1, c2 = st.columns([1, 3])
    with c1:
        asof = kit.s_date("Analysis date", "asof_screen", "asof", idx[-1].date(), idx[0].date(), idx[-1].date())
    asof = pd.Timestamp(asof)
    feats = F.compute_features(panel, [asof], F.Params(S["ma_days"], S["min_history"], S["benchmark"]))
    if len(feats.dates) == 0:
        st.warning("No data on or before this date.")
        st.stop()
    d0 = feats.dates[0]
    snap = feats.snapshot(d0)
    rets = panel.prices.pct_change(fill_method=None).loc[:d0]
    with c2:
        all_t = sorted(snap.index)
        held = kit.s_multi("ETFs I already hold (optional)", "held_now", "held_now", all_t, [], format_func=lambda t: f"{t} – {names.get(t, '')[:40]}")
        blocked = kit.s_multi("ETFs I cannot buy on my platform (optional)", "blocked", "blocked", all_t, [], format_func=lambda t: f"{t} – {names.get(t, '')[:40]}")
    research, corr = PF.build_portfolio(snap, rets, S["benchmark"], S["params"], held=held, names=names)
    port, _ = PF.build_portfolio(snap, rets, S["benchmark"], S["params"], held=held, exclude=set(blocked), names=names)
    port["Allocation"] = port["PortfolioWeight"] * S["value"]
    n_act = int((port["Role"] != "Core").sum())

    kit.kpi_row([
        ("Eligible ETFs", str(int(snap["EligibleSignal"].sum())), "How many ETFs currently pass every entry rule (trend + positive excess return).", None),
        ("Selected", f"{n_act} of {S['n_active']}", "Momentum ETFs chosen. Unused slots stay in the core – the model never buys weak ETFs to fill slots.", None),
        ("Core / unused", kit.fmt_pct(float(port.loc[port['Role'] == 'Core', 'PortfolioWeight'].sum())), "Share of the portfolio sitting in the benchmark (or cash).", None),
        ("Universe ranked", str(len(snap)), "ETFs with enough price history and a current price.", None),
        ("Data as of", str(d0.date()), "Last trading date used. Nothing after this date influences the numbers.", None),
    ])
    if n_act == 0:
        kit.note("No ETF passes the entry rules on this date, so the model portfolio is 100% benchmark. This is a valid output – the model is not forced to pick.")
    flagged = qrep[qrep["Severity"].isin(["WARN", "FAIL"])] if not qrep.empty else pd.DataFrame()
    if len(flagged):
        kit.note(f"{len(flagged)} series have data-quality warnings (FAIL series are excluded). See <b>Data Center</b>.", "caution")

    t1, t2, t3, t4 = st.tabs(["Model portfolio", "Full ranking", "Decision audit", "Risk & overlap"])
    with t1:
        show = port[["Role", "Yahoo_Ticker", "Fund_Name", "MomentumPercentile", "Cluster", "PortfolioWeight", "Allocation"]].copy()
        kit.dataframe(show, {
            "Role": dict(label="Role", help=content.COL["Role"]), "Yahoo_Ticker": dict(label="Ticker"), "Fund_Name": dict(label="ETF"),
            "MomentumPercentile": dict(label="Momentum rank", kind="num", help=content.COL["MomentumPercentile"]),
            "Cluster": dict(label="Cluster", kind="int", help=content.COL["Cluster"]),
            "PortfolioWeight": dict(label="Weight", kind="pct", help=content.COL["PortfolioWeight"]),
            "Allocation": dict(label=f"Amount ({S['base']})", kind="money", help="Weight × portfolio value."),
        })
        if blocked:
            changed = set(research.Yahoo_Ticker) ^ set(port.Yahoo_Ticker)
            if changed:
                st.caption("Because of your platform restrictions the actionable portfolio differs from the pure research portfolio for: " + ", ".join(sorted(changed)))
        w = port.set_index("Yahoo_Ticker")["PortfolioWeight"]
        kit.chart(charts.bars(w.index, w.values, "Portfolio weights", fmt=".1%", diverging=False, height=max(240, 30 * len(w) + 90)))
        kit.explain("How this portfolio was built",
                    "1. ETFs must be above their long-term trend and have positive excess returns (skipping the latest month).\n"
                    "2. New purchases must rank in the top group (entry rank); ETFs you already hold may stay until they drop below the exit rank.\n"
                    "3. ETFs that move almost identically are grouped into clusters; the best of each cluster is picked first.\n"
                    "4. Money is split by risk (calmer ETFs get more) with caps per ETF and per cluster.\n"
                    "5. Anything unallocated stays in the benchmark (or cash).")
    with t2:
        meta_cols = [c for c in ["Asset_Class", "Region", "Exposure_Type"] if c in uni.columns]
        full = snap.join(uni.set_index("Yahoo_Ticker")[["Fund_Name"] + meta_cols])
        f1, f2 = st.columns(2)
        ac = f1.multiselect("Filter: asset class", sorted(full["Asset_Class"].dropna().unique()), help="Show only these asset classes.") if "Asset_Class" in full else []
        only_elig = f2.toggle("Only eligible ETFs", value=False, help="Hide ETFs that fail any entry rule.")
        if ac:
            full = full[full["Asset_Class"].isin(ac)]
        if only_elig:
            full = full[full["EligibleSignal"]]
        full = full.sort_values("MomentumPercentile", ascending=False).reset_index()
        spec = {"Yahoo_Ticker": dict(label="Ticker"), "Fund_Name": dict(label="ETF"),
                "MomentumPercentile": dict(label="Momentum rank", kind="num", help=content.COL["MomentumPercentile"]),
                "EligibleSignal": dict(label="Eligible", kind="bool", help=content.COL["EligibleSignal"]),
                "TrendGate": dict(label="Above trend", kind="bool", help=content.COL["TrendGate"])}
        for c, lab in [("Excess_6M", "6M excess"), ("Excess_12M", "12M excess"), ("Excess_6M_Skip1M", "6M excess (skip 1M)"),
                       ("Excess_12M_Skip1M", "12M excess (skip 1M)"), ("Ret_1M", "1M return"), ("Ret_3M", "3M return"),
                       ("Ret_12M", "12M return"), ("Dist_52wHigh", "vs 52w high"), ("Vol_63D", "Volatility"), ("DownsideVol_63D", "Downside vol")]:
            spec[c] = dict(label=lab, kind="pct_signed" if "Excess" in c or "Ret" in c or "Dist" in c else "pct", help=content.COL.get(c))
        spec["RelReg_R2"] = dict(label="Trend smoothness R²", kind="num2", help=content.COL["RelReg_R2"])
        cols = ["Yahoo_Ticker", "Fund_Name"] + meta_cols + ["MomentumPercentile", "EligibleSignal", "TrendGate", "Excess_6M", "Excess_12M",
                                                            "Excess_6M_Skip1M", "Excess_12M_Skip1M", "Ret_1M", "Ret_3M", "Ret_12M", "Dist_52wHigh",
                                                            "RelReg_R2", "Vol_63D", "DownsideVol_63D"]
        kit.dataframe(full[[c for c in cols if c in full.columns]], spec, height=560)
        st.caption(f"{len(full)} ETFs shown. Column headers have hover explanations. Sort by clicking a header.")
    with t3:
        failed = set(qrep.loc[qrep["Severity"].eq("FAIL"), "Ticker"]) if not qrep.empty else set()
        aud = PF.decision_audit(uni, snap, set(panel.prices.columns), failed, research, S["benchmark"], S["params"], S["min_history"])
        summ = aud["Decision"].value_counts().rename_axis("Outcome").reset_index(name="ETFs")
        kit.dataframe(summ, {"Outcome": dict(label="Outcome"), "ETFs": dict(label="ETFs", kind="int")})
        st.caption("Every ETF in the universe is accounted for. The reason is the FIRST rule that excluded it.")
        kit.dataframe(aud.sort_values("MomentumPercentile", ascending=False), {
            "Yahoo_Ticker": dict(label="Ticker"), "Fund_Name": dict(label="ETF"), "Decision": dict(label="Outcome"),
            "MomentumPercentile": dict(label="Momentum rank", kind="num", help=content.COL["MomentumPercentile"])}, height=420)
    with t4:
        if corr is None or corr.empty:
            st.info("No active candidates → no correlation pool to show.")
        else:
            st.caption(f"Correlation of daily returns over the last {S['corr_days']} days among the candidate pool. Values near 1 = the same bet.")
            kit.chart(charts.heatmap(corr, None, zmid=0.0, fmt=".2f", zmin=-1, zmax=1, height=max(360, 28 * len(corr) + 140)))
            pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:] if pd.notna(corr.loc[a, b])]
            if pairs:
                top = pd.DataFrame(pairs, columns=["ETF 1", "ETF 2", "Correlation"]).sort_values("Correlation", ascending=False).head(15)
                kit.dataframe(top, {"Correlation": dict(label="Correlation", kind="num2", help="1 = move identically.")})
        if "Economic_Risk_Tags" in uni.columns:
            sel_t = [t for t in port["Yahoo_Ticker"] if t in set(uni["Yahoo_Ticker"]) and t != S["benchmark"]]
            if len(sel_t) >= 2:
                tg = {r.Yahoo_Ticker: {x.strip().lower() for x in str(r.Economic_Risk_Tags or "").split(";") if x.strip()} for r in uni[uni.Yahoo_Ticker.isin(sel_t)].itertuples()}
                st.markdown("**Theme overlap of the selected ETFs** (shared descriptive tags – not holdings overlap)")
                kit.chart(charts.heatmap(jaccard(tg), None, zmid=None, fmt=".2f", diverging=False, zmin=0, zmax=1))
    kit.disclaimer_footer()
