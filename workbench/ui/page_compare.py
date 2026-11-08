from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config, content, data_macro as DM, quality, relationships as RL, sectors as SC
from . import charts, controls, data as D, kit
from .page_asset import _spans

SECTOR_SET = list(config.SPDR_SECTORS) + ["SPY"]


def _label(t: str, names: dict) -> str:
    if t in config.SPDR_SECTORS:
        return f"{t} – {config.SPDR_SECTORS[t]} (US sector, USD)"
    if t == "SPY":
        return "SPY – S&P 500 ETF (US market, USD)"
    return f"{t} – {names.get(t, '')[:40]}".rstrip(" –")


def _pool(S: dict):
    """ticker -> (price Series, source).  ETFs come filtered by the quality rules; cached stocks are screened here."""
    pool, notes, names = {}, [], {}
    try:
        uni = D.load_universe(S["universe"])
        names = dict(zip(uni["Yahoo_Ticker"], uni["Fund_Name"]))
        store = D.etf_store()
        panel, _, _ = D.etf_panel(S["universe"], S["base"], S["benchmark"], D.mtime(store.f_prices), D.mtime(store.f_meta))
        if panel is not None:
            for t in panel.prices.columns:
                pool[t] = (panel.prices[t].dropna(), "ETF universe")
    except Exception as e:  # noqa: BLE001
        notes.append(f"ETF universe unavailable: {type(e).__name__}")
    native = D.stock_store().load()
    if not native.empty:
        native, repaired = quality.repair_unit_frame(native)
        q = quality.quality_report(native, native.index.max(), repaired=repaired)
        bad = quality.failing_tickers(q)
        for t in native.columns:
            if t in pool:
                continue
            if t in bad:
                notes.append(f"{t} excluded: failed data-quality checks")
                continue
            pool[t] = (native[t].dropna(), "Sector ETF" if t in SECTOR_SET else "Stock cache")
    return pool, names, notes


def render():
    S = controls.strategy_sidebar()
    kit.page_header("Compare with Economy", "Which ETFs, stocks and sectors move with an economic measure – and is the link strong enough to take seriously, or just a coincidence?")
    scheme_q = 0.80
    T = D.macro_tables(D.macro_stamp(), "level", scheme_q)
    if T is None:
        kit.note("No economic data yet. Open <b>Data Center → Update economic data</b>.", "caution")
        st.stop()
    pool, names, notes = _pool(S)
    X = pd.concat([T["comp"], T["feats"]], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    comps = list(T["comp"].columns)
    opts = comps + [c for c in T["feats"].columns if c not in comps]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ind = kit.s_select("Economic indicator", "cmp_indicator", "cmp_indicator", opts, comps[0],
                           format_func=lambda c: f"Composite: {c}" if c in comps else f"{DM.BY_ID[c].name if c in DM.BY_ID else c} ({c})")
    with c2:
        h = int(kit.s_slider("Return window (months)", "cmp_h", "cmp_h", 1, 12, 3))
    with c3:
        lag = int(kit.s_slider("Indicator lead (months)", "cmp_lag", "cmp_lag", 0, 12, 0))
    avail = [t for t in pool if t not in SECTOR_SET]
    d1, d2 = st.columns([3, 1])
    default = [S["benchmark"]] if S["benchmark"] in avail else avail[:1]
    with d1:
        chosen = kit.s_multi("Assets to compare", "cmp_assets", "cmp_assets", avail, default, format_func=lambda t: _label(t, names))
    with d2:
        use_sec = kit.s_toggle("Add sector ETFs", "cmp_sector", "cmp_sector", True)
    have_sec = [t for t in SECTOR_SET if t in pool]
    young_ok = {"XLRE", "XLC"}                      # these two really only exist since 2015 / 2018
    too_short = [t_ for t_ in have_sec if t_ not in young_ok and pool[t_][0].index.min() > pd.Timestamp("2005-01-01")]
    if use_sec and (len(have_sec) < len(SECTOR_SET) or too_short):
        if too_short:
            kit.note("Your saved sector ETFs only go back to " + str(min(pool[t_][0].index.min() for t_ in too_short).year) + ", but they have been trading since 1998. Extending them gives the comparison many more years (and more recessions) to learn from.", "caution")
        else:
            kit.note(f"Sector ETFs are not downloaded yet ({len(have_sec)} of {len(SECTOR_SET)} present). They have 25+ years of free history – far more reliable evidence than recently launched ETFs.", "caution")
        if kit.wide(st.button, "Download sector ETFs (free)" if not too_short else "Extend sector ETFs to 25 years (free)", type="primary"):
            with st.spinner("Downloading sector ETFs…"):
                res = D.stock_store().update(SECTOR_SET, history_years=25)
            st.cache_data.clear()
            if res.errors:
                st.warning("Problems:\n- " + "\n- ".join(res.errors[:10]))
            st.rerun()
    book = SC.SectorBook()
    stocks = [t_ for t_ in chosen if pool[t_][1] == "Stock cache"]
    match = {t_: book.etf(t_)[0] for t_ in stocks}
    todo = [t_ for t_, e in match.items() if e is None and book.etf(t_)[1] == "unknown"]
    if stocks:
        with st.expander("Sector ETF matched to each stock", expanded=bool(todo)):
            st.caption("Each stock is drawn next to its own sector ETF in the Overlay tab, so you can see whether it moves with its sector or on its own.")
            if todo and kit.wide(st.button, f"Look up sector for {len(todo)} stock(s) on Yahoo"):
                with st.spinner("Asking Yahoo Finance…"):
                    fails = book.lookup_missing(todo)
                st.session_state["cmp_sector_fail"] = fails
                st.rerun()
            for t_ in stocks:
                e, how = book.etf(t_)
                cur = f"{e} – {config.SPDR_SECTORS[e]}" if e else "(none)"
                pick = st.selectbox(f"{t_}", SC.MANUAL_CHOICES, index=SC.MANUAL_CHOICES.index(cur) if cur in SC.MANUAL_CHOICES else 0,
                                    key=f"cmp_sec_{t_}", help=f"Found via: {how}. Change it if the match is wrong; your choice is remembered.")
                new = None if pick == "(none)" else pick.split(" – ")[0]
                if new != e and not (new is None and e is None):
                    book.set_manual(t_, new if new else "(none)")
                    st.rerun()
            for t_, why in st.session_state.get("cmp_sector_fail", {}).items():
                st.caption(f"Could not find a sector for {t_}: {why}. Choose one by hand above.")
    missing_etf = sorted({e for e in match.values() if e and e not in pool})
    if missing_etf:
        kit.note("The sector ETF(s) matching your stock(s) – " + ", ".join(missing_etf) + " – are not downloaded yet, so they cannot be drawn. Use <b>Download sector ETFs</b> above.", "caution")
    sel = list(dict.fromkeys(chosen + [e for e in match.values() if e and e in pool] + (have_sec if use_sec else [])))
    if not sel:
        kit.note("Choose at least one asset (or download the sector ETFs).")
        st.stop()
    e1, e2 = st.columns(2)
    with e1:
        min_m = int(kit.s_slider("Minimum months of data", "cmp_minmonths", "cmp_minmonths", 36, 240, 84, 12))
    with e2:
        max_lag = int(kit.s_slider("Largest lead to scan", "cmp_maxlag", "cmp_maxlag", 3, 24, 12))
    me = pd.concat({t: RL.month_end(pool[t][0]) for t in sel}, axis=1)
    x = X[ind].dropna()
    table, excluded = RL.compare_assets(x, me, h, lag, min_m)
    if x.empty or table.empty:
        kit.note("None of the selected assets has enough overlapping history with this indicator at these settings. Lower ‘Minimum months’, change the indicator, or add the sector ETFs (long history).", "caution")
        if excluded:
            st.caption("Excluded for too little data: " + ", ".join(f"{t} ({n} months)" for t, n in excluded))
        st.stop()
    n_tests = len(table)
    nsup = int((table["Verdict"] == "Supported").sum())
    kit.kpi_row([("Assets compared", f"{n_tests}", "Assets with enough data at these settings.", None),
                 ("Supported links", f"{nsup}", "Pass all three checks: false-discovery-adjusted, stable across halves, stable across neighbouring lags.", None),
                 ("Indicator last known", str(x.index[-1].date()), "Latest month for which the indicator was published at the time.", None),
                 ("Typical history", f"{int(table['Months'].median())} months", None, None)])
    if nsup == 0:
        kit.verdict("No asset shows a link that survives the reliability checks at these settings. That is a normal and informative result – most apparent links are coincidence.", "neutral")
    tabs = st.tabs(["Overlay", "Ranking", "Lag map", "Rolling stability", "Data reliability"])
    shade = _spans(T["recession"]) if len(T["recession"]) else None
    with tabs[0]:
        shown_assets = list(table["Asset"])
        wanted = []
        for a in chosen:
            if a in shown_assets:
                wanted.append(a)
            e = match.get(a)
            if e and e in shown_assets:
                wanted.append(e)
        wanted = list(dict.fromkeys(wanted))[:6] or shown_assets[:1]
        # the key changes with the selection, so the picker follows what you chose above instead of sticking to an old pick
        show = st.multiselect("Assets to overlay", shown_assets, default=wanted, format_func=lambda t: _label(t, names),
                              key="cmp_overlay_" + "|".join(chosen) + "|" + "|".join(f"{a}>{match.get(a)}" for a in chosen) + f"|{h}|{lag}|{use_sec}|{len(shown_assets)}", max_selections=6,
                              help="Starts with the assets chosen above plus each stock's own sector ETF. Add or remove any compared asset (up to 6).")
        if not show:
            st.info("Pick at least one asset to draw.")
        else:
            cols = {}
            for a in show:
                r12 = me[a].pct_change(12)
                role = ""
                if a in match.values() and a not in chosen:
                    role = " (sector of " + ", ".join(k for k, v in match.items() if v == a) + ")"
                cols[f"{a}{role}: 12-month return"] = (r12 - r12.mean()) / r12.std()
            cols[f"{ind}: indicator"] = (x - x.mean()) / x.std()
            z = pd.DataFrame(cols)
            # each line is drawn over its OWN history; a short-history ETF must not cut everyone else down to a few years
            z = z.loc[z.iloc[:, :-1].notna().any(axis=1)]
            short = [c for c in z.columns[:-1] if z[c].first_valid_index() is not None and z[c].first_valid_index() > z.index.min() + pd.DateOffset(years=2)]
            if short:
                st.caption("Shorter history: " + ", ".join(f"{c.split(':')[0]} starts {z[c].first_valid_index().date()}" for c in short) + ". Each line is drawn for as long as its data exists.")
            kit.chart(charts.line(z, "All lines rescaled to the same units (0 = its own average) so they can share one axis", shade=shade, hline=0.0, hover_fmt=".2f", height=400), key="cmp_ov_chart_" + "|".join(show) + f"|{ind}|{h}|{lag}")
            st.caption("Grey bands: NBER recessions (hindsight, display only). Rescaling is for viewing only; all statistics use the unscaled monthly data.")
    with tabs[1]:
        kit.explain("How to read the Verdict", "**Supported** = significant after correcting for the number of assets compared (q < 0.10) AND same direction in both halves of the history AND same direction at the neighbouring lags. "
                    "**Suggestive only** = raw p < 0.05 but at least one check failed. **No reliable link** = indistinguishable from chance.")
        t = table.copy()
        t["Asset"] = t["Asset"].map(lambda a: _label(a, names))
        kit.dataframe(t, {"Asset": dict(label="Asset"), "Verdict": dict(label="Verdict"), "Correlation": dict(label="Correlation", kind="num2", help="+1: indicator and later return rise together; -1: opposite. Near 0: no linear link."),
                          "t-stat (HAC)": dict(label="t-stat", kind="num2", help="Allows for overlapping windows. |t| above 2 is interesting; above 3 is strong."),
                          "p-value": dict(label="p-value", kind="num2", help="Chance of a link this strong by coincidence, judged against the same two series slid out of alignment (so slow-moving series are not over-credited)."), "q-value (false-discovery adj.)": dict(label="q-value", kind="num2", help="p-value corrected for comparing many assets at once."),
                          "Months": dict(label="Months", kind="int"), "Independent obs (≈)": dict(label="Independent obs", kind="num", help="Overlapping windows are not independent; roughly months ÷ window."),
                          "First half": dict(label="First half", kind="num2"), "Second half": dict(label="Second half", kind="num2"),
                          "Same direction in both halves": dict(label="Stable halves", kind="bool"), "Same direction at neighbouring lags": dict(label="Stable lags", kind="bool"),
                          "Slope": dict(label="Slope", kind="num2", help="Change in the asset's next-window return per 1-unit change in the indicator.")}, height=420)
        if excluded:
            st.caption("Excluded for too little data: " + ", ".join(f"{a} ({n} months)" for a, n in excluded))
        kit.note(f"You are comparing {n_tests} assets, and every other indicator or lead you try is another comparison. The q-value corrects for the assets here, <b>not</b> for the indicators you browse – so treat even ‘Supported’ as a lead to investigate, and ask whether there is an economic reason.")
    with tabs[2]:
        corr, q = RL.lag_grid(x, me, h, max_lag, min_m)
        shown = corr.where(q < 0.10)
        shown.index = [_label(a, names)[:34] for a in shown.index]
        shown.columns = [f"{c}m" for c in shown.columns]
        if not shown.notna().to_numpy().any():
            kit.note("No lead/asset combination survives the false-discovery correction for this indicator – i.e. nothing here is distinguishable from chance. That is a result, not an error.", "note")
        kit.chart(charts.heatmap(shown, "Correlation by lead – only cells that survive the false-discovery correction are coloured", zmid=0.0, fmt=".2f", height=max(320, 28 * len(shown) + 120), colorbar_title="corr"))
        st.caption("Blank = not distinguishable from chance after correcting for all cells shown. A real relationship usually forms a smooth block, not one isolated cell.")
    with tabs[3]:
        ra = list(table["Asset"])
        who = st.selectbox("Asset", ra, index=ra.index(chosen[-1]) if chosen and chosen[-1] in ra else 0, format_func=lambda t: _label(t, names),
                           key="cmp_roll_asset_" + "|".join(chosen), help="Asset for the rolling correlation. Follows the last asset you chose above.")
        win = int(kit.s_slider("Window (months)", "cmp_window", "cmp_window", 24, 120, 60, 6))
        fwd = me[who].shift(-h) / me[who] - 1.0
        rc = RL.rolling_corr(x.shift(lag), fwd, win).dropna()
        if rc.empty:
            st.info("Not enough history for this window.")
        else:
            kit.chart(charts.line(rc.to_frame("Rolling correlation"), f"Rolling {win}-month correlation with the later {h}-month return", shade=shade, hline=0.0, hover_fmt=".2f", height=340), key=f"cmp_roll_chart_{who}|{ind}|{h}|{lag}|{win}")
            share = float((rc > 0).mean())
            st.write(f"The correlation was positive {share:.0%} of the time. A dependable link keeps the same sign most of the time; one that flips is not something to build a decision on.")
    with tabs[4]:
        rows = [{"Asset": _label(t, names), "Source": pool[t][1], "First price": pool[t][0].index[0].date(), "Last price": pool[t][0].index[-1].date(),
                 "Months used": int(table.loc[table["Asset"] == t, "Months"].iloc[0]) if (table["Asset"] == t).any() else 0,
                 "Status": "Used" if (table["Asset"] == t).any() else "Excluded: too little overlap"} for t in sel]
        kit.dataframe(pd.DataFrame(rows), {"Asset": dict(label="Asset"), "Source": dict(label="Source"), "First price": dict(label="First price", kind="date"), "Last price": dict(label="Last price", kind="date"),
                                           "Months used": dict(label="Months used", kind="int"), "Status": dict(label="Status")})
        for n in notes[:8]:
            st.caption(n)
        kit.note("Data safeguards: price series that fail the quality checks (impossible jumps, unit switches, stale feeds) are excluded automatically; the indicator is only used from the date it would really have been published; sector ETFs are USD while your ETFs are in your base currency, which matters little for correlations with US-centred indicators but is worth remembering.")
    kit.disclaimer_footer()
