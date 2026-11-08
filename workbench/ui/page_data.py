from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from .. import config, content, data_macro as DM, data_prices as DP, data_ref as DR, selftest
from . import controls, data as D, kit

log = logging.getLogger("workbench.ui")


def _all_fx_tickers(ct: pd.DataFrame) -> list[str]:
    out = set()
    for base in config.BASE_CURRENCIES:
        out.update(DP.required_fx(ct, base))
    return sorted(out)


def update_etf_prices(universe_name: str, full: bool, years: int, detect: bool):
    uni = D.load_universe(universe_name)
    store = D.etf_store()
    tickers = list(dict.fromkeys(uni["Yahoo_Ticker"].tolist() + list(config.BENCHMARKS.values())))
    meta = store.load_meta()
    cached = meta.get("currency", {})
    bar = st.progress(0.0, text="Checking trading currencies…")
    if detect:
        todo = [t for t in tickers if t not in cached and t not in D.read_overrides()]
        for i, t in enumerate(todo):
            c = DP.yahoo_currency_lookup(t)
            if c:
                cached[t] = c
            bar.progress((i + 1) / max(len(todo), 1) * 0.3, text=f"Detecting currency {i + 1}/{len(todo)}: {t}")
        meta["currency"] = cached
        store.save_meta(meta)
    ct = DP.resolve_currencies(tickers, overrides=D.read_overrides(), cached=cached)
    fx = _all_fx_tickers(ct)
    bar.progress(0.35, text="Downloading prices…")
    res = store.update(tickers + fx, history_years=years, full=full, progress=lambda m: bar.progress(0.5, text=m))
    meta = store.load_meta()
    meta["currency"] = cached
    store.save_meta(meta)
    bar.progress(1.0, text="Done")
    st.cache_data.clear()
    return res


def render():
    kit.page_header("Data Center", "Download free market and economic data, see exactly what was received, and verify that the calculation engine is behaving.")
    S = controls.strategy_sidebar()
    store = D.etf_store()
    age = store.age_hours()
    macro_n = len(D.macro_raw(D.macro_stamp()))
    fac = D.french_factors(D.factor_stamp())
    kit.kpi_row([
        ("ETF price cache", "none" if age is None else f"{age:.0f} h old", "Time since the last successful ETF price download.", None),
        ("Economic series", f"{macro_n} loaded", "FRED series cached on this computer.", None),
        ("Style factors", "loaded" if fac is not None else "not loaded", "Fama-French daily factors, needed for the ‘why did it move’ factor analysis of stocks.", None),
        ("Stock cache", f"{len(D.stock_store().load().columns)} tickers" if D.stock_store().f_prices.exists() else "empty", "Stock prices downloaded through the Group Explorer.", None),
    ])
    tabs = st.tabs(["① Prices", "② Economy", "③ Style factors", "④ Connection test", "⑤ Data quality", "⑥ Self-test"])
    with tabs[0]:
        uname = S["universe"]
        c1, c2, c3 = st.columns(3)
        full = c1.toggle("Re-download everything", value=False, help=content.HELP["refresh_full"])
        years = c2.number_input("History (years)", 3, 30, 12, help=content.HELP["history_years"])
        detect = c3.toggle("Detect trading currencies", value=True, help="Asks Yahoo which currency each ETF line trades in (cached afterwards). Strongly recommended: London lines are often quoted in USD, and a wrong guess distorts rankings.")
        if kit.wide(st.button, "Update ETF prices", type="primary"):
            try:
                res = update_etf_prices(uname, full, int(years), detect)
                summ = res.status["Source"].value_counts()
                st.success("Update finished: " + ", ".join(f"{v} {k.lower().replace('_', ' ')}" for k, v in summ.items()))
                if res.errors:
                    st.warning("Problems:\n- " + "\n- ".join(res.errors[:25]))
                bad = res.status[res.status["Source"].isin(["UNRESOLVED", "STALE_CACHE"])]
                if len(bad):
                    kit.dataframe(bad, {"Ticker": dict(label="Ticker"), "Source": dict(label="Source"), "Detail": dict(label="Detail")})
            except Exception as e:  # noqa: BLE001
                log.exception("ETF update failed")
                st.error(f"Update failed: {type(e).__name__}: {e}")
        kit.note("Only new days are downloaded after the first run. If old and new data disagree in shape, the series is re-downloaded in full automatically. Trading currencies can be overridden in <code>universe/currency_overrides.csv</code>.")
    with tabs[1]:
        extra = kit.s_text("Extra FRED series IDs (optional)", "fred_extra", "fred_custom", "")
        ids = [s.id for s in DM.CATALOG] + [DM.RECESSION_ID] + [x.strip().upper() for x in extra.split(",") if x.strip()]
        force = st.toggle("Force re-download", value=False, help="Ignore the 12-hour freshness window.")
        if kit.wide(st.button, "Update economic data", type="primary"):
            with st.spinner(f"Downloading {len(ids)} FRED series…"):
                stat = DM.MacroStore().refresh(ids, force=force)
            st.cache_data.clear()
            st.session_state["macro_status"] = stat
        stat = st.session_state.get("macro_status")
        if stat is not None:
            ok = int(stat["Status"].isin(["DOWNLOADED", "CACHED_FRESH"]).sum())
            failed = stat[stat["Status"].isin(["FAILED", "STALE_CACHE"])]
            (st.success if not len(failed) else st.warning)(f"{ok} of {len(stat)} series up to date; {len(failed)} problem(s).")
            kit.dataframe(stat, {"Series": dict(label="FRED ID"), "Status": dict(label="Status"), "Observations": dict(label="Observations", kind="int"),
                                 "Last observation": dict(label="Last observation", kind="date"), "Detail": dict(label="Detail")})
        kit.note("Source: FRED (Federal Reserve Bank of St. Louis), public CSV endpoint – no account or API key. The app tolerates individual series failing; they simply drop out of the composites.")
    with tabs[2]:
        st.write("Kenneth French's free data library provides daily market, size, value, profitability, investment and momentum factors (US-based, in USD).")
        if kit.wide(st.button, "Update style factors", type="primary"):
            try:
                with st.spinner("Downloading factor files…"):
                    df, src = DR.fetch_factors(DR.default_http_bytes)
                st.cache_data.clear()
                st.success(f"Loaded {len(df):,} daily observations ({src}); last date {df.index.max().date()}.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not download factors: {type(e).__name__}: {e}")
    with tabs[3]:
        st.write("Checks that each free data source is reachable from this computer. Corporate networks and some countries block some of them.")
        if kit.wide(st.button, "Run connection test"):
            rows = []
            tests = {
                "Yahoo Finance (prices)": lambda: len(DP.yahoo_fetcher(["SPY"], (pd.Timestamp.today() - pd.Timedelta(days=10)).date().isoformat(), (pd.Timestamp.today() + pd.Timedelta(days=1)).date().isoformat())),
                "FRED (economy)": lambda: len(DM.parse_fred_csv(DM.default_http_get(DM.FRED_URL.format(id="UNRATE")))),
                "Wikipedia (index lists)": lambda: len(DR.parse_constituents(DM.default_http_get(DR.INDEXES["Dow Jones 30"]["url"]), "Dow Jones 30")),
                "Ken French library (factors)": lambda: len(DR.read_french_zip(DR.default_http_bytes(DR.FF_URLS["mom"]))),
            }
            for name, fn in tests.items():
                try:
                    n = fn()
                    rows.append({"Source": name, "Result": "OK" if n else "EMPTY", "Detail": f"{n} rows"})
                except Exception as e:  # noqa: BLE001
                    rows.append({"Source": name, "Result": "FAILED", "Detail": f"{type(e).__name__}: {e}"[:200]})
            kit.dataframe(pd.DataFrame(rows), {"Source": dict(label="Source"), "Result": dict(label="Result"), "Detail": dict(label="Detail")})
    with tabs[4]:
        uni, panel, qrep, ccy, names = controls.load_context(S)
        sev = st.multiselect("Show severity", ["FAIL", "WARN", "INFO", "OK"], default=["FAIL", "WARN"], help="FAIL = excluded from all signals. WARN = used but check it. INFO = informational.")
        q = qrep[qrep["Severity"].isin(sev)] if sev else qrep
        counts = qrep["Severity"].value_counts()
        st.write(f"**{len(qrep)} series:** " + ", ".join(f"{counts.get(k, 0)} {k}" for k in ["OK", "INFO", "WARN", "FAIL"]))
        kit.dataframe(q, {"Ticker": dict(label="Ticker"), "Severity": dict(label="Severity"), "Observations": dict(label="Observations", kind="int"),
                          "First date": dict(label="First", kind="date"), "Last date": dict(label="Last", kind="date"), "Issues": dict(label="Issues found"),
                          "Currency": dict(label="Currency", help="Currency the line trades in."),
                          "CurrencySource": dict(label="Currency source", help="WORKBOOK/CACHED_YAHOO/YAHOO are reliable. GUESSED_FROM_SUFFIX is a guess – verify it, as it can distort rankings."),
                          "Status": dict(label="FX status")}, height=420)
        guessed = qrep[qrep.get("CurrencySource", pd.Series(dtype=object)) == "GUESSED_FROM_SUFFIX"] if "CurrencySource" in qrep else pd.DataFrame()
        if len(guessed):
            kit.note(f"{len(guessed)} ETFs use a <b>guessed</b> trading currency. Tick ‘Detect trading currencies’ and update prices, or list them in <code>universe/currency_overrides.csv</code>.", "caution")
    with tabs[5]:
        st.write("Runs the calculation engine against independently derived answers (drawdown maths, regression standard errors, cost accounting, return reconciliation, look-ahead and leakage tests…).")
        if kit.wide(st.button, "Run self-test"):
            with st.spinner("Running checks…"):
                df = selftest.run_all()
            (st.success if (df.Result == "PASS").all() else st.error)(f"{int((df.Result == 'PASS').sum())} of {len(df)} checks passed.")
            kit.dataframe(df, {"Check": dict(label="Check"), "Result": dict(label="Result"), "Detail": dict(label="What was verified"), "Seconds": dict(label="Seconds", kind="num2")})
    kit.disclaimer_footer()
