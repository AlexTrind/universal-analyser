"""Headless smoke test: runs every page's render() against synthetic data with a fake streamlit/plotly.
It cannot judge appearance, but it catches wrong function calls, missing columns, duplicate widget keys, bad widget
defaults, missing help text and exceptions in page logic."""
from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import fake_streamlit as FS  # noqa: E402

PAGES = ["page_home", "page_learn", "page_data", "page_screener", "page_backtest", "page_map", "page_group", "page_asset",
         "page_macro", "page_compare", "page_prob", "page_audit"]
END = pd.Timestamp("2026-11-06")


def _walk(idx, rng, n, mkt):
    cols = {}
    for i in range(n):
        r = rng.uniform(0.5, 1.3) * mkt + rng.normal(0.0001, 0.009, len(idx))
        cols[i] = 50 * np.exp(np.cumsum(r))
    return cols


@pytest.fixture(scope="module")
def env():
    tmp = Path(tempfile.mkdtemp(prefix="wb_smoke_"))
    st = FS.install()
    from workbench import config
    config.DATA_DIR = tmp / "data"
    config.DATA_DIR.mkdir()
    for m in [m for m in sys.modules if m.startswith("workbench.ui")]:
        del sys.modules[m]
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=END, periods=252 * 16)
    mkt = rng.normal(0.0003, 0.008, len(idx))
    uni = pd.read_excel(ROOT / "universe" / "ETF_Momentum_Universe_v7_8_FINAL_TEST.xlsx", sheet_name="ETF_Master")
    tick = list(dict.fromkeys(uni["Yahoo_Ticker"].dropna().tolist() + ["SSAC.L", "VWRP.L"]))
    from workbench.ui import data as D
    etf = pd.DataFrame({t: v for t, v in zip(tick, _walk(idx, rng, len(tick), mkt).values())}, index=idx)
    etf["SSAC.L"] = 50 * np.exp(np.cumsum(mkt))
    ctry = dict.fromkeys(tick, "GBP")
    store = D.etf_store()
    store.save(etf.round(4))
    store.save_meta({"currency": ctry, "last_update": str(END)})
    # stocks
    grp = pd.read_csv(ROOT / "universe" / "sample_stock_groups.csv")
    stk = pd.DataFrame({t: v for t, v in zip(list(grp["Ticker"]) + ["^GSPC"], _walk(idx, rng, len(grp) + 1, mkt).values())}, index=idx)
    D.stock_store().save(stk.round(4))
    # macro
    from workbench import data_macro as DM
    ms = DM.MacroStore()
    midx = pd.date_range("1997-01-31", "2026-09-30", freq="ME")
    for s in DM.CATALOG:
        lvl = np.cumsum(rng.normal(0, 0.3, len(midx))) + (100 if s.transform in ("yoy",) else 2.5)
        if s.transform == "yoy":
            lvl = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, len(midx))))
        pd.Series(lvl, index=midx).to_csv(ms.path(s.id), header=[s.id])
    rec = pd.Series(((np.arange(len(midx)) % 90) < 8).astype(float), index=midx)
    rec.to_csv(ms.path(DM.RECESSION_ID), header=[DM.RECESSION_ID])
    fi = idx[-2500:]
    fac = pd.DataFrame(rng.normal(0, 0.007, (len(fi), 6)), index=fi, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    fac["RF"] = 0.0001
    fac.to_csv(config.DATA_DIR / "french_factors_daily.csv.gz")
    yield st
    shutil.rmtree(tmp, ignore_errors=True)


def run_page(st, name, press=()):
    st.new_run()
    st.press = set(press)
    st.session_state.pop("_sticky", None) if False else None
    mod = importlib.import_module(f"workbench.ui.{name}")
    stopped = False
    try:
        mod.render()
    except FS.StopApp:
        stopped = True
    return stopped


@pytest.mark.parametrize("name", PAGES)
def test_page_renders(env, name):
    st = env
    press = {"page_backtest": ["Run backtest"], "page_prob": ["Calculate probabilities"], "page_data": ["Run self-test"]}.get(name, [])
    stopped = run_page(st, name, press)
    errs = [m for m in st.messages if m[0] in ("error", "exception")]
    assert not errs, errs[:3]
    nohelp = [m for m in st.messages if m[0] == "nohelp"]
    assert not nohelp, nohelp
    assert not stopped, f"{name} stopped early (missing data?) -> {st.messages[-3:]}"


@pytest.mark.parametrize("name", [p for p in PAGES if p not in ("page_home", "page_learn")])
def test_page_draws_something(env, name):
    st = env
    press = {"page_backtest": ["Run backtest"], "page_prob": ["Calculate probabilities"]}.get(name, [])
    run_page(st, name, press)
    assert st.charts + st.tables > 0, f"{name} drew no chart or table"


def test_all_help_keys_exist():
    from workbench import content
    import re
    used = set()
    for f in (ROOT / "workbench" / "ui").glob("*.py"):
        used |= set(re.findall(r'content\.HELP\["(\w+)"\]', f.read_text()))
        # s_* helpers: third positional argument is the help key
        for m in re.finditer(r'kit\.s_(?:slider|number|select|multi|text|toggle|date)\(\s*"[^"]*",\s*[^,]+,\s*"(\w+)"', f.read_text()):
            used.add(m.group(1))
    missing = sorted(k for k in used if k not in content.HELP)
    assert not missing, missing


def test_stock_source_in_asset_lab_and_probabilities(env):
    st = env
    st.radio_index = 1
    try:
        for name, press in (("page_asset", []), ("page_prob", ["Calculate probabilities"])):
            stopped = run_page(st, name, press)
            errs = [m for m in st.messages if m[0] in ("error", "exception")]
            assert not errs and not stopped, (name, errs, st.messages[-2:])
    finally:
        st.radio_index = None


def test_pages_survive_empty_data(env, tmp_path, monkeypatch):
    from workbench import config
    from workbench.ui import data as D
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(D, "ETF_STORE_DIR", tmp_path / "etf")
    monkeypatch.setattr(D, "STOCK_STORE_DIR", tmp_path / "stocks")
    st = env
    for name in PAGES:
        run_page(st, name)          # a clean stop() is fine; a crash is not
        errs = [m for m in st.messages if m[0] in ("error", "exception")]
        assert not errs, (name, errs[:2])


def test_download_buttons_with_fake_network(env, tmp_path, monkeypatch):
    """Press every 'Update…' button with the network replaced by synthetic data; nothing may crash and caches must fill."""
    from workbench import config, data_macro as DM, data_prices as DP, data_ref as DR
    from workbench.ui import data as D

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(D, "ETF_STORE_DIR", tmp_path / "etf")
    monkeypatch.setattr(D, "STOCK_STORE_DIR", tmp_path / "stocks")
    rng = np.random.default_rng(3)

    def fetcher(tickers, start, end):
        idx = pd.bdate_range(start, min(pd.Timestamp(end), END))
        out = {}
        for t in tickers:
            base = 0.8 if t.endswith("=X") else 50
            out[t] = base * np.exp(np.cumsum(rng.normal(0, 0.006 if t.endswith("=X") else 0.01, len(idx))))
        return pd.DataFrame(out, index=idx)

    monkeypatch.setattr(DP, "yahoo_fetcher", fetcher)
    monkeypatch.setattr(DP, "yahoo_currency_lookup", lambda t: "GBP" if t.endswith(".L") else "USD")
    monkeypatch.setattr(DM, "default_http_get", lambda url: "DATE,X\n" + "\n".join(f"{d.date()},{1 + 0.01 * i}" for i, d in enumerate(pd.date_range("2015-01-01", periods=140, freq="MS"))))
    st = env
    run_page(st, "page_data", ["Update ETF prices", "Update economic data"])
    errs = [m for m in st.messages if m[0] in ("error", "exception")]
    assert not errs, errs[:3]
    assert D.etf_store().f_prices.exists() and D.etf_store().load().shape[1] > 100
    assert len(list((tmp_path / "macro").glob("*.csv"))) >= 20
    meta = D.etf_store().load_meta()
    assert meta["currency"], "currencies not cached"
    # group page download
    run_page(st, "page_compare", ["Download sector ETFs"])
    cols = set(D.stock_store().load().columns)
    assert {"XLK", "XLE", "SPY"} <= cols, cols
    errs = [m for m in st.messages if m[0] in ("error", "exception")]
    assert not errs, errs[:3]
    run_page(st, "page_group", ["Download / update prices"])
    assert D.stock_store().f_prices.exists()
    errs = [m for m in st.messages if m[0] in ("error", "exception")]
    assert not errs, errs[:3]


def test_compare_stock_overlays_with_its_sector_etf(env):
    """Choosing a stock must redraw the overlay and bring in the stock's own sector ETF (reported bug)."""
    from workbench import sectors as SC
    from workbench.ui import data as D
    st = env
    native = D.stock_store().load()
    rng = np.random.default_rng(11)
    for t in ("AAA", "BBB"):
        native[t] = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(native))))
    for t in ("XLK", "XLE", "XLB"):
        if t not in native:
            native[t] = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, len(native))))
    D.stock_store().save(native)
    st.cache_data.clear()
    book = SC.SectorBook()
    book.remember(pd.DataFrame({"Ticker": ["AAA"], "Sector": ["Information Technology"]}))
    book.set_manual("BBB", "XLE")
    for pick in (["AAA"], ["BBB"], ["AAA", "BBB"]):
        st.session_state.setdefault("_sticky", {})["cmp_assets"] = pick
        stopped = run_page(st, "page_compare")
        errs = [m for m in st.messages if m[0] in ("error", "exception")]
        assert not errs, (pick, errs[:2])
        assert not stopped, pick
        keys = [w["key"] for w in st.widgets if w["label"] == "Assets to overlay"]
        assert keys and "|".join(pick) in keys[0], (pick, keys)       # overlay picker follows the selection
        assert st.charts >= 1
    assert book.etf("AAA")[0] == "XLK" and book.etf("BBB")[0] == "XLE"


def test_overlay_picker_refreshes_when_the_sector_match_changes(env):
    """Reported: stock shown alone; its sector ETF (matched later) never joined the picker."""
    from workbench import sectors as SC
    st = env
    book = SC.SectorBook()
    st.session_state.setdefault("_sticky", {})["cmp_assets"] = ["AAA"]
    book.set_manual("AAA", "(none)")
    run_page(st, "page_compare")
    k1 = [w["key"] for w in st.widgets if w["label"] == "Assets to overlay"][0]
    book.set_manual("AAA", "XLK")
    run_page(st, "page_compare")
    k2 = [w["key"] for w in st.widgets if w["label"] == "Assets to overlay"][0]
    assert k1 != k2
