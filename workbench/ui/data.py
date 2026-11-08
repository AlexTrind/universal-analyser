"""Cached data loading for the app (disk → DataFrames → Panel).  All heavy lifting is keyed on file timestamps,
so pages are instant after the first load and reload automatically after a data refresh."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .. import config, data_macro, data_prices as DP, data_ref, features as F, quality

ETF_STORE_DIR = config.DATA_DIR / "etf"
STOCK_STORE_DIR = config.DATA_DIR / "stocks"
OVERRIDES_CSV = config.UNIVERSE_DIR / "currency_overrides.csv"


def universe_path(name: str) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    for base in (config.UNIVERSE_DIR, config.ROOT):
        if (base / name).exists():
            return base / name
    return config.UNIVERSE_DIR / name


def mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def read_universe(path: str, stamp: float) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="ETF_Master")
    need = ["Fund_Name", "Yahoo_Ticker"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"ETF_Master sheet is missing columns: {miss}")
    df["Yahoo_Ticker"] = df["Yahoo_Ticker"].astype(str).str.strip()
    df = df[~df["Yahoo_Ticker"].isin(["", "nan"])].drop_duplicates("Yahoo_Ticker").reset_index(drop=True)
    return df


def load_universe(name: str) -> pd.DataFrame:
    p = universe_path(name)
    return read_universe(str(p), mtime(p))


def read_overrides() -> dict[str, str]:
    if not OVERRIDES_CSV.exists():
        return {}
    try:
        d = pd.read_csv(OVERRIDES_CSV, comment="#")
        return {str(r.Ticker).strip(): str(r.Currency).strip() for r in d.itertuples() if str(r.Currency) not in ("nan", "")}
    except Exception:  # noqa: BLE001
        return {}


def etf_store() -> DP.PriceStore:
    return DP.PriceStore(ETF_STORE_DIR)


def stock_store() -> DP.PriceStore:
    return DP.PriceStore(STOCK_STORE_DIR)


@st.cache_data(show_spinner="Preparing price panel…", max_entries=4)
def etf_panel(universe_name: str, base: str, benchmark: str, price_stamp: float, meta_stamp: float):
    """Returns (Panel, quality_report, currency_status).  FAIL-quality series are excluded from the panel."""
    uni = load_universe(universe_name)
    st_ = etf_store()
    native = st_.load()
    if native.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    tickers = list(dict.fromkeys(uni["Yahoo_Ticker"].tolist() + list(config.BENCHMARKS.values())))
    meta = st_.load_meta()
    ct = DP.resolve_currencies(tickers, overrides=read_overrides(), cached=meta.get("currency", {}))
    native, repaired = quality.repair_unit_frame(native)
    prices, ccy_status = DP.to_base_currency(native, ct, base)
    if prices.empty:
        return None, pd.DataFrame(), ccy_status
    q = quality.quality_report(prices, prices.index.max(), repaired=repaired)
    bad = quality.failing_tickers(q) - {benchmark}
    panel = F.build_panel(prices.drop(columns=[c for c in bad if c in prices.columns]))
    q = q.merge(ccy_status[["Ticker", "Currency", "CurrencySource", "Status"]], on="Ticker", how="left")
    return panel, q, ccy_status


@st.cache_data(show_spinner=False)
def macro_raw(stamp: float) -> dict[str, pd.Series]:
    store = data_macro.MacroStore()
    ids = [s.id for s in data_macro.CATALOG] + [data_macro.RECESSION_ID]
    extra = sorted(p.stem for p in store.folder.glob("*.csv") if p.stem not in ids)
    return store.load_many(ids + extra)


def macro_stamp() -> float:
    store = data_macro.MacroStore()
    files = list(store.folder.glob("*.csv"))
    return max([mtime(p) for p in files], default=0.0) + len(files)


@st.cache_data(show_spinner="Building macro regimes…", max_entries=6)
def macro_tables(stamp: float, scheme: str, stress_q: float):
    from .. import regimes as RG

    raw = macro_raw(stamp)
    if not raw:
        return None
    feats, Z = data_macro.build_monthly_table(raw)
    comp = RG.composites(Z)
    labels = RG.classify(comp, feats, scheme, stress_q)
    return {"raw": raw, "feats": feats, "Z": Z, "comp": comp, "labels": labels, "recession": data_macro.recession_monthly(raw)}


@st.cache_data(show_spinner=False)
def french_factors(stamp: float):
    f = config.DATA_DIR / "french_factors_daily.csv.gz"
    if not f.exists():
        return None
    return pd.read_csv(f, index_col=0, parse_dates=True)


def factor_stamp() -> float:
    return mtime(config.DATA_DIR / "french_factors_daily.csv.gz")


@st.cache_data(show_spinner="Preparing stock panel…", max_entries=6)
def stock_panel(tickers: tuple[str, ...], price_stamp: float):
    native = stock_store().load()
    if native.empty:
        return None, pd.DataFrame()
    cols = [t for t in tickers if t in native.columns]
    if not cols:
        return None, pd.DataFrame()
    prices, repaired = quality.repair_unit_frame(native[cols])
    q = quality.quality_report(prices, prices.index.max(), repaired=repaired)
    bad = quality.failing_tickers(q)
    return F.build_panel(prices.drop(columns=[c for c in bad if c in prices.columns])), q
