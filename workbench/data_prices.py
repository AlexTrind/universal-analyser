"""Market-price acquisition (Yahoo Finance via `yfinance`), local cache, currency conversion.

Design rules
------------
* Prices are *adjusted close* (dividends reinvested) = total return, which is what momentum should use.
* Updating is incremental, and an update never blindly splices old and new history: because adjusted
  prices are rescaled whenever a dividend is paid, old and new data are matched on their overlap.  If they
  disagree in shape (not just scale) the ticker is re-downloaded in full.
* Every ticker gets a status. Nothing is silently invented or silently dropped.
* All network calls go through an injectable `fetcher`, so the logic is testable offline.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .quality import repair_bad_ticks

log = logging.getLogger("workbench.prices")

Fetcher = Callable[[list[str], str, str], pd.DataFrame]


# ----------------------------------------------------------------------------
# yfinance plumbing
# ----------------------------------------------------------------------------
def flatten_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Turn whatever yfinance returned into a wide DataFrame (index=date, columns=tickers)."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(map(str, raw.columns.get_level_values(0)))
        lv1 = set(map(str, raw.columns.get_level_values(1)))
        field_level = 0 if ("Close" in lv0 or "Adj Close" in lv0) else (1 if ("Close" in lv1 or "Adj Close" in lv1) else None)
        if field_level is None:
            return pd.DataFrame()
        names = lv0 if field_level == 0 else lv1
        key = "Close" if "Close" in names else "Adj Close"
        out = raw.xs(key, axis=1, level=field_level).copy()
    else:
        key = "Close" if "Close" in raw.columns else ("Adj Close" if "Adj Close" in raw.columns else None)
        if key is None:
            return pd.DataFrame()
        out = raw[[key]].copy()
        out.columns = tickers[:1]
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def yahoo_fetcher(tickers: list[str], start: str, end: str) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf

    parts = []
    for i in range(0, len(tickers), 20):
        chunk = tickers[i : i + 20]
        for attempt in range(3):
            try:
                raw = yf.download(
                    chunk, start=start, end=end, interval="1d", auto_adjust=True, actions=False,
                    progress=False, threads=False, group_by="column", timeout=45,
                )
                p = flatten_download(raw, chunk)
                if not p.empty:
                    parts.append(p)
                    break
            except Exception as e:  # noqa: BLE001 - provider errors are many and varied
                log.warning("Yahoo chunk %s attempt %d failed: %s", chunk[:3], attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
        time.sleep(0.3)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()].sort_index()


def yahoo_currency_lookup(ticker: str) -> str | None:  # pragma: no cover - network
    import yfinance as yf

    try:
        c = yf.Ticker(ticker).fast_info.get("currency")
        return str(c) if c else None
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Safe merging of old (cached) and new (fresh) adjusted histories
# ----------------------------------------------------------------------------
def merge_adjusted(old: pd.Series, new: pd.Series, shape_tol: float = 2e-3) -> tuple[pd.Series, str]:
    """Merge a cached adjusted-close series with a fresher download covering a later window.

    Returns (merged, status) where status is one of
      'NEW_ONLY'     – nothing cached
      'APPENDED'     – overlap agreed exactly; new rows appended
      'RESCALED'     – overlap agreed in *shape* but differed by a constant factor (a dividend/split
                       adjustment since last download): history was multiplied by that factor
      'NO_OVERLAP'   – cannot verify continuity; the new data are appended unscaled (flagged)
      'INCONSISTENT' – overlap shapes disagree; caller must re-download the full history
    """
    old = old.dropna()
    new = new.dropna()
    if old.empty:
        return new, "NEW_ONLY"
    if new.empty:
        return old, "APPENDED"
    ov = old.index.intersection(new.index)
    if len(ov) < 3:
        keep = old[old.index < new.index.min()]
        return pd.concat([keep, new]).sort_index(), "NO_OVERLAP"
    ratio = (new.loc[ov] / old.loc[ov]).to_numpy()
    med = float(np.median(ratio))
    spread = float(np.max(np.abs(ratio / med - 1.0)))
    if spread > shape_tol:
        return old, "INCONSISTENT"
    keep = old[old.index < ov.min()]
    merged = pd.concat([keep * med, new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    status = "APPENDED" if abs(med - 1.0) < 1e-6 else "RESCALED"
    return merged, status


# ----------------------------------------------------------------------------
# Persistent store
# ----------------------------------------------------------------------------
@dataclass
class UpdateResult:
    prices: pd.DataFrame
    status: pd.DataFrame
    errors: list[str] = field(default_factory=list)


class PriceStore:
    """Wide CSV cache of native-currency adjusted closes plus a small JSON metadata file."""

    def __init__(self, folder: Path | str | None = None):
        self.folder = Path(folder) if folder else config.DATA_DIR
        self.folder.mkdir(parents=True, exist_ok=True)
        self.f_prices = self.folder / "prices_native_adjclose.csv.gz"
        self.f_meta = self.folder / "price_meta.json"

    # -- persistence -----------------------------------------------------
    def load(self) -> pd.DataFrame:
        if not self.f_prices.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.f_prices, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).normalize()
            return df.sort_index()
        except Exception as e:  # noqa: BLE001
            log.error("Cannot read price cache: %s", e)
            return pd.DataFrame()

    def save(self, df: pd.DataFrame) -> None:
        tmp = self.f_prices.with_suffix(".tmp")
        df.to_csv(tmp, compression="gzip")
        tmp.replace(self.f_prices)  # atomic: a crash can never leave a half-written cache

    def load_meta(self) -> dict:
        try:
            return json.loads(self.f_meta.read_text())
        except Exception:  # noqa: BLE001
            return {"currency": {}, "last_update": None}

    def save_meta(self, meta: dict) -> None:
        self.f_meta.write_text(json.dumps(meta, indent=1, default=str))

    # -- update ----------------------------------------------------------
    def update(
        self,
        tickers: list[str],
        history_years: int = 15,
        full: bool = False,
        fetcher: Fetcher | None = None,
        today: pd.Timestamp | None = None,
        overlap_days: int = 20,
        progress: Callable[[str], None] | None = None,
    ) -> UpdateResult:
        fetcher = fetcher or yahoo_fetcher
        today = (today or pd.Timestamp.today()).normalize()
        tickers = sorted(set(tickers))
        cached = self.load()
        errors: list[str] = []
        status_rows: dict[str, dict] = {t: {"Ticker": t, "Source": "MISSING", "Detail": ""} for t in tickers}
        end = (today + pd.Timedelta(days=2)).date().isoformat()
        full_start = (today - pd.DateOffset(years=history_years)).date().isoformat()

        # Which tickers need a full download and which an incremental one?
        need_full, need_inc = [], []
        meta0 = self.load_meta()
        asked = meta0.get("asked_from", {})
        for t in tickers:
            if full or t not in cached.columns or cached[t].dropna().empty:
                need_full.append(t)
                continue
            first = cached[t].dropna().index.min()
            # Cached history starts later than what is wanted now AND we never asked that far back before
            # (e.g. a sector ETF first fetched with 12 years, now wanted with 25): fetch the longer history once.
            too_short = first > pd.Timestamp(full_start) + pd.Timedelta(days=45)
            already_asked = asked.get(t) is not None and pd.Timestamp(asked[t]) <= pd.Timestamp(full_start) + pd.Timedelta(days=1)
            (need_full if (too_short and not already_asked) else need_inc).append(t)

        result = cached.copy()

        def put(t: str, s: pd.Series):
            nonlocal result
            if t in result.columns:
                result = result.drop(columns=[t])
            result = result.join(s.rename(t), how="outer")

        if progress:
            progress(f"Downloading {len(need_full)} full + {len(need_inc)} incremental series…")
        fresh_full = fetcher(need_full, full_start, end) if need_full else pd.DataFrame()
        for t in need_full:
            s = fresh_full[t].dropna() if t in fresh_full.columns else pd.Series(dtype=float)
            if len(s) >= 2:
                put(t, s)
                status_rows[t].update(Source="FRESH_FULL", Detail=f"{len(s)} obs")
            elif t in cached.columns and not cached[t].dropna().empty:
                status_rows[t].update(Source="STALE_CACHE", Detail="download failed; using cached history")
                errors.append(f"{t}: download failed, cached history kept")
            else:
                status_rows[t].update(Source="UNRESOLVED", Detail="download failed and no cache")
                errors.append(f"{t}: unresolved")

        if need_inc:
            last = {t: cached[t].dropna().index.max() for t in need_inc}
            inc_start = (min(last.values()) - pd.Timedelta(days=overlap_days)).date().isoformat()
            fresh_inc = fetcher(need_inc, inc_start, end)
            redo = []
            for t in need_inc:
                new = fresh_inc[t].dropna() if t in fresh_inc.columns else pd.Series(dtype=float)
                if new.empty:
                    status_rows[t].update(Source="STALE_CACHE", Detail="no fresh rows returned")
                    continue
                merged, st = merge_adjusted(cached[t], new)
                if st == "INCONSISTENT":
                    redo.append(t)
                    continue
                put(t, merged)
                status_rows[t].update(Source="FRESH_INCREMENTAL", Detail=st)
            if redo:
                fr = fetcher(redo, full_start, end)
                for t in redo:
                    s = fr[t].dropna() if t in fr.columns else pd.Series(dtype=float)
                    if len(s) >= 2:
                        put(t, s)
                        status_rows[t].update(Source="FRESH_FULL", Detail="re-downloaded: cached history was inconsistent")
                    else:
                        status_rows[t].update(Source="STALE_CACHE", Detail="re-download failed; old history kept")
                        errors.append(f"{t}: re-download failed")

        result = result.sort_index()
        result = result[~result.index.duplicated(keep="last")]
        # Remove provider glitches (spike that reverses next day) – and record how many.
        for t in tickers:
            if t in result.columns:
                cleaned, bad = repair_bad_ticks(result[t])
                if bad:
                    result[t] = cleaned
                    status_rows[t]["Detail"] += f" | {len(bad)} bad tick(s) removed"
        self.save(result)
        meta = self.load_meta()
        meta["last_update"] = pd.Timestamp.now(tz="UTC").isoformat()
        asked = dict(meta.get("asked_from", {}))
        for t in need_full:
            if status_rows[t]["Source"] == "FRESH_FULL":
                asked[t] = full_start if t not in asked else min(asked[t], full_start)   # remember how far back we have asked
        meta["asked_from"] = asked
        self.save_meta(meta)
        status = pd.DataFrame(status_rows.values())
        return UpdateResult(result[[t for t in tickers if t in result.columns]], status, errors)

    def age_hours(self) -> float | None:
        m = self.load_meta().get("last_update")
        if not m:
            return None
        try:
            t = pd.Timestamp(m)
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            return (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 3600.0
        except (ValueError, TypeError):
            return None


# ----------------------------------------------------------------------------
# Currency handling
# ----------------------------------------------------------------------------
def guess_currency(ticker: str) -> str:
    t = str(ticker).upper()
    for suf, ccy in config.SUFFIX_CURRENCY.items():
        if t.endswith(suf):
            return ccy
    return "USD"


def resolve_currencies(
    tickers: list[str],
    overrides: dict[str, str] | None = None,
    cached: dict[str, str] | None = None,
    lookup: Callable[[str], str | None] | None = None,
) -> pd.DataFrame:
    """Decide each listing's trading currency.  Priority: workbook override > cache > Yahoo > suffix guess.

    Pence-quoted lines ('GBp'/'GBX') are mapped to GBP with scale 0.01.
    """
    overrides = overrides or {}
    cached = cached or {}
    rows = []
    for t in tickers:
        raw, src = None, None
        if overrides.get(t):
            raw, src = str(overrides[t]), "WORKBOOK"
        elif cached.get(t):
            raw, src = str(cached[t]), "CACHED_YAHOO"
        elif lookup is not None:
            raw = lookup(t)
            src = "YAHOO" if raw else None
        if not raw:
            raw, src = guess_currency(t), "GUESSED_FROM_SUFFIX"
        scale = 0.01 if raw in config.PENCE_CODES else 1.0
        ccy = "GBP" if raw in config.PENCE_CODES else raw.upper()
        rows.append({"Ticker": t, "Currency": ccy, "Scale": scale, "CurrencySource": src})
    return pd.DataFrame(rows)


def fx_ticker(ccy: str, base: str) -> str:
    return f"{ccy}{base}=X"


def required_fx(currency_table: pd.DataFrame, base: str) -> list[str]:
    return sorted({fx_ticker(c, base) for c in currency_table["Currency"].unique() if c != base})


def to_base_currency(
    native: pd.DataFrame, currency_table: pd.DataFrame, base: str, fx_limit: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert native-currency adjusted closes into `base` currency.  Returns (prices, status)."""
    ct = currency_table.set_index("Ticker")
    out, rows = {}, []
    for t in ct.index:
        ccy = ct.at[t, "Currency"]
        scale = float(ct.at[t, "Scale"])
        src = ct.at[t, "CurrencySource"]
        if t not in native.columns or native[t].dropna().empty:
            rows.append((t, ccy, "PRICE_MISSING", False, src))
            continue
        s = native[t].dropna() * scale
        if ccy == base:
            out[t] = s
            rows.append((t, ccy, "NO_FX_NEEDED", True, src))
            continue
        fx = fx_ticker(ccy, base)
        if fx in native.columns and not native[fx].dropna().empty:
            f = native[fx].dropna().reindex(native.index.union(s.index)).ffill(limit=fx_limit).reindex(s.index)
            conv = (s * f).dropna()
            out[t] = conv
            rows.append((t, ccy, f"FX_CONVERTED_{ccy}_TO_{base}", len(conv) > 1, src))
        else:
            rows.append((t, ccy, f"FX_MISSING_{ccy}_{base}", False, src))
    prices = pd.DataFrame(out).sort_index() if out else pd.DataFrame()
    status = pd.DataFrame(rows, columns=["Ticker", "Currency", "Status", "Comparable", "CurrencySource"])
    return prices, status
