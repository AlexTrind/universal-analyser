"""Which US sector ETF belongs with a stock?  (Used to co-plot a stock with its own sector.)

Sector labels come from, in order: a choice the user made by hand, a saved lookup, the index lists that
Group Explorer downloaded (GICS sectors from Wikipedia), and finally a one-off Yahoo Finance lookup.
Everything found is saved in data/stocks/sector_map.json so it is only ever looked up once.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config

# GICS names (Wikipedia / S&P) and Yahoo's own names, normalised, -> SPDR sector ETF
_ETF_FOR = {
    "information technology": "XLK", "technology": "XLK",
    "financials": "XLF", "financial": "XLF", "financial services": "XLF",
    "health care": "XLV", "healthcare": "XLV",
    "energy": "XLE",
    "consumer discretionary": "XLY", "consumer cyclical": "XLY",
    "consumer staples": "XLP", "consumer defensive": "XLP",
    "industrials": "XLI", "industrial": "XLI",
    "materials": "XLB", "basic materials": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
    "communication services": "XLC", "communications": "XLC", "telecommunications": "XLC",
}
MANUAL_CHOICES = ["(none)"] + [f"{k} – {v}" for k, v in config.SPDR_SECTORS.items()]


def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).lower().replace("&", "and")).strip()


def etf_for_label(label: str | None) -> str | None:
    """Map any sector wording to a SPDR ticker (None if it is not a recognised sector)."""
    if not label or str(label).lower() in {"nan", "none", "?", "unclassified"}:
        return None
    n = _norm(label)
    if n in _ETF_FOR:
        return _ETF_FOR[n]
    for key, etf in _ETF_FOR.items():          # e.g. "Information Technology – Software"
        if n.startswith(key):
            return etf
    return None


def yahoo_sector(ticker: str) -> str | None:  # pragma: no cover - network
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    return info.get("sector") or None


class SectorBook:
    def __init__(self, folder: Path | str | None = None, constituents_dir: Path | str | None = None):
        self.folder = Path(folder) if folder else config.DATA_DIR / "stocks"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = self.folder / "sector_map.json"
        self.cdir = Path(constituents_dir) if constituents_dir else config.DATA_DIR / "constituents"
        self._d = self._read()

    def _read(self) -> dict:
        try:
            return json.loads(self.file.read_text())
        except Exception:  # noqa: BLE001
            return {"manual": {}, "found": {}}

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._d, indent=1))
        tmp.replace(self.file)

    def _from_lists(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in sorted(self.cdir.glob("*.csv")) if self.cdir.exists() else []:
            try:
                df = pd.read_csv(f)
                if {"Ticker", "Sector"} <= set(df.columns):
                    out.update(dict(zip(df["Ticker"].astype(str), df["Sector"].astype(str))))
            except Exception:  # noqa: BLE001
                continue
        return out

    # -- reading -----------------------------------------------------------
    def etf(self, ticker: str) -> tuple[str | None, str]:
        """(sector ETF or None, where it came from)."""
        m = self._d["manual"].get(ticker)
        if m:
            return (None, "chosen by you") if m == "(none)" else (m, "chosen by you")
        for src, d in (("saved lookup", self._d["found"]), ("index list", self._from_lists())):
            e = etf_for_label(d.get(ticker))
            if e:
                return e, src
        return None, "unknown"

    def label(self, ticker: str) -> str | None:
        return self._d["found"].get(ticker) or self._from_lists().get(ticker)

    # -- writing -----------------------------------------------------------
    def set_manual(self, ticker: str, etf: str | None) -> None:
        if etf in (None, ""):                      # forget my choice -> go back to the automatic match
            self._d["manual"].pop(ticker, None)
        else:
            self._d["manual"][ticker] = etf
        self._save()

    def remember(self, members: pd.DataFrame) -> None:
        """Keep the sectors of a downloaded group (Ticker/Sector columns)."""
        if members is None or members.empty or "Sector" not in members:
            return
        for t, s in zip(members["Ticker"].astype(str), members["Sector"].astype(str)):
            if etf_for_label(s):
                self._d["found"][t] = s
        self._save()

    def lookup_missing(self, tickers: list[str], fetch: Callable[[str], str | None] | None = None) -> dict[str, str]:
        """Ask Yahoo for tickers with no known sector.  Returns {ticker: message} for failures."""
        fetch = fetch or yahoo_sector
        fails: dict[str, str] = {}
        for t in tickers:
            if self.etf(t)[0] or t.startswith("^"):
                continue
            try:
                s = fetch(t)
                if etf_for_label(s):
                    self._d["found"][t] = s
                else:
                    fails[t] = f"Yahoo gave no usable sector ({s!r})"
            except Exception as e:  # noqa: BLE001
                fails[t] = f"{type(e).__name__}"
        self._save()
        return fails
