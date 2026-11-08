"""Reference data: index constituent lists (Wikipedia) and Fama-French factors (Ken French data library)."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config

HttpGet = Callable[[str], str]

INDEXES = {
    "S&P 500": dict(url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", min_rows=400, suffix="", benchmark="^GSPC", equal_proxy="RSP"),
    "Nasdaq-100": dict(url="https://en.wikipedia.org/wiki/Nasdaq-100", min_rows=90, suffix="", benchmark="^NDX", equal_proxy="QQEW"),
    "Dow Jones 30": dict(url="https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", min_rows=28, suffix="", benchmark="^DJI", equal_proxy=None),
    "FTSE 100": dict(url="https://en.wikipedia.org/wiki/FTSE_100_Index", min_rows=90, suffix=".L", benchmark="^FTSE", equal_proxy=None),
    "DAX 40": dict(url="https://en.wikipedia.org/wiki/DAX", min_rows=38, suffix=".DE", benchmark="^GDAXI", equal_proxy=None),
    "CAC 40": dict(url="https://en.wikipedia.org/wiki/CAC_40", min_rows=38, suffix=".PA", benchmark="^FCHI", equal_proxy=None),
}


def _norm_symbol(sym: str, suffix: str) -> str:
    s = re.sub(r"\[.*?\]", "", str(sym)).strip().upper()
    s = re.sub(r"^(NASDAQ|NYSE|LON|ETR|EPA)[:\s]+", "", s)
    s = s.replace(".", "-") if suffix == "" else s.rstrip(".").replace(".", "-")
    if suffix and not s.endswith(suffix):
        s += suffix
    return s


def parse_constituents(html: str, index_key: str) -> pd.DataFrame:
    """Find the constituents table in a Wikipedia page and return Ticker / Name / Sector."""
    spec = INDEXES[index_key]
    tables = pd.read_html(io.StringIO(html))
    for t in tables:
        t.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in t.columns]
        tick_col = next((c for c in t.columns if re.match(r"^(Symbol|Ticker|Code)", c, re.I)), None)
        name_col = next((c for c in t.columns if re.match(r"^(Security|Company|Name|Constituent)", c, re.I)), None)
        if tick_col is None or name_col is None or len(t) < spec["min_rows"] * 0.9:
            continue
        sec_col = next((c for c in t.columns if re.search(r"GICS Sector|Sector|Industry|ICB", c, re.I) and "Sub" not in c), None)
        sub_col = next((c for c in t.columns if re.search(r"Sub-Industry", c, re.I)), None)
        out = pd.DataFrame({
            "Ticker": [_norm_symbol(x, spec["suffix"]) for x in t[tick_col]],
            "Name": t[name_col].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip(),
            "Sector": t[sec_col].astype(str).str.strip() if sec_col else "Unclassified",
            "SubIndustry": t[sub_col].astype(str).str.strip() if sub_col else "",
        })
        out = out[out.Ticker.str.len() > 0].drop_duplicates("Ticker").reset_index(drop=True)
        return out
    raise ValueError(f"Could not find a constituents table for {index_key} (page layout may have changed)")


def fetch_constituents(index_key: str, http_get: HttpGet, cache_dir: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Download (and cache) an index list.  Falls back to the last cached copy and says so."""
    cache_dir = Path(cache_dir or config.DATA_DIR / "constituents")
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / (re.sub(r"\W+", "_", index_key) + ".csv")
    try:
        df = parse_constituents(http_get(INDEXES[index_key]["url"]), index_key)
        df.to_csv(f, index=False)
        return df, "LIVE"
    except Exception as e:  # noqa: BLE001
        if f.exists():
            return pd.read_csv(f), f"CACHED (live fetch failed: {type(e).__name__})"
        raise


def parse_custom_list(text: str) -> pd.DataFrame:
    """User-pasted tickers: one per line / comma separated, optionally `TICKER,Name,Group` rows."""
    rows = []
    for line in re.split(r"[\n;]+", text):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 1 and " " in parts[0]:
            parts = parts[0].split()
        if len(parts) == 1:
            rows.append((parts[0].upper(), parts[0].upper(), "Custom"))
        else:
            rows.append((parts[0].upper(), parts[1] if len(parts) > 1 else parts[0], parts[2] if len(parts) > 2 else "Custom"))
    df = pd.DataFrame(rows, columns=["Ticker", "Name", "Sector"]).drop_duplicates("Ticker")
    df["SubIndustry"] = ""
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Fama–French factors
# ----------------------------------------------------------------------------
FF_URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV (header lines, blank line, `,Col1,Col2…`, then YYYYMMDD rows, then a footer)."""
    lines = text.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.strip().startswith(",")), None)
    if hdr is None:
        raise ValueError("Header row not found in factor file")
    cols = [c.strip() for c in lines[hdr].split(",")[1:]]
    rows, idx = [], []
    for l in lines[hdr + 1 :]:
        parts = [p.strip() for p in l.split(",")]
        if not parts or not re.fullmatch(r"\d{8}", parts[0]):
            if rows:
                break
            continue
        idx.append(pd.Timestamp(parts[0]))
        rows.append([np.nan if (p == "" or float(p) <= -99.0) else float(p) / 100.0 for p in parts[1 : 1 + len(cols)]])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=cols)


def read_french_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return parse_french_csv(z.read(name).decode("latin-1"))


def fetch_factors(http_bytes: Callable[[str], bytes], cache_dir: Path | None = None) -> tuple[pd.DataFrame, str]:  # pragma: no cover - network wrapper
    cache_dir = Path(cache_dir or config.DATA_DIR)
    f = cache_dir / "french_factors_daily.csv.gz"
    try:
        ff5 = read_french_zip(http_bytes(FF_URLS["ff5"]))
        mom = read_french_zip(http_bytes(FF_URLS["mom"]))
        mom.columns = ["Mom"]
        df = ff5.join(mom, how="left")
        df.to_csv(f, compression="gzip")
        return df, "LIVE"
    except Exception as e:  # noqa: BLE001
        if f.exists():
            return pd.read_csv(f, index_col=0, parse_dates=True), f"CACHED (live fetch failed: {type(e).__name__})"
        raise


def default_http_bytes(url: str) -> bytes:  # pragma: no cover - network
    import requests

    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (research workbench)"})
    r.raise_for_status()
    return r.content
