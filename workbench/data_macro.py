"""Open macro-economic data (FRED public CSV endpoint – no API key, no paid database).

Point-in-time discipline
------------------------
Economic numbers are published *after* the period they describe (CPI for March appears in mid-April).  Each
series therefore carries a `lag_days` – a conservative publication delay – and every value is only treated as
"known" lag_days after its observation date.  Monthly feature tables are built from what was KNOWN at each
month-end.

Honest caveat (also shown in the app): FRED serves the latest *revised* figures, not the first-published ones,
so back-tests that use revised GDP/payrolls/production data are slightly flattering.  Market-based series
(yields, spreads, VIX) are never revised and are the cleanest inputs.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("workbench.macro")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
HttpGet = Callable[[str], str]


@dataclass(frozen=True)
class MacroSeries:
    id: str
    name: str
    group: str          # Growth | Inflation | Rates & curve | Financial conditions | Money & liquidity | Markets | Housing & consumers
    lag_days: int       # conservative publication delay after the observation date
    transform: str      # how it enters the monthly table: level | yoy | diff12 | chg3
    agg: str            # month aggregation of what was known: last | mean
    why: str            # plain-English reason it matters (shown in the app)
    revised: bool       # True if later revised (look-ahead risk)


CATALOG: list[MacroSeries] = [
    # --- growth / labour -------------------------------------------------------------
    MacroSeries("INDPRO", "Industrial production index (US)", "Growth", 50, "yoy", "last", "Output of factories, mines and utilities – a fast read on the real economy.", True),
    MacroSeries("PAYEMS", "Non-farm payrolls (US jobs)", "Growth", 40, "yoy", "last", "Total jobs. Falling job growth is one of the most reliable recession warnings.", True),
    MacroSeries("UNRATE", "Unemployment rate (US)", "Growth", 40, "diff12", "last", "Share of workers without a job. A sharp 12-month rise usually marks a downturn.", True),
    MacroSeries("ICSA", "Initial jobless claims (US, weekly)", "Growth", 6, "yoy", "mean", "New unemployment claims – the earliest weekly sign of layoffs.", True),
    MacroSeries("RSAFS", "Retail sales (US)", "Growth", 50, "yoy", "last", "Consumer spending is ~70% of US GDP.", True),
    MacroSeries("CFNAI", "Chicago Fed National Activity Index", "Growth", 55, "level", "last", "One number summarising 85 monthly indicators; above 0 means above-trend growth.", True),
    MacroSeries("USALOLITOAASTSAM", "OECD leading indicator (US)", "Growth", 75, "level", "last", "Built to turn before the economy does; 100 = long-run trend.", True),
    MacroSeries("GDPC1", "Real GDP (US, quarterly)", "Growth", 120, "yoy", "last", "The official size of the economy – slow and heavily revised, so treated with a long delay.", True),
    # --- inflation ------------------------------------------------------------------------
    MacroSeries("CPIAUCSL", "Consumer prices – CPI (US)", "Inflation", 45, "yoy", "last", "Headline inflation. Drives central-bank interest-rate decisions.", False),
    MacroSeries("CPILFESL", "Core CPI excl. food & energy (US)", "Inflation", 45, "yoy", "last", "Underlying inflation with volatile items removed.", False),
    MacroSeries("PCEPILFE", "Core PCE price index (US)", "Inflation", 60, "yoy", "last", "The Federal Reserve's preferred inflation gauge.", True),
    MacroSeries("T10YIE", "10-year breakeven inflation (market)", "Inflation", 1, "level", "last", "Inflation the bond market expects over 10 years (never revised).", False),
    # --- rates & curve ------------------------------------------------------------------
    MacroSeries("FEDFUNDS", "Fed funds rate (monthly avg)", "Rates & curve", 32, "level", "last", "The US policy interest rate – the price of money.", False),
    MacroSeries("DGS10", "10-year Treasury yield", "Rates & curve", 1, "level", "last", "The benchmark long-term interest rate; moves all asset valuations.", False),
    MacroSeries("DGS2", "2-year Treasury yield", "Rates & curve", 1, "level", "last", "Reflects where the market expects policy rates to go.", False),
    MacroSeries("T10Y2Y", "Yield curve: 10y minus 2y", "Rates & curve", 1, "level", "last", "Negative (inverted) curves have preceded most US recessions.", False),
    MacroSeries("T10Y3M", "Yield curve: 10y minus 3m", "Rates & curve", 1, "level", "last", "The curve measure the New York Fed uses for recession probability.", False),
    MacroSeries("DFII10", "10-year real yield (TIPS)", "Rates & curve", 1, "level", "last", "Interest rate after inflation – high real yields squeeze valuations.", False),
    # --- financial conditions -------------------------------------------------------------
    MacroSeries("NFCI", "Chicago Fed financial conditions index", "Financial conditions", 5, "level", "last", "Above 0 = tighter than average credit/financing conditions.", False),
    MacroSeries("STLFSI4", "St. Louis Fed financial stress index", "Financial conditions", 6, "level", "last", "Above 0 = above-average market stress.", False),
    MacroSeries("BAA10Y", "Corporate (Baa) bond spread over 10y Treasury", "Financial conditions", 1, "level", "last", "Extra yield investors demand to hold riskier companies – widens before/during trouble.", False),
    MacroSeries("VIXCLS", "VIX – stock-market fear gauge", "Financial conditions", 1, "level", "last", "Expected stock volatility. Spikes in sell-offs.", False),
    # --- money & liquidity ----------------------------------------------------------------
    MacroSeries("M2SL", "Money supply M2 (US)", "Money & liquidity", 50, "yoy", "last", "Broad money growth; shrinking money supply is historically unusual and bearish.", True),
    MacroSeries("WALCL", "Federal Reserve balance sheet", "Money & liquidity", 3, "yoy", "last", "When the Fed buys assets liquidity rises; when it shrinks the balance sheet it falls.", False),
    # --- markets / commodities / dollar -------------------------------------------------
    MacroSeries("DCOILWTICO", "WTI crude oil price", "Markets", 2, "yoy", "last", "Energy costs feed inflation and company profits.", False),
    MacroSeries("DTWEXBGS", "Trade-weighted US dollar (broad)", "Markets", 2, "yoy", "last", "A strong dollar tightens global financial conditions.", False),
    # --- housing & consumers ---------------------------------------------------------------
    MacroSeries("PERMIT", "Housing building permits (US)", "Housing & consumers", 50, "yoy", "last", "Housing is highly interest-rate sensitive and leads the cycle.", True),
    MacroSeries("UMCSENT", "Consumer sentiment (Michigan)", "Housing & consumers", 30, "level", "last", "How households feel; low readings often coincide with market lows.", True),
]
BY_ID = {s.id: s for s in CATALOG}
# NBER recession dummy – known only with a long delay, so it is used for *hindsight* shading only.
RECESSION_ID = "USREC"


def parse_fred_csv(text: str) -> pd.Series:
    """Parse the FRED `fredgraph.csv` format (header `observation_date,<ID>` or older `DATE,<ID>`)."""
    if text.lstrip().startswith("<"):
        raise ValueError("FRED returned an HTML page instead of CSV")
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("Unexpected FRED CSV layout")
    d = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    v = pd.to_numeric(df.iloc[:, 1], errors="coerce")     # '.' means missing
    s = pd.Series(v.to_numpy(), index=d, name=str(df.columns[1])).dropna()
    s = s[~s.index.isna()]
    return s[~s.index.duplicated(keep="last")].sort_index()


def default_http_get(url: str) -> str:  # pragma: no cover - network
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (research workbench)"})
    r.raise_for_status()
    return r.text


class MacroStore:
    def __init__(self, folder: Path | str | None = None):
        self.folder = Path(folder) if folder else config.DATA_DIR / "macro"
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, sid: str) -> Path:
        return self.folder / f"{sid}.csv"

    def load(self, sid: str) -> pd.Series:
        p = self.path(sid)
        if not p.exists():
            return pd.Series(dtype=float, name=sid)
        s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = sid
        return s

    def age_hours(self, sid: str) -> float | None:
        p = self.path(sid)
        return (time.time() - p.stat().st_mtime) / 3600 if p.exists() else None

    def refresh(
        self, ids: list[str], http_get: HttpGet | None = None, max_age_hours: float = 12.0, force: bool = False
    ) -> pd.DataFrame:
        """Download any series older than `max_age_hours`; return a status table.  Failures keep the cache."""
        http_get = http_get or default_http_get
        rows = []
        for sid in ids:
            age = self.age_hours(sid)
            if not force and age is not None and age < max_age_hours:
                s = self.load(sid)
                rows.append({"Series": sid, "Status": "CACHED_FRESH", "Observations": len(s), "Last observation": s.index.max() if len(s) else None, "Detail": f"{age:.1f}h old"})
                continue
            try:
                s = parse_fred_csv(http_get(FRED_URL.format(id=sid)))
                if len(s) < 2:
                    raise ValueError("no observations")
                s.to_csv(self.path(sid), header=[sid])
                rows.append({"Series": sid, "Status": "DOWNLOADED", "Observations": len(s), "Last observation": s.index.max(), "Detail": ""})
            except Exception as e:  # noqa: BLE001
                s = self.load(sid)
                st = "STALE_CACHE" if len(s) else "FAILED"
                rows.append({"Series": sid, "Status": st, "Observations": len(s), "Last observation": s.index.max() if len(s) else None, "Detail": f"{type(e).__name__}: {e}"[:160]})
            time.sleep(0.05)
        return pd.DataFrame(rows)

    def load_many(self, ids: list[str]) -> dict[str, pd.Series]:
        out = {}
        for sid in ids:
            s = self.load(sid)
            if len(s):
                out[sid] = s
        return out


# ----------------------------------------------------------------------------
# Point-in-time monthly feature table
# ----------------------------------------------------------------------------
def known_monthly(s: pd.Series, lag_days: int, agg: str = "last") -> pd.Series:
    """Value that was KNOWN at each month-end (observations are released `lag_days` after their date)."""
    avail = s.copy()
    avail.index = avail.index + pd.Timedelta(days=lag_days)
    r = avail.resample("ME")
    m = r.mean() if agg == "mean" else r.last()
    return m.ffill()


def transform(m: pd.Series, how: str) -> pd.Series:
    if how == "yoy":
        return m.pct_change(12, fill_method=None)
    if how == "diff12":
        return m.diff(12)
    if how == "chg3":
        return m.diff(3)
    return m


def expanding_z(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """z-score using only history up to each date (no look-ahead)."""
    mu = s.expanding(min_periods).mean()
    sd = s.expanding(min_periods).std(ddof=1)
    return (s - mu) / sd.replace(0, np.nan)


def build_monthly_table(raw: dict[str, pd.Series], end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, zscores): transformed monthly series known at each month-end, and their expanding z-scores."""
    feats = {}
    for sid, s in raw.items():
        meta = BY_ID.get(sid)
        if meta is None or sid == RECESSION_ID:
            continue
        m = known_monthly(s, meta.lag_days, meta.agg)
        feats[sid] = transform(m, meta.transform)
    F_ = pd.DataFrame(feats).sort_index()
    # Never publish a month that has not finished yet: a half-month average of a daily series (VIX, yields…)
    # would be treated as a full month and could swing the "current regime" on a few days of data.
    latest = max((s.index.max() for s in raw.values() if len(s)), default=None)
    if latest is not None:
        F_ = F_.loc[F_.index <= latest]
        month_end = latest + pd.offsets.MonthEnd(0)
        if latest < month_end:                       # latest observation is mid-month -> that month is incomplete
            F_ = F_.loc[F_.index < month_end]
    if end is not None:
        F_ = F_.loc[:end]
    Z = F_.apply(expanding_z)
    return F_, Z


def recession_monthly(raw: dict[str, pd.Series]) -> pd.Series:
    s = raw.get(RECESSION_ID)
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.resample("ME").max().rename("NBER_recession")
