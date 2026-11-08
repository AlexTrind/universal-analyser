"""Central configuration: paths, defaults, benchmark choices."""
from __future__ import annotations

from pathlib import Path

from . import __version__, APP_NAME

APP_VERSION = __version__
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "reports"
UNIVERSE_DIR = ROOT / "universe"
for _d in (DATA_DIR, LOG_DIR, REPORT_DIR, UNIVERSE_DIR):
    _d.mkdir(exist_ok=True)

DEFAULT_UNIVERSE = "ETF_Momentum_Universe_v7_8_FINAL_TEST.xlsx"

BENCHMARKS = {
    "iShares MSCI ACWI UCITS ETF (SSAC.L)": "SSAC.L",
    "Vanguard FTSE All-World UCITS ETF (VWRP.L)": "VWRP.L",
}
BASE_CURRENCIES = ["GBP", "USD", "EUR"]
DEFAULT_BASE = "GBP"

# Heuristic currency of a listing when neither the workbook nor Yahoo tells us.
# London lines are often quoted in USD, so ".L" is only a *guess* – the app flags it.
SUFFIX_CURRENCY = {
    ".L": "GBP", ".DE": "EUR", ".AS": "EUR", ".PA": "EUR", ".MI": "EUR", ".MC": "EUR", ".BR": "EUR",
    ".SW": "CHF", ".TO": "CAD", ".AX": "AUD", ".T": "JPY", ".HK": "HKD", ".ST": "SEK", ".CO": "DKK",
}
PENCE_CODES = {"GBp", "GBX", "GBPp"}

TRADING_DAYS_YEAR = 252

# Scoring weights (identical to v7.8 so results stay comparable).
SCORE_WEIGHTS = {
    "Excess_6M": 0.20,
    "Excess_12M": 0.30,
    "Excess_6M_Skip1M": 0.15,
    "Excess_12M_Skip1M": 0.20,
    "RelReg_Score": 0.15,
}

# Tickers used for context / factors regardless of universe
MARKET_PROXIES = {
    "S&P 500 (price index)": "^GSPC",
    "S&P 500 ETF (SPY)": "SPY",
    "VIX": "^VIX",
    "US 10y yield (Yahoo)": "^TNX",
    "Gold futures": "GC=F",
    "WTI crude futures": "CL=F",
    "US dollar index": "DX-Y.NYB",
}

SPDR_SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care", "XLE": "Energy",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}
