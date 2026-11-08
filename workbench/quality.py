"""Data-quality screening.  Philosophy: never silently use a suspicious series – flag it, and
exclude 'FAIL' series from signals so a data error cannot become an investment decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEV_ORDER = {"OK": 0, "INFO": 1, "WARN": 2, "FAIL": 3}


def repair_bad_ticks(s: pd.Series, spike: float = 0.25, revert_tol: float = 0.05) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Blank out single-day spikes that fully reverse the next day (typical provider glitch).

    A real crash/rally does not snap back to the previous level within a day, so a move of more than
    `spike` that returns to within `revert_tol` of the pre-spike level is treated as a bad tick.
    """
    x = s.dropna()
    if len(x) < 5:
        return s, []
    v = x.to_numpy(dtype=float)
    bad = []
    for i in range(1, len(v) - 1):
        up = v[i] / v[i - 1] - 1.0
        back = v[i + 1] / v[i - 1] - 1.0
        if abs(up) > spike and abs(back) < revert_tol:
            bad.append(x.index[i])
    if not bad:
        return s, []
    out = s.copy()
    out.loc[bad] = np.nan
    return out, bad


def repair_unit_errors(s: pd.Series, band: float = 0.6, max_fix: int = 12) -> tuple[pd.Series, list[str]]:
    """Fix pence/pound (×100) scale errors, working back from the newest prices.

    Yahoo sometimes quotes a London ETF in pounds for a few days (or for its whole early history)
    and in pence otherwise.  A ×100 / ÷100 step is rescaled so the series joins up with its latest
    scale.  The fix is accepted ONLY if every repaired step then looks like an ordinary move (within
    ±`band` of flat); otherwise the series is returned untouched and the quality screen will FAIL it.
    """
    x = s.dropna()
    if len(x) < 5:
        return s, []
    v = x.to_numpy(dtype=float).copy()
    if (v <= 0).any():
        return s, []
    notes: list[str] = []
    for _ in range(max_fix):
        ratio = v[1:] / v[:-1]
        hit = np.where(((ratio > 85) & (ratio < 115)) | ((ratio > 1 / 115) & (ratio < 1 / 85)))[0]
        if len(hit) == 0:
            break
        i = int(hit[-1])  # latest step first
        f = 100.0 if ratio[i] > 1 else 0.01
        v[: i + 1] *= f
        if not (1 - band < v[i + 1] / v[i] < 1 + band / (1 - band)):
            return s, []  # not a clean ×100 step – refuse to guess
        notes.append(f"×{f:g} applied to prices up to {x.index[i].date()}")
    else:
        return s, []
    if not notes:
        return s, []
    out = s.copy()
    out.loc[x.index] = v
    return out, notes


def repair_unit_frame(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Apply `repair_unit_errors` to every column; returns the repaired frame and {ticker: notes}."""
    out, log = prices.copy(), {}
    for t in prices.columns:
        fixed, n = repair_unit_errors(prices[t])
        if n:
            out[t] = fixed
            log[t] = n
    return out, log


def assess_series(s: pd.Series, ref_last: pd.Timestamp, min_history: int = 252) -> list[tuple[str, str, str]]:
    """Return a list of (severity, code, message) for one price series."""
    issues: list[tuple[str, str, str]] = []
    x = s.dropna()
    if x.empty:
        return [("FAIL", "NO_DATA", "No price data returned")]
    if (x <= 0).any():
        issues.append(("FAIL", "NONPOSITIVE", "Zero or negative prices"))
    r = x.pct_change().dropna()
    if len(r):
        ratio = (x / x.shift(1)).dropna()
        unit = ratio[(ratio.between(85, 115)) | (ratio.between(1 / 115, 1 / 85))]
        if len(unit):
            issues.append(("FAIL", "UNIT_SWITCH", f"Price jumped ×100 or ÷100 on {unit.index[0].date()} (pence/pound mix-up?)"))
        # any other one-day move that no ETF or stock can really make (more than +150% or less than -60%)
        absurd = ratio[((ratio > 2.5) | (ratio < 0.4)) & ~ratio.index.isin(unit.index)]
        if len(absurd):
            issues.append(("FAIL", "IMPLAUSIBLE_JUMP", f"Price moved ×{absurd.iloc[0]:.3g} in one day on {absurd.index[0].date()} – almost certainly a data error"))
        big = r[r.abs() > 0.35]
        if len(big):
            issues.append(("WARN", "LARGE_MOVE", f"{len(big)} daily move(s) beyond ±35% (largest {big.abs().max():.0%} on {big.abs().idxmax().date()}) – verify splits/data"))
        zero = float((r == 0).mean())
        if zero > 0.40 and len(r) > 60:
            issues.append(("WARN", "FLAT_PRICES", f"{zero:.0%} of days show no price change – illiquid or stale feed"))
    if len(r) > 120:
        zr = (r == 0).groupby(r.index.year).agg(["mean", "size"])
        flat = zr[(zr["mean"] > 0.5) & (zr["size"] >= 100)]
        if len(flat):
            yrs = ", ".join(str(y) for y in flat.index)
            issues.append(("WARN", "FLAT_PERIOD", f"Prices barely move in {yrs} (stale or placeholder history) – returns before this are unreliable"))
    stale_days = (ref_last - x.index[-1]).days
    if stale_days > 20:
        issues.append(("FAIL", "STALE", f"Last price is {stale_days} days old"))
    elif stale_days > 7:
        issues.append(("WARN", "STALE", f"Last price is {stale_days} days old"))
    gaps = x.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > 14:
        g = gaps.idxmax()
        issues.append(("WARN", "GAP", f"Gap of {int(gaps.max())} days before {g.date()}"))
    if len(x) < min_history:
        issues.append(("INFO", "SHORT_HISTORY", f"Only {len(x)} observations"))
    return issues


def quality_report(prices: pd.DataFrame, ref_last: pd.Timestamp | None = None, min_history: int = 252, repaired: dict | None = None) -> pd.DataFrame:
    """One row per ticker: observations, first/last date, worst severity, and issue text."""
    ref_last = ref_last or prices.index.max()
    rows = []
    for t in prices.columns:
        s = prices[t]
        iss = assess_series(s, ref_last, min_history)
        if repaired and t in repaired:
            iss.append(("WARN", "UNIT_REPAIRED", "Pence/pound scale error in the provider data was corrected automatically (" + "; ".join(repaired[t]) + ") – check the chart"))
        sev = max((i[0] for i in iss), key=lambda k: SEV_ORDER[k], default="OK")
        x = s.dropna()
        rows.append(
            {
                "Ticker": t,
                "Severity": sev,
                "Observations": int(len(x)),
                "First date": x.index.min().date() if len(x) else None,
                "Last date": x.index.max().date() if len(x) else None,
                "Issues": " | ".join(f"{c}: {m}" for _, c, m in iss),
            }
        )
    return pd.DataFrame(rows)


def failing_tickers(report: pd.DataFrame) -> set[str]:
    return set(report.loc[report["Severity"].eq("FAIL"), "Ticker"])
