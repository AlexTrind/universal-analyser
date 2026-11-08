"""Long-run behaviour of an asset: profile, seasonality, drawdown anatomy, trend persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S


def asset_profile(p: pd.Series) -> dict:
    p = p.dropna()
    r = p.pct_change().dropna()
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    out = {"First date": p.index[0].date(), "Last date": p.index[-1].date(), "Years of data": yrs,
           "Annualised return (CAGR)": (p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
           "Annualised volatility": r.std() * np.sqrt(252), "Worst drawdown": S.max_drawdown(p / p.iloc[0]),
           "Best day": r.max(), "Worst day": r.min()}
    for y in (1, 3, 5, 10):
        if yrs >= y:
            sub = p[p.index >= p.index[-1] - pd.DateOffset(years=y)]
            out[f"Return, last {y}y (annualised)"] = (sub.iloc[-1] / sub.iloc[0]) ** (1 / y) - 1
    m = p.resample("ME").last().pct_change().dropna()
    if len(m) > 12:
        out["% of months positive"] = float((m > 0).mean())
        out["Skew (monthly)"] = float(m.skew())
        out["Excess kurtosis (monthly)"] = float(m.kurt())
    return out


def calendar_year_returns(p: pd.Series) -> pd.Series:
    ye = p.dropna().resample("YE").last()
    first = p.dropna().iloc[0]
    prev = ye.shift(1)
    prev.iloc[0] = first
    r = ye / prev - 1
    r.index = r.index.year
    return r.rename("Return")


def seasonality(p: pd.Series) -> pd.DataFrame:
    """Average return for each calendar month with a t-statistic.  Most 'seasonal patterns' are NOT significant."""
    m = p.dropna().resample("ME").last().pct_change().dropna()
    rows = []
    for k in range(1, 13):
        x = m[m.index.month == k]
        if len(x) < 5:
            continue
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if x.std(ddof=1) > 0 else np.nan
        rows.append({"Month": pd.Timestamp(2000, k, 1).strftime("%b"), "k": k, "Average": x.mean(), "Median": x.median(),
                     "% positive": (x > 0).mean(), "Years": len(x), "t-stat": t})
    out = pd.DataFrame(rows)
    if not out.empty:
        # 12 simultaneous looks: Bonferroni-style cutoff for 'really unusual'
        out["Stands out after multiple-testing?"] = out["t-stat"].abs() > 3.0
    return out


def trend_persistence(p: pd.Series, ma: int = 200) -> pd.DataFrame:
    """How long does the asset typically stay above / below its moving average, and what happens next?"""
    p = p.dropna()
    ma_s = p.rolling(ma, min_periods=ma).mean()
    above = (p > ma_s)[ma_s.notna()]
    if above.empty:
        return pd.DataFrame()
    grp = (above != above.shift()).cumsum()
    runs = above.groupby(grp).agg(["first", "size"])
    runs.columns = ["Above", "Days"]
    fwd = p.pct_change(63).shift(-63)
    state = above.map({True: "Above 200-day average", False: "Below 200-day average"})
    rows = []
    for s_lab, flag in (("Above 200-day average", True), ("Below 200-day average", False)):
        d = runs[runs.Above == flag]["Days"]
        f = fwd[above.index][above == flag].dropna()
        rows.append({"State": s_lab, "% of time": float((above == flag).mean()), "Typical spell (days, median)": float(d.median()) if len(d) else np.nan,
                     "Longest spell (days)": int(d.max()) if len(d) else 0, "Avg next-3M return": float(f.mean()) if len(f) else np.nan,
                     "% of next-3M returns positive": float((f > 0).mean()) if len(f) else np.nan})
    return pd.DataFrame(rows)
