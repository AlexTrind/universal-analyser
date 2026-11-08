"""Vectorised, point-in-time momentum features.

Every feature for decision date *d* uses only prices on or before *d*.  The calculations reproduce the
v7.8 definitions (so results stay comparable) but run on whole matrices instead of per-ticker Python
loops, which is what makes walk-forward backtests and robustness grids ~50-100× faster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

REG_WINDOW = 125


@dataclass
class Panel:
    """Prices on one common trading calendar (base currency), plus bookkeeping."""
    prices: pd.DataFrame            # master calendar, forward-filled up to `ffill_limit` days
    ffill_limit: int

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.prices.index

    def pos(self, dates) -> np.ndarray:
        """Integer position of the last calendar day on or before each date (-1 if none)."""
        return self.index.searchsorted(pd.DatetimeIndex(dates), side="right") - 1


def build_panel(base_prices: pd.DataFrame, ffill_limit: int = 5, min_cov: float = 0.5) -> Panel:
    """Align all series to a master calendar.

    A date belongs to the master calendar if at least `min_cov` of the series that exist on that date have a
    price (so one market's holiday doesn't create a fake zero-return day for everyone else).
    Short gaps (≤ ffill_limit days) are forward-filled; longer gaps stay missing – never invented.
    """
    raw = base_prices.sort_index().dropna(axis=1, how="all")
    raw = raw[~raw.index.duplicated(keep="last")]
    if raw.empty:
        return Panel(raw, ffill_limit)
    first = raw.apply(lambda s: s.first_valid_index())
    last = raw.apply(lambda s: s.last_valid_index())
    idx = raw.index.values[:, None]
    exists = (idx >= first.values.astype("datetime64[ns]")[None, :]) & (idx <= last.values.astype("datetime64[ns]")[None, :])
    cov = raw.notna().to_numpy().sum(1) / np.maximum(exists.sum(1), 1)
    master = raw.index[cov >= min_cov]
    P = raw.reindex(master).ffill(limit=ffill_limit)
    return Panel(P, ffill_limit)


@dataclass
class Params:
    ma_days: int = 200
    min_history_days: int = 252
    benchmark: str = "SSAC.L"


@dataclass
class Features:
    dates: pd.DatetimeIndex
    pos: np.ndarray
    mats: dict[str, pd.DataFrame]       # name -> (dates × tickers)
    universe: pd.DataFrame              # bool (dates × tickers): ETFs that may be ranked on that date
    benchmark: str

    def snapshot(self, date, names: list[str] | None = None) -> pd.DataFrame:
        d = pd.Timestamp(date)
        i = self.dates.get_loc(d)
        u = self.universe.iloc[i]
        tick = u.index[u.to_numpy()]
        cols = {k: v.iloc[i][tick] for k, v in self.mats.items() if names is None or k in names}
        snap = pd.DataFrame(cols)
        snap.index.name = "Yahoo_Ticker"
        return snap


def _ret_at(A: np.ndarray, pos: np.ndarray, days: int, skip: int) -> np.ndarray:
    """Simple return from t-(days+skip) to t-skip, in observations, for every date/ticker."""
    end = pos - skip
    start = pos - skip - days
    ok = (start >= 0) & (end >= 0)
    out = np.full((len(pos), A.shape[1]), np.nan)
    if ok.any():
        out[ok] = A[end[ok]] / A[start[ok]] - 1.0
    return out


def _regression(logA: np.ndarray, logB: np.ndarray, pos: np.ndarray, n: int = REG_WINDOW, chunk: int = 64):
    """Slope & R² of log(asset/benchmark) against time over the last `n` observations (masked OLS)."""
    D, N = len(pos), logA.shape[1]
    slope = np.full((D, N), np.nan)
    r2 = np.full((D, N), np.nan)
    need = max(50, int(n * 0.7))
    off = np.arange(-(n - 1), 1)
    x_full = np.arange(n, dtype=float)
    for c0 in range(0, D, chunk):
        sl = slice(c0, min(D, c0 + chunk))
        win = pos[sl, None] + off[None, :]                     # (d, n)
        inb = win >= 0
        w = np.clip(win, 0, None)
        y = logA[w] - logB[w][:, :, None]                      # (d, n, N)
        y = np.where(inb[:, :, None], y, np.nan)
        valid = np.isfinite(y)
        cnt = valid.sum(1).astype(float)
        y0 = np.where(valid, y, 0.0)
        x = np.where(valid, x_full[None, :, None], 0.0)
        Sx, Sy = x.sum(1), y0.sum(1)
        Sxy, Sxx, Syy = (x * y0).sum(1), (x * x).sum(1), (y0 * y0).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vx = cnt * Sxx - Sx**2
            vy = cnt * Syy - Sy**2
            cxy = cnt * Sxy - Sx * Sy
            s = cxy / vx
            r = cxy**2 / (vx * vy)
        bad = cnt < need
        s[bad] = np.nan
        r[bad] = np.nan
        slope[sl], r2[sl] = s, r
    return slope, r2


def compute_features(panel: Panel, dates, params: Params) -> Features:
    P = panel.prices
    dates = pd.DatetimeIndex(dates)
    pos = panel.pos(dates)
    keep = pos >= 0
    dates, pos = dates[keep], pos[keep]
    A = P.to_numpy(dtype=float)
    tick = list(P.columns)
    if params.benchmark not in P.columns:
        raise KeyError(f"Benchmark {params.benchmark} has no price data")
    b_i = tick.index(params.benchmark)
    B = A[:, [b_i]]

    def frame(arr):
        return pd.DataFrame(arr, index=dates, columns=tick)

    mats: dict[str, np.ndarray] = {}
    # price & trend ----------------------------------------------------------
    ma = P.rolling(params.ma_days, min_periods=params.ma_days).mean().to_numpy()
    mats["Price"] = A[pos]
    mats["MA"] = ma[pos]
    with np.errstate(invalid="ignore"):
        mats["TrendGate"] = (A[pos] > ma[pos]).astype(float)   # NaN compares False -> 0
    # trailing excess returns -------------------------------------------------
    raw_ret = {}
    for nm, days, skip in [("6M", 126, 0), ("12M", 252, 0), ("6M_Skip1M", 126, 21), ("12M_Skip1M", 252, 21)]:
        ra = _ret_at(A, pos, days, skip)
        rb = _ret_at(B, pos, days, skip)
        raw_ret[nm] = ra
        mats["Excess_" + nm] = (1 + ra) / (1 + rb) - 1
    mats["Ret_1M"] = _ret_at(A, pos, 21, 0)
    mats["Ret_3M"] = _ret_at(A, pos, 63, 0)
    mats["Ret_12M"] = raw_ret["12M"]
    mats["Bench_Ret_12M"] = np.repeat(_ret_at(B, pos, 252, 0), A.shape[1], axis=1)
    hi = P.rolling(252, min_periods=126).max().to_numpy()
    mats["Dist_52wHigh"] = A[pos] / hi[pos] - 1
    # regression trend quality ------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        logA, logB = np.log(A), np.log(B[:, 0])
    slope, r2 = _regression(logA, logB, pos)
    with np.errstate(over="ignore"):
        ann = np.expm1(slope * 252)
    mats["RelReg_AnnSlope"] = ann
    mats["RelReg_R2"] = r2
    mats["RelReg_Score"] = ann * r2
    # risk ---------------------------------------------------------------------
    ret = P.pct_change(fill_method=None)
    lr = np.log(P).diff()
    mats["Vol_63D"] = (lr.rolling(63, min_periods=20).std() * np.sqrt(252)).to_numpy()[pos]
    d2 = ret.clip(upper=0.0) ** 2
    cnt = ret.rolling(63, min_periods=20).count()
    mats["DownsideVol_63D"] = np.sqrt(d2.rolling(63, min_periods=20).sum() / cnt * 252).to_numpy()[pos]
    mats["ATRpct_20D_proxy"] = ret.abs().rolling(20, min_periods=20).mean().to_numpy()[pos]
    # history requirement ---------------------------------------------------------
    nobs = P.notna().cumsum().to_numpy()[pos]
    first = P.apply(lambda s: s.first_valid_index())
    first_ns = first.values.astype("datetime64[ns]")
    cutoff = dates.values[:, None] - np.timedelta64(int(params.min_history_days * 1.35), "D")
    hist_ok = (nobs >= params.min_history_days) & (first_ns[None, :] <= cutoff)
    has_px = np.isfinite(A[pos])
    uni = hist_ok & has_px
    uni[:, b_i] = False
    U = frame(uni)

    F = {k: frame(v) for k, v in mats.items()}
    F["TrendGate"] = F["TrendGate"].astype(bool)
    # scoring (percentiles are ranked only among rankable ETFs on that date) -------------
    pct = {}
    for k in ["Excess_6M", "Excess_12M", "Excess_6M_Skip1M", "Excess_12M_Skip1M", "RelReg_Score"]:
        pct[k] = F[k].where(U).rank(axis=1, pct=True, method="average") * 100
        F[k + "_Pct"] = pct[k]
    w = config.SCORE_WEIGHTS
    score = sum(w[k] * pct[k] for k in w)
    F["MomentumScore"] = score
    F["MomentumPercentile"] = score.rank(axis=1, pct=True, method="average") * 100
    F["EligibleSignal"] = F["TrendGate"] & (F["Excess_6M_Skip1M"] > 0) & (F["Excess_12M_Skip1M"] > 0) & U
    return Features(dates, pos, F, U, params.benchmark)


def rebalance_dates(index: pd.DatetimeIndex, start, end, freq: str) -> list[pd.Timestamp]:
    """Last trading day of each week/month/quarter inside [start, end]."""
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if idx.empty:
        return []
    s = pd.Series(idx, index=idx)
    key = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q"}[freq]
    return list(pd.DatetimeIndex(s.groupby(idx.to_period(key)).last().values))
