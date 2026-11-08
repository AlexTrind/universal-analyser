"""How assets relate to the economy: regime-conditional returns, lead-lag, predictive regressions, macro betas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S


def month_end(prices: pd.Series | pd.DataFrame):
    """Month-end prices; the final (possibly partial) month uses the latest available close."""
    return prices.resample("ME").last().dropna(how="all")


def forward_returns(me_prices: pd.Series, h: int) -> pd.Series:
    """Simple return from this month-end to h months later (NaN where the future is not yet known)."""
    return me_prices.shift(-h) / me_prices - 1.0


def confidence_label(n_eff: float) -> str:
    return "Very low (anecdotal)" if n_eff < 5 else "Low" if n_eff < 10 else "Moderate" if n_eff < 25 else "Reasonable"


def regime_table(me_prices: pd.Series, labels: pd.Series, h: int = 3, n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    """What happened to the asset over the NEXT h months after each regime was observed.

    n_eff ≈ months/h because overlapping h-month windows are not independent observations.
    p-value: circular-shift permutation test (keeps the persistence of regimes).
    """
    fwd = forward_returns(me_prices, h).rename("fwd")
    df = pd.concat([fwd, labels.rename("lab")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()
    base_mean = df["fwd"].mean()
    rows = []
    for lab, g in df.groupby("lab"):
        n = len(g)
        n_eff = max(n / h, 1.0)
        hit = int((g["fwd"] > 0).sum())
        lo, hi = S.wilson_interval(round(hit / h) if h > 1 else hit, max(int(round(n_eff)), 1)) if h > 1 else S.wilson_interval(hit, n)
        t = S.circular_shift_test(df["fwd"].to_numpy(), df["lab"].to_numpy(), lab, n_perm=n_perm, seed=seed)
        rows.append({
            "Regime": lab, "Months observed": n, "Independent obs (≈)": round(n_eff, 1),
            f"Avg {h}M forward return": g["fwd"].mean(), "Median": g["fwd"].median(),
            "% of periods positive": hit / n, "Worst": g["fwd"].min(), "Best": g["fwd"].max(),
            "vs all months": g["fwd"].mean() - base_mean, "p-value (luck?)": t["p"],
            "Confidence": confidence_label(n_eff),
        })
    return pd.DataFrame(rows).sort_values(f"Avg {h}M forward return", ascending=False).reset_index(drop=True)


def lead_lag(x: pd.Series, me_prices: pd.Series, h: int = 3, max_lag: int = 12) -> pd.DataFrame:
    """Does the indicator, observed L months earlier, line up with the asset's next-h-month return?"""
    y = forward_returns(me_prices, h)
    rows = []
    for L in range(0, max_lag + 1):
        d = pd.concat([x.shift(L).rename("x"), y.rename("y")], axis=1).dropna()
        if len(d) < 36:
            continue
        res = S.ols_hac(d["y"].to_numpy(), d["x"].to_numpy(), lags=h + L + 1, names=["x"])
        corr, p_null = shift_null_p(d["x"].to_numpy(float), d["y"].to_numpy(float), min_shift=max(12, 2 * h))
        rows.append({"Indicator lead (months)": L, "Correlation": corr,
                     "Slope": res.coef[1], "t-stat (HAC)": res.t[1], "p-value": p_null, "Months": len(d)})
    return pd.DataFrame(rows)


def predictive_regression(X: pd.DataFrame, me_prices: pd.Series, h: int = 3, min_train: int = 84) -> dict:
    """Predict the next-h-month return from indicators known today.

    In-sample: HAC regression.  Out-of-sample (the honest test): expanding window, training only on outcomes
    already realised at each date (purged by h months); score = Campbell–Thompson OOS R² versus the
    historical-average forecast.  Positive OOS R² means the indicators beat 'just use the average'.
    """
    y = forward_returns(me_prices, h).rename("y")
    d = pd.concat([y, X], axis=1)
    full = d.dropna()
    if len(full) < min_train + 24:
        return {"ok": False, "reason": f"Only {len(full)} usable months; need ≥ {min_train + 24}"}
    ins = S.ols_hac(full["y"].to_numpy(), full.drop(columns="y").to_numpy(), lags=h + 2, names=list(X.columns))
    idx = full.index
    preds, bench, actual, when = [], [], [], []
    Xv, yv = full.drop(columns="y").to_numpy(), full["y"].to_numpy()
    mo = (idx.year * 12 + idx.month).to_numpy()
    for t in range(min_train, len(full)):
        # only rows whose outcome (month_i + h) is already known at month_t
        tr = np.where(mo <= mo[t] - h)[0]
        if len(tr) < Xv.shape[1] + 10:
            continue
        A = np.column_stack([np.ones(len(tr)), Xv[tr]])
        beta = np.linalg.lstsq(A, yv[tr], rcond=None)[0]
        preds.append(float(np.r_[1.0, Xv[t]] @ beta))
        bench.append(float(yv[tr].mean()))
        actual.append(float(yv[t]))
        when.append(idx[t])
    p, b, a = np.array(preds), np.array(bench), np.array(actual)
    sse_m, sse_b = float(((a - p) ** 2).sum()), float(((a - b) ** 2).sum())
    oos = 1 - sse_m / sse_b
    diff = (a - b) ** 2 - (a - p) ** 2
    ci = S.bootstrap_ci(diff, np.mean, n_boot=2000, mean_block=max(h + 2, 4), seed=0)
    return {
        "ok": True, "in_sample": ins, "oos_r2": float(oos), "oos_months": len(a),
        "oos_gain_ci": (ci["lo"], ci["hi"]), "oos_gain_p_le0": ci["p_le_0"],
        "series": pd.DataFrame({"prediction": p, "historical_mean": b, "actual": a}, index=pd.DatetimeIndex(when)),
        "verdict": ("Indicators beat the historical average out-of-sample" if (oos > 0 and ci["lo"] > 0)
                    else "No reliable out-of-sample edge – treat any in-sample link as unproven"),
    }


def macro_betas(asset_me: pd.Series, drivers: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Sensitivity of monthly asset returns to changes in macro/market drivers (HAC t-stats)."""
    r = asset_me.pct_change().rename("r")
    d = pd.concat([r, drivers], axis=1).dropna()
    if len(d) < 36:
        return pd.DataFrame()
    res = S.ols_hac(d["r"].to_numpy(), d.drop(columns="r").to_numpy(), lags=lags, names=list(drivers.columns))
    t = res.table().drop(index="const")
    t["Significant (|t|>2)"] = t["t"].abs() > 2
    t["R² (all drivers)"] = res.r2
    return t


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 36) -> pd.Series:
    return a.rolling(window).corr(b)


def _pair(x: pd.Series, y: pd.Series, lag: int) -> pd.DataFrame:
    return pd.concat([x.shift(lag).rename("x"), y.rename("y")], axis=1).dropna()


def shift_null_p(xv: np.ndarray, yv: np.ndarray, min_shift: int = 12) -> tuple[float, float]:
    """Persistence-proof p-value for a correlation.

    Both series are slid past each other in a circle (every possible offset except the near-aligned ones). That keeps each
    series' own persistence and the overlap of the return windows, but destroys any real link, so the spread of the
    correlations at those offsets shows what 'pure coincidence' looks like for THESE two series. The p-value is the
    normal-tail probability of the observed correlation against that spread.  (Ordinary t-tests badly overstate
    significance when both series are slow-moving - the classic 'spurious regression'.)
    Returns (correlation, p).
    """
    n = len(xv)
    xz = (xv - xv.mean()) / xv.std()
    yz = (yv - yv.mean()) / yv.std()
    cc = np.fft.irfft(np.fft.rfft(xz) * np.conj(np.fft.rfft(yz)), n) / n
    k = np.arange(n)
    allowed = np.minimum(k, n - k) >= min_shift
    if allowed.sum() < 20:
        return float(cc[0]), float("nan")
    sd = float(cc[allowed].std(ddof=1))
    from scipy.stats import norm

    return float(cc[0]), float(2 * norm.sf(abs(cc[0]) / sd)) if sd > 0 else float("nan")


def _fit(d: pd.DataFrame, h: int, lag: int) -> dict | None:
    if len(d) < 36 or d["x"].std() == 0 or d["y"].std() == 0:
        return None
    res = S.ols_hac(d["y"].to_numpy(), d["x"].to_numpy(), lags=h + lag + 1, names=["x"])
    corr, p = shift_null_p(d["x"].to_numpy(float), d["y"].to_numpy(float), min_shift=max(12, 2 * h))
    return {"corr": corr, "slope": float(res.coef[1]), "t": float(res.t[1]), "p": p, "n": len(d)}


def compare_assets(x: pd.Series, me: pd.DataFrame, h: int = 3, lag: int = 0, min_months: int = 60) -> tuple[pd.DataFrame, list]:
    """Rank assets by how their NEXT-h-month return lines up with an indicator observed `lag` months earlier.

    Each result is checked three ways before it is called 'Supported':
      1. false-discovery-adjusted p-value (q) across all assets compared,
      2. same direction in both halves of the history,
      3. same direction at the neighbouring lags (a real link is not a one-lag accident).
    Assets with fewer than `min_months` usable months are excluded and returned separately.
    """
    fwd = me.shift(-h) / me - 1.0
    rows, excluded = [], []
    for c in me.columns:
        d = _pair(x, fwd[c], lag)
        if len(d) < min_months:
            excluded.append((c, len(d)))
            continue
        f = _fit(d, h, lag)
        if f is None:
            excluded.append((c, len(d)))
            continue
        mid = len(d) // 2
        halves = [float(np.corrcoef(g["x"], g["y"])[0, 1]) if g["x"].std() > 0 and g["y"].std() > 0 else 0.0 for g in (d.iloc[:mid], d.iloc[mid:])]
        sign = np.sign(f["corr"])
        nb = []
        for L in (lag - 1, lag + 1):
            if L < 0:
                continue
            dd = _pair(x, fwd[c], L)
            ff = _fit(dd, h, L) if len(dd) >= min_months else None
            if ff:
                nb.append(ff["corr"])
        rows.append({"Asset": c, "Months": f["n"], "Independent obs (≈)": round(f["n"] / h, 1), "Correlation": f["corr"], "Slope": f["slope"],
                     "t-stat (HAC)": f["t"], "p-value": f["p"], "First half": halves[0], "Second half": halves[1],
                     "Same direction in both halves": bool(all(np.sign(v) == sign for v in halves)),
                     "Same direction at neighbouring lags": bool(nb) and all(np.sign(v) == sign for v in nb)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, excluded
    df["q-value (false-discovery adj.)"] = S.benjamini_hochberg(df["p-value"].to_numpy())
    q = df["q-value (false-discovery adj.)"]
    ok = (q < 0.10) & df["Same direction in both halves"] & df["Same direction at neighbouring lags"]
    df["Verdict"] = np.where(ok, "Supported", np.where(df["p-value"] < 0.05, "Suggestive only", "No reliable link"))
    return df.sort_values("Correlation", key=lambda s: -s.abs()).reset_index(drop=True), excluded


def lag_grid(x: pd.Series, me: pd.DataFrame, h: int = 3, max_lag: int = 12, min_months: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assets × lags correlation grid and the matching false-discovery q-values (adjusted over ALL cells)."""
    fwd = me.shift(-h) / me - 1.0
    corr = pd.DataFrame(np.nan, index=me.columns, columns=range(0, max_lag + 1))
    pv = corr.copy()
    for c in me.columns:
        for L in range(0, max_lag + 1):
            d = _pair(x, fwd[c], L)
            f = _fit(d, h, L) if len(d) >= min_months else None
            if f:
                corr.loc[c, L], pv.loc[c, L] = f["corr"], f["p"]
    q = pd.DataFrame(S.benjamini_hochberg(pv.to_numpy().ravel()).reshape(pv.shape), index=pv.index, columns=pv.columns)
    return corr, q
