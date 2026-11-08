"""Analysis of a group of stocks (an index, a sector, or your own list) and of single assets:
ranking, breadth, concentration, dispersion/correlation, and 'why did it move' decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from . import stats as S


def group_snapshot(panel: F.Panel, members: pd.DataFrame, index_ticker: str, asof, ma_days: int = 200,
                   min_history_days: int = 252) -> pd.DataFrame:
    """Rank every member versus the index using the same point-in-time momentum model as the ETF screener."""
    prices = panel.prices
    have = [t for t in members["Ticker"] if t in prices.columns and t != index_ticker]
    sub = F.Panel(prices[have + [index_ticker]], panel.ffill_limit)
    feats = F.compute_features(sub, [pd.Timestamp(asof)], F.Params(ma_days, min_history_days, index_ticker))
    snap = feats.snapshot(feats.dates[0])
    ret = sub.prices.pct_change(fill_method=None).loc[: pd.Timestamp(asof)].tail(252)
    ix = ret[index_ticker]
    extra = pd.DataFrame(index=snap.index)
    extra["Vol_1Y"] = ret[snap.index].std() * np.sqrt(252)
    extra["Beta_1Y"] = [ret[t].cov(ix) / ix.var() if ret[t].count() > 100 else np.nan for t in snap.index]
    extra["Corr_1Y"] = [ret[t].corr(ix) for t in snap.index]
    wealth3y = sub.prices.loc[: pd.Timestamp(asof)].tail(756)
    extra["MaxDD_3Y"] = [S.max_drawdown(wealth3y[t].dropna() / wealth3y[t].dropna().iloc[0]) if wealth3y[t].count() > 100 else np.nan for t in snap.index]
    out = snap.join(extra).join(members.drop_duplicates("Ticker").set_index("Ticker")[["Name", "Sector"]])
    out.index.name = "Ticker"
    return out.sort_values("MomentumScore", ascending=False)


def breadth_series(prices: pd.DataFrame, ma_list=(50, 200)) -> pd.DataFrame:
    """Share of group members that are above their moving averages / making 52-week highs and lows, over time."""
    out = {}
    valid = prices.notna()
    for m in ma_list:
        ma = prices.rolling(m, min_periods=m).mean()
        ok = ma.notna() & valid
        out[f"% above {m}-day average"] = ((prices > ma) & ok).sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)
    hi = prices.rolling(252, min_periods=126).max()
    lo = prices.rolling(252, min_periods=126).min()
    n = valid.sum(axis=1).replace(0, np.nan)
    out["% at 52-week high"] = ((prices >= hi) & hi.notna()).sum(axis=1) / n
    out["% at 52-week low"] = ((prices <= lo) & lo.notna()).sum(axis=1) / n
    r126 = prices / prices.shift(126) - 1
    out["% with positive 6M return"] = ((r126 > 0) & r126.notna()).sum(axis=1) / r126.notna().sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(out) * 100


def equal_weight_composite(prices: pd.DataFrame) -> pd.Series:
    """Daily-rebalanced equal-weight wealth index of whichever members have a price that day."""
    r = prices.pct_change(fill_method=None)
    return (1 + r.mean(axis=1).fillna(0)).cumprod()


def composite_vs_index(prices: pd.DataFrame, index_series: pd.Series) -> pd.DataFrame:
    ew = equal_weight_composite(prices)
    ix = index_series.reindex(ew.index).ffill()
    df = pd.DataFrame({"Equal-weight members": ew, "Index": ix / ix.dropna().iloc[0]}).dropna()
    df = df / df.iloc[0]
    df["Equal ÷ Index"] = df.iloc[:, 0] / df.iloc[:, 1]
    return df


def avg_pairwise_corr(prices: pd.DataFrame, window: int = 63, step: int = 21) -> pd.Series:
    r = prices.pct_change(fill_method=None)
    pts = {}
    for i in range(window, len(r), step):
        w = r.iloc[i - window + 1 : i + 1]
        w = w.loc[:, w.count() >= int(window * 0.8)]
        if w.shape[1] < 5:
            continue
        c = w.corr().to_numpy()
        n = c.shape[0]
        pts[r.index[i]] = (np.nansum(c) - n) / (n * (n - 1))
    return pd.Series(pts, name="Average pairwise correlation")


def dispersion_series(prices: pd.DataFrame, window: int = 63) -> pd.Series:
    r = prices / prices.shift(window) - 1
    return r.std(axis=1).rename("Cross-sectional dispersion of 3M returns")


def contribution(prices: pd.DataFrame, start, end) -> pd.DataFrame:
    """Equal-weight buy-and-hold contribution of each member to the group return between two dates."""
    p = prices.loc[pd.Timestamp(start): pd.Timestamp(end)]
    p = p.loc[:, p.iloc[0].notna()]
    rel = p.ffill().iloc[-1] / p.iloc[0] - 1
    w = 1.0 / len(rel)
    return pd.DataFrame({"Return": rel, "Weight": w, "Contribution": w * rel}).sort_values("Contribution", ascending=False)


def sector_summary(snap: pd.DataFrame) -> pd.DataFrame:
    g = snap.groupby("Sector")
    return pd.DataFrame({
        "Members": g.size(),
        "Avg 1M": g["Ret_1M"].mean(), "Avg 3M": g["Ret_3M"].mean(), "Avg 12M": g["Ret_12M"].mean(),
        "% above 200d trend": g["TrendGate"].mean() * 100,
        "Avg momentum percentile": g["MomentumPercentile"].mean(),
    }).sort_values("Avg 3M", ascending=False)


# ----------------------------------------------------------------------------
# "Why did it behave like that?"
# ----------------------------------------------------------------------------
def market_regression(r: pd.Series, mkt: pd.Series, window: int | None = None) -> dict:
    d = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna()
    if window:
        d = d.tail(window)
    if len(d) < 60:
        return {}
    res = S.ols_hac(d["r"].to_numpy(), d["m"].to_numpy(), lags=5, names=["Market"])
    resid = d["r"] - (res.coef[0] + res.coef[1] * d["m"])
    up, dn = d[d["m"] > 0], d[d["m"] < 0]
    beta = lambda x: float(np.cov(x["r"], x["m"])[0, 1] / x["m"].var()) if len(x) > 20 else np.nan
    return {
        "beta": res.coef[1], "beta_t": res.t[1], "alpha_annual": float(res.coef[0] * 252), "alpha_t": res.t[0],
        "r2": res.r2, "idio_vol": float(resid.std() * np.sqrt(252)), "total_vol": float(d["r"].std() * np.sqrt(252)),
        "up_beta": beta(up), "down_beta": beta(dn), "n": len(d),
    }


def factor_regression(r: pd.Series, factors: pd.DataFrame, rf: pd.Series | None = None, window: int | None = 756) -> dict:
    """Regress daily (excess) returns on market/size/value/profitability/investment/momentum factors.

    Attribution of the window's return: Σ β_k·f_k over the window (approximate, additive).  Valid for USD
    returns of US-listed stocks; for other markets the US factors are only a rough lens.
    """
    fcols = [c for c in factors.columns if c != "RF"]
    rf_s = factors["RF"] if "RF" in factors else (rf if rf is not None else 0.0)
    d = pd.concat([r.rename("r"), factors[fcols]], axis=1).dropna()
    if window:
        d = d.tail(window)
    if len(d) < 120:
        return {}
    y = d["r"] - (rf_s.reindex(d.index).fillna(0.0) if hasattr(rf_s, "reindex") else rf_s)
    res = S.ols_hac(y.to_numpy(), d[fcols].to_numpy(), lags=5, names=fcols)
    tab = res.table().drop(index="const")
    contrib = {c: float(res.coef[1 + i] * d[c].sum()) for i, c in enumerate(fcols)}
    total = float(y.sum())
    alpha_c = float(res.coef[0] * len(d))
    contrib["Alpha (unexplained)"] = alpha_c
    resid_c = total - sum(contrib.values())
    return {
        "betas": tab, "alpha_annual": float(res.coef[0] * 252), "alpha_t": float(res.t[0]), "r2": res.r2,
        "contrib": pd.Series(contrib), "total_excess": total, "residual": resid_c, "n": len(d),
        "start": d.index[0], "end": d.index[-1],
    }


def rolling_beta(r: pd.Series, m: pd.Series, window: int = 252) -> pd.Series:
    d = pd.concat([r.rename("r"), m.rename("m")], axis=1).dropna()
    return (d["r"].rolling(window).cov(d["m"]) / d["m"].rolling(window).var()).dropna().rename("Rolling beta")
