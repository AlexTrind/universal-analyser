"""Statistical toolkit (pure numpy/scipy/pandas; no hidden dependencies).

Everything here is deliberately small and independently testable against textbook
formulas. Nothing in this module touches the network or the UI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats as sps

EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------
# Basic return / risk metrics
# ----------------------------------------------------------------------------
def max_drawdown(wealth: pd.Series | np.ndarray, include_start: bool = True) -> float:
    """Largest peak-to-trough fall of a wealth index (negative number)."""
    w = np.asarray(pd.Series(wealth).dropna(), dtype=float)
    if w.size == 0:
        return float("nan")
    if include_start:
        w = np.concatenate([[1.0], w]) if w[0] != 1.0 else w
    peak = np.maximum.accumulate(w)
    return float((w / peak - 1.0).min())


def drawdown_series(wealth: pd.Series) -> pd.Series:
    w = wealth.dropna()
    return w / w.cummax() - 1.0


def drawdown_table(wealth: pd.Series, top: int = 5) -> pd.DataFrame:
    """Worst drawdown episodes: peak date, trough date, recovery date, depth, durations."""
    w = wealth.dropna()
    if w.empty:
        return pd.DataFrame()
    peak = w.cummax()
    dd = w / peak - 1.0
    episodes = []
    in_dd = False
    for i, (d, v) in enumerate(dd.items()):
        if v < 0 and not in_dd:
            in_dd = True
            start_i = i - 1 if i > 0 else 0
        if in_dd and v == 0:
            episodes.append((start_i, i))
            in_dd = False
    if in_dd:
        episodes.append((start_i, None))
    rows = []
    for s, e in episodes:
        seg = dd.iloc[s : (e if e is not None else len(dd))]
        if seg.empty:
            continue
        trough_date = seg.idxmin()
        rows.append(
            {
                "Peak date": dd.index[s],
                "Trough date": trough_date,
                "Recovery date": dd.index[e] if e is not None else pd.NaT,
                "Depth": float(seg.min()),
                "Days peak→trough": (trough_date - dd.index[s]).days,
                "Days to recover": (dd.index[e] - trough_date).days if e is not None else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Depth").head(top).reset_index(drop=True)


def periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    years = (index[-1] - index[0]).days / 365.25
    return (len(index) - 1) / years if years > 0 else 252.0


def performance_metrics(
    ret: pd.Series,
    bench: pd.Series | None = None,
    rf_annual: float = 0.0,
    ppy: float | None = None,
) -> dict:
    """Standard performance statistics from a periodic simple-return series."""
    r = ret.dropna().astype(float)
    n = len(r)
    if n < 3:
        return {}
    ppy = ppy or periods_per_year(r.index)
    wealth = (1 + r).cumprod()
    years = n / ppy
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1) if years > 0 and wealth.iloc[-1] > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(ppy))
    rf_p = (1 + rf_annual) ** (1 / ppy) - 1
    ex = r - rf_p
    sharpe = float(ex.mean() / r.std(ddof=1) * math.sqrt(ppy)) if r.std(ddof=1) > 0 else float("nan")
    dn = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    sortino = float(ex.mean() / dn * math.sqrt(ppy)) if dn > 0 else float("nan")
    mdd = max_drawdown(wealth)
    out = {
        "Total return": float(wealth.iloc[-1] - 1),
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown": mdd,
        "Calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "Best period": float(r.max()),
        "Worst period": float(r.min()),
        "Hit rate": float((r > 0).mean()),
        "Skew": float(sps.skew(r, bias=False)) if n > 3 else float("nan"),
        "Excess kurtosis": float(sps.kurtosis(r, bias=False)) if n > 4 else float("nan"),
        "Observations": n,
    }
    if bench is not None:
        j = pd.concat([r.rename("a"), bench.rename("b")], axis=1).dropna()
        if len(j) >= 3:
            act = j["a"] - j["b"]
            te = float(act.std(ddof=1) * math.sqrt(ppy))
            bw = (1 + j["b"]).cumprod()
            bcagr = float(bw.iloc[-1] ** (ppy / len(j)) - 1)
            acagr = float((1 + j["a"]).cumprod().iloc[-1] ** (ppy / len(j)) - 1)
            up = j["b"] > 0
            dn_m = j["b"] < 0
            upc = (
                (1 + j.loc[up, "a"]).prod() ** (1 / up.sum()) - 1
            ) / ((1 + j.loc[up, "b"]).prod() ** (1 / up.sum()) - 1) if up.sum() else float("nan")
            dnc = (
                (1 + j.loc[dn_m, "a"]).prod() ** (1 / dn_m.sum()) - 1
            ) / ((1 + j.loc[dn_m, "b"]).prod() ** (1 / dn_m.sum()) - 1) if dn_m.sum() else float("nan")
            out.update(
                {
                    "Benchmark CAGR": bcagr,
                    "CAGR minus benchmark": acagr - bcagr,
                    "Tracking error": te,
                    "Information ratio": float(act.mean() * ppy / te) if te > 0 else float("nan"),
                    "Upside capture": float(upc),
                    "Downside capture": float(dnc),
                    "Beta": float(np.cov(j["a"], j["b"], ddof=1)[0, 1] / j["b"].var(ddof=1)) if j["b"].var() > 0 else float("nan"),
                }
            )
    return out


# ----------------------------------------------------------------------------
# Regression with Newey–West (HAC) standard errors
# ----------------------------------------------------------------------------
@dataclass
class OLSResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    r2: float
    n: int
    lags: int

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.coef, "se": self.se, "t": self.t, "p": self.p}, index=self.names)


def ols_hac(y, X, lags: int | None = None, add_const: bool = True, names: list[str] | None = None) -> OLSResult:
    """OLS with Newey–West (Bartlett kernel) heteroskedasticity-and-autocorrelation-consistent SEs.

    lags=None -> floor(4*(n/100)^(2/9)) (Newey–West 1994 rule of thumb).
    Use lags >= horizon-1 when the dependent variable is built from overlapping windows.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if names is None:
        names = [f"x{i+1}" for i in range(X.shape[1])]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(names)
    n, k = X.shape
    if n <= k:
        raise ValueError("Not enough observations for the number of regressors")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    if lags is None:
        lags = int(math.floor(4 * (n / 100.0) ** (2 / 9)))
    lags = int(max(0, min(lags, n - 2)))
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * (1 - sps.norm.cdf(np.abs(t)))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((u**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return OLSResult(names, beta, se, t, p, r2, n, lags)


# ----------------------------------------------------------------------------
# Resampling / significance helpers
# ----------------------------------------------------------------------------
def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap index matrix (n_boot × n)."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    jumps = rng.random((n_boot, n)) < p
    new = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jumps[:, t], new[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_ci(x, stat=np.mean, n_boot: int = 2000, mean_block: float = 6.0, alpha: float = 0.05, seed: int = 0) -> dict:
    """Stationary-bootstrap confidence interval for a statistic of a (possibly autocorrelated) series."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 8:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_le_0": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, rng)
    draws = np.apply_along_axis(stat, 1, x[idx])
    return {
        "estimate": float(stat(x)),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        # share of bootstrap worlds in which the statistic is <= 0 (one-sided)
        "p_le_0": float((draws <= 0).mean()),
        "n": n,
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than Wald for small n)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z**2 / n
    centre = (ph + z**2 / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def circular_shift_test(x: np.ndarray, label: np.ndarray, target, n_perm: int = 2000, seed: int = 0) -> dict:
    """Does the mean of x differ when `label == target`? p-value by circularly shifting labels.

    Circular shifts keep the persistence (autocorrelation) of the regime labels, which a plain
    shuffle would destroy – giving falsely small p-values for slowly-changing regimes.
    """
    x = np.asarray(x, dtype=float)
    lab = np.asarray(label)
    m = ~np.isnan(x)
    x, lab = x[m], lab[m]
    n = len(x)
    sel = lab == target
    if sel.sum() < 3 or (~sel).sum() < 3:
        return {"diff": float("nan"), "p": float("nan"), "n_in": int(sel.sum()), "n_out": int((~sel).sum())}
    obs = x[sel].mean() - x[~sel].mean()
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        k = int(rng.integers(1, n))
        s2 = np.roll(sel, k)
        d = x[s2].mean() - x[~s2].mean() if s2.sum() >= 1 and (~s2).sum() >= 1 else 0.0
        cnt += abs(d) >= abs(obs) - 1e-15
    return {"diff": float(obs), "p": float((cnt + 1) / (n_perm + 1)), "n_in": int(sel.sum()), "n_out": int((~sel).sum())}


# ----------------------------------------------------------------------------
# Sharpe-ratio inference (Bailey & López de Prado)
# ----------------------------------------------------------------------------
def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """PSR: probability the true (per-period) Sharpe exceeds `sr_benchmark`.  `kurt` is NON-excess."""
    if n < 3:
        return float("nan")
    den = 1 - skew * sr + (kurt - 1) / 4.0 * sr**2
    if den <= 0:
        return float("nan")
    return float(sps.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(den)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period Sharpe among `n_trials` unskilled strategies (null SR=0)."""
    if n_trials <= 1:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(var_sr) * ((1 - g) * sps.norm.ppf(1 - 1.0 / n_trials) + g * sps.norm.ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(returns: pd.Series, n_trials: int, var_sr_trials: float) -> dict:
    """Deflated Sharpe ratio: PSR against the best-of-N-trials luck benchmark."""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return {"sr": float("nan"), "sr0": float("nan"), "dsr": float("nan")}
    sr = float(r.mean() / r.std(ddof=1))
    sk = float(sps.skew(r, bias=False))
    ku = float(sps.kurtosis(r, fisher=False, bias=False))
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    return {"sr": sr, "sr0": sr0, "dsr": probabilistic_sharpe(sr, n, sk, ku, sr0)}


# ----------------------------------------------------------------------------
# Regularised multinomial logistic regression (L2) – numpy/scipy only
# ----------------------------------------------------------------------------
class MultinomialLogit:
    """Softmax regression minimising  sum_i -log p(y_i|x_i) + (1/(2C))·||W||²  (intercepts unpenalised).

    Same objective as scikit-learn's LogisticRegression(multinomial, penalty='l2', C=C), so the two
    can be cross-checked (see tests).  Features are standardised internally using training data only.
    """

    def __init__(self, C: float = 0.3, max_iter: int = 500):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y, n_classes: int):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.K = n_classes
        self.mu_ = X.mean(0)
        sd = X.std(0)
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        Z = (X - self.mu_) / self.sd_
        n, d = Z.shape
        Y = np.zeros((n, self.K))
        Y[np.arange(n), y] = 1.0
        lam = 1.0 / self.C

        def unpack(theta):
            return theta[: self.K * d].reshape(self.K, d), theta[self.K * d :]

        def fun(theta):
            W, b = unpack(theta)
            S = Z @ W.T + b
            lse = special.logsumexp(S, axis=1)
            nll = float((lse - (S * Y).sum(1)).sum())
            P = np.exp(S - lse[:, None])
            G = P - Y
            gW = G.T @ Z + lam * W
            gb = G.sum(0)
            return nll + 0.5 * lam * float((W**2).sum()), np.concatenate([gW.ravel(), gb])

        theta0 = np.zeros(self.K * d + self.K)
        res = optimize.minimize(fun, theta0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter, "gtol": 1e-9, "ftol": 1e-13})
        self.W_, self.b_ = unpack(res.x)
        self.converged_ = bool(res.success)
        self.n_train_ = n
        return self

    def predict_proba(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        S = Z @ self.W_.T + self.b_
        return np.exp(S - special.logsumexp(S, axis=1)[:, None])


# ----------------------------------------------------------------------------
# Probability-forecast scoring
# ----------------------------------------------------------------------------
def brier_score(P: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score (lower is better; range 0..2)."""
    P = np.asarray(P, dtype=float)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return float(((P - Y) ** 2).sum(1).mean())


def log_loss(P: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    P = np.clip(np.asarray(P, dtype=float), eps, 1.0)
    return float(-np.log(P[np.arange(len(y)), np.asarray(y, dtype=int)]).mean())


def calibration_table(p: np.ndarray, hit: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Reliability table for one outcome: bin the forecast probability, compare with realised frequency."""
    p = np.asarray(p, dtype=float)
    hit = np.asarray(hit, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    edges = np.unique(edges)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"Forecast (avg)": p[m].mean(), "Observed frequency": hit[m].mean(), "Count": int(m.sum())})
    return pd.DataFrame(rows)


def benjamini_hochberg(p) -> np.ndarray:
    """False-discovery-rate adjusted p-values (q-values). A q of 0.10 means: among results flagged at this level,
    about 10% are expected to be flukes. NaNs are ignored and returned as NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q
