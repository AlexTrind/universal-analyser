"""Probability of Down / Sideways / Up over the next h months – with an honest scorecard.

Three forecasts are always produced side by side:
  1. BASE RATE            – how often each outcome happened historically (the benchmark to beat),
  2. REGIME-CONDITIONAL   – the same frequencies, but only in months resembling today's economic regime,
  3. MODEL                – regularised multinomial logistic regression on price + macro indicators.

The model's walk-forward (out-of-sample) record is scored against the base rate.  If it does not clearly
beat the base rate, the app says so and tells the user to rely on the base rate instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import stats as S
from .relationships import forward_returns

CLASSES = ["Down", "Sideways", "Up"]


def make_labels(fwd: pd.Series, me_prices: pd.Series, h: int, mode: str = "vol", band: float = 0.5, trailing: int = 36) -> pd.Series:
    """0=Down, 1=Sideways, 2=Up.  'vol' band = ±band × trailing monthly σ × √h (known at the decision date); 'fixed' = ±band."""
    if mode == "fixed":
        thr = pd.Series(band, index=fwd.index)
    else:
        sig = me_prices.pct_change().rolling(trailing, min_periods=24).std()
        thr = band * sig * np.sqrt(h)
    lab = pd.Series(np.where(fwd > thr, 2, np.where(fwd < -thr, 0, 1)), index=fwd.index, dtype=float)
    lab[fwd.isna() | thr.isna()] = np.nan
    return lab


def price_features(me_prices: pd.Series) -> pd.DataFrame:
    p = me_prices
    r = p.pct_change()
    return pd.DataFrame({
        "Momentum 12-1": p.shift(1) / p.shift(12) - 1,
        "Momentum 3M": p / p.shift(3) - 1,
        "Trend vs 10M avg": p / p.rolling(10, min_periods=10).mean() - 1,
        "Vol ratio 3M/36M": r.rolling(3).std() / r.rolling(36, min_periods=24).std(),
        "Drawdown from 12M high": p / p.rolling(12, min_periods=12).max() - 1,
    })


@dataclass
class ForecastResult:
    ok: bool
    reason: str = ""
    horizon: int = 3
    base_rate: pd.Series | None = None
    base_ci: pd.DataFrame | None = None
    regime_rate: pd.DataFrame | None = None
    model_probs: pd.Series | None = None
    oos: pd.DataFrame | None = None
    scores: dict = field(default_factory=dict)
    calibration: pd.DataFrame | None = None
    coef: pd.DataFrame | None = None
    verdict: str = ""
    n_train: int = 0
    features_used: list[str] = field(default_factory=list)
    label_counts: pd.Series | None = None


def walk_forward(X: pd.DataFrame, y: pd.Series, h: int, min_train: int = 72, C: float = 0.3, refit_every: int = 1) -> pd.DataFrame:
    """Out-of-sample class probabilities.  At month t the model is trained ONLY on rows whose outcome was already
    observable at t (row i is usable iff i + h ≤ t) – no leakage of overlapping future returns."""
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    Xv, yv = d.drop(columns="y").to_numpy(), d["y"].to_numpy().astype(int)
    mo = (d.index.year * 12 + d.index.month).to_numpy()          # calendar-month counter (robust to gaps)
    out, model, last_fit = [], None, -10**9
    for t in range(min_train + h, len(d)):
        tr = np.where(mo <= mo[t] - h)[0]                          # outcome of row i is known at t iff month_i + h <= month_t
        if len(tr) < 30:
            continue
        if t - last_fit >= refit_every or model is None:
            model = S.MultinomialLogit(C=C).fit(Xv[tr], yv[tr], 3)
            last_fit = t
        pm = model.predict_proba(Xv[[t]])[0]
        cnt = np.bincount(yv[tr], minlength=3) + 1.0           # Laplace-smoothed base rate
        out.append({"date": d.index[t], **{f"p_{c}": pm[k] for k, c in enumerate(CLASSES)},
                    **{f"b_{c}": (cnt / cnt.sum())[k] for k, c in enumerate(CLASSES)}, "y": yv[t]})
    return pd.DataFrame(out).set_index("date") if out else pd.DataFrame()


def score_oos(oos: pd.DataFrame, h: int, seed: int = 0) -> dict:
    P = oos[[f"p_{c}" for c in CLASSES]].to_numpy()
    B = oos[[f"b_{c}" for c in CLASSES]].to_numpy()
    y = oos["y"].to_numpy().astype(int)
    bm, bb = S.brier_score(P, y), S.brier_score(B, y)
    per_m = ((P - np.eye(3)[y]) ** 2).sum(1)
    per_b = ((B - np.eye(3)[y]) ** 2).sum(1)
    ci = S.bootstrap_ci(per_b - per_m, np.mean, n_boot=2000, mean_block=max(h + 2, 4), seed=seed)
    return {
        "months": len(y), "brier_model": bm, "brier_base": bb, "brier_skill": 1 - bm / bb,
        "logloss_model": S.log_loss(P, y), "logloss_base": S.log_loss(B, y),
        "acc_model": float((P.argmax(1) == y).mean()), "acc_base": float((B.argmax(1) == y).mean()),
        "gain_ci": (ci["lo"], ci["hi"]), "p_no_skill": ci["p_le_0"],
        "effective_obs": len(y) / h,
    }


def verdict_text(sc: dict) -> str:
    lo, hi = sc["gain_ci"]
    if lo > 0 and sc["brier_skill"] > 0:
        return (f"The model beat the naive base rate out-of-sample (Brier skill {sc['brier_skill']:+.1%}; 95% interval for the gain "
                f"excludes zero). Even so, treat the probabilities as a modest tilt, not a certainty.")
    if hi < 0:
        return "The model did WORSE than the naive base rate out-of-sample. Use the base rate and ignore the model."
    return ("No reliable edge: the model's out-of-sample record is statistically indistinguishable from the naive base rate "
            f"(Brier skill {sc['brier_skill']:+.1%}, 95% interval for the gain includes zero). Rely on the base rate / regime history.")


def run_forecast(
    me_prices: pd.Series,
    macro_X: pd.DataFrame | None,
    regime_labels: pd.Series | None,
    h: int = 3,
    mode: str = "vol",
    band: float = 0.5,
    use_price: bool = True,
    use_macro: bool = True,
    min_train: int = 72,
    C: float = 0.3,
) -> ForecastResult:
    me = me_prices.dropna()
    if len(me) < 48 + h + 24:
        return ForecastResult(False, f"Only {len(me)} months of history – at least {48 + h + 24} are needed (signals need a warm-up, then training, then a test period). Pick an asset with a longer history.", h)
    fwd = forward_returns(me, h)
    y = make_labels(fwd, me, h, mode, band)
    parts = []
    if use_price:
        parts.append(price_features(me))
    if use_macro and macro_X is not None and not macro_X.empty:
        parts.append(macro_X.reindex(me.index))
    if not parts:
        return ForecastResult(False, "No features selected", h)
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.notna().mean() > 0.6]
    Xc = X.dropna()
    n_ov = len(Xc.index.intersection(y.dropna().index))
    if n_ov < min_train + h + 24:
        room = n_ov - h - 24
        if room >= 24:
            tip = f"Lower ‘Min training months’ to {room} or less."
        else:
            tip = "Turn off ‘Use economic signals’ (their history is shorter) or pick an asset with a longer history."
        return ForecastResult(False, f"Only {n_ov} months have both usable signals and a known outcome (signals need a warm-up period); {min_train + h + 24} are needed. {tip}", h)
    lab = y.dropna()
    counts = lab.value_counts().reindex([0, 1, 2]).fillna(0)
    counts.index = CLASSES
    base = (counts + 1) / (counts + 1).sum()
    ci = pd.DataFrame([S.wilson_interval(int(counts[c]), int(counts.sum())) for c in CLASSES], index=CLASSES, columns=["lo", "hi"])
    # regime-conditional frequencies
    reg_tab = None
    if regime_labels is not None and regime_labels.notna().any():
        cur = regime_labels.dropna().iloc[-1]
        same = regime_labels[regime_labels == cur].index.intersection(lab.index)
        n = len(same)
        rc = lab.loc[same].value_counts().reindex([0, 1, 2]).fillna(0)
        rows = []
        for k, c in enumerate(CLASSES):
            lo, hi = S.wilson_interval(int(round(rc.iloc[k] / h)), max(int(round(n / h)), 1))
            rows.append({"Outcome": c, "Probability": rc.iloc[k] / n if n else np.nan, "Low": lo, "High": hi})
        reg_tab = pd.DataFrame(rows).set_index("Outcome")
        reg_tab.attrs.update(regime=cur, months=n, independent=round(n / h, 1))
    oos = walk_forward(X, y, h, min_train, C)
    if oos.empty:
        return ForecastResult(False, "Walk-forward produced no out-of-sample months", h)
    sc = score_oos(oos, h)
    # final model on everything with a known outcome, applied to the latest feature row
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    final = S.MultinomialLogit(C=C).fit(d.drop(columns="y").to_numpy(), d["y"].to_numpy().astype(int), 3)
    x_now = X.dropna().iloc[[-1]]
    pm = pd.Series(final.predict_proba(x_now.to_numpy())[0], index=CLASSES)
    coef = pd.DataFrame(final.W_.T, index=X.columns, columns=CLASSES)
    cal = S.calibration_table(oos["p_Up"].to_numpy(), (oos["y"] == 2).to_numpy(), bins=5)
    return ForecastResult(True, "", h, base, ci, reg_tab, pm, oos, sc, cal, coef, verdict_text(sc), len(d),
                          list(X.columns), counts)
