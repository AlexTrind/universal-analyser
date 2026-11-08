"""Economic regime classification from open macro data (transparent rules – no black box)."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROWTH_SIGNS = {"INDPRO": 1, "PAYEMS": 1, "RSAFS": 1, "ICSA": -1, "UNRATE": -1, "CFNAI": 1, "USALOLITOAASTSAM": 1, "PERMIT": 1, "GDPC1": 1}
INFLATION_SIGNS = {"CPIAUCSL": 1, "CPILFESL": 1, "PCEPILFE": 1, "T10YIE": 1}
STRESS_SIGNS = {"NFCI": 1, "STLFSI4": 1, "BAA10Y": 1, "VIXCLS": 1}
LIQUIDITY_SIGNS = {"M2SL": 1, "WALCL": 1}

QUADRANTS = {
    (True, False): "Goldilocks (growth up, inflation down)",
    (True, True): "Overheating (growth up, inflation up)",
    (False, True): "Stagflation (growth down, inflation up)",
    (False, False): "Slowdown (growth down, inflation down)",
}
QUADRANT_ORDER = list(QUADRANTS.values())


def composite(Z: pd.DataFrame, signs: dict[str, int], min_components: int = 2) -> pd.Series:
    cols = [c for c in signs if c in Z.columns]
    if not cols:
        return pd.Series(np.nan, index=Z.index)
    parts = pd.concat([Z[c] * signs[c] for c in cols], axis=1)
    out = parts.mean(axis=1)
    out[parts.notna().sum(axis=1) < min_components] = np.nan
    return out


def composites(Z: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Growth": composite(Z, GROWTH_SIGNS),
        "Inflation": composite(Z, INFLATION_SIGNS),
        "Financial stress": composite(Z, STRESS_SIGNS),
        "Liquidity": composite(Z, LIQUIDITY_SIGNS, 1),
    })


def classify(comp: pd.DataFrame, feats: pd.DataFrame, scheme: str = "level", stress_quantile: float = 0.80) -> pd.DataFrame:
    """Return labelled regimes per month-end.

    scheme='level'     : growth/inflation above or below their own long-run average (z>0).
    scheme='direction' : growth/inflation rising or falling over the last 3 months.
    Everything uses only information known at each month-end.
    """
    g, i = comp["Growth"], comp["Inflation"]
    if scheme == "direction":
        gu, iu = g.diff(3) > 0, i.diff(3) > 0
        valid = g.diff(3).notna() & i.diff(3).notna()
    else:
        gu, iu = g > 0, i > 0
        valid = g.notna() & i.notna()
    quad = pd.Series([QUADRANTS[(bool(a), bool(b))] for a, b in zip(gu, iu)], index=comp.index).where(valid)
    st = comp["Financial stress"]
    thr = st.expanding(60).quantile(stress_quantile)
    stress = pd.Series(np.where(st > thr, "Stress", "Calm"), index=comp.index).where(st.notna() & thr.notna())
    curve_col = "T10Y3M" if "T10Y3M" in feats.columns else ("T10Y2Y" if "T10Y2Y" in feats.columns else None)
    if curve_col:
        c = feats[curve_col]
        inv_12 = (c < 0).rolling(12, min_periods=1).max().astype(bool)
        curve = pd.Series(np.where(c < 0, "Inverted now", np.where(inv_12, "Was inverted in last 12m", "Normal")), index=comp.index).where(c.notna())
    else:
        curve = pd.Series(np.nan, index=comp.index, dtype=object)
    if "FEDFUNDS" in feats.columns:
        d6 = feats["FEDFUNDS"].diff(6)
        policy = pd.Series(np.where(d6 > 0.5, "Tightening", np.where(d6 < -0.5, "Easing", "On hold")), index=comp.index).where(d6.notna())
    else:
        policy = pd.Series(np.nan, index=comp.index, dtype=object)
    return pd.DataFrame({"Growth/Inflation": quad, "Market stress": stress, "Yield curve": curve, "Policy stance": policy})


def current_summary(labels: pd.DataFrame, raw: dict[str, pd.Series], today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Latest label per dimension, how long it has lasted, and how fresh the underlying data are."""
    rows = []
    for col in labels.columns:
        s = labels[col].dropna()
        if s.empty:
            rows.append({"Dimension": col, "Current": "n/a", "Months in this state": np.nan, "Since": None})
            continue
        cur = s.iloc[-1]
        run = (s != cur)[::-1].cumsum()
        n = int((run == 0).sum())
        rows.append({"Dimension": col, "Current": cur, "Months in this state": n, "Since": s.index[-n]})
    return pd.DataFrame(rows)


def data_freshness(raw: dict[str, pd.Series], today: pd.Timestamp | None = None) -> pd.DataFrame:
    today = today or pd.Timestamp.today().normalize()
    from .data_macro import BY_ID

    rows = []
    for sid, s in raw.items():
        last = s.index.max()
        rows.append({"Series": sid, "Name": BY_ID[sid].name if sid in BY_ID else sid, "Last observation": last.date(),
                     "Days old": (today - last).days, "Typical delay (days)": BY_ID[sid].lag_days if sid in BY_ID else np.nan})
    return pd.DataFrame(rows).sort_values("Days old", ascending=False)
