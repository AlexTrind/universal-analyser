"""Portfolio construction: selection (with hysteresis), correlation clusters, risk-based sizing, caps."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CASH = "CASH"
SIZING_METHODS = ["Realized volatility", "Downside volatility", "ATR% proxy", "Equal weight"]


@dataclass
class PortfolioParams:
    active_fraction: float = 0.20     # share of the whole portfolio given to the momentum sleeve
    n_active: int = 6
    entry_pct: float = 80.0           # new holdings must rank at/above this momentum percentile
    exit_pct: float = 60.0            # existing holdings may stay until they fall below this
    sizing: str = "Realized volatility"
    corr_days: int = 126
    corr_threshold: float = 0.80
    max_cluster: float = 0.40         # max share of the ACTIVE sleeve in one correlation cluster
    max_single: float = 0.30          # max share of the ACTIVE sleeve in one ETF
    residual_to: str = "Benchmark"    # where weight goes when caps/eligibility leave room: 'Benchmark' or 'Cash'


def cap_weights(raw: np.ndarray, clusters: np.ndarray, single_cap: float, cluster_cap: float, iters: int = 100) -> np.ndarray:
    """Normalise to 1, then enforce single and cluster caps, redistributing excess only where there is room.

    Guarantees (property-tested): every cap is respected, weights are ≥ 0 and sum ≤ 1.  If the caps make
    100% unreachable (e.g. 2 names × 30% cap) the shortfall is left unallocated for the caller to park.
    """
    raw = np.asarray(raw, dtype=float)
    w = raw / raw.sum()
    cl = np.asarray(clusters)
    for _ in range(iters):
        w = np.minimum(w, single_cap)
        for c in np.unique(cl):
            m = cl == c
            tot = w[m].sum()
            if tot > cluster_cap:
                w[m] *= cluster_cap / tot
        resid = 1.0 - w.sum()
        if resid < 1e-12:
            break
        room = np.clip(single_cap - w, 0, None)
        for c in np.unique(cl):
            m = cl == c
            cr = max(cluster_cap - w[m].sum(), 0.0)
            rs = room[m].sum()
            if rs > cr:
                room[m] *= cr / rs if rs > 0 else 0.0
        if room.sum() < 1e-12:
            break
        w = w + room * min(1.0, resid / room.sum())
    return w


def greedy_clusters(order: list[str], corr: pd.DataFrame, threshold: float) -> dict[str, int]:
    """Walk down the ranked list; join the first cluster containing any member correlated ≥ threshold."""
    members: list[list[str]] = []
    cid: dict[str, int] = {}
    for t in order:
        placed = False
        for k, mem in enumerate(members):
            vals = [corr.at[t, m] for m in mem if t in corr.index and m in corr.columns and pd.notna(corr.at[t, m])]
            if vals and max(vals) >= threshold:
                mem.append(t)
                cid[t] = k + 1
                placed = True
                break
        if not placed:
            members.append([t])
            cid[t] = len(members)
    return cid


def risk_measure(row: pd.Series, method: str) -> float:
    if method == "Realized volatility":
        return row["Vol_63D"]
    if method == "Downside volatility":
        return row["DownsideVol_63D"]
    if method == "ATR% proxy":
        return row["ATRpct_20D_proxy"] * np.sqrt(252) if pd.notna(row["ATRpct_20D_proxy"]) else np.nan
    return 1.0


def build_portfolio(
    snap: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark: str,
    p: PortfolioParams,
    held: list[str] | None = None,
    exclude: set[str] | None = None,
    names: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (portfolio, corr).  `snap` is indexed by ticker.  `returns` = daily simple returns up to the decision date.

    Selection rules
      1. NEW candidates: EligibleSignal and MomentumPercentile ≥ entry_pct.
      2. HELD names (hysteresis): kept while above their trend line and MomentumPercentile ≥ exit_pct.
      3. Kept names fill slots first; remaining slots go to the best name of each *not yet represented*
         correlation cluster, then to the best remaining names.  Unused slots are NOT filled with failing ETFs.
    """
    names = names or {}
    exclude = exclude or set()
    held = [t for t in (held or []) if t in snap.index and t != benchmark]
    residual_ticker = CASH if p.residual_to == "Cash" else benchmark

    def core_row(weight: float) -> dict:
        return {"Yahoo_Ticker": residual_ticker, "Fund_Name": "Cash (0% return)" if residual_ticker == CASH else names.get(benchmark, "Global benchmark core"),
                "Cluster": 0, "MomentumPercentile": np.nan, "MomentumScore": np.nan, "RiskMeasure": np.nan,
                "ActiveWeight": np.nan, "PortfolioWeight": weight, "Role": "Core"}

    s = snap.copy()
    fresh = s[(s["EligibleSignal"]) & (s["MomentumPercentile"] >= p.entry_pct)].index.difference(list(exclude))
    keep = [t for t in held if bool(s.at[t, "TrendGate"]) and s.at[t, "MomentumPercentile"] >= p.exit_pct and t not in exclude]
    if (len(fresh) == 0 and len(keep) == 0) or p.active_fraction <= 0:
        return pd.DataFrame([core_row(1.0)]), pd.DataFrame()

    pool_n = max(p.n_active * 4, p.n_active)
    pool = list(dict.fromkeys(list(s.loc[fresh].sort_values("MomentumScore", ascending=False).index[:pool_n]) + keep))
    pool = sorted(pool, key=lambda t: -s.at[t, "MomentumScore"] if pd.notna(s.at[t, "MomentumScore"]) else 1e9)
    rets = returns[[t for t in pool if t in returns.columns]].tail(p.corr_days)
    corr = rets.corr(min_periods=max(20, p.corr_days // 2))
    cid = greedy_clusters(pool, corr, p.corr_threshold)

    chosen: list[str] = [t for t in pool if t in keep][: p.n_active]
    used = {cid[t] for t in chosen}
    for t in pool:                                   # pass 1: new clusters first
        if len(chosen) >= p.n_active:
            break
        if t not in chosen and cid[t] not in used:
            chosen.append(t); used.add(cid[t])
    for t in pool:                                   # pass 2: best remaining
        if len(chosen) >= p.n_active:
            break
        if t not in chosen:
            chosen.append(t)

    sel = s.loc[chosen].copy()
    sel["Cluster"] = [cid[t] for t in chosen]
    sel["RiskMeasure"] = sel.apply(lambda r: risk_measure(r, p.sizing), axis=1)
    if p.sizing == "Equal weight":
        raw = np.ones(len(sel))
    else:
        rm = sel["RiskMeasure"].replace([np.inf, -np.inf], np.nan)
        fb = rm.median() if rm.notna().any() else 1.0
        rm = rm.fillna(fb).clip(lower=max(fb * 0.1, 1e-6))
        raw = 1.0 / rm.to_numpy()
    aw = cap_weights(raw, sel["Cluster"].to_numpy(), p.max_single, p.max_cluster)
    sel["ActiveWeight"] = aw
    sel["PortfolioWeight"] = p.active_fraction * aw
    sel["Role"] = ["Held (hysteresis)" if t in keep else "New entry" for t in sel.index]
    sel["Fund_Name"] = [names.get(t, t) for t in sel.index]
    sel.index.name = "Yahoo_Ticker"
    sel = sel.reset_index()
    core_w = 1.0 - sel["PortfolioWeight"].sum()
    cols = ["Yahoo_Ticker", "Fund_Name", "Cluster", "MomentumPercentile", "MomentumScore", "RiskMeasure",
            "ActiveWeight", "PortfolioWeight", "Role"]
    out = pd.concat([pd.DataFrame([core_row(core_w)]), sel[cols]], ignore_index=True)
    return out, corr


def decision_audit(universe: pd.DataFrame, snap: pd.DataFrame, panel_cols: set[str], failed: set[str],
                   port: pd.DataFrame, benchmark: str, p: PortfolioParams, min_history_days: int) -> pd.DataFrame:
    """One row per ETF in the universe: the single rule that decided its fate (rule-based, fully reproducible)."""
    sel = set(port.loc[port["Role"] != "Core", "Yahoo_Ticker"]) if "Role" in port else set()
    rows = []
    for _, m in universe.iterrows():
        t = m["Yahoo_Ticker"]
        if t == benchmark:
            reason = "BENCHMARK / CORE HOLDING"
        elif t in failed:
            reason = "EXCLUDED: FAILED DATA-QUALITY CHECK"
        elif t not in panel_cols:
            reason = "EXCLUDED: NO USABLE PRICE DATA"
        elif t not in snap.index:
            reason = f"EXCLUDED: LESS THAN {min_history_days} DAYS OF HISTORY (or no current price)"
        else:
            r = snap.loc[t]
            if not bool(r["TrendGate"]):
                reason = "EXCLUDED: BELOW LONG-TERM TREND LINE"
            elif not (r["Excess_6M_Skip1M"] > 0):
                reason = "EXCLUDED: 6M EXCESS RETURN (SKIP 1M) NOT POSITIVE"
            elif not (r["Excess_12M_Skip1M"] > 0):
                reason = "EXCLUDED: 12M EXCESS RETURN (SKIP 1M) NOT POSITIVE"
            elif not (r["MomentumPercentile"] >= p.entry_pct):
                reason = f"NOT SELECTED: MOMENTUM RANK BELOW ENTRY LEVEL ({p.entry_pct:.0f})"
            elif t in sel:
                reason = "SELECTED"
            else:
                reason = "ELIGIBLE BUT NOT SELECTED: SLOT LIMIT, CLUSTER DIVERSIFICATION OR CAPS"
        if t in sel and reason not in ("SELECTED",):
            reason = "SELECTED (held under exit rule)" if t in snap.index else reason
        rows.append({"Yahoo_Ticker": t, "Fund_Name": m.get("Fund_Name", t), "Decision": reason,
                     "MomentumPercentile": snap.at[t, "MomentumPercentile"] if t in snap.index else np.nan})
    return pd.DataFrame(rows)
