from __future__ import annotations

import hashlib
import io
import json
import platform
import sys

import pandas as pd
import streamlit as st

from .. import APP_NAME, __version__, content, features as F, portfolio as PF
from . import controls, data as D, kit


def _versions() -> pd.DataFrame:
    import numpy, scipy
    rows = {"App": __version__, "Python": platform.python_version(), "pandas": pd.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "streamlit": st.__version__}
    return pd.DataFrame({"Component": list(rows), "Version": list(rows.values())})


def render():
    S = controls.strategy_sidebar()
    kit.page_header("Audit & Export", "Everything needed to reproduce a result: settings, data fingerprint, rankings, portfolio and backtest – in one Excel file.")
    uni, panel, qrep, ccy, names = controls.load_context(S)
    asof = panel.index[-1]
    feats = F.compute_features(panel, [asof], F.Params(S["ma_days"], S["min_history"], S["benchmark"]))
    d0 = feats.dates[0]
    snap = feats.snapshot(d0)
    rets = panel.prices.pct_change(fill_method=None).loc[:d0]
    port, _ = PF.build_portfolio(snap, rets, S["benchmark"], S["params"], names=names)
    port["Allocation"] = port["PortfolioWeight"] * S["value"]
    failed = set(qrep.loc[qrep["Severity"] == "FAIL", "Ticker"]) if not qrep.empty else set()
    audit = PF.decision_audit(uni, snap, set(panel.prices.columns), failed, port, S["benchmark"], S["params"], S["min_history"])
    fp = hashlib.sha256(pd.util.hash_pandas_object(panel.prices.round(6), index=True).to_numpy().tobytes()).hexdigest()[:16]
    params = {"data_as_of": str(d0.date()), "data_fingerprint": fp, "universe": S["universe"], "base_currency": S["base"], "benchmark": S["benchmark"],
              "ma_days": S["ma_days"], "min_history": S["min_history"], "cost_bps": S["cost_bps"], "lag_days": S["lag_days"],
              **{k: v for k, v in vars(S["params"]).items()}}
    kit.kpi_row([("Data as of", str(d0.date()), None, None), ("Data fingerprint", fp, "Changes if any price changes – proves two runs used identical data.", None),
                 ("Version", __version__, None, None), ("Series with warnings", str(0 if qrep.empty else int(qrep['Severity'].isin(['WARN', 'FAIL']).sum())), None, None)])
    t1, t2, t3 = st.tabs(["Parameters", "Decision audit", "Export"])
    with t1:
        kit.dataframe(pd.DataFrame({"Setting": list(params), "Value": [str(v) for v in params.values()]}), {"Setting": dict(label="Setting"), "Value": dict(label="Value")})
        kit.dataframe(_versions(), {"Component": dict(label="Component"), "Version": dict(label="Version")})
    with t2:
        kit.dataframe(audit, {c: dict(label=c) for c in audit.columns}, height=520)
    with t3:
        res = st.session_state.get("bt_res")
        sheets = {"Parameters": pd.DataFrame({"Setting": list(params), "Value": [str(v) for v in params.values()]}), "Versions": _versions(),
                  "Ranking": snap.reset_index(), "Portfolio": port, "Decision_audit": audit, "Data_quality": qrep}
        if res is not None:
            sheets["Backtest_metrics"] = res.metrics().reset_index()
            sheets["Backtest_periods"] = res.periods.reset_index() if hasattr(res.periods, "reset_index") else res.periods
            sheets["Backtest_flags"] = res.flags.reset_index() if hasattr(res.flags, "reset_index") else res.flags
        else:
            kit.note("Run a backtest in <b>Backtest Lab</b> first to include its results here.")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for name, df in sheets.items():
                d = df.copy()
                for c in d.columns:
                    if pd.api.types.is_datetime64_any_dtype(d[c]):
                        d[c] = d[c].dt.tz_localize(None) if getattr(d[c].dt, "tz", None) else d[c]
                d.to_excel(xw, sheet_name=name[:31], index=False)
        kit.wide(st.download_button, "Download Excel report", buf.getvalue(), f"{APP_NAME.replace(' ', '_')}_{d0.date()}.xlsx",
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", help="All tables on this page plus the backtest if you ran one.")
        st.download_button("Download run manifest (JSON)", json.dumps(params, default=str, indent=2), f"manifest_{d0.date()}.json", "application/json",
                           help="Machine-readable record of settings and data fingerprint.")
    kit.disclaimer_footer()
