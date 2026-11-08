from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import content, forecast as FC, relationships as RL
from . import assets, charts, controls, data as D, kit


def render():
    S = controls.strategy_sidebar()
    kit.page_header("Probabilities", "How likely is the asset to go down, sideways or up over the next few months? Three views – and an honest check of whether the model deserves any trust.")
    p, label, tick = assets.pick_asset(S, "pr", "fc_asset")
    T = D.macro_tables(D.macro_stamp(), "level", 0.80)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        h = int(kit.s_slider("Horizon (months)", "fc_h", "fc_h", 1, 12, 3))
    with c2:
        mode_l = kit.s_select("Sideways band", "fc_band_mode", "fc_band_mode", ["Volatility-scaled", "Fixed %"], "Volatility-scaled")
    with c3:
        band = float(kit.s_number("Band size", "fc_band", "fc_band", 0.05, 20.0, 0.5, 0.05))
    with c4:
        min_train = int(kit.s_slider("Min training months", "fc_min_train", "fc_min_train", 24, 144, 72, 6))
    d1, d2, d3 = st.columns(3)
    use_price = d1.toggle("Use price signals", value=True, help=content.HELP["fc_price"])
    use_macro = d2.toggle("Use economic signals", value=T is not None, disabled=T is None, help=content.HELP["fc_macro"])
    C = float(d3.select_slider("Model caution (regularisation)", options=[0.03, 0.1, 0.3, 1.0, 3.0], value=0.3, help=content.HELP["fc_C"]))
    mode = "vol" if mode_l.startswith("Vol") else "fixed"
    me = RL.month_end(p)
    macro_X = reg = None
    if T is not None:
        macro_X = T["comp"].copy()
        for c in ("T10Y3M", "T10Y2Y"):
            if c in T["feats"].columns:
                macro_X[c] = T["feats"][c]
                break
        reg = T["labels"]["Growth/Inflation"]
    key = repr((tick, h, mode, band, use_price, use_macro, min_train, C, len(me)))
    if kit.wide(st.button, "Calculate probabilities", type="primary"):
        with st.spinner("Running walk-forward test (this can take a minute)…"):
            st.session_state["fc"] = (key, FC.run_forecast(me, macro_X, reg, h, mode, band, use_price, use_macro, min_train, C))
    got = st.session_state.get("fc")
    if not got:
        kit.note("Choose settings and press <b>Calculate probabilities</b>.")
        st.stop()
    if got[0] != key:
        kit.note("Settings changed since the last calculation – press the button to refresh.", "caution")
    R = got[1]
    if not R.ok:
        kit.note(R.reason, "caution")
        st.stop()
    tone = "good" if (R.scores["gain_ci"][0] > 0) else ("bad" if R.scores["gain_ci"][1] < 0 else "neutral")
    kit.verdict(R.verdict, tone)
    sc = R.scores
    kit.kpi_row([("Out-of-sample months", f"{sc['months']}", "Months where the model forecast something it had not seen.", None),
                 ("Brier skill", f"{sc['brier_skill']:+.1%}", "Improvement over the naive base rate. Zero or negative = no better than history's average.", None),
                 ("Accuracy: model", kit.fmt_pct(sc["acc_model"], 0), None, None), ("Accuracy: base rate", kit.fmt_pct(sc["acc_base"], 0), None, None),
                 ("Independent periods", f"{sc['effective_obs']:.0f}", "Overlapping windows count less: months ÷ horizon.", None)])
    v1, v2, v3 = st.tabs(["① Base rate", "② Same economic regime", "③ Model"])
    order = FC.CLASSES
    with v1:
        df = pd.DataFrame({"Outcome": order, "Probability": R.base_rate.reindex(order).to_numpy(), "Low": R.base_ci["lo"].to_numpy(), "High": R.base_ci["hi"].to_numpy()})
        kit.chart(charts.bars(order, df["Probability"].to_numpy(), f"How often each outcome occurred historically over {h} months", fmt=".0%", diverging=False))
        kit.dataframe(df, {"Outcome": dict(label="Outcome"), "Probability": dict(label="Probability", kind="pct"), "Low": dict(label="95% low", kind="pct"), "High": dict(label="95% high", kind="pct", help="Range of plausible true values given the number of independent observations.")})
        st.caption("Start here: if you knew nothing else, this is the honest answer. Everything else must beat it.")
    with v2:
        rt = R.regime_rate
        if rt is None:
            st.info("Needs economic data (Data Center) – not available.")
        else:
            a = rt.attrs
            st.write(f"Current growth/inflation regime: **{a.get('regime')}** – seen in {a.get('months')} months (≈ {a.get('independent')} independent periods).")
            d = rt.reset_index()
            kit.chart(charts.bars(order, d["Probability"].to_numpy(), "What followed in the same regime", fmt=".0%", diverging=False))
            kit.dataframe(d, {"Outcome": dict(label="Outcome"), "Probability": dict(label="Probability", kind="pct"), "Low": dict(label="95% low", kind="pct"), "High": dict(label="95% high", kind="pct")})
            if a.get("independent", 0) < 10:
                kit.note("Fewer than ~10 independent periods: treat as anecdote, not evidence.", "caution")
    with v3:
        mp = R.model_probs.reindex(order)
        kit.chart(charts.bars(order, mp.to_numpy(), "Model probabilities for the next period", fmt=".0%", diverging=False))
        cmp_ = pd.DataFrame({"Outcome": order, "Model": mp.to_numpy(), "Base rate": R.base_rate.reindex(order).to_numpy()})
        kit.dataframe(cmp_, {"Outcome": dict(label="Outcome"), "Model": dict(label="Model", kind="pct"), "Base rate": dict(label="Base rate", kind="pct")})
        if R.calibration is not None and len(R.calibration):
            kit.chart(charts.calibration(R.calibration))
            st.caption("Points near the dotted line = when the model said ‘30%’, it happened about 30% of the time. Few, tiny points = too little data to judge.")
        with st.expander("What drove the model? (coefficients)"):
            kit.dataframe(R.coef.reset_index().rename(columns={"index": "Input"}), {"Input": dict(label="Input")} | {c: dict(label=c, kind="num2") for c in order})
            st.caption("Positive = pushes probability of that outcome up (inputs are standardised). Large values with a no-skill verdict mean the model is fitting noise.")
    kit.disclaimer_footer()
