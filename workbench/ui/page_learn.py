from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import content, data_macro as DM
from . import kit


def render():
    kit.page_header("Learn", "Plain-English explanations of how the tool works, what the numbers mean, and how to avoid the classic traps.")
    tabs = st.tabs(["How it works", "Tutorials", "Glossary", "Indicator guide", "Pitfalls", "FAQ"])
    with tabs[0]:
        st.markdown(content.HOW_IT_WORKS)
        try:
            st.graphviz_chart(content.HOW_DOT)
        except Exception:  # noqa: BLE001
            st.caption("(diagram unavailable in this browser)")
    with tabs[1]:
        titles = [t for t, _ in content.TUTORIALS]
        pick = st.selectbox("Choose a tutorial", titles, help="Short lessons in recommended order.")
        st.markdown(dict(content.TUTORIALS)[pick])
        with st.expander("Show all tutorials on one page"):
            for t, body in content.TUTORIALS:
                st.markdown(f"## {t}")
                st.markdown(body)
    with tabs[2]:
        q = st.text_input("Search the glossary", "", help="Type part of a word, e.g. ‘drawdown’ or ‘bias’.")
        items = [(k, v) for k, v in sorted(content.GLOSSARY.items()) if q.lower() in k.lower() or q.lower() in v.lower()]
        for k, v in items:
            st.markdown(f"**{k}** — {v}")
        if not items:
            st.info("No matches.")
    with tabs[3]:
        st.write("Every economic indicator the tool can use, why it matters, and how it enters the analysis.")
        rows = [{"Group": s.group, "Indicator": s.name, "FRED ID": s.id, "Why it matters": s.why,
                 "Delay assumed (days)": s.lag_days, "Revised later?": "yes" if s.revised else "no"} for s in DM.CATALOG]
        kit.dataframe(pd.DataFrame(rows), {"Group": dict(label="Group"), "Indicator": dict(label="Indicator"), "FRED ID": dict(label="FRED ID"),
                                           "Why it matters": dict(label="Why it matters"),
                                           "Delay assumed (days)": dict(label="Publication delay assumed (days)", kind="int", help="The tool treats a value as unknown until this many days after its date, so tests cannot peek at unpublished data."),
                                           "Revised later?": dict(label="Revised later?", help="Revised series flatter backtests slightly because FRED serves the final numbers, not the first-published ones.")}, height=560)
    with tabs[4]:
        for t, d in content.PITFALLS:
            st.markdown(f"**{t}** — {d}")
    with tabs[5]:
        for q_, a_ in content.FAQ:
            with st.expander(q_):
                st.write(a_)
    kit.disclaimer_footer()
