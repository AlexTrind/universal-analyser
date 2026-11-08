"""ETF & Stock Momentum Workbench – entry point.  Run:  streamlit run app.py"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from workbench import APP_NAME, __version__, config  # noqa: E402
from workbench.ui import theme  # noqa: E402

config.LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=config.LOG_DIR / "workbench.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown(theme.CSS, unsafe_allow_html=True)


def _page(modname: str):
    def run():
        import importlib

        mod = importlib.import_module(f"workbench.ui.{modname}")
        try:
            mod.render()
        except st.errors.StreamlitAPIException:
            raise
        except Exception as e:  # noqa: BLE001  - show a readable message instead of a raw traceback
            logging.getLogger("workbench.ui").exception("page %s failed", modname)
            st.error(f"Something went wrong on this page: {type(e).__name__}: {e}")
            with st.expander("Technical details (send these if you ask for help)"):
                st.exception(e)
    run.__name__ = f"page_{modname}"
    return run


PAGES = {
    "Start": [st.Page(_page("page_home"), title="Home", icon="🏠", default=True, url_path="home"),
              st.Page(_page("page_learn"), title="Learn", icon="📚", url_path="learn")],
    "ETFs": [st.Page(_page("page_screener"), title="ETF Screener", icon="🔎", url_path="screener"),
             st.Page(_page("page_backtest"), title="Backtest Lab", icon="🧪", url_path="backtest"),
             st.Page(_page("page_map"), title="Market Map", icon="🗺️", url_path="map")],
    "Stocks & assets": [st.Page(_page("page_group"), title="Group Explorer", icon="🏢", url_path="group"),
                        st.Page(_page("page_asset"), title="Asset Lab", icon="🔬", url_path="asset")],
    "Economy": [st.Page(_page("page_macro"), title="Macro & Cycles", icon="🌍", url_path="macro"),
                st.Page(_page("page_compare"), title="Compare with Economy", icon="🔗", url_path="compare"),
                st.Page(_page("page_prob"), title="Probabilities", icon="🎲", url_path="prob")],
    "Tools": [st.Page(_page("page_data"), title="Data Center", icon="🗄️", url_path="data"),
              st.Page(_page("page_audit"), title="Audit & Export", icon="📋", url_path="audit")],
}
nav = st.navigation(PAGES)
with st.sidebar:
    st.caption(f"{APP_NAME} · v{__version__}")
nav.run()
