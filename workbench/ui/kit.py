"""Small UI toolkit: every control is created through here so it ALWAYS carries hover-help."""
from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from .. import content
from . import theme


# ---------------------------------------------------------------------------- Streamlit-version compatibility
def stretch_kwargs(fn: Callable) -> dict:
    """Streamlit renamed `use_container_width=True` to `width="stretch"`.  Pick whichever this install supports."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    p = params.get("width")
    if p is not None and p.default in ("stretch", "content"):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def wide(fn: Callable, *args, **kwargs):
    for extra in (stretch_kwargs(fn), {"use_container_width": True}, {"width": "stretch"}, {}):
        try:
            return fn(*args, **kwargs, **extra)
        except TypeError:
            continue
    return fn(*args, **kwargs)


def chart(fig, key: str | None = None):
    kw = {"key": key} if key else {}
    return wide(st.plotly_chart, fig, **kw)


def dataframe(df: pd.DataFrame, spec: dict | None = None, height: int | None = None, key: str | None = None):
    """Render a DataFrame with per-column labels, number formats and hover help.

    spec = {column: dict(label=, kind='pct'|'pct_signed'|'num'|'num2'|'int'|'money'|'text'|'date'|'bool', help=)}
    Fractions marked 'pct' are multiplied by 100 for display.
    """
    d = df.copy()
    cfg = {}
    for col, s in (spec or {}).items():
        if col not in d.columns:
            continue
        kind = s.get("kind", "text")
        label = s.get("label", col)
        h = s.get("help")
        if kind in ("pct", "pct_signed"):
            d[col] = pd.to_numeric(d[col], errors="coerce") * 100
            cfg[col] = st.column_config.NumberColumn(label, format="%+.1f%%" if kind == "pct_signed" else "%.1f%%", help=h)
        elif kind == "num":
            cfg[col] = st.column_config.NumberColumn(label, format="%.1f", help=h)
        elif kind == "num2":
            cfg[col] = st.column_config.NumberColumn(label, format="%.2f", help=h)
        elif kind == "int":
            cfg[col] = st.column_config.NumberColumn(label, format="%d", help=h)
        elif kind == "money":
            cfg[col] = st.column_config.NumberColumn(label, format="%.2f", help=h)
        elif kind == "bool":
            cfg[col] = st.column_config.CheckboxColumn(label, help=h)
        elif kind == "date":
            cfg[col] = st.column_config.DateColumn(label, help=h)
        else:
            cfg[col] = st.column_config.TextColumn(label, help=h)
    kw: dict[str, Any] = {"hide_index": True, "column_config": cfg}
    if height:
        kw["height"] = height
    if key:
        kw["key"] = key
    return wide(st.dataframe, d, **kw)


def col_help(*cols: str) -> dict:
    return {c: content.COL.get(c) for c in cols}


# ---------------------------------------------------------------------------- sticky widgets (settings survive page changes)
def _store() -> dict:
    return st.session_state.setdefault("_sticky", {})


def _make(kind: Callable, label: str, key: str, help_key: str, default, **kw):
    help_text = content.HELP.get(help_key, help_key)
    assert help_text, f"Missing help text for {help_key}"
    store = _store()
    val = store.get(key, default)
    wkey = f"w_{key}"

    def _cb():
        store[key] = st.session_state[wkey]

    return val, wkey, _cb, help_text


def s_slider(label, key, help_key, minimum, maximum, default, step=None, **kw):
    val, wkey, cb, h = _make(st.slider, label, key, help_key, default)
    args = dict(min_value=minimum, max_value=maximum, value=val, key=wkey, on_change=cb, help=h)
    if step is not None:
        args["step"] = step
    return st.slider(label, **args, **kw)


def s_number(label, key, help_key, minimum, maximum, default, step=1.0, **kw):
    val, wkey, cb, h = _make(st.number_input, label, key, help_key, default)
    return st.number_input(label, min_value=minimum, max_value=maximum, value=val, step=step, key=wkey, on_change=cb, help=h, **kw)


def s_select(label, key, help_key, options, default=None, **kw):
    options = list(options)
    val, wkey, cb, h = _make(st.selectbox, label, key, help_key, default if default is not None else options[0])
    idx = options.index(val) if val in options else 0
    return st.selectbox(label, options, index=idx, key=wkey, on_change=cb, help=h, **kw)


def s_multi(label, key, help_key, options, default, **kw):
    options = list(options)
    val, wkey, cb, h = _make(st.multiselect, label, key, help_key, default)
    val = [v for v in val if v in options]
    return st.multiselect(label, options, default=val, key=wkey, on_change=cb, help=h, **kw)


def s_text(label, key, help_key, default, **kw):
    val, wkey, cb, h = _make(st.text_input, label, key, help_key, default)
    return st.text_input(label, value=val, key=wkey, on_change=cb, help=h, **kw)


def s_toggle(label, key, help_key, default=False, **kw):
    val, wkey, cb, h = _make(st.toggle, label, key, help_key, default)
    return st.toggle(label, value=val, key=wkey, on_change=cb, help=h, **kw)


def s_date(label, key, help_key, default, min_value, max_value, **kw):
    val, wkey, cb, h = _make(st.date_input, label, key, help_key, default)
    val = min(max(val, min_value), max_value)
    return st.date_input(label, value=val, min_value=min_value, max_value=max_value, key=wkey, on_change=cb, help=h, **kw)


# ---------------------------------------------------------------------------- layout helpers
def page_header(title: str, subtitle: str):
    st.title(title)
    st.markdown(f'<p class="wb-sub">{subtitle}</p>', unsafe_allow_html=True)


def note(text: str, kind: str = "note"):
    st.markdown(f'<div class="wb-{"caution" if kind == "caution" else "note"}">{text}</div>', unsafe_allow_html=True)


def verdict(text: str, tone: str = "neutral"):
    st.markdown(f'<div class="wb-verdict-{tone}">{text}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "ok") -> str:
    return f'<span class="wb-pill wb-{tone}">{text}</span>'


def explain(title: str, body: str, expanded: bool = False):
    with st.expander(f"ⓘ {title}", expanded=expanded):
        st.markdown(body)


def kpi_row(items: list[tuple[str, str, str | None, str | None]]):
    """items = (label, value_text, help, delta_text)"""
    cols = st.columns(len(items))
    for c, (label, val, h, delta) in zip(cols, items):
        with c:
            st.metric(label, val, delta, help=h)


def fmt_pct(x, d=1, signed=False):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:+.{d}%}" if signed else f"{x:.{d}%}")


def fmt_num(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def disclaimer_footer():
    st.divider()
    st.caption(content.DISCLAIMER)
