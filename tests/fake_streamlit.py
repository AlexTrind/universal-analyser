"""Minimal stand-ins for streamlit and plotly so every page's render() can be executed in a headless test.

The fake enforces the Streamlit rules that most often break real apps: unique widget keys within one run, consistent numeric
types in sliders/number inputs, a default that lies inside [min, max], options that contain the chosen index, and a help
text on every input widget."""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager


class StopApp(Exception):
    pass


class DuplicateKey(Exception):
    pass


class Anything:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Anything()

    def __call__(self, *a, **k):
        return Anything()

    def __iter__(self):
        return iter(())

    def __getitem__(self, k):
        return Anything()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Figure(Anything):
    def __init__(self, *a, **k):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self


class Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self._st

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Cache:
    def __call__(self, *a, **k):
        def deco(fn):
            fn.clear = lambda *x, **y: None
            return fn
        return deco(a[0]) if a and callable(a[0]) else deco

    def clear(self):
        return None


class FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.__version__ = "1.99.0"
        self.session_state = State()
        self.press: set[str] = set()
        self.radio_index = None
        self.widgets: list[dict] = []
        self.keys: set[str] = set()
        self.messages: list[tuple[str, str]] = []
        self.charts = 0
        self.tables = 0
        self.errors = types.SimpleNamespace(StreamlitAPIException=type("StreamlitAPIException", (Exception,), {}))
        self.column_config = types.SimpleNamespace(**{n: (lambda n=n: (lambda label=None, **k: {"type": n, "label": label, **k}))() for n in
                                                      ["NumberColumn", "TextColumn", "CheckboxColumn", "DateColumn", "LinkColumn", "ProgressColumn"]})
        self.sidebar = Ctx(self)
        self.cache_data = self.cache_resource = _Cache()

    def new_run(self):
        self.keys = set()
        self.widgets.clear()
        self.messages.clear()
        self.charts = self.tables = 0

    # -- helpers
    def _reg(self, kind, label, key, kw, need_help=True):
        if key is not None:
            if key in self.keys:
                raise DuplicateKey(f"duplicate widget key {key!r} ({label})")
            self.keys.add(key)
        self.widgets.append({"kind": kind, "label": label, "key": key, "help": kw.get("help")})
        if need_help and not kw.get("help") and kind not in ("button", "download_button", "radio"):
            self.messages.append(("nohelp", f"{kind}: {label}"))

    # -- inputs
    def slider(self, label, min_value=0, max_value=100, value=None, step=None, key=None, **kw):
        kinds = {("f" if isinstance(x, float) else "i") for x in (min_value, max_value, value, step) if x is not None}
        if len(kinds) > 1:
            raise TypeError(f"slider {label!r}: mixed int/float arguments")
        value = min_value if value is None else value
        if not (min_value <= value <= max_value):
            raise ValueError(f"slider {label!r}: value {value} outside [{min_value}, {max_value}]")
        self._reg("slider", label, key, kw)
        return value

    def number_input(self, label, min_value=None, max_value=None, value=0, step=None, key=None, **kw):
        ts = {type(x) for x in (min_value, max_value, value, step) if x is not None}
        if len(ts) > 1:
            raise TypeError(f"number_input {label!r}: mixed numeric types {ts}")
        if min_value is not None and not (min_value <= value <= max_value):
            raise ValueError(f"number_input {label!r}: value {value} outside [{min_value}, {max_value}]")
        self._reg("number_input", label, key, kw)
        return value

    def select_slider(self, label, options=(), value=None, key=None, **kw):
        opts = list(options)
        if key in self.session_state and self.session_state[key] in opts:
            value = self.session_state[key]
        self._reg("select_slider", label, key, kw)
        return value if value in opts else opts[-1]

    def selectbox(self, label, options=(), index=0, key=None, **kw):
        opts = list(options)
        if opts and not (0 <= index < len(opts)):
            raise IndexError(f"selectbox {label!r}: index {index} of {len(opts)}")
        self._reg("selectbox", label, key, kw)
        return opts[index] if opts else None

    def radio(self, label, options=(), index=0, key=None, **kw):
        self._reg("radio", label, key, kw)
        return list(options)[self.radio_index if self.radio_index is not None else index]

    def multiselect(self, label, options=(), default=None, key=None, **kw):
        opts = list(options)
        bad = [d for d in (default or []) if d not in opts]
        if bad:
            raise ValueError(f"multiselect {label!r}: defaults not in options: {bad[:3]}")
        self._reg("multiselect", label, key, kw)
        return list(default or [])

    def text_input(self, label, value="", key=None, **kw):
        self._reg("text_input", label, key, kw)
        return value

    text_area = text_input

    def toggle(self, label, value=False, key=None, **kw):
        self._reg("toggle", label, key, kw)
        return value

    checkbox = toggle

    def date_input(self, label, value=None, min_value=None, max_value=None, key=None, **kw):
        if min_value is not None and not (min_value <= value <= max_value):
            raise ValueError(f"date_input {label!r}: {value} outside range")
        self._reg("date_input", label, key, kw)
        return value

    def button(self, label, key=None, **kw):
        self._reg("button", label, key, kw)
        return any(p in label for p in self.press)

    def download_button(self, label, data=None, file_name=None, mime=None, key=None, **kw):
        self._reg("download_button", label, key, kw)
        assert data is not None and len(data) > 0, "empty download"
        return False

    # -- layout
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx(self) for _ in range(n)]

    def tabs(self, labels):
        return [Ctx(self) for _ in labels]

    def expander(self, *a, **k):
        return Ctx(self)

    container = expander
    spinner = expander
    form = expander

    def progress(self, *a, **k):
        return Anything()

    # -- output
    def _msg(kind):
        def f(self, body="", *a, **k):
            self.messages.append((kind, str(body)))
        return f

    error = _msg("error")
    warning = _msg("warning")
    success = _msg("success")
    info = _msg("info")

    def exception(self, e, *a, **k):
        self.messages.append(("exception", repr(e)))

    def plotly_chart(self, fig, *a, **k):
        assert isinstance(fig, Figure), f"plotly_chart got {type(fig)}"
        self.charts += 1

    def dataframe(self, df, *a, column_config=None, **k):
        import pandas as pd
        assert isinstance(df, pd.DataFrame), type(df)
        self.tables += 1
        if column_config:
            extra = set(column_config) - set(df.columns)
            assert not extra, f"column_config for missing columns {extra}"
        assert df.columns.is_unique, "duplicate column names"
        return None

    def metric(self, label, value, delta=None, **k):
        assert isinstance(value, str), f"metric {label!r} value must be str, got {type(value)}"

    def stop(self):
        raise StopApp()

    def rerun(self):
        raise StopApp()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self


class Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self._st

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Cache:
    def __call__(self, *a, **k):
        def deco(fn):
            fn.clear = lambda *x, **y: None
            return fn
        return deco(a[0]) if a and callable(a[0]) else deco

    def clear(self):
        return None


class FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.__version__ = "1.99.0"
        self.session_state = State()
        self.press: set[str] = set()
        self.radio_index = None
        self.widgets: list[dict] = []
        self.keys: set[str] = set()
        self.messages: list[tuple[str, str]] = []
        self.charts = 0
        self.tables = 0
        self.errors = types.SimpleNamespace(StreamlitAPIException=type("StreamlitAPIException", (Exception,), {}))
        self.column_config = types.SimpleNamespace(**{n: (lambda n=n: (lambda label=None, **k: {"type": n, "label": label, **k}))() for n in
                                                      ["NumberColumn", "TextColumn", "CheckboxColumn", "DateColumn", "LinkColumn", "ProgressColumn"]})
        self.sidebar = Ctx(self)
        self.cache_data = self.cache_resource = _Cache()

    def new_run(self):
        self.keys = set()
        self.widgets.clear()
        self.messages.clear()
        self.charts = self.tables = 0

    # -- helpers
    def _reg(self, kind, label, key, kw, need_help=True):
        if key is not None:
            if key in self.keys:
                raise DuplicateKey(f"duplicate widget key {key!r} ({label})")
            self.keys.add(key)
        self.widgets.append({"kind": kind, "label": label, "key": key, "help": kw.get("help")})
        if need_help and not kw.get("help") and kind not in ("button", "download_button", "radio"):
            self.messages.append(("nohelp", f"{kind}: {label}"))

    # -- inputs
    def slider(self, label, min_value=0, max_value=100, value=None, step=None, key=None, **kw):
        kinds = {("f" if isinstance(x, float) else "i") for x in (min_value, max_value, value, step) if x is not None}
        if len(kinds) > 1:
            raise TypeError(f"slider {label!r}: mixed int/float arguments")
        value = min_value if value is None else value
        if not (min_value <= value <= max_value):
            raise ValueError(f"slider {label!r}: value {value} outside [{min_value}, {max_value}]")
        self._reg("slider", label, key, kw)
        return value

    def number_input(self, label, min_value=None, max_value=None, value=0, step=None, key=None, **kw):
        ts = {type(x) for x in (min_value, max_value, value, step) if x is not None}
        if len(ts) > 1:
            raise TypeError(f"number_input {label!r}: mixed numeric types {ts}")
        if min_value is not None and not (min_value <= value <= max_value):
            raise ValueError(f"number_input {label!r}: value {value} outside [{min_value}, {max_value}]")
        self._reg("number_input", label, key, kw)
        return value

    def select_slider(self, label, options=(), value=None, key=None, **kw):
        opts = list(options)
        if key in self.session_state and self.session_state[key] in opts:
            value = self.session_state[key]
        self._reg("select_slider", label, key, kw)
        return value if value in opts else opts[-1]

    def selectbox(self, label, options=(), index=0, key=None, **kw):
        opts = list(options)
        if opts and not (0 <= index < len(opts)):
            raise IndexError(f"selectbox {label!r}: index {index} of {len(opts)}")
        self._reg("selectbox", label, key, kw)
        return opts[index] if opts else None

    def radio(self, label, options=(), index=0, key=None, **kw):
        self._reg("radio", label, key, kw)
        return list(options)[self.radio_index if self.radio_index is not None else index]

    def multiselect(self, label, options=(), default=None, key=None, **kw):
        opts = list(options)
        bad = [d for d in (default or []) if d not in opts]
        if bad:
            raise ValueError(f"multiselect {label!r}: defaults not in options: {bad[:3]}")
        self._reg("multiselect", label, key, kw)
        return list(default or [])

    def text_input(self, label, value="", key=None, **kw):
        self._reg("text_input", label, key, kw)
        return value

    text_area = text_input

    def toggle(self, label, value=False, key=None, **kw):
        self._reg("toggle", label, key, kw)
        return value

    checkbox = toggle

    def date_input(self, label, value=None, min_value=None, max_value=None, key=None, **kw):
        if min_value is not None and not (min_value <= value <= max_value):
            raise ValueError(f"date_input {label!r}: {value} outside range")
        self._reg("date_input", label, key, kw)
        return value

    def button(self, label, key=None, **kw):
        self._reg("button", label, key, kw)
        return any(p in label for p in self.press)

    def download_button(self, label, data=None, file_name=None, mime=None, key=None, **kw):
        self._reg("download_button", label, key, kw)
        assert data is not None and len(data) > 0, "empty download"
        return False

    # -- layout
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx(self) for _ in range(n)]

    def tabs(self, labels):
        return [Ctx(self) for _ in labels]

    def expander(self, *a, **k):
        return Ctx(self)

    container = expander
    spinner = expander
    form = expander

    def progress(self, *a, **k):
        return Anything()

    # -- output
    def _msg(kind):
        def f(self, body="", *a, **k):
            self.messages.append((kind, str(body)))
        return f

    error = _msg("error")
    warning = _msg("warning")
    success = _msg("success")
    info = _msg("info")

    def exception(self, e, *a, **k):
        self.messages.append(("exception", repr(e)))

    def plotly_chart(self, fig, *a, **k):
        assert isinstance(fig, Figure), f"plotly_chart got {type(fig)}"
        self.charts += 1

    def dataframe(self, df, *a, column_config=None, **k):
        import pandas as pd
        assert isinstance(df, pd.DataFrame), type(df)
        self.tables += 1
        if column_config:
            extra = set(column_config) - set(df.columns)
            assert not extra, f"column_config for missing columns {extra}"
        assert df.columns.is_unique, "duplicate column names"
        return None

    def metric(self, label, value, delta=None, **k):
        assert isinstance(value, str), f"metric {label!r} value must be str, got {type(value)}"

    def stop(self):
        raise StopApp()

    def rerun(self):
        raise StopApp()

    def cache_data(self, *a, **k):
        def deco(fn):
            fn.clear = lambda *x, **y: None
            return fn
        return deco(a[0]) if a and callable(a[0]) else deco

    cache_resource = cache_data
    clear = lambda self: None  # noqa: E731

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: Anything()


def install():
    st = FakeSt()
    sys.modules["streamlit"] = st
    go = types.ModuleType("plotly.graph_objects")
    for n in ["Figure", "Scatter", "Bar", "Heatmap", "Treemap", "Pie", "Candlestick", "Histogram", "Scattergl"]:
        setattr(go, n, Figure)
    sub = types.ModuleType("plotly.subplots")
    sub.make_subplots = lambda *a, **k: Figure()
    px = types.ModuleType("plotly.express")
    px.treemap = lambda *a, **k: Figure()
    px.scatter = px.line = px.bar = px.treemap
    px.Constant = lambda v, *a, **k: v
    pl = types.ModuleType("plotly")
    pl.graph_objects, pl.subplots, pl.express = go, sub, px
    sys.modules.update({"plotly": pl, "plotly.graph_objects": go, "plotly.subplots": sub, "plotly.express": px})
    return st
