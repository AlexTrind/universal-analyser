"""Scenario auditor: drives every page through many user choices with the fake streamlit and reports problems
that a plain 'does it crash' smoke test misses – empty/truncated charts, NaN in headline numbers, widgets that
ignore a new choice, error messages, missing help text.

Used by tests/test_audit_scenarios.py (synthetic data) and runnable on real data:
    python tests/audit_harness.py REAL_DIR     (REAL_DIR = folder holding the downloaded cache files)"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import fake_streamlit as FS  # noqa: E402

PAGES = ["page_home", "page_learn", "page_data", "page_screener", "page_backtest", "page_map", "page_group", "page_asset",
         "page_macro", "page_compare", "page_prob", "page_audit"]


@dataclass
class Finding:
    page: str
    scenario: str
    kind: str
    detail: str

    def __str__(self):
        return f"[{self.kind}] {self.page} | {self.scenario} | {self.detail}"


@dataclass
class Audit:
    st: object
    findings: list = field(default_factory=list)
    charts: list = field(default_factory=list)
    kpis: list = field(default_factory=list)
    tables: list = field(default_factory=list)

    # ------------------------------------------------------------------ instrumentation
    def install_probes(self):
        from workbench.ui import charts, kit
        st = self.st
        self.overrides: dict = {}
        audit = self

        for name in [n for n in dir(charts) if not n.startswith("_") and callable(getattr(charts, n))]:
            fn = getattr(charts, name)
            if getattr(fn, "_probed", False) or not hasattr(fn, "__code__") or fn.__module__ != charts.__name__:
                continue

            def make(fn=fn, name=name):
                def probe(*a, **k):
                    audit.charts.append((name, a, k))
                    return fn(*a, **k)
                probe._probed = True
                return probe
            setattr(charts, name, make())

        orig_kpi = kit.kpi_row
        if not getattr(orig_kpi, "_probed", False):
            def kpi(items, *a, **k):
                audit.kpis.extend(items)
                return orig_kpi(items, *a, **k)
            kpi._probed = True
            kit.kpi_row = kpi
        orig_df = kit.dataframe
        if not getattr(orig_df, "_probed", False):
            def dfp(df, *a, **k):
                audit.tables.append(df)
                return orig_df(df, *a, **k)
            dfp._probed = True
            kit.dataframe = dfp

        # user choices by widget label (raw st.* widgets otherwise always return their default)
        for meth in ("selectbox", "multiselect", "radio", "toggle", "checkbox", "slider", "text_input", "number_input", "select_slider"):
            orig = getattr(st, meth)

            def wrap(orig=orig, meth=meth):
                def f(label, *a, **k):
                    out = orig(label, *a, **k)
                    if label in audit.overrides:
                        want = audit.overrides[label]
                        opts = list(a[0]) if a else list(k.get("options", ()))
                        if meth in ("selectbox", "radio", "select_slider", "multiselect") and opts:
                            wants = want if isinstance(want, list) else [want]
                            miss = [w for w in wants if w not in opts]
                            if miss:
                                audit.note("override_not_offered", f"{label!r}: {miss[:3]} not among {len(opts)} options")
                                return out
                        return want
                    return out
                return f
            setattr(st, meth, wrap())

    def note(self, kind, detail):
        self.findings.append(Finding(self._page, self._scn, kind, detail))

    # ------------------------------------------------------------------ analysis
    def _check_charts(self):
        for name, a, k in self.charts:
            df = a[0] if a else None
            if name in ("line", "drawdown", "stacked_area") and isinstance(df, pd.DataFrame):
                if df.empty or df.dropna(how="all").empty:
                    self.note("empty_chart", f"{name} '{k.get('title') or (a[1] if len(a) > 1 else '')}' has no data")
                    continue
                if np.isinf(df.select_dtypes("number").to_numpy(dtype=float)).any():
                    self.note("inf_in_chart", f"{name} contains infinite values")
                spans = {}
                for c in df.columns:
                    v = df[c].dropna()
                    if len(v):
                        spans[c] = (v.index.max() - v.index.min()).days
                if len(spans) > 1:
                    longest = max(spans.values())
                    short = [str(c) for c, d in spans.items() if longest > 365 and d < 0.4 * longest]
                    if short:
                        self.note("short_series_in_chart", f"{name}: {short[:3]} cover <40% of the longest line (cut-off or late start?)")
                span_all = (df.dropna(how="all").index.max() - df.dropna(how="all").index.min()).days
                if len(df.columns) > 1 and len(df.dropna()) < 0.5 * len(df.dropna(how="all")) and span_all > 730:
                    self.note("overlap_cut", f"{name}: only {len(df.dropna())} of {len(df.dropna(how='all'))} rows have ALL lines")
            elif name == "heatmap" and isinstance(df, pd.DataFrame) and (df.empty or df.isna().all().all()):
                self.note("empty_chart", "heatmap empty (may be fine if nothing is significant)")

    def _check_kpis(self):
        for it in self.kpis:
            label, val = it[0], str(it[1])
            if val.lower() in ("nan", "none", "inf", "-inf", "nan%", "inf%"):
                self.note("bad_kpi", f"{label} = {val}")

    def _check_tables(self):
        for df in self.tables:
            if isinstance(df, pd.DataFrame) and len(df) and df.select_dtypes("number").shape[1]:
                num = df.select_dtypes("number")
                if np.isinf(num.to_numpy(dtype=float)).any():
                    self.note("inf_in_table", f"columns {list(num.columns[np.isinf(num.to_numpy(dtype=float)).any(axis=0)])[:4]}")

    # ------------------------------------------------------------------ running
    def run(self, page: str, scenario: str = "default", press=(), overrides=None, sticky=None, radio=None, expect_stop_ok=True):
        st = self.st
        self._page, self._scn = page, scenario
        st.new_run()
        st.press = set(press)
        st.radio_index = radio
        self.overrides = dict(overrides or {})
        self.charts.clear(); self.kpis.clear(); self.tables.clear()
        for k, v in (sticky or {}).items():
            st.session_state.setdefault("_sticky", {})[k] = v
        mod = importlib.import_module(f"workbench.ui.{page}")
        stopped = False
        try:
            mod.render()
        except FS.StopApp:
            stopped = True
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)[-1]
            self.note("CRASH", f"{type(e).__name__}: {e} @ {Path(tb.filename).name}:{tb.lineno}")
            return None
        for kind, msg in st.messages:
            if kind in ("error", "exception"):
                self.note("error_shown", msg[:200])
            elif kind == "nohelp":
                self.note("no_help", msg)
            elif kind == "warning":
                self.note("warning_shown", msg[:160])
        self._check_charts(); self._check_kpis(); self._check_tables()
        return stopped


# ---------------------------------------------------------------------- environments
def build_env(real_dir: str | None = None):
    tmp = Path(tempfile.mkdtemp(prefix="wb_audit_"))
    st = FS.install()
    from workbench import config
    config.DATA_DIR = tmp / "data"
    config.DATA_DIR.mkdir()
    for m in [m for m in sys.modules if m.startswith("workbench.ui")]:
        del sys.modules[m]
    if real_dir:
        real = Path(real_dir)
        (config.DATA_DIR / "etf").mkdir(); (config.DATA_DIR / "stocks").mkdir(); (config.DATA_DIR / "macro").mkdir()
        shutil.copy(real / "etf_prices.csv.gz", config.DATA_DIR / "etf" / "prices_native_adjclose.csv.gz")
        shutil.copy(real / "price_meta.json", config.DATA_DIR / "etf" / "price_meta.json")
        shutil.copy(real / "stock_prices.csv.gz", config.DATA_DIR / "stocks" / "prices_native_adjclose.csv.gz")
        for f in (real / "macro").glob("*.csv"):
            shutil.copy(f, config.DATA_DIR / "macro" / f.name)
    return st, tmp
