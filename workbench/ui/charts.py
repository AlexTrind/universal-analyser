"""Plotly figure builders (kept deliberately simple: lines, areas, bars, heat-maps)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme

COL = theme.CATEGORICAL


def _base(fig: go.Figure, title: str | None = None, height: int = 420, yfmt: str | None = None, ytitle: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=theme.NAVY)) if title else None,
        height=height, margin=dict(l=60, r=24, t=60 if title else 24, b=44),
        paper_bgcolor=theme.SURFACE, plot_bgcolor=theme.SURFACE,
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color=theme.INK),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=theme.GRID, ticks="outside", tickcolor=theme.GRID)
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False, tickformat=yfmt, title_text=ytitle)
    return fig


def line(df: pd.DataFrame, title: str | None = None, yfmt: str | None = None, ytitle: str | None = None, height: int = 420,
         log: bool = False, hover_fmt: str = ".2f", colors: dict | None = None, shade: list | None = None, hline: float | None = None) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        color = (colors or {}).get(c, COL[i % len(COL)])
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c), line=dict(color=color, width=2),
                                 hovertemplate="%{y:" + hover_fmt + "}"))
    lo, hi = (df.index.min(), df.index.max()) if len(df) else (None, None)
    for x0, x1 in (shade or []):
        if lo is not None:                        # only draw bands inside the data, and never let them stretch the axis
            if x1 < lo or x0 > hi:
                continue
            x0, x1 = max(x0, lo), min(x1, hi)
        fig.add_vrect(x0=x0, x1=x1, fillcolor="#8896A5", opacity=0.15, line_width=0, layer="below")
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color=theme.MUTED, line_width=1)
    _base(fig, title, height, yfmt, ytitle, legend=df.shape[1] > 1)
    if lo is not None:
        fig.update_xaxes(range=[lo, hi], autorange=False)   # the axis always follows the data being shown
    if log:
        fig.update_yaxes(type="log")
    return fig


def drawdown(dd: pd.DataFrame, title: str = "Drawdown from previous peak", height: int = 300) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(dd.columns):
        fig.add_trace(go.Scatter(x=dd.index, y=dd[c], mode="lines", name=str(c), line=dict(color=COL[i % len(COL)], width=1.6),
                                 fill="tozeroy" if i == 0 else None, fillcolor="rgba(42,120,214,0.12)" if i == 0 else None,
                                 hovertemplate="%{y:.1%}"))
    return _base(fig, title, height, ".0%")


def stacked_area(df: pd.DataFrame, title: str, height: int = 420, hv: bool = True) -> go.Figure:
    """df: index = dates, columns = holdings, values = weights (fractions)."""
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c), stackgroup="one",
                                 line=dict(width=0.5, shape="hv" if hv else "linear", color=COL[i % len(COL)] if i < len(COL) else None),
                                 hovertemplate="%{y:.1%}"))
    return _base(fig, title, height, ".0%")


def bars(labels, values, title: str | None = None, horizontal: bool = True, fmt: str = "+.1%", height: int | None = None,
         diverging: bool = True, xtitle: str | None = None) -> go.Figure:
    vals = np.asarray(values, dtype=float)
    colors = [theme.POS if v >= 0 else theme.NEG for v in vals] if diverging else [COL[0]] * len(vals)
    texts = [format(v, fmt) for v in vals]
    fig = go.Figure(go.Bar(
        x=vals if horizontal else list(labels), y=list(labels) if horizontal else vals,
        orientation="h" if horizontal else "v", marker=dict(color=colors), text=texts, textposition="outside",
        hovertemplate="%{text}<extra>%{y}</extra>" if horizontal else "%{text}<extra>%{x}</extra>", cliponaxis=False,
    ))
    h = height or (max(260, 28 * len(vals) + 90) if horizontal else 380)
    _base(fig, title, h, legend=False)
    fig.update_layout(hovermode="closest")
    if horizontal:
        fig.update_xaxes(tickformat=".0%" if "%" in fmt else None, title_text=xtitle, gridcolor=theme.GRID)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    else:
        fig.update_yaxes(tickformat=".0%" if "%" in fmt else None)
    return fig


def heatmap(Z: pd.DataFrame, title: str | None = None, zmid: float | None = 0.0, fmt: str = ".2f", height: int | None = None,
            diverging: bool = True, zmin: float | None = None, zmax: float | None = None, colorbar_title: str | None = None) -> go.Figure:
    z = Z.to_numpy(dtype=float)
    text = np.where(np.isfinite(z), np.vectorize(lambda v: format(v, fmt))(np.nan_to_num(z)), "")
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in Z.columns], y=[str(i) for i in Z.index],
        colorscale=[[0, theme.NEG], [0.5, theme.MID], [1, theme.POS]] if diverging else [[0, "#EEF3F8"], [1, theme.ACCENT]],
        zmid=zmid if diverging else None, zmin=zmin, zmax=zmax, text=text, texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>", colorbar=dict(title=colorbar_title, thickness=12),
    ))
    h = height or max(320, 26 * len(Z) + 120)
    _base(fig, title, h, legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(side="top")
    return fig


def monthly_heatmap(m: pd.Series, title: str = "Monthly returns") -> go.Figure:
    df = m.to_frame("r")
    df["Year"], df["Month"] = df.index.year, df.index.month
    piv = df.pivot(index="Year", columns="Month", values="r").reindex(columns=range(1, 13))
    piv.columns = [pd.Timestamp(2000, k, 1).strftime("%b") for k in piv.columns]
    lim = float(np.nanpercentile(np.abs(piv.to_numpy()), 95)) if piv.notna().any().any() else 0.05
    return heatmap(piv, title, zmid=0.0, fmt="+.1%", zmin=-lim, zmax=lim, height=max(260, 30 * len(piv) + 110))


def scatter(df: pd.DataFrame, x: str, y: str, hover: list[str] | None = None, color: str | None = None, title: str | None = None,
            xfmt: str | None = None, yfmt: str | None = None, height: int = 460, text: str | None = None) -> go.Figure:
    fig = go.Figure()
    groups = [(None, df)] if color is None else list(df.groupby(color))
    for i, (g, d) in enumerate(groups):
        cd = d[hover].to_numpy() if hover else None
        tmpl = "<br>".join([f"{h}: %{{customdata[{k}]}}" for k, h in enumerate(hover)]) if hover else ""
        fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers+text" if text else "markers", name=str(g) if g is not None else y,
                                 text=d[text] if text else None, textposition="top center", customdata=cd,
                                 marker=dict(size=10, color=COL[i % len(COL)], line=dict(width=1, color="#FFFFFF")),
                                 hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<br>{tmpl}<extra>{g if g is not None else ''}</extra>"))
    _base(fig, title, height, yfmt, legend=color is not None)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickformat=xfmt, showgrid=True, gridcolor=theme.GRID, title_text=x)
    fig.update_yaxes(title_text=y)
    return fig


def calibration(cal: pd.DataFrame, title: str = "Calibration: forecast vs what happened") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color=theme.MUTED, dash="dot")))
    fig.add_trace(go.Scatter(x=cal["Forecast (avg)"], y=cal["Observed frequency"], mode="markers+lines", name="Model",
                             marker=dict(size=[max(8, min(26, c / 4)) for c in cal["Count"]], color=COL[0]), line=dict(color=COL[0]),
                             customdata=cal["Count"], hovertemplate="Forecast %{x:.0%} → observed %{y:.0%} (n=%{customdata})"))
    _base(fig, title, 380, ".0%")
    fig.update_xaxes(tickformat=".0%", title_text="Forecast probability of ‘Up’", range=[0, 1])
    fig.update_yaxes(title_text="How often it actually happened", range=[0, 1])
    fig.update_layout(hovermode="closest")
    return fig


def timeline(wealth: pd.DataFrame, dd: pd.DataFrame, alloc_w: pd.DataFrame, start=None, end=None) -> go.Figure:
    """Three aligned panels: growth, drawdown, allocation (used by the Market Map page)."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.38, 0.24, 0.38],
                        subplot_titles=["Growth of 1", "Drawdown from previous peak", "Portfolio weights"])
    for i, c in enumerate(wealth.columns):
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth[c], mode="lines", name=str(c), legendgroup=str(c), line=dict(color=COL[i], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd[c], mode="lines", name=str(c), legendgroup=str(c), showlegend=False, line=dict(color=COL[i], width=1.5)), row=2, col=1)
    for j, c in enumerate(alloc_w.columns):
        fig.add_trace(go.Scatter(x=alloc_w.index, y=alloc_w[c], mode="lines", name=str(c), stackgroup="alloc",
                                 line=dict(width=0.5, shape="hv"), hovertemplate="%{y:.1%}"), row=3, col=1)
    if start is not None and end is not None:
        for r in (1, 2, 3):
            fig.add_vrect(x0=start, x1=end, fillcolor="#EDA100", opacity=0.14, line_width=0, row=r, col=1)
    _base(fig, None, 900)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    return fig
