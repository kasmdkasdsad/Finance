"""Plotly figure builders following the terminal's data-viz rules.

* Categorical hues come from a validated palette in a fixed order (light and dark steps selected
  separately); scatter-type charts use at most three hues plus marker shape.
* Magnitude uses a single-hue blue ramp; signed quantities use blue↔red around a neutral gray.
* One y-axis per chart, hairline solid grids, 2px lines, 4px rounded bar ends, legends for ≥2 series.
* Every chart is paired with a table view in the page (colour is never the only channel).
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

LIGHT = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "muted": "#898781",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "neutral": "#f0efec",
    "surface": "#fcfcfb",
}
DARK = {
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    "grid": "#2c2c2a",
    "axis": "#383835",
    "muted": "#898781",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "neutral": "#383835",
    "surface": "#1a1a19",
}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_RED = ["#e34948", "#ec8a89", "#f5c7c6"]
DIVERGING_BLUE = ["#b7d3f6", "#6da7ec", "#2a78d6"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def theme() -> dict[str, Any]:
    try:
        is_dark = st.context.theme.type == "dark"
    except Exception:
        is_dark = False
    return DARK if is_dark else LIGHT


def series(i: int) -> str:
    return theme()["series"][i % 8]


def diverging_scale(neutral: str | None = None) -> list[list[Any]]:
    mid = neutral or theme()["neutral"]
    stops = [*DIVERGING_RED, mid, *DIVERGING_BLUE]
    n = len(stops) - 1
    return [[i / n, c] for i, c in enumerate(stops)]


def sequential_scale() -> list[list[Any]]:
    n = len(SEQUENTIAL_BLUE) - 1
    return [[i / n, c] for i, c in enumerate(SEQUENTIAL_BLUE)]


def base_layout(fig: go.Figure, title: str | None = None, height: int = 360, **kwargs: Any) -> go.Figure:
    t = theme()
    # Title sits in the top band; the legend gets its own row beneath it (left-aligned) so they never collide.
    layout: dict[str, Any] = {
        "title": {
            "text": title,
            "font": {"size": 15, "color": t["ink"]},
            "x": 0,
            "xanchor": "left",
            "xref": "paper",
            "y": 1.0,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 8},
        }
        if title
        else None,
        "height": height,
        "margin": {"l": 8, "r": 8, "t": 76 if title else 30, "b": 8},
        "font": {
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": t["ink2"],
            "size": 12,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "traceorder": "normal",
        },
        "hoverlabel": {"font": {"family": 'system-ui, -apple-system, "Segoe UI", sans-serif'}},
        "bargap": 0.18,
    }
    layout.update(kwargs)  # caller overrides win
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], tickcolor=t["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=t["grid"], gridwidth=1, zeroline=False, linecolor=t["axis"])
    return fig


def show(fig: go.Figure, key: str | None = None, toolbar: bool = True) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={"displaylogo": False, "responsive": True, "displayModeBar": "hover" if toolbar else False},
    )


def rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r,g,b,a)`` (for translucent bands of a single series hue)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def equity(points: list[tuple[Any, float]], title: str | None = None, height: int = 240) -> go.Figure:
    """An account's equity over time: one line over a soft gradient (green when it ended above where it
    started, red below), the y-axis fitted to the data rather than to zero."""
    xs = [p[0] for p in points]
    ys = [float(p[1]) for p in points]
    up = ys[-1] >= ys[0]
    color = STATUS["good"] if up else STATUS["critical"]
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.15, abs(hi) * 0.002, 1.0)
    floor = lo - pad
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[floor] * len(xs), mode="lines", line={"width": 0}, hoverinfo="skip",
                             showlegend=False))  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="Equity",
            line={"color": color, "width": 2.5, "shape": "spline", "smoothing": 0.6},
            fill="tonexty",
            fillgradient={
                "type": "vertical",
                "colorscale": [[0, rgba(color, 0.0)], [1, rgba(color, 0.22)]],
            },
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
            showlegend=False,
        )
    )
    base_layout(fig, title, height=height, margin={"l": 8, "r": 8, "t": 40 if title else 8, "b": 8})
    fig.update_yaxes(range=[floor, hi + pad], tickprefix="$", tickformat=",.0f")
    fig.update_layout(hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def reference_line(fig: go.Figure, y: float, label: str | None = None) -> None:
    """A solid hairline reference (zero, 50%, the diagonal's anchor) in muted ink, never dashed."""
    fig.add_hline(
        y=y,
        line_width=1,
        line_color=theme()["axis"],
        annotation_text=label,
        annotation_position="bottom right",
        annotation_font_color=theme()["muted"],
    )
