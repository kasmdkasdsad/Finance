"""The dashboard's building blocks: one header, KPI cards, a status line, quiet facts and sections, so every
page looks and reads the same on a desktop and on a phone. Text from the API is always HTML-escaped."""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import streamlit as st

STYLE = Path(__file__).with_name("style.css")


def inject_css() -> None:
    """The stylesheet: sent once per run to the page's event container (it takes no space)."""
    st.html(STYLE)


def md(text: Any) -> str:
    """Escape ``$``: Streamlit Markdown renders text between two dollars as a formula."""
    return str(text).replace("$", "\\$")


def header(title: str, subtitle: str | None = None) -> None:
    """The page title with the PAPER pill on the same line, and one muted sentence under it."""
    with st.container(key="qp_header", horizontal=True, vertical_alignment="center", gap="small"):
        st.title(title, anchor=False, width="content")
        st.badge(
            "Paper", icon=":material/science:", color="orange", help="Alpaca paper account: simulated money"
        )
    if subtitle:
        st.caption(md(subtitle))


class Kpi(NamedTuple):
    label: str
    value: Any
    delta: Any = None
    help: str | None = None
    delta_color: str = "normal"
    arrow: str = "auto"  # "off" for a delta that is a word, not a change


def kpis(items: Sequence[Kpi], key: str) -> None:
    """A row of bordered KPI cards; two per line on a phone."""
    with st.container(key=f"kpis_{key}"):
        cols = st.columns(len(items))
        for col, k in zip(cols, items, strict=True):
            col.metric(
                k.label,
                k.value,
                k.delta,
                delta_color=k.delta_color,  # type: ignore[arg-type]
                delta_arrow=k.arrow,  # type: ignore[arg-type]
                help=k.help,
                border=True,
            )


def status(light: str, headline: str, detail: str | None = None) -> None:
    """One sentence with a coloured dot: green (all good), yellow (attention), red (stopped or broken)."""
    tone = light if light in ("green", "yellow", "red") else "gray"
    small = f"<small>{html.escape(detail)}</small>" if detail else ""
    st.html(
        f'<div class="qp-status qp-{tone}" role="status"><span class="qp-dot"></span>'
        f"<div>{html.escape(headline)}{small}</div></div>"
    )


def facts(items: Iterable[tuple[str, Any]]) -> None:
    """Small label/value pairs in a grid: the details nobody needs in large type."""
    cells = "".join(
        f"<div><span>{html.escape(str(label))}</span>{html.escape(str(value))}</div>"
        for label, value in items
        if value not in (None, "")
    )
    if cells:
        st.html(f'<div class="qp-facts">{cells}</div>')


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title, anchor=False)
    if caption:
        st.caption(md(caption))


def when(ts: str | None) -> str:
    """``2026-10-01T14:05:09Z`` → ``Oct 1, 14:05 UTC``."""
    if not ts:
        return "—"
    try:
        moment = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts[:16].replace("T", " ")
    return f"{moment:%b} {moment.day}, {moment:%H:%M} UTC"


def ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    s = int(seconds)
    if s < 90:
        return f"{s}s ago"
    if s < 5400:
        return f"{s // 60} min ago"
    if s < 172800:
        return f"{s // 3600} h ago"
    return f"{s // 86400} d ago"
