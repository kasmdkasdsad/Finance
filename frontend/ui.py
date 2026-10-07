"""The dashboard's building blocks: one header, KPI cards, a status line, quiet facts and sections, so every
page looks and reads the same on a desktop and on a phone. Text from the API is always HTML-escaped."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

STYLE = Path(__file__).with_name("style.css")
ET = ZoneInfo("America/New_York")  # every time on the dashboard is New York time, the market's own clock
TIME_FORMAT = "MMM D, h:mm A"  # tables (momentJS)


def inject_css() -> None:
    """The stylesheet: sent once per run to the page's event container (it takes no space), plus the current
    theme's background for the phone tab bar (Streamlit exposes no CSS variable for it)."""
    st.html(STYLE)
    theme = getattr(st.context, "theme", None)
    dark = getattr(theme, "type", None) == "dark"
    bg, line = ("#0B0F17", "#262E3B") if dark else ("#FFFFFF", "#E3E7ED")
    st.html(f"<style>:root {{ --qp-bg: {bg}; --qp-line: {line}; }}</style>")


def link(path: str, label: str) -> None:
    """A link to another page; nothing when that page is not in this session's navigation (frontend/dashboard.py
    registers its pages by URL path in ``st.session_state["qp_pages"]``)."""
    target = (st.session_state.get("qp_pages") or {}).get(path)
    if target is not None:
        st.page_link(target, label=label, icon=":material/arrow_forward:")


def tab_bar(pages: Sequence[Any]) -> None:
    """The main pages as a tab bar fixed to the bottom of a phone's screen (hidden on wider screens)."""
    with st.container(key="qp_tabbar", horizontal=True):
        for p in pages:
            st.page_link(p)


def plain(text: Any) -> str:
    """Drop the setting names in brackets, e.g. "(QP_TRADING_SCHEDULER_REQUIRES_ARMING)": they mean nothing to
    a reader; the documentation and System have them."""
    return re.sub(r"\s*\((?:set )?QP_[^)]*\)", "", str(text)).strip()


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


class Step(NamedTuple):
    label: str
    n: int
    note: str = ""
    live: bool = False  # a stage that trades (shadow or paper): drawn in the accent colour


def ladder(steps: Sequence[Step]) -> None:
    """Stages left to right, each with how many items stand there; filled stages stand out, and the stages
    that trade are drawn in the accent colour. Wraps onto several lines on a phone."""
    cells = "".join(
        f'<div class="qp-step{" qp-step-live" if s.live else ""}{" qp-step-on" if s.n else ""}">'
        f"<b>{int(s.n)}</b><span>{html.escape(s.label)}</span>"
        + (f"<small>{html.escape(s.note)}</small>" if s.note else "")
        + "</div>"
        for s in steps
    )
    st.html(f'<div class="qp-ladder" role="list">{cells}</div>')


def eastern(ts: str | datetime | None) -> datetime | None:
    """An API time (ISO 8601, UTC unless it says otherwise) in New York time; ``None`` if there is none."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        moment = ts
    else:
        try:
            moment = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(ET)


def when(ts: str | datetime | None, seconds: bool = False) -> str:
    """``2026-10-01T14:05:09Z`` → ``Oct 1, 10:05 AM ET`` (Eastern time, daylight saving included)."""
    m = eastern(ts)
    if m is None:
        return str(ts)[:16].replace("T", " ") if ts else "—"
    clock = f"{m.hour % 12 or 12}:{m:%M}" + (f":{m:%S}" if seconds else "")
    return f"{m:%b} {m.day}, {clock} {m:%p} ET"


def day(ts: str | datetime | None) -> str:
    """The New York date of an API time: ``2026-10-02T01:30:00Z`` → ``2026-10-01``."""
    m = eastern(ts)
    return m.date().isoformat() if m is not None else (str(ts)[:10] if ts else "—")


def et_times(df: pd.DataFrame, columns: Iterable[str] | None = None) -> dict[str, Any]:
    """Convert a table's time columns (``at`` and ``…_at``, or ``columns``) to New York time in place.
    Returns their column config, labelled "(ET)"; they stay real times, so they still sort."""
    cols = (
        list(columns)
        if columns is not None
        else [c for c in df.columns if c == "at" or str(c).endswith("_at")]
    )
    config: dict[str, Any] = {}
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601").dt.tz_convert(ET)
            label = "When" if c == "at" else str(c).removesuffix("_at").replace("_", " ").capitalize()
            config[c] = st.column_config.DatetimeColumn(f"{label} (ET)", format=TIME_FORMAT)
    return config


def since(ts: str | None) -> str:
    """``2026-10-01T14:05:09Z`` → ``12 min ago``."""
    if not ts:
        return "never"
    try:
        moment = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return ago(max(0.0, (datetime.now(UTC) - moment).total_seconds()))


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
