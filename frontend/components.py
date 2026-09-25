"""Shared UI pieces: provenance badges, number formatting and error boundaries."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

import streamlit as st

from frontend.api_client import ApiClient, ApiError, Pending

T = TypeVar("T")

# Status is always icon + label (never colour alone).
STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    "live": ("green", ":material/bolt:", "LIVE"),
    "cached": ("blue", ":material/cached:", "CACHED"),
    "stale": ("orange", ":material/history:", "STALE"),
    "synthetic": ("red", ":material/science:", "SYNTHETIC"),
}


@st.cache_resource
def _client(base_url: str, token: str | None) -> ApiClient:
    return ApiClient(base_url, token)


def api() -> ApiClient:
    return _client(
        st.session_state.get("api_url", ApiClient().base_url), st.session_state.get("api_token") or None
    )


def _age(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        moment = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ""
    seconds = max(0, int((datetime.now(UTC) - moment).total_seconds()))
    if seconds < 90:
        return f"{seconds}s ago"
    if seconds < 5400:
        return f"{seconds // 60}m ago"
    if seconds < 172800:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def status_badge(meta: dict[str, Any], label: str | None = None) -> None:
    """Render a provenance badge for an ``Envelope.meta`` (single source)."""
    color, icon, text = STATUS_STYLE.get(meta.get("status", "synthetic"), STATUS_STYLE["synthetic"])
    provider = meta.get("provider", "")
    if provider.lower() == text.lower():
        provider = ""
    parts = [p for p in (label, text, provider, _age(meta.get("as_of")) if meta.get("as_of") else None) if p]
    st.badge(" · ".join(parts), icon=icon, color=color, help=_help(meta))


def _help(meta: dict[str, Any]) -> str:
    lines = []
    if meta.get("message"):
        lines.append(meta["message"])
    for a in meta.get("attempts", []):
        mark = "✓" if a.get("ok") else "✗"
        lines.append(f"{mark} {a.get('provider')}: {a.get('error') or 'ok'}")
    return "\n\n".join(lines) or "Data provenance"


def composite_badges(meta: dict[str, Any]) -> None:
    """Overall badge for a ``CompositeMeta`` plus a per-source breakdown."""
    color, icon, text = STATUS_STYLE.get(meta.get("status", "synthetic"), STATUS_STYLE["synthetic"])
    sources = meta.get("sources", {})
    worst = [k for k, v in sources.items() if v.get("status") == meta.get("status")]
    st.badge(
        f"{text} · {len(sources)} sources",
        icon=icon,
        color=color,
        help="Worst source: " + ", ".join(worst[:6]),
    )
    with st.expander("Data sources", icon=":material/lan:"):
        for name, prov in sources.items():
            status_badge(prov, label=name)


def job_progress(job: dict[str, Any]) -> None:
    """Live progress for a background job; the page reruns itself when the job finishes."""

    @st.fragment(run_every=2)
    def poll() -> None:
        current = job
        with contextlib.suppress(ApiError):  # a restarted API forgets jobs: keep showing the last state
            current = api().get(f"/jobs/{job['id']}")
        if current["status"] == "done":
            st.rerun()
        elif current["status"] == "failed":
            st.error(f"{current['description']} failed: {current.get('error')}", icon=":material/error:")
            return
        st.progress(
            float(current["progress"]),
            text=f"{current['description']} · {current['progress']:.0%} · {current['stage']} "
            f"({current['elapsed_seconds']:.0f}s)",
        )

    st.info(
        "This runs in the background: the first run for a large universe downloads years of prices and SEC data, "
        "later ones take about a minute. The page updates by itself when it is ready.",
        icon=":material/hourglass_top:",
    )
    poll()


def guarded(fn: Callable[[], T], what: str = "request") -> T | None:
    """Run an API call and turn failures into an inline, actionable error (never a stack trace).
    A request the API is still computing (202) shows the job's live progress instead."""
    try:
        return fn()
    except Pending as pending:
        job_progress(pending.job)
        return None
    except ApiError as exc:
        if exc.status == 0:
            st.error(
                f"{exc.message}. Start it with `make api` (or `quantpulse-api`).", icon=":material/cloud_off:"
            )
        elif exc.status == 409:
            st.warning(exc.message, icon=":material/science:")
        else:
            st.error(f"{what.capitalize()} failed ({exc.status}): {exc.message}", icon=":material/error:")
        return None


def money(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(v)
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if v >= scale:
            return f"{sign}${v / scale:,.{digits}f}{unit}"
    return f"{sign}${v:,.{digits}f}"


def pct(v: float | None, digits: int = 2, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"


def num(v: float | None, digits: int = 2) -> str:
    return "—" if v is None else f"{v:,.{digits}f}"
