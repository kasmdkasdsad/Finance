"""System: provider health, circuit breakers, cache, rate limiters, poller and ingestion log."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components import api, guarded


def render() -> None:
    st.title("System Status")
    s = guarded(lambda: api().get("/system/status"), "status")
    if not s:
        return
    c = st.columns(4)
    c[0].metric("Version", s["version"])
    c[1].metric("Live data", "enabled" if s["live_data_enabled"] else "disabled")
    c[2].metric("Market session", s["market_session"])
    db = s["database"]
    c[3].metric(
        "DB schema",
        db["revision"],
        "up to date" if db["revision"] == db["head"] else f"head is {db['head']}",
        delta_color="off" if db["revision"] == db["head"] else "inverse",
    )
    st.subheader("Credentials configured")
    st.dataframe(pd.DataFrame([s["credentials"]]), hide_index=True)
    st.subheader("Providers")
    rows = []
    for name, p in s["gateway"]["providers"].items():
        br = p["breaker"]
        rows.append(
            {
                "provider": name,
                "breaker": br["state"],
                "retry_in_s": br["retry_in_sec"],
                "calls": p["calls"],
                "ok": p["successes"],
                "failed": p["failures"],
                "avg_ms": p["avg_latency_ms"],
                "last_success": p["last_success_at"],
                "last_error": p["last_error"],
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info(
            "No provider has been called yet"
            + ("" if s["live_data_enabled"] else " — live data is disabled (QP_ENABLE_LIVE_DATA=false)")
            + "."
        )
    a, b = st.columns(2)
    with a:
        st.subheader("Cache")
        st.json(s["cache"])
        st.subheader("Rate limiters")
        st.dataframe(pd.DataFrame(s["rate_limiters"].values()), hide_index=True)
    with b:
        st.subheader("Background poller")
        st.json(s["poller"])
        st.subheader("Recent ingestions")
        events = guarded(lambda: api().get("/system/ingestions", limit=25), "ingestions")
        if events:
            st.dataframe(pd.DataFrame(events), hide_index=True)
        elif events is not None:
            st.caption("No live data has been written to the warehouse yet.")
