"""System: provider health, circuit breakers, cache, rate limiters, poller and ingestion log."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import ui
from frontend.components import api, guarded


def render() -> None:
    ui.header("System status", "Data providers, background jobs and the server's internals.")
    s = guarded(lambda: api().get("/system/status"), "status")
    if not s:
        return
    db = s["database"]
    ui.kpis(
        [
            ui.Kpi("Version", s["version"]),
            ui.Kpi("Live data", "enabled" if s["live_data_enabled"] else "disabled"),
            ui.Kpi("Market session", str(s["market_session"]).replace("_", " ")),
            ui.Kpi(
                "DB schema",
                db["revision"],
                "up to date" if db["revision"] == db["head"] else f"head is {db['head']}",
                delta_color="off" if db["revision"] == db["head"] else "inverse",
                arrow="off",
            ),
        ],
        key="system_status",
    )
    with st.expander("Credentials configured", icon=":material/key:"):
        st.dataframe(pd.DataFrame([s["credentials"]]), hide_index=True)
    ui.section("Providers")
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
    ui.section("Background jobs")
    jobs = guarded(lambda: api().get("/jobs"), "jobs")
    if jobs:
        st.dataframe(
            pd.DataFrame(jobs)[
                ["id", "description", "status", "progress", "stage", "elapsed_seconds", "error"]
            ],
            hide_index=True,
            column_config={
                "progress": st.column_config.ProgressColumn(
                    "progress", min_value=0.0, max_value=1.0, format="percent"
                )
            },
        )
    elif jobs is not None:
        st.caption("No model runs or replays have been started since the API started.")
    with st.expander("Cache, rate limiters and the background poller", icon=":material/memory:"):
        a, b = st.columns(2)
        with a:
            st.markdown("**Cache**")
            st.json(s["cache"], expanded=False)
            st.markdown("**Rate limiters**")
            st.dataframe(pd.DataFrame(s["rate_limiters"].values()), hide_index=True)
        with b:
            st.markdown("**Background poller**")
            st.json(s["poller"], expanded=False)
    ui.section("Recent ingestions")
    events = guarded(lambda: api().get("/system/ingestions", limit=25), "ingestions")
    if events:
        st.dataframe(pd.DataFrame(events), hide_index=True)
    elif events is not None:
        st.caption("No live data has been written to the warehouse yet.")
