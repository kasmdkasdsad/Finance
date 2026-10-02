"""Market Evolution: structural changes (with competing explanations, none assumed), the history of the
relationships strategies rely on, and the versioned model registry."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend import ui
from frontend.components import api, guarded


def _table(rows: list[dict[str, Any]] | None, cols: list[str], empty: str) -> None:
    if not rows:
        st.caption(empty)
        return
    df = pd.DataFrame(rows)
    st.dataframe(df[[c for c in cols if c in df.columns]], width="stretch", hide_index=True)


def render() -> None:
    ui.header("Market evolution", "Structural changes in the market, each with its competing explanations.")
    status = guarded(lambda: api().get("/evolution/status"), "evolution status")
    if status is None:
        return
    ui.kpis(
        [
            ui.Kpi("Days measured", status.get("days_measured", 0)),
            ui.Kpi("Active changes", status.get("active_changes", 0)),
            ui.Kpi("Last day", status.get("last_day") or "—"),
        ],
        key="evolution_status",
    )
    if status.get("note"):
        st.caption(status["note"])
    tabs = st.tabs(["Changes", "Relationships", "Model registry"])
    with tabs[0]:
        st.caption(
            "Reported only after a false-discovery-rate control across everything measured; no explanation is "
            "assumed."
        )
        rows = guarded(lambda: api().get("/evolution/changes"), "changes") or []
        _table(rows, ["detected_at", "dimension", "subject", "metric", "timescale", "kind", "effect_sd", "q_value",
                      "persisted", "status"], "No structural change detected.")  # fmt: skip
        for r in rows[:10]:
            with st.expander(f"{r['subject']} {r['metric']} ({r['timescale']}): {r['kind']}"):
                st.write(r.get("summary"))
                _table(r.get("hypotheses"), ["name", "verdict", "detail", "identifiable_from_prices"], "—")
                if (r.get("revalidation") or {}).get("strategies"):
                    st.caption("Re-validated: " + ", ".join(r["revalidation"]["strategies"]))
    with tabs[1]:
        rels = guarded(lambda: api().get("/evolution/relationships"), "relationships") or []
        _table(rels, ["recorded_at", "key", "subject", "status", "slope", "r", "n", "z"],
               "No relationship estimated yet (needs enough measured days).")  # fmt: skip
    with tabs[2]:
        models = guarded(lambda: api().get("/registry/models"), "registry") or []
        st.caption("No model becomes authoritative on in-sample results: out-of-sample, walk-forward, stress and "
                   "a live shadow record that beats the champion — and a person's approval for AI models.")  # fmt: skip
        _table(
            models, ["id", "slot", "version", "kind", "name", "stage", "role", "approved_by"], "No models."
        )
