"""Research (24/7): what the Brain is doing now, whether execution is ready, its research queue and history, what
it has concluded (with the evidence — UNPROVEN until enough), and the improvement lifecycle with the promotion
decision that only a person makes."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend import ui
from frontend.components import api, guarded

STATUS_ICON = {"SUPPORTED": "🟢", "REFUTED": "🔴", "INCONCLUSIVE": "⚪", "UNPROVEN": "🟡"}


def _table(rows: list[dict[str, Any]] | None, cols: list[str], empty: str) -> None:
    if not rows:
        st.caption(empty)
        return
    df = pd.DataFrame(rows)
    st.dataframe(df[[c for c in cols if c in df.columns]], width="stretch", hide_index=True)


def _operating(status: dict[str, Any]) -> None:
    op = status.get("operating") or {}
    mode = op.get("mode", "?")
    ready = op.get("readiness")
    if ready is None:
        ui.status(
            "gray",
            "Execution readiness has not been checked today",
            "It runs from 08:30 New York, and before the first order.",
        )
    elif ready.get("passed"):
        ui.status("green", f"Execution readiness passed ({ui.when(str(ready.get('at')))})")
    else:
        st.error("Execution readiness HELD: " + ", ".join(ready.get("failed") or []))
    q = status.get("queue") or {}
    ui.kpis(
        [
            ui.Kpi("Mode", str(mode).replace("_", " ").capitalize()),
            ui.Kpi("Research running", len(status.get("running") or [])),
            ui.Kpi("Queued", q.get("queued", 0)),
            ui.Kpi("Done", q.get("done", 0)),
        ],
        key="research_status",
    )
    st.caption("Loop: " + " → ".join(op.get("loop") or []))
    if ready:
        with st.expander("Readiness checks", icon=":material/checklist:"):
            _table(ready.get("steps"), ["step", "ok", "required", "detail"], "—")
    res = status.get("resources") or {}
    lim = res.get("limits") or {}
    st.caption(  # the server's resources, small print
        f"Resources: memory {res.get('memory_pct')}% ({res.get('memory_source')}), load {res.get('load_per_cpu')}/core, "
        f"this process {res.get('rss_mb')} MB · new jobs start below {lim.get('start_below_memory_pct')}% memory, "
        f"running jobs stop above {lim.get('stop_above_memory_pct')}% · at most {lim.get('max_concurrent')} at once · "
        f"{res.get('detail')}"
    )


def _learnings() -> None:
    status = st.selectbox(
        "Status", ["all", "SUPPORTED", "REFUTED", "INCONCLUSIVE", "UNPROVEN"], key="research_status"
    )
    params = {} if status == "all" else {"status": status}
    rows = guarded(lambda: api().get("/brain/research/learnings", **params), "learnings") or []
    st.caption(
        "Every conclusion carries its sample size, period, regime, benchmark, test, confidence and limitations. "
        "Its status is computed from that evidence: UNPROVEN until the sample reaches the minimum and a test supports it."
    )
    for r in rows[:100]:
        icon = STATUS_ICON.get(r["status"], "")
        with st.expander(f"{icon} {r['status']} · {r['claim']} (n={r['sample_size']}/{r['min_sample']})"):
            period = r.get("period") or {}
            st.write(
                f"**Topic** {r['topic']} · **period** {period.get('start')} → {period.get('end')} · **regime** "
                f"{r['regime']} · **benchmark** {r['benchmark']}"
            )
            st.write(f"**Method** {r['method']} · **confidence** {r['confidence']}")
            st.json(r.get("statistics") or {}, expanded=False)
            st.write("**Limitations**")
            for lim in r.get("limitations") or []:
                st.write(f"- {lim}")


def _hypotheses() -> None:
    rows = guarded(lambda: api().get("/brain/research/hypotheses"), "hypotheses") or []
    st.caption(
        "DISCOVERED → HYPOTHESIS → BACKTEST → WALK_FORWARD → STRESS_TEST → PAPER_SHADOW → EVALUATION → "
        "(a person promotes) → PRODUCTION. One tested stage at a time; protected controls are never a subject."
    )
    _table(rows, ["id", "kind", "title", "stage", "next_stage", "protected_control", "updated_at"],
           "No hypothesis yet: research discovers them while the market is closed.")  # fmt: skip
    waiting = [r for r in rows if r.get("awaiting_person")]
    if not waiting:
        return
    st.subheader("Awaiting your decision")
    for r in waiting:
        with st.expander(f"#{r['id']} {r['title']}"):
            st.json(r.get("history") or [], expanded=False)
            by = st.text_input("Your name", key=f"by_{r['id']}")
            note = st.text_area("Why the evidence is (or is not) enough", key=f"note_{r['id']}")
            a, b = st.columns(2)
            base = f"/brain/research/hypotheses/{r['id']}"
            decision = {"by": by, "note": note}
            if a.button("Promote to production", key=f"promote_{r['id']}", disabled=not (by and note)):
                out = guarded(lambda p=f"{base}/promote", d=decision: api().post(p, d), "promote")
                if out:
                    st.success(f"Promoted: {out['stage']}")
            if b.button("Reject", key=f"reject_{r['id']}", disabled=not (by and note)):
                out = guarded(lambda p=f"{base}/reject", d=decision: api().post(p, d), "reject")
                if out:
                    st.warning("Rejected")


def _queue() -> None:
    tabs = st.tabs(["Running & queued", "History"])
    with tabs[0]:
        running = guarded(lambda: api().get("/brain/research/jobs", status="running"), "running") or []
        queued = guarded(lambda: api().get("/brain/research/jobs", status="queued", limit=50), "queued") or []
        _table(running, ["id", "kind", "question", "started_at", "attempts"], "Nothing running.")
        _table(
            queued, ["id", "kind", "question", "priority", "source", "not_before", "attempts"], "Queue empty."
        )
    with tabs[1]:
        history = guarded(lambda: api().get("/brain/research/jobs", limit=100), "history") or []
        _table([h for h in history if h["status"] not in ("queued", "running")],
               ["id", "kind", "status", "finished_at", "duration_ms", "peak_rss_mb", "error"], "No finished job yet.")  # fmt: skip
        for h in history[:20]:
            if h.get("result"):
                with st.expander(f"#{h['id']} {h['kind']} — {h['status']}"):
                    st.json(h["result"], expanded=False)
    catalog = guarded(lambda: api().get("/brain/research/catalog"), "catalog") or []
    with st.expander("Ask the Brain a question"):
        kinds = [c["kind"] for c in catalog]
        kind = st.selectbox("Research job", kinds, format_func=lambda k: next(
            (f"{c['phase']}: {c['question']}" for c in catalog if c["kind"] == k), k))  # fmt: skip
        if st.button("Queue it"):
            out = guarded(lambda: api().post("/brain/research/questions", {"kind": kind}), "ask")
            if out:
                st.success(f"Queued as #{out['id']} (priority {out['priority']})")


def render() -> None:
    ui.header(
        "Research",
        "While the market is closed the Brain grades, analyses and tests ideas. Research never sends an order or "
        "changes a setting, a limit or a strategy in production.",
    )
    status = guarded(lambda: api().get("/brain/research/status"), "research status")
    if status is None:
        return
    _operating(status)
    tabs = st.tabs(["What it learned", "Improvements", "Queue"])
    with tabs[0]:
        _learnings()
    with tabs[1]:
        _hypotheses()
    with tabs[2]:
        _queue()
