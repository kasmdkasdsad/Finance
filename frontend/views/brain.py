"""The Brain: what QuantPulse's agents see, think and propose — and what the risk engine says about it.

Four layers are kept visibly apart on this page:

1. **Agent analysis** — each agent's own opinion (stance, confidence, evidence, what would prove it wrong);
2. **Consensus** — the agents' views combined per subject, with the disagreement kept visible;
3. **Risk preview** — the deterministic risk engine's verdict on each proposed trade;
4. **Broker execution** — none: the Brain never sends an order. Orders only come from the Paper Trading
   (Alpaca) page, through its own risk checks and order manager.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.components import api, guarded, money, num, pct

BASE = "/brain"
STANCE_ICON = {"bullish": "▲", "bearish": "▼", "neutral": "■", "abstain": "·"}
STATUS_LABEL = {
    "recommended": "Risk engine: would allow (recommendation only)",
    "dry_run_approved": "Risk engine: would allow (dry run)",
    "risk_rejected": "Risk engine: rejected",
    "blocked": "Blocked by a data veto",
    "not_checked": "Not checked (account unreadable)",
    "no_trade": "No trade proposed",
}
QUALITY_COLOR = {
    "fresh": "green",
    "live": "green",
    "market_closed": "gray",
    "stale": "orange",
    "unavailable": "red",
    "provider_error": "red",
}


def _md(text: str) -> str:
    return str(text).replace("$", "\\$")


def _stance(value: str) -> str:
    return f"{STANCE_ICON.get(value, '')} {value}"


def _layers_banner() -> None:
    st.info(
        "**How to read this page.** ① *Agent analysis* is what each specialist concludes on its own. "
        "② *Consensus* combines them and shows where they disagree. ③ *Risk preview* is the deterministic risk "
        "engine's verdict on each proposed trade — the same checks that guard real orders. ④ *Broker execution*: "
        "**none** — the Brain never sends an order; orders only come from the Paper Trading (Alpaca) page.",
        icon=":material/psychology:",
    )


def _status(status: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Mode", status["mode"].replace("_", " "))
    agents = status["agents"]
    cols[1].metric("Agents enabled", f"{agents['enabled']} / {agents['registered']}")
    last = status.get("last_cycle")
    cols[2].metric("Last cycle", f"#{last['id']} · {last['status']}" if last else "none yet")
    cols[3].metric("Open predictions", status["open_predictions"])
    cols[4].metric("Orders sent by the Brain", 0)
    st.caption(f"Orders: {status['orders']}. Learning: {status['learning']}.")


def _run_controls() -> None:
    with st.form("brain-run", border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        symbols = c1.text_input(
            "Also study these symbols (optional)",
            placeholder="NVDA, MSFT",
            help="Holdings and the best of a quick pre-screen are always studied.",
        )
        kind = c2.selectbox("Kind", ["full", "portfolio", "deep"], index=0)
        go = c3.form_submit_button("Run a cycle now", icon=":material/play_arrow:", use_container_width=True)
    if go:
        body = {"kind": kind, "symbols": [s.strip().upper() for s in symbols.split(",") if s.strip()][:25]}
        out = guarded(lambda: api().post(f"{BASE}/run", body, wait=60), "brain cycle")
        if out:
            st.session_state["brain_cycle_id"] = out["id"]
            st.success(
                f"Cycle #{out['id']} {out['status']}: {out['summary'].get('trades_proposed', 0)} trades proposed, "
                "0 orders sent.",
                icon=":material/check_circle:",
            )


def _overview(cycle: dict[str, Any]) -> None:
    summary = cycle.get("summary") or {}
    regime = cycle.get("regime") or {}
    market = cycle.get("market") or {}
    c = st.columns(4)
    c[0].metric("Regime", (regime.get("label") or "unknown").replace("_", " "))
    c[1].metric("Market", "open" if market.get("open") else "closed", help=f"clock: {market.get('clock')}")
    c[2].metric("Session", cycle["session"].replace("_", " "))
    c[3].metric("Duration", f"{(cycle.get('duration_ms') or 0) / 1000:.1f}s")
    if regime.get("description"):
        st.markdown(_md(f"**{regime['description']}.** " + "; ".join(regime.get("reasons") or [])))
    c = st.columns(6)
    c[0].metric("Agents run", summary.get("agents_run", 0))
    c[1].metric("Failed", summary.get("agents_failed", 0))
    c[2].metric("Skipped", summary.get("agents_skipped", 0))
    c[3].metric("Subjects", summary.get("subjects", 0))
    c[4].metric('"I don\'t know"', summary.get("unknown", 0))
    c[5].metric("Disagreements", summary.get("disagreements", 0))
    c = st.columns(4)
    c[0].metric("Trades proposed", summary.get("trades_proposed", 0))
    c[1].metric("Risk engine would allow", summary.get("risk_approved", 0))
    c[2].metric("Predictions recorded", summary.get("predictions_recorded", 0))
    c[3].metric("Orders sent", summary.get("orders_sent", 0))

    left, right = st.columns(2)
    with left:
        st.subheader("Portfolio (Alpaca paper, read only)")
        pf = cycle.get("portfolio") or {}
        if not pf.get("available"):
            st.warning(_md(f"Paper account unavailable: {pf.get('error')}"), icon=":material/cloud_off:")
        else:
            st.markdown(
                f"Equity **{money(pf.get('equity'))}** · cash {money(pf.get('cash'))} · "
                f"{len(pf.get('positions') or {})} positions · {pf.get('open_orders', 0)} open orders"
            )
            rows = [{"symbol": s, **p} for s, p in (pf.get("positions") or {}).items()]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            cons = pf.get("constraints") or {}
            if cons:
                st.caption(
                    f"Exposure {pct(cons.get('exposure'), 1)} · spendable cash {money(cons.get('spendable_cash'))} · "
                    f"free slots {cons.get('free_slots')} · beta {num(cons.get('beta'))}"
                )
    with right:
        st.subheader("Data quality")
        dq = cycle.get("data_quality") or {}
        market_view = dq.get("market") or {}
        if market_view.get("veto"):
            st.warning(_md(f"Market-wide veto: {market_view['veto']}"), icon=":material/block:")
        elif market_view:
            st.success(_md(market_view.get("thesis", "")), icon=":material/verified:")
        states = dq.get("states") or {}
        if states:
            counts = pd.Series(states).value_counts()
            for state, n in counts.items():
                st.badge(f"{state}: {n}", color=QUALITY_COLOR.get(str(state), "gray"))
        for source, err in (dq.get("provider_errors") or {}).items():
            st.caption(_md(f"✗ {source}: {err}"))
    if cycle.get("focus"):
        st.subheader("What it studied")
        st.dataframe(pd.DataFrame(cycle["focus"]), hide_index=True, use_container_width=True)
    for note in cycle.get("notes") or []:
        st.caption(_md(f"• {note}"))


def _agents(cycle: dict[str, Any], agents: list[dict[str, Any]]) -> None:
    st.markdown("**① Agent analysis** — who ran this cycle, who was skipped and why.")
    this_cycle = {a["agent_id"]: a for a in cycle.get("agents") or []}
    rows = []
    for a in agents:
        run = this_cycle.get(a["id"], {})
        runs = a.get("runs") or {}
        rows.append(
            {
                "agent": a["name"],
                "id": a["id"],
                "role": (a.get("spec") or {}).get("role"),
                "enabled": a["enabled"],
                "this cycle": run.get("status", "—"),
                "why / error": run.get("reason") or run.get("error") or "",
                "opinions": run.get("opinions", 0),
                "runs": runs.get("runs", 0),
                "failures": runs.get("failures", 0),
                "avg ms": round(runs.get("avg_ms") or 0, 1),
                "track record": "measured"
                if a.get("performance")
                else "unproven (no evaluated predictions yet)",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Switch an agent on or off", icon=":material/toggle_on:"):
        ids = [a["id"] for a in agents]
        pick = st.selectbox("Agent", ids, key="brain-agent-pick")
        current = next((a["enabled"] for a in agents if a["id"] == pick), True)
        clicked = st.button("Disable" if current else "Enable", key="brain-agent-toggle")
        if clicked and guarded(
            lambda: api().post(f"{BASE}/agents/{pick}", {"enabled": not current}), "switch"
        ):
            st.rerun()
    with st.expander("What each agent does", icon=":material/info:"):
        for a in agents:
            spec = a.get("spec") or {}
            st.markdown(
                _md(
                    f"**{a['name']}** (`{a['id']}`, {spec.get('role')}, horizon {spec.get('horizon_days')}d) — "
                    f"{spec.get('description', '')}"
                )
            )


def _consensus(cycle: dict[str, Any]) -> None:
    st.markdown(
        "**② Consensus** — agents' forecasts combined per subject (weight = confidence × data quality × "
        "measured reliability; every agent is *unproven* until its predictions are graded)."
    )
    items = cycle.get("consensus") or []
    if not items:
        st.info("No consensus this cycle.")
        return
    rows = [
        {
            "subject": c["subject"],
            "view": "I don't know" if c["unknown"] else _stance(c["stance"]),
            "score": round(c["score"], 3),
            "confidence": round(c["confidence"], 3),
            "supporting": c["supporting"],
            "neutral": c["neutral"],
            "opposing": c["opposing"],
            "abstaining": c["abstaining"],
            "disagreement": round(c["disagreement"], 2),
            "data": c["data_quality"],
            "vetoes": "; ".join(v["reason"] for v in c.get("vetoes") or []),
        }
        for c in items
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    subject = st.selectbox("Look inside one subject", [c["subject"] for c in items], key="brain-subject")
    chosen = next(c for c in items if c["subject"] == subject)
    detail = chosen.get("detail") or {}
    dispute = detail.get("primary_disagreement")
    if dispute:
        left, right = st.columns(2)
        left.success(
            _md(f"**For** ({dispute['for']['agent_id']}): {dispute['for']['thesis']}"),
            icon=":material/thumb_up:",
        )
        right.error(
            _md(f"**Against** ({dispute['against']['agent_id']}): {dispute['against']['thesis']}"),
            icon=":material/thumb_down:",
        )
    votes = detail.get("votes") or []
    if votes:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "agent": v["agent_id"],
                        "stance": _stance(v["stance"]),
                        "score": round(v["score"], 3),
                        "confidence": round(v["confidence"], 3),
                        "weight": round(v["weight"], 3),
                        "reliability": f"{v['reliability']['status']} (n={v['reliability']['n']})",
                        "thesis": v["thesis"],
                    }
                    for v in votes
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    for reason in chosen.get("reasons") or []:
        st.caption(_md(f"• {reason}"))
    _opinions(cycle, subject)


def _opinions(cycle: dict[str, Any], subject: str) -> None:
    ops = [o for o in cycle.get("opinions") or [] if o["subject"] == subject]
    if not ops:
        return
    st.markdown(f"**① What each agent concluded about {subject}**")
    for o in ops:
        label = f"{o['agent_id']} · {_stance(o['stance'])} · confidence {o['confidence']:.2f} · data {o['data_quality']}"
        with st.expander(label):
            st.markdown(_md(o["thesis"]))
            if o.get("veto"):
                st.warning(_md(f"Veto: {o['veto']}"), icon=":material/block:")
            if o.get("invalidation"):
                st.caption(_md(f"Would be proven wrong by: {o['invalidation']}"))
            if o.get("evidence"):
                st.dataframe(pd.DataFrame(o["evidence"]), hide_index=True, use_container_width=True)
            if o.get("data_missing"):
                st.caption(_md("Missing: " + ", ".join(o["data_missing"])))


def _decisions(cycle: dict[str, Any]) -> None:
    decisions = cycle.get("decisions") or []
    if not decisions:
        st.info(
            "No proposed actions (research-only mode, or nothing to do).", icon=":material/do_not_disturb_on:"
        )
        return
    st.markdown(
        "**Proposed portfolio actions → ③ risk preview → ④ execution.** The risk preview is the deterministic "
        'risk engine\'s answer; *execution is always "not sent"* — the Brain has no order access.'
    )
    rows = [
        {
            "subject": d["subject"],
            "action": d["action"].upper(),
            "quantity": d.get("quantity"),
            "est. price": d.get("est_price"),
            "notional": d.get("notional"),
            "weight now → target": f"{pct(d.get('current_weight'), 1)} → {pct(d.get('target_weight'), 1)}",
            "confidence": round(d.get("confidence") or 0, 2),
            "③ risk preview": STATUS_LABEL.get(d["status"], d["status"]),
            "④ execution": "not sent (the Brain never sends orders)",
            "why": "; ".join((d.get("rationale") or {}).get("reasons") or []),
        }
        for d in decisions
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    trades = [d for d in decisions if d.get("risk")]
    for d in trades:
        risk = d["risk"]
        mark = "✓" if risk.get("approved") else "✗"
        with st.expander(
            f"{mark} {d['action'].upper()} {d.get('quantity')} {d['subject']} — {risk.get('summary')}"
        ):
            blocked = (d.get("rationale") or {}).get("blocked_by") or []
            if blocked:
                st.warning(_md("Blocked by: " + "; ".join(blocked)), icon=":material/block:")
            checks = risk.get("checks") or []
            if checks:
                st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)


def _memory() -> None:
    tier = st.segmented_control(
        "Memory tier", ["short_term", "working", "long_term", "strategy", "agent"], default="long_term"
    )
    rows = guarded(lambda: api().get(f"{BASE}/memory", tier=tier, limit=100), "memory") or []
    if not rows:
        st.info("Nothing remembered in this tier yet.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "when": r["updated_at"],
                    "kind": r["kind"],
                    "subject": r["subject"],
                    "summary": r["summary"],
                    "importance": r["importance"],
                    "cycle": r["cycle_id"],
                }
                for r in rows
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _history(cycles: list[dict[str, Any]]) -> None:
    if not cycles:
        st.info("No cycles yet.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "cycle": c["id"],
                    "started": c["started_at"],
                    "kind": c["kind"],
                    "trigger": c["trigger"],
                    "session": c["session"],
                    "mode": c["mode"],
                    "status": c["status"],
                    "regime": (c.get("regime") or {}).get("label"),
                    "trades proposed": (c.get("summary") or {}).get("trades_proposed"),
                    "risk would allow": (c.get("summary") or {}).get("risk_approved"),
                    "unknown": (c.get("summary") or {}).get("unknown"),
                    "orders sent": 0,
                }
                for c in cycles
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def render() -> None:
    st.title("QuantPulse Brain", anchor=False)
    st.caption("Specialist agents · consensus · proposals checked by the risk engine · Alpaca PAPER only")
    _layers_banner()
    status = guarded(lambda: api().get(f"{BASE}/status"), "brain status")
    if status is None:
        return
    _status(status)
    _run_controls()
    cycles = guarded(lambda: api().get(f"{BASE}/cycles", limit=50), "cycle history") or []
    if not cycles:
        st.info("The Brain has not run yet. Run a cycle above.", icon=":material/lightbulb:")
        return
    ids = [c["id"] for c in cycles]
    wanted = st.session_state.get("brain_cycle_id")
    cycle_id = st.selectbox(
        "Cycle",
        ids,
        index=ids.index(wanted) if wanted in ids else 0,
        format_func=lambda i: next(
            f"#{c['id']} · {c['started_at'][:16].replace('T', ' ')} UTC · {c['kind']} · {c['status']}"
            for c in cycles
            if c["id"] == i
        ),
    )
    cycle = guarded(lambda: api().get(f"{BASE}/cycles/{cycle_id}"), "cycle")
    if cycle is None:
        return
    if cycle["status"] == "failed":
        st.error(_md(f"This cycle failed: {cycle.get('error')}"), icon=":material/error:")
    agents = guarded(lambda: api().get(f"{BASE}/agents"), "agents") or []
    tabs = st.tabs(["Overview", "Agents", "Consensus & opinions", "Proposed actions", "Memory", "History"])
    with tabs[0]:
        _overview(cycle)
    with tabs[1]:
        _agents(cycle, agents)
    with tabs[2]:
        _consensus(cycle)
    with tabs[3]:
        _decisions(cycle)
    with tabs[4]:
        _memory()
    with tabs[5]:
        _history(cycles)
