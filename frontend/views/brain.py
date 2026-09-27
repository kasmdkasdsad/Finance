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
    "invalid": "red",
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
    st.caption(
        _md(
            f"Orders: {status['orders']}. Learning: {status['learning']}. "
            f"Language models: {status.get('language_models', 'unknown')}."
        )
    )


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
    situation = market.get("situation") or {}
    if situation:
        posture = situation.get("posture", "normal")
        color = {"normal": "green", "cautious": "orange", "defensive": "red"}.get(posture, "gray")
        st.badge(f"Risk posture: {posture}", icon=":material/shield:", color=color)
        if situation.get("reasons"):
            st.caption(_md("Why: " + "; ".join(situation["reasons"])))
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
        feed = dq.get("feed") or {}
        if feed:
            st.markdown(_md(f"**What the data is:** {feed.get('headline', '')}"))
            for cause in (feed.get("causes") or [])[1:]:
                st.caption(_md(f"• {cause}"))
            skew = feed.get("clock_skew_s")
            if skew is not None:
                st.caption(f"This computer's clock vs Alpaca's: {skew:+.1f}s")
        for source, err in (dq.get("provider_errors") or {}).items():
            st.caption(_md(f"✗ {source}: {err}"))
    diagnosis = (cycle.get("data_quality") or {}).get("diagnosis") or {}
    if diagnosis:
        with st.expander("Quote diagnosis for the symbols studied"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "symbol": d["symbol"],
                            "status": d["status"].replace("_", " "),
                            "priced on": d["coverage"],
                            "last print (s)": d.get("trade_age_s"),
                            "bid/ask (s)": d.get("quote_age_s"),
                            "spread (bp)": d.get("spread_bps"),
                            "spread checked on": d.get("spread_source"),
                            "why": "; ".join(d.get("reasons") or []),
                        }
                        for d in diagnosis.values()
                    ]
                ).astype(str),
                hide_index=True,
                use_container_width=True,
            )
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
    with st.expander("What each agent does (its charter)", icon=":material/info:"):
        for a in agents:
            spec = a.get("spec") or {}
            st.markdown(
                _md(
                    f"**{a['name']}** (`{a['id']}`, {spec.get('role')}, horizon {spec.get('horizon_days')}d, "
                    f"source: {spec.get('source') or '—'}) — {spec.get('description', '')}"
                )
            )
            st.caption(
                _md(
                    f"Reads: {', '.join(spec.get('inputs') or [])} · writes: {', '.join(spec.get('outputs') or [])}"
                    f" · if it cannot run: {spec.get('failure') or '—'}"
                )
            )


def _consensus(cycle: dict[str, Any]) -> None:
    st.markdown(
        "**② Consensus** — agents' forecasts combined per subject (weight = confidence × data quality × "
        "measured reliability; every agent is *unproven* until its predictions are graded). Agents that rest on "
        "the same information count as **one source**: a confident view needs at least two independent sources."
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
            "independent sources": (c.get("detail") or {}).get("independent_sources"),
            "data": c["data_quality"],
            "vetoes": "; ".join(v["reason"] for v in c.get("vetoes") or []),
        }
        for c in items
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    subjects = [c["subject"] for c in items]
    first_stock = next((i for i, s in enumerate(subjects) if not s.startswith("@")), 0)
    subject = st.selectbox("Look inside one subject", subjects, index=first_stock, key="brain-subject")
    chosen = next(c for c in items if c["subject"] == subject)
    detail = chosen.get("detail") or {}
    dispute = detail.get("primary_disagreement")
    if dispute:
        left, right = st.columns(2)
        left.success(
            _md(f"**For** ({dispute['for']['agent_id']}): {dispute['for']['thesis']}"),
            icon=":material/thumb_up:",
        )
        right.warning(
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
    sources = detail.get("sources") or {}
    if sources:
        st.markdown("**Evidence by source** (agents sharing a source count once)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "source": k,
                        "score": v["score"],
                        "weight": v["weight"],
                        "agents": ", ".join(v["agents"]),
                    }
                    for k, v in sources.items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    uncertainty = detail.get("uncertainty") or []
    if uncertainty:
        st.markdown("**Why it is uncertain**")
        for u in uncertainty:
            st.caption(_md(f"• {u}"))
    missing = detail.get("missing") or []
    if missing:
        with st.expander(f"Agents with no view here ({len(missing)})"):
            st.dataframe(pd.DataFrame(missing), hide_index=True, use_container_width=True)
    _debate(cycle, subject)
    _opinions(cycle, subject)


VERDICT_COLOR = {"stands": "green", "weakened": "orange", "challenged": "red", "no view to challenge": "gray"}


def _debate(cycle: dict[str, Any], subject: str) -> None:
    debate = next((d for d in cycle.get("debates") or [] if d["subject"] == subject), None)
    if debate is None:
        return
    st.markdown(
        "**Debate** — the strongest case each way, then the devil's advocate attacks the leading view."
    )
    st.badge(
        f"Devil's advocate: {debate['verdict']} · confidence {debate['confidence_before']:.2f} → "
        f"{debate['confidence_after']:.2f}",
        color=VERDICT_COLOR.get(debate["verdict"], "gray"),
        icon=":material/gavel:",
    )
    bull, bear = st.columns(2)
    with bull:
        st.markdown("🐂 **Bull case**")
        for a in debate.get("bull") or []:
            st.markdown(_md(f"- {a['text']} *({a['agent_id']})*"))
        if not debate.get("bull"):
            st.caption("No argument for.")
    with bear:
        st.markdown("🐻 **Bear case**")
        for a in debate.get("bear") or []:
            st.markdown(_md(f"- {a['text']} *({a['agent_id']})*"))
        if not debate.get("bear"):
            st.caption("No argument against.")
    objections = debate.get("objections") or []
    if objections:
        st.dataframe(
            pd.DataFrame(
                [
                    {"objection": o["text"], "severity": o["severity"], "confidence ×": o["haircut"]}
                    for o in objections
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    if debate.get("change_our_mind"):
        st.caption(_md("What would change our mind: " + "; ".join(debate["change_our_mind"])))


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
                ev = pd.DataFrame(o["evidence"])
                ev["value"] = ev["value"].map(
                    lambda v: "" if v is None else str(v)
                )  # mixed types: show as text
                st.dataframe(ev, hide_index=True, use_container_width=True)
            if o.get("data_missing"):
                st.caption(_md("Missing: " + ", ".join(o["data_missing"])))


def _fit(fit: dict[str, Any]) -> str:
    if not fit:
        return ""
    return ("fits" if fit.get("ok") else "poor fit") + (
        f": {'; '.join(fit['notes'])}" if fit.get("notes") else ""
    )


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
            "portfolio fit": _fit((d.get("rationale") or {}).get("fit") or {}),
            "④ execution": "not sent (the Brain never sends orders)",
            "why": "; ".join((d.get("rationale") or {}).get("reasons") or []),
        }
        for d in decisions
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    controls = (cycle.get("portfolio") or {}).get("trading_controls") or {}
    if controls:
        st.caption(
            _md(
                "Trading controls at the time of this cycle (read only): "
                + (
                    "orders placed through the trading service would reach the Alpaca paper account."
                    if controls.get("orders_would_reach_alpaca")
                    else "orders would NOT reach Alpaca — " + "; ".join(controls.get("blockers") or [])
                )
                + " The Brain itself never submits."
            )
        )
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


OPP_STATUS = {
    "recommended": "risk engine would allow (recommendation)",
    "dry_run_approved": "risk engine would allow (dry run)",
    "risk_rejected": "risk engine rejected",
    "watch": "watching (not acted on)",
    "no_action": "no action",
    "hold": "held: no change",
    "not_analysed": "not analysed (focus budget)",
    "rejected_data": "rejected: data not trustworthy",
    "context": "market context",
    "no_view": "no agent had a view",
}


def _opportunities(cycle: dict[str, Any]) -> None:
    ops = cycle.get("opportunities") or []
    st.markdown(
        "Ideas the Brain found **by itself** this cycle. Finding one is not a recommendation: each goes through "
        "data validation → the relevant agents → research → bull / bear / devil's advocate → consensus → portfolio "
        "fit → risk preview, and most stop along the way."
    )
    if not ops:
        st.info("Nothing unusual detected this cycle.", icon=":material/search_off:")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "kind": o["kind"].replace("_", " "),
                    "subject": o["subject"],
                    "idea": {1: "long", -1: "avoid / reduce", 0: "look closer"}.get(o["direction"], ""),
                    "strength": round(o["strength"], 2),
                    "what": o["headline"],
                    "outcome": OPP_STATUS.get(o["status"], o["status"]),
                }
                for o in ops
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    pick = st.selectbox(
        "Follow one idea through the pipeline",
        range(len(ops)),
        format_func=lambda i: f"{ops[i]['kind'].replace('_', ' ')} · {ops[i]['subject']}",
        key="brain-opp",
    )
    for stage in ops[pick]["stages"]:
        extra = {k: v for k, v in stage.items() if k not in ("stage", "result") and v}
        st.markdown(_md(f"**{stage['stage'].replace('_', ' ')}** — {stage['result']}"))
        if extra:
            st.caption(_md("; ".join(f"{k}: {v}" for k, v in extra.items())))


QUADRANT = {
    "earned": "good decision · good outcome",
    "unlucky": "good decision · bad outcome (variance)",
    "lucky": "weak decision · good outcome (luck)",
    "process_failure": "weak decision · bad outcome (fix the process)",
    "block_saved_money": "blocked idea · would have lost",
    "block_cost_opportunity": "blocked idea · would have gained",
    "inconclusive": "move too small to judge",
}


def _learning() -> None:
    st.markdown(
        "The Brain learns **only from matured predictions graded against real closing prices**. An agent has no "
        "track record — and weighs 1.0 in the consensus — until enough of its own calls have been graded. "
        "Decisions are judged on process and outcome separately: a good decision can lose and a bad one can win."
    )
    state = guarded(lambda: api().get(f"{BASE}/learning"), "learning status")
    if state is None:
        return
    preds = state["predictions"]
    c = st.columns(5)
    c[0].metric("Open predictions", preds["open"])
    c[1].metric("Graded", preds["evaluated"])
    c[2].metric("Hit rate", pct(preds["hit_rate"], 0) if preds["hit_rate"] is not None else "—")
    c[3].metric("Next due", preds["next_due"] or "—")
    c[4].metric(
        "Measured agents", len(state["measured_agents"]), help=f"≥ {state['min_observations']} graded calls"
    )
    last = state.get("last_run")
    st.caption(
        f"Last learning pass: {last['at'][:16].replace('T', ' ')} UTC · graded {last['evaluated']}, reflections "
        f"{last['reflections']}"
        if last
        else "No learning pass yet."
    )
    if st.button("Grade matured predictions now", icon=":material/school:", key="brain-learn"):
        out = guarded(lambda: api().post(f"{BASE}/learn", wait=120), "learning pass")
        if out:
            st.success(f"Graded {out['evaluated']} predictions, wrote {out['reflections']} reflections.")
    if state["calibration"]:
        st.markdown("**Consensus calibration** — does a more confident consensus hit more often?")
        st.dataframe(pd.DataFrame(state["calibration"]), hide_index=True, use_container_width=True)
    rows = guarded(lambda: api().get(f"{BASE}/performance", window="all"), "track records") or []
    overall = [r for r in rows if r["regime"] == "all"]
    if overall:
        st.markdown(
            "**Track records** (all regimes, all time). Calls on one name whose horizons overlap share an outcome "
            "and count once (*independent*). A verdict needs enough independent calls and a false-discovery-"
            "adjusted q below 10%; a weight moves only then, and only to the conservative end of the interval."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "agent": r["agent_id"],
                        "version": r["agent_version"],
                        "calls": r["n"],
                        "independent": r.get("n_effective"),
                        "hit rate": r["hit_rate"],
                        "95% interval": f"{pct(r.get('ci_low'), 0)}–{pct(r.get('ci_high'), 0)}"
                        if r.get("ci_low") is not None
                        else "—",
                        "q": r.get("q_value"),
                        "verdict": r.get("verdict") or "unproven",
                        "mean excess": pct(r.get("mean_excess"), 2),
                        "in risk units": r.get("mean_excess_z"),
                        "Brier (0.25 = coin flip)": r["brier"],
                        "rank IC": r["ic"],
                        "consensus weight": r["reliability"]
                        if r["reliability"] is not None
                        else "unproven (1.0)",
                    }
                    for r in overall
                ]
            ).astype(str),
            hide_index=True,
            use_container_width=True,
        )
    reflections = guarded(lambda: api().get(f"{BASE}/reflections", limit=200), "reflections") or []
    decisions = [r for r in reflections if r["subject_type"] == "decision"]
    if decisions:
        st.markdown("**Decision vs outcome**")
        counts = pd.Series([QUADRANT.get(r["category"], r["category"]) for r in decisions]).value_counts()
        st.dataframe(counts.rename("decisions").to_frame(), use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "when": r["created_at"][:10],
                        "subject": (r.get("evidence") or {}).get("subject"),
                        "quadrant": QUADRANT.get(r["category"], r["category"]),
                        "decision": r["decision_quality"],
                        "outcome": r["outcome_quality"],
                        "lesson": "; ".join(r["lessons"][:2]),
                    }
                    for r in decisions
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    for r in (x for x in reflections if x["category"] == "failure_analysis"):
        with st.expander(f"Failure analysis · {r['created_at'][:10]}", icon=":material/troubleshoot:"):
            for lesson in r["lessons"]:
                st.markdown(_md(f"- {lesson}"))
    if not decisions and not overall:
        st.info(
            "Nothing has matured yet: predictions are graded once their horizon has passed.",
            icon=":material/hourglass_top:",
        )


STATUS_BADGE = {
    "proposed": "gray",
    "validated": "blue",
    "rejected": "red",
    "paper": "orange",
    "promoted": "green",
    "retired": "gray",
}


def _lab() -> None:
    st.markdown(
        "Strategies are **proposed → validated → paper-tracked → promoted**. Validation needs every gate to pass: "
        "enough out-of-sample history, walk-forward value over the equal-weight universe, a deflated Sharpe "
        "that survives the number of variants tried, better than random portfolios, and stress tests. One "
        "attractive backtest is never enough, and **only a person can promote**. A promoted strategy becomes one "
        "voice in the consensus; nothing here places an order."
    )
    rows = guarded(lambda: api().get(f"{BASE}/lab/strategies"), "strategy lab") or []
    propose = st.button("Propose untried templates", icon=":material/add_circle:", key="lab-propose")
    if propose and guarded(lambda: api().post(f"{BASE}/lab/propose"), "propose") is not None:
        st.rerun()
    if not rows:
        st.info("No strategies yet.", icon=":material/science:")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "strategy": r["key"],
                    "name": r["name"],
                    "status": r["status"],
                    "verdict": (r["validation"] or {}).get("verdict"),
                    "gates passed": f"{sum(g['passed'] for g in (r['validation'] or {}).get('gates', []))}"
                    f"/{len((r['validation'] or {}).get('gates', []))}",
                    "OOS active Sharpe": ((r["validation"] or {}).get("walk_forward") or {}).get(
                        "oos_active_sharpe"
                    ),
                    "deflated Sharpe": ((r["validation"] or {}).get("walk_forward") or {}).get("dsr"),
                    "paper sessions": (r["paper"] or {}).get("sessions"),
                    "paper excess": (r["paper"] or {}).get("excess_return"),
                }
                for r in rows
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    keys = [r["key"] for r in rows]
    key = st.selectbox("Strategy", keys, key="lab-pick")
    chosen = next(r for r in rows if r["key"] == key)
    st.badge(chosen["status"], color=STATUS_BADGE.get(chosen["status"], "gray"))
    st.caption(_md(chosen["spec"].get("description", "")))
    sid, ver = chosen["strategy_id"], chosen["version"]
    c = st.columns(4)
    if c[0].button("Validate", key="lab-validate", disabled=chosen["status"] == "retired"):
        guarded(lambda: api().post(f"{BASE}/lab/strategies/{sid}/{ver}/validate", wait=300), "validation")
        st.rerun()
    for col, status, label in (
        (c[1], "paper", "Start paper tracking"),
        (c[2], "promoted", "Promote"),
        (c[3], "retired", "Retire"),
    ):
        pressed = col.button(label, key=f"lab-{status}")
        if pressed and guarded(
            lambda s=status: api().post(f"{BASE}/lab/strategies/{sid}/{ver}/status", {"status": s}), label
        ):
            st.rerun()
    v = chosen["validation"] or {}
    if v.get("gates"):
        st.dataframe(pd.DataFrame(v["gates"]), hide_index=True, use_container_width=True)
        bt, wf = v.get("backtest") or {}, v.get("walk_forward") or {}
        st.caption(
            f"Backtest {bt.get('start')} → {bt.get('end')}: Sharpe {bt.get('sharpe')}, max drawdown "
            f"{pct(bt.get('max_drawdown'), 1)}, excess {pct(bt.get('excess_annual'), 1)}/yr · walk-forward "
            f"in-sample active Sharpe {wf.get('is_active_sharpe')} → out-of-sample {wf.get('oos_active_sharpe')} · "
            f"{(v.get('data') or {}).get('symbols')} symbols; {(v.get('data') or {}).get('survivorship_bias')}"
        )


def _improvements() -> None:
    st.markdown(
        "The Brain reviews its own record and writes **proposals** — problem, evidence, proposed change, expected "
        "improvement and validation plan. **Nothing is applied automatically**; changes are built as new versions and "
        "must pass PROPOSE → VERSION → TEST → BACKTEST → WALK-FORWARD → PAPER EVALUATION → COMPARE → PROMOTE ONLY IF "
        "VALIDATED. Risk controls are never a subject."
    )
    review = st.button("Review the record now", icon=":material/rule:", key="improve-review")
    if review and guarded(lambda: api().post(f"{BASE}/improvements/review"), "review") is not None:
        st.rerun()
    rows = guarded(lambda: api().get(f"{BASE}/improvements"), "improvements") or []
    if not rows:
        st.info("No proposals: the record does not show a problem yet (or has too few graded calls).")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "target": r["target"],
                    "problem": r["title"],
                    "status": r["status"],
                }
                for r in rows
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    pick = st.selectbox(
        "Proposal",
        [r["id"] for r in rows],
        format_func=lambda i: next(f"#{r['id']} {r['title']}" for r in rows if r["id"] == i),
        key="improve-pick",
    )
    chosen = next(r for r in rows if r["id"] == pick)
    p = chosen["proposal"]
    st.markdown(_md(f"**Problem:** {chosen['title']}"))
    st.markdown(_md(f"**Evidence:** {chosen['evidence']}"))
    st.markdown(_md(f"**Proposed change:** {p.get('change')}"))
    st.markdown(_md(f"**Expected improvement:** {p.get('expected_improvement')}"))
    st.markdown(
        "**Validation plan:**\n"
        + "\n".join(f"{i + 1}. {_md(s)}" for i, s in enumerate(p.get("validation_plan", [])))
    )
    note = st.text_input("Note (optional)", key="improve-note")
    cols = st.columns(4)
    for col, status in zip(cols, ("testing", "validated", "rejected", "applied"), strict=True):
        pressed = col.button(status.capitalize(), key=f"improve-{status}")
        if pressed and guarded(
            lambda s=status: api().post(f"{BASE}/improvements/{pick}", {"status": s, "note": note or None}),
            "decision",
        ):
            st.rerun()


def _operations() -> None:
    st.markdown(
        "While the server runs, the **supervisor** decides what the Brain does: full cycles during the session, a "
        "quote monitor, focused cycles when events happen (rate-limited), learning after the close and research at "
        "weekends. It is analysis only — the Brain never sends orders."
    )
    sup = guarded(lambda: api().get(f"{BASE}/supervisor"), "supervisor")
    if sup is not None:
        c = st.columns(4)
        c[0].metric("Supervisor", "off" if not sup["enabled"] else "paused" if sup["paused"] else "running")
        c[1].metric("Session", sup["session"].replace("_", " "))
        c[2].metric("Queued wake-ups", len(sup["queue"]))
        c[3].metric(
            "Event cycles (last hour)",
            f"{sup['event_cycles_last_hour']} / {sup['limits']['max_event_cycles_per_hour']}",
        )
        if sup["enabled"]:
            label = "Resume" if sup["paused"] else "Pause"
            toggle = {"paused": not sup["paused"]}
            clicked = st.button(label, icon=":material/pause_circle:", key="brain-pause")
            if clicked and guarded(lambda: api().post(f"{BASE}/supervisor", toggle), "supervisor"):
                st.rerun()
        if sup["queue"]:
            st.dataframe(pd.DataFrame(sup["queue"]), hide_index=True, use_container_width=True)
        if sup["recent"]:
            st.markdown("**Recent work**")
            st.dataframe(
                pd.DataFrame(sup["recent"][::-1]).astype(str), hide_index=True, use_container_width=True
            )
        st.caption(
            _md("Last runs: " + ", ".join(f"{k} {v[:16].replace('T', ' ')}" for k, v in sup["last"].items()))
        )
    _models()
    kind = st.selectbox(
        "Events",
        [
            "all",
            "PriceMoveDetected",
            "VolumeSpikeDetected",
            "QuoteBecameStale",
            "OpportunityDetected",
            "EarningsApproaching",
            "MarketRegimeChanged",
            "PortfolioChanged",
            "PositionChanged",
            "OrderSubmitted",
            "OrderFilled",
            "OrderCanceled",
            "RiskLimitTriggered",
            "AgentFailed",
            "AgentCompleted",
            "PredictionMatured",
            "TradeOutcomeAvailable",
            "MarketDataUpdated",
            "NewsEventDetected",
        ],
        key="brain-event-kind",
    )
    rows = (
        guarded(
            lambda: api().get(f"{BASE}/events", type=None if kind == "all" else kind, limit=200), "events"
        )
        or []
    )
    if rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "when": r["created_at"][:19].replace("T", " "),
                        "event": r["type"],
                        "subject": r["subject"],
                        "source": (r.get("payload") or {}).get("source"),
                        "detail": ", ".join(
                            f"{k}={v}"
                            for k, v in (r.get("payload") or {}).items()
                            if k != "source" and v is not None
                        )[:160],
                    }
                    for r in rows
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No events of this kind yet.", icon=":material/notifications_off:")


def _models() -> None:
    m = guarded(lambda: api().get(f"{BASE}/models"), "language models")
    if m is None:
        return
    st.markdown("**Language models**")
    if not m["available"]:
        st.info(
            _md(
                f"Not in use: {m['reason']}. Every analysis is deterministic; the briefing agent skips itself. "
                "Calculations (indicators, risk, sizing, spreads, quote ages, the account) never go to a model."
            ),
            icon=":material/memory:",
        )
        return
    u = m["usage"]
    c = st.columns(4)
    c[0].metric("Provider", m["provider"])
    c[1].metric("Tokens today", f"{u['tokens'] + u['estimated']:,} / {m['daily_token_budget']:,}")
    c[2].metric("Calls (cached)", f"{u['calls']} ({u['cached']})")
    c[3].metric("Failed / refused", f"{u['failed']} / {u['refused']}")
    st.caption(
        _md(
            f"Fast: {m['models']['fast'] or '—'} · strong: {m['models']['strong'] or '—'} · output cap "
            f"{m['max_output_tokens']} tokens · cache {m['cache_minutes']} min. Model output is context only: it "
            "casts no vote and sets no number."
        )
    )
    if m["recent"]:
        st.dataframe(pd.DataFrame(m["recent"]).astype(str), hide_index=True, use_container_width=True)


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
    tabs = st.tabs(
        [
            "Overview",
            "Opportunities",
            "Agents",
            "Consensus & debate",
            "Proposed actions",
            "Learning",
            "Strategy lab",
            "Improvements",
            "Supervisor & events",
            "Memory",
            "History",
        ]
    )
    with tabs[0]:
        _overview(cycle)
    with tabs[1]:
        _opportunities(cycle)
    with tabs[2]:
        _agents(cycle, agents)
    with tabs[3]:
        _consensus(cycle)
    with tabs[4]:
        _decisions(cycle)
    with tabs[5]:
        _learning()
    with tabs[6]:
        _lab()
    with tabs[7]:
        _improvements()
    with tabs[8]:
        _operations()
    with tabs[9]:
        _memory()
    with tabs[10]:
        _history(cycles)
