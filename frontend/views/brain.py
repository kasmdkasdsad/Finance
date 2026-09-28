"""The Brain: what QuantPulse's agents see, think and decide — and what happened to each decision.

Four layers are kept visibly apart on this page:

1. **Agent analysis** — each agent's own opinion (stance, confidence, evidence, what would prove it wrong);
2. **Consensus** — the agents' views combined per subject, with the disagreement kept visible;
3. **Risk preview** — the deterministic risk engine's verdict on each proposed trade;
4. **Broker execution** — when the Brain owns the Alpaca **paper** account (``QP_BRAIN_MODE=paper_execution``)
   its decisions are executed by the trading service (reconciliation, fresh quotes, the same risk engine,
   the order manager, every trading switch); otherwise none — proposals only, simulated in its paper book.

The Brain kill switch at the top stops new Brain orders at once.
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
    "approved": "Risk engine: would allow",
    "risk_rejected": "Risk engine: rejected",
    "blocked": "Blocked by a data veto",
    "not_checked": "Not checked (account unreadable)",
    "no_trade": "No trade proposed",
    "halted": "Halted: no new positions (see the execution panel)",
    "skipped": "Not traded (a working order or the cooldown)",
    "duplicate_prevented": "Already sent in this slot: not resent",
    "failed": "Execution failed (nothing sent)",
    "submitted": "Sent to Alpaca paper",
    "accepted": "Sent to Alpaca paper: working",
    "partially_filled": "Alpaca paper: partially filled",
    "filled": "Alpaca paper: filled",
    "canceled": "Alpaca paper: canceled",
    "expired": "Alpaca paper: expired",
    "rejected": "Rejected by Alpaca",
    "unknown": "Sent: outcome unknown (reconciliation settles it)",
    "risk_approved": "Risk engine: approved (dry run: not sent)",
    "blocked_at_submit": "Stopped at submission (a switch changed): not sent",
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


def _layers_banner(owns: bool) -> None:
    execution = (
        "the trading service executes the Brain's decisions on the Alpaca **paper** account — after "
        "reconciling, on fresh quotes, through the same risk engine and order manager as every order, and only "
        "while every trading switch allows it."
        if owns
        else "**none** — the Brain never sends an order in this mode (proposals only, simulated in its paper "
        "book); orders only come from the Paper Trading (Alpaca) page."
    )
    st.info(
        "**How to read this page.** ① *Agent analysis* is what each specialist concludes on its own. "
        "② *Consensus* combines them and shows where they disagree. ③ *Risk preview* is the deterministic risk "
        f"engine's verdict on each proposed trade — the same checks that guard real orders. ④ *Broker execution*: "
        f"{execution}",
        icon=":material/psychology:",
    )


def _execution_panel(ex: dict[str, Any]) -> None:
    """Who owns the account, the Brain kill switch (always one click away) and what stops Brain orders."""
    kill = ex["brain_kill_switch"]
    last = ex.get("last_cycle") or {}
    if not ex["owns_account"]:
        st.caption(
            _md(
                f"QP_BRAIN_MODE={ex['mode']}: the Brain proposes only (its paper book); the strategy owns the "
                "Alpaca paper account."
            )
        )
    elif kill["active"]:
        st.error(
            _md(f"BRAIN KILL SWITCH ON — no new Brain orders ({kill.get('reason') or kill['source']})."),
            icon=":material/block:",
        )
    elif ex["blockers_scheduled"]:
        st.warning(
            _md(
                "The Brain owns the Alpaca PAPER account; its orders are not sent right now: "
                + "; ".join(ex["blockers_scheduled"])
            ),
            icon=":material/pause_circle:",
        )
    else:
        st.success(
            "The Brain owns the Alpaca PAPER account and executes its decisions through the trading service.",
            icon=":material/smart_toy:",
        )
    halts = last.get("entry_halts") or []
    if ex["owns_account"] and halts:
        data = [h for h in halts if h["code"] == "data_quality"]
        if data:
            st.error(_md(data[0]["reason"]), icon=":material/signal_disconnected:")
        others = [h for h in halts if h["code"] != "data_quality"]
        if others:
            st.warning(
                _md("New positions halted (exits still allowed): " + "; ".join(h["reason"] for h in others)),
                icon=":material/front_hand:",
            )
    if kill["active"]:
        if kill["source"] == "env":
            st.caption("Set by QP_BRAIN_KILL_SWITCH=true: change the setting and restart to release it.")
        elif st.button(
            "Allow Brain orders again", icon=":material/lock_open:", key="brain_release"
        ) and guarded(lambda: api().post(f"{BASE}/kill-switch", {"active": False}), "Brain kill switch"):
            st.rerun()
    elif ex["owns_account"]:
        c1, c2 = st.columns([3, 1])
        reason = c1.text_input("Reason (optional)", key="brain_kill_reason", label_visibility="collapsed",
                               placeholder="Why stop the Brain? (optional)")  # fmt: skip
        body = {"active": True, "reason": reason or None, "cancel_open_orders": True}
        stop = c2.button(
            "STOP BRAIN ORDERS",
            icon=":material/block:",
            type="primary",
            key="brain_kill",
            use_container_width=True,
        )
        if stop and guarded(lambda: api().post(f"{BASE}/kill-switch", body), "Brain kill switch"):
            st.rerun()
        st.caption(
            "Stops every new Brain-originated order at once and cancels the Brain's working orders. Positions "
            "stay as they are; the trading kill switch and close-all on the Paper Trading page still work."
        )


def _status(status: dict[str, Any], ex: dict[str, Any] | None) -> None:
    cols = st.columns(5)
    cols[0].metric("Mode", status["mode"].replace("_", " "))
    agents = status["agents"]
    cols[1].metric("Agents enabled", f"{agents['enabled']} / {agents['registered']}")
    last = status.get("last_cycle")
    cols[2].metric("Last cycle", f"#{last['id']} · {last['status']}" if last else "none yet")
    cols[3].metric("Open predictions", status["open_predictions"])
    sent = ((ex or {}).get("last_cycle") or {}).get("orders_sent", 0)
    cols[4].metric("Orders sent by the Brain", sent, help="in the latest cycle (through the trading service)")
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
            summary = out.get("summary") or {}
            st.success(
                f"Cycle #{out['id']} {out['status']}: {summary.get('trades_proposed', 0)} trades proposed, "
                f"{summary.get('orders_sent', 0)} orders sent.",
                icon=":material/check_circle:",
            )


def _why(cycle: dict[str, Any]) -> None:
    """Why the Brain traded in this cycle, or why it did not — in plain words."""
    decision = (cycle.get("summary") or {}).get("decision") or {}
    if not decision:
        return
    headline = _md(decision.get("headline") or "")
    if decision.get("outcome") == "traded":
        st.success(headline, icon=":material/swap_horiz:")
    else:
        st.info(headline, icon=":material/do_not_disturb_on:")
    for o in decision.get("orders") or []:
        st.caption(_md(f"• {o['action']} {o['qty']:g} {o['subject']} — {'; '.join(o.get('why') or [])}"))
    for r in decision.get("reasons") or []:
        st.caption(_md(f"• {r}"))


def _overview(cycle: dict[str, Any]) -> None:
    summary = cycle.get("summary") or {}
    regime = cycle.get("regime") or {}
    market = cycle.get("market") or {}
    _why(cycle)
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
        st.subheader("The Brain's paper book (hypothetical)")
        pf = cycle.get("portfolio") or {}
        st.caption(
            "Owned by the Brain: its decisions are simulated here at modelled prices after the risk engine allows "
            "them. Never sent to a broker."
        )
        st.markdown(
            f"Equity **{money(pf.get('equity'))}** · cash {money(pf.get('cash'))} · "
            f"{len(pf.get('positions') or {})} positions · {len((pf.get('book') or {}).get('fills') or [])} "
            "simulated fills this cycle"
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
        alpaca = pf.get("alpaca_account") or {}
        owner = alpaca.get("owner") or "the trading strategy (the Brain only reads it)"
        with st.expander(_md(f"Alpaca paper account — owned by {owner}")):
            if not alpaca.get("available"):
                st.warning(_md(f"Unavailable: {alpaca.get('error')}"), icon=":material/cloud_off:")
            else:
                st.markdown(
                    f"Equity {money(alpaca.get('equity'))} · cash {money(alpaca.get('cash'))} · "
                    f"{len(alpaca.get('positions') or {})} positions · {alpaca.get('open_orders', 0)} open orders. "
                    + (
                        "The Brain manages this account; its orders go through the trading service."
                        if "Brain" in owner
                        else "The Brain makes no decisions for this account."
                    )
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


def _execution(d: dict[str, Any]) -> str:
    ex = d.get("execution") or {}
    fill = ex.get("book")
    if fill:
        return (
            f"paper book: {fill['side']} {fill['qty']:g} @ ${fill['fill_price']:,.2f} "
            f"({fill['slippage_bps']:+.1f}bp) — not sent to Alpaca"
        )
    if ex.get("duplicate_prevented"):
        return f"not resent: {ex.get('client_order_id')} was already sent in this slot"
    if ex.get("sent"):
        filled = (
            f", filled {ex['filled_qty']:g} @ ${ex['filled_avg_price']:,.2f}"
            if ex.get("filled_qty") and ex.get("filled_avg_price")
            else ""
        )
        return f"Alpaca paper: {ex.get('stage') or ex.get('status')}{filled} ({ex.get('client_order_id')})"
    if ex.get("stage") == "risk_rejected":
        return f"trading service: risk engine rejected ({ex.get('risk')})"
    reason = str(ex.get("reason") or "not sent")
    return reason if reason.startswith("not sent") else f"not sent: {reason}"


def _execution_tab() -> None:
    """The final execution audit, what the Brain's orders did at Alpaca paper, and how well they executed."""
    audits = guarded(lambda: api().get(f"{BASE}/execution-audit"), "execution audit") or {}
    latest = audits.get("latest")
    st.markdown(
        "**Final execution audit** — run before the first Brain order of each day and of each start, and at "
        "startup. Scheduled Brain execution is armed only when every check passes; each order is checked "
        "again immediately before it is sent (kill switches, paper endpoint and key, dry run, .env)."
    )
    if not latest:
        st.caption("No execution audit yet (the first one runs when the Brain first wants to send an order).")
    else:
        ok = latest["ok"]
        (st.success if ok else st.error)(
            _md(
                f"{latest['purpose'].replace('_', ' ')} audit at {latest['at'][:19].replace('T', ' ')} UTC: "
                + ("every check passed" if ok else "FAILED — " + "; ".join(latest["failed"]))
            ),
            icon=":material/verified_user:" if ok else ":material/gpp_bad:",
        )
        c = st.columns(4)
        c[0].metric("Brain mode", str(latest.get("mode") or "—").replace("_", " "))
        c[1].metric("Endpoint", "paper" if "paper-api" in str(latest.get("endpoint")) else "NOT PAPER")
        c[2].metric(
            "Live trading possible", "no" if latest.get("live_trading_possible") is False else "CHECK"
        )
        c[3].metric("Agents", str((latest.get("agents") or {}).get("enabled", "—")))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "check": x["name"],
                        "result": "pass" if x["ok"] else "FAIL" if x["ok"] is False else "—",
                        "detail": x["detail"],
                    }
                    for x in latest.get("checks") or []
                ]
            ).astype(str),
            hide_index=True,
            use_container_width=True,
        )
        if latest.get("orders"):
            st.caption("What was about to go when it ran:")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "symbol": o["subject"],
                            "action": o["action"],
                            "qty": o["quantity"],
                            "price": o["est_price"],
                            "risk preview": o.get("risk_preview"),
                            "why": "; ".join(o.get("reasons") or []),
                        }
                        for o in latest["orders"]
                    ]
                ).astype(str),
                hide_index=True,
                use_container_width=True,
            )
        if latest.get("outcome"):
            st.caption(_md(f"Outcome: {latest['outcome']}"))
    quality = guarded(lambda: api().get(f"{BASE}/execution-quality"), "execution quality") or {}
    st.markdown(
        "**Execution quality** — judged on its own, never merged with whether a trade made money: slippage "
        "against the decision's price, cost against the quote as the order left (graded against half the "
        "spread), time to fill, latency."
    )
    if quality.get("sent"):
        c = st.columns(5)
        c[0].metric("Orders sent", quality.get("sent", 0))
        c[1].metric("Fill rate", pct(quality.get("fill_rate"), 0))
        c[2].metric("Mean slippage", f"{num(quality.get('slippage_bps_mean'))}bp")
        c[3].metric("Cost vs quote", f"{num(quality.get('cost_vs_quote_bps_mean'))}bp")
        c[4].metric("Median latency", f"{num(quality.get('submit_latency_ms_median'))}ms")
        st.caption(_md(f"Grades: {quality.get('grades') or {}} · median spread paid "
                       f"{num(quality.get('spread_bps_median'))}bp · median quote age "
                       f"{num(quality.get('quote_age_s_median'))}s"))  # fmt: skip
    else:
        st.caption("No Brain order has been sent yet: execution quality is unproven.")
    ledger = guarded(lambda: api().get(f"{BASE}/executions", limit=200), "execution ledger") or {}
    rows = ledger.get("executions") or []
    st.markdown("**Execution ledger** — every Brain order from the decision to its final state.")
    if rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "decided": (r.get("decided_at") or "")[:19].replace("T", " "),
                        "symbol": r["symbol"],
                        "action": r["action"],
                        "qty": r["qty"],
                        "status": r["status"],
                        "filled": r["filled_qty"],
                        "expected": r["expected_price"],
                        "fill": r["filled_avg_price"],
                        "slippage (bp)": r["slippage_bps"],
                        "vs quote (bp)": r["cost_vs_quote_bps"],
                        "spread (bp)": r["spread_bps"],
                        "quote age (s)": r["quote_age_s"],
                        "latency (ms)": r["submit_latency_ms"],
                        "to fill (s)": r["seconds_to_fill"],
                        "grade": r["grade"],
                        "brain cycle": r["brain_cycle_id"],
                        "order": r["client_order_id"],
                        "why": r["reason"],
                    }
                    for r in rows
                ]
            ).astype(str),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No Brain order has been sent yet.")
    sessions = (guarded(lambda: api().get(f"{BASE}/sessions", limit=1), "sessions") or {}).get(
        "sessions"
    ) or []
    near = (sessions[0].get("near_close") or {}) if sessions else {}
    if near:
        st.markdown(_md(f"**Near-close review** ({sessions[0]['day']}): held overnight "
                        f"{', '.join(near.get('held_overnight') or []) or 'nothing'}; reduced "
                        f"{', '.join(near.get('reduced') or []) or 'nothing'}; earnings before the next "
                        f"session: {', '.join(near.get('earnings_overnight') or []) or 'none'}."))  # fmt: skip


def _positions() -> None:
    st.markdown(
        "**Positions and their theses** — every position on the Alpaca paper account the Brain owns: why it is "
        "held, what would prove it wrong, its stop and (once calibrated) target, the agents for and against, "
        "and how it has done against the benchmark since entry. A **broken** thesis is exited without waiting "
        "for a new signal; a **weakening** one is first in line to be replaced by a stronger idea."
    )
    data = guarded(lambda: api().get(f"{BASE}/positions"), "positions")
    if data is None:
        return
    if "Brain" not in data["owner"]:
        st.caption(_md(f"Owner: {data['owner']}. See the paper book tab for the Brain's own positions."))
        return
    for sym in data.get("unexpected") or []:
        c1, c2 = st.columns([4, 1])
        c1.warning(
            _md(
                f"{sym}: a position the Brain did not open or adopt — no new positions until it is adopted or gone."
            ),
            icon=":material/help:",
        )
        if c2.button(f"Adopt {sym}", key=f"adopt_{sym}") and guarded(
            lambda sym=sym: api().post(f"{BASE}/positions/{sym}/adopt"), "adopt"
        ):
            st.rerun()
    rows = data.get("open") or []
    if not rows:
        st.caption("No open positions.")
    else:
        table = [
            {
                "symbol": r["symbol"],
                "check": (r.get("check") or {}).get("status", "—"),
                "origin": r["origin"],
                "qty": r["qty"],
                "weight": pct(r.get("weight"), 1),
                "P&L": money(r.get("unrealized_pnl")),
                "return": pct(r.get("return_pct"), 1),
                "vs benchmark": pct(r.get("relative_return"), 1),
                "stop": money(r.get("stop_price")),
                "target": money(r.get("target_price")) if r.get("target_price") else "uncalibrated",
                "horizon": r.get("horizon_days"),
                "confidence": num(r.get("confidence")),
                "for": ", ".join(r.get("supporting") or []) or "—",
                "against": ", ".join(r.get("opposing") or []) or "—",
                "regime": r.get("regime") or "—",
                "sector": r.get("sector") or "—",
            }
            for r in rows
        ]
        st.dataframe(pd.DataFrame(table).astype(str), hide_index=True, use_container_width=True)
        for r in rows:
            check = r.get("check") or {}
            with st.expander(
                _md(f"{r['symbol']} — {check.get('status', 'not checked yet')}: {r['thesis'][:90]}")
            ):
                st.markdown(_md(f"**Thesis:** {r['thesis']}"))
                st.markdown(_md(f"**Invalidation:** {r.get('invalidation') or 'below the stop'}"))
                if check.get("reasons"):
                    st.markdown(_md("**Check:** " + "; ".join(check["reasons"])))
                st.caption(
                    _md(
                        f"Opened {r['opened_at'][:16].replace('T', ' ')} UTC at {money(r['entry_price'])} · "
                        f"entry order {r.get('entry_order_id') or '—'} · decision #{r.get('entry_decision_id') or '—'}"
                    )
                )
    closed = data.get("closed") or []
    if closed:
        st.markdown("**Closed**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "symbol": r["symbol"],
                        "opened": r["opened_at"][:10],
                        "closed": (r.get("closed_at") or "")[:10],
                        "entry": money(r["entry_price"]),
                        "exit": money(r.get("exit_price")),
                        "realised": money(r.get("realized_pnl")),
                        "why": r.get("exit_reason") or "—",
                    }
                    for r in closed
                ]
            ).astype(str),
            hide_index=True,
            use_container_width=True,
        )


STAGE_ICON = {"done": "✅", "pending": "⏳", "missing": "❌", "none": "—", "n/a": "·"}


def _audit() -> None:
    st.markdown(
        "**Audit trail** — one decision followed from the idea to what it taught: opportunity → data → agents → "
        "evidence → disagreement → consensus → debate → portfolio fit → decision → risk check → order → Alpaca "
        "→ execution → fill → position → P&L → benchmark-relative outcome → prediction grade → decision quality "
        "→ lesson. Everything shown was recorded at the time: ⏳ comes later, ❌ is a gap in the record."
    )
    traces = guarded(lambda: api().get(f"{BASE}/traces", limit=200), "traceability") or {}
    if traces:
        (st.error if traces.get("with_gaps") else st.caption)(_md(f"Traceability: {traces['headline']}."))
        if traces.get("gaps_by_stage"):
            st.caption(_md(f"Missing links by stage: {traces['gaps_by_stage']}"))
    trades = guarded(lambda: api().get(f"{BASE}/trades", limit=100), "trades") or []
    if not trades:
        st.caption("No trade decisions yet.")
        return
    labels = {
        t["decision_id"]: f"#{t['decision_id']} · {t['at'][:16].replace('T', ' ')} · {t['action']} "
        f"{t['quantity']:g} {t['subject']} · {'sent' if t['sent'] else 'not sent'} ({t['status']})"
        for t in trades
    }
    chosen = st.selectbox("Decision", list(labels), format_func=labels.__getitem__, key="audit_decision")
    trail = guarded(lambda: api().get(f"{BASE}/decisions/{chosen}/audit"), "audit trail")
    if trail is None:
        return
    for stage in trail["stages"]:
        icon = STAGE_ICON.get(stage["status"], "·")
        with st.expander(
            _md(f"{icon} {stage['stage'].replace('_', ' ').upper()} — {stage['summary'][:160]}")
        ):
            st.json(stage["detail"], expanded=False)


def _data_report() -> None:
    st.markdown(
        "**Market data and SIP** — how often market data stopped the Brain, what it stopped, and what real-time "
        "SIP data would and would not change. The quote-age and spread limits are never loosened to trade "
        "more; buying data is your decision."
    )
    days = st.select_slider("Window (days)", [5, 10, 20, 60, 120], value=20, key="data_days")
    rep = guarded(lambda: api().get(f"{BASE}/data-report", days=days), "data report")
    if rep is None:
        return
    how = rep["how_often"]
    (st.error if how.get("data_blocked_cycles") else st.info)(
        _md(rep["headline"]), icon=":material/monitoring:"
    )
    c = st.columns(4)
    c[0].metric("Cycles in session", how["cycles_in_session"])
    c[1].metric(
        "Data-blocked cycles", how["data_blocked_cycles"], pct(how.get("share"), 0), delta_color="off"
    )
    q = rep["quote_age"]
    c[2].metric("Stale IEX quotes", q["iex_quiet"], f"of {q['focus_quotes']} focus quotes", delta_color="off")
    c[3].metric(
        "IEX last trade (median)",
        f"{q['iex_trade_age_median_s']:,.0f}s" if q.get("iex_trade_age_median_s") is not None else "—",
        f"limit {q['limit_s']:,.0f}s",
        delta_color="off",
    )
    f = rep["functions_affected"]
    rows = [{"what": "cycles with new positions halted", "count": f["new_positions_halted_cycles"]}]
    rows += [{"what": k, "count": v} for k, v in f["trade_decisions_stopped"].items()]
    rows += [{"what": "opportunities stopped at the data stage", "count": f["opportunities_stopped_at_data"]}]
    rows += [{"what": f"agent {a}: opinions on non-executable data", "count": n}
             for a, n in f["agents_on_non_executable_data"].items()]  # fmt: skip
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, use_container_width=True)
    if how.get("by_day"):
        st.bar_chart(pd.DataFrame(how["by_day"]).set_index("day")[["cycles", "data_blocked"]])
    sip = rep["sip"]
    st.markdown(_md(f"**SIP report** (configured feed: `{sip['configured_feed']}`)"))
    st.markdown(_md(f"*Current limitation.* {sip['limitation']}"))
    st.markdown(_md("*What SIP would solve:*\n" + "\n".join(f"- {x}" for x in sip["would_solve"])))
    st.markdown(_md("*What it would not solve:*\n" + "\n".join(f"- {x}" for x in sip["would_not_solve"])))
    st.markdown(_md(f"*Expected benefit.* {sip['expected_benefit']}"))
    st.markdown(_md(f"*Cost.* {sip['cost']}"))
    st.markdown(_md(f"*Decision.* {sip['decision']}"))


def _evaluation() -> None:
    st.markdown(
        "**60-session evaluation** — the Brain's trading days on the Alpaca paper account against the benchmark "
        "and against the strategy it replaced (run as a shadow on its own hypothetical portfolio). A report for "
        "your review, not a target: nothing in the Brain optimises for it."
    )
    ev = guarded(lambda: api().get(f"{BASE}/evaluation"), "evaluation")
    if ev is None:
        return
    (st.success if ev["sessions"] >= ev["target_sessions"] else st.info)(
        _md(ev["status"]), icon=":material/fact_check:"
    )
    rows = []
    for name, key in (
        ("Brain", "brain"),
        ("Benchmark", "benchmark"),
        ("Previous strategy (shadow)", "previous_strategy"),
    ):
        m = ev.get(key) or {}
        rows.append(
            {
                "": name,
                "sessions": m.get("sessions", 0),
                "return": pct(m.get("total_return"), 2),
                "volatility": pct(m.get("volatility"), 1),
                "Sharpe": num(m.get("sharpe")),
                "Sortino": num(m.get("sortino")),
                "max drawdown": pct(m.get("max_drawdown"), 2),
                "excess (annual)": pct(m.get("excess_return_annual"), 2),
                "information ratio": num(m.get("information_ratio")),
                "beta": num(m.get("beta")),
            }
        )
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, use_container_width=True)
    st.caption(
        _md(
            f"Turnover {num(ev.get('turnover'))}× (Brain) · {num((ev.get('previous_strategy') or {}).get('turnover'))}× "
            f"(previous strategy). {(ev.get('previous_strategy') or {}).get('note', '')}"
        )
    )
    if ev.get("regime_performance"):
        st.markdown("**By regime** (mean daily excess return)")
        st.dataframe(
            pd.DataFrame(
                [
                    {"regime": k, "days": v["days"], "mean excess": pct(v["mean_excess"], 3)}
                    for k, v in ev["regime_performance"].items()
                ]
            ).astype(str),
            hide_index=True,
            use_container_width=True,
        )
    if ev.get("sector_exposure"):
        st.caption(
            _md("Sector exposure: " + ", ".join(f"{k} {pct(v, 0)}" for k, v in ev["sector_exposure"].items()))
        )
    card = ev["scorecard"]
    st.markdown(
        "**Learning, measured separately** (each with its sample size; *unproven* until it is large enough)"
    )
    acc, dec, ex = card["prediction_accuracy"], card["decision_quality"], card["execution_quality"]
    c = st.columns(4)
    c[0].metric("Prediction accuracy", pct(acc.get("hit_rate"), 0), acc["status"], delta_color="off")
    c[1].metric("Sound decisions", pct(dec.get("sound_decisions"), 0), dec["status"], delta_color="off")
    c[2].metric(
        "Luck share", pct(card["luck"].get("share"), 0), f"{card['luck']['judged']} judged", delta_color="off"
    )
    c[3].metric(
        "Fill slippage",
        f"{ex['slippage_bps_mean']:+.1f}bp" if ex.get("slippage_bps_mean") is not None else "—",
        f"{ex['filled']} of {ex['sent']} filled",
        delta_color="off",
    )
    rel, risk, agents = card["benchmark_relative"], card["risk_outcome"], card["agent_reliability"]
    st.caption(
        _md(
            f"Decision mix: {dec['mix']}. Closed positions {rel['closed_positions']}: beat the benchmark "
            f"{pct(rel.get('beat_benchmark'), 0)}, win rate {pct(rel.get('win_rate'), 0)} ({rel['status']}). "
            f"Stopped out {risk['stopped_out']}; halts {risk['halts'] or 'none'}. Agents with a record: "
            f"{agents['agents_with_a_record']} ({agents['verdicts'] or 'none yet'})."
        )
    )
    for c_ in ev.get("caveats") or []:
        st.caption(_md(f"• {c_}"))


def _experiment() -> None:
    """The long-term paper experiment: checkpoints, reviews, what the record says, rejected ideas, behaviour, data."""
    st.markdown(
        "**The paper experiment** — measured, never declared: every figure carries its sample, and nothing here "
        "calls the Brain successful or unsuccessful. Nothing on this tab changes a rule, a threshold or a limit."
    )
    cp = guarded(lambda: api().get(f"{BASE}/checkpoints"), "checkpoints") or {}
    if cp:
        st.markdown(f"**20 / 40 / 60-session checkpoints** — {cp.get('sessions', 0)} session(s) recorded")
        rows = []
        for key, w in [*((f"first {k}", v) for k, v in (cp.get("checkpoints") or {}).items()),
                       ("last 20", cp.get("rolling")), ("so far", cp.get("so_far"))]:  # fmt: skip
            if not w or "brain" not in w:
                rows.append({"window": key, "status": (w or {}).get("status", "—")})
                continue
            rows.append({
                "window": key, "sessions": w["sessions"], "Brain": pct(w["brain"].get("total_return"), 2),
                "benchmark": pct(w["benchmark"].get("total_return"), 2), "shadow": pct(w["shadow"].get("total_return"), 2),
                "max drawdown": pct(w["brain"].get("max_drawdown"), 2), "volatility": pct(w["brain"].get("volatility"), 1),
                "turnover/session": pct((w.get("turnover") or {}).get("per_session"), 1),
                "status": w.get("status", ""), "statistics": w["significance"].get("statement", ""),
            })  # fmt: skip
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, use_container_width=True)
    reviews = guarded(lambda: api().get(f"{BASE}/reviews", limit=10), "reviews") or []
    st.markdown("**Automatic reviews** (daily after the close, weekly after the week's last session)")
    if not reviews:
        st.caption("No review yet (the supervisor writes one after the close).")
    for r in reviews[:4]:
        with st.expander(_md(f"{r['kind'].upper()} · {r['headline']}")):
            for lesson in r["lessons"]:
                st.markdown(
                    _md(
                        f"- **{lesson['topic']}** ({lesson['strength']}, sample {lesson['sample']}): {lesson['lesson']}"
                    )
                )
            for p in r.get("proposals") or []:
                st.caption(_md(f"proposal ({p['status']}): {p['title']}"))
    lr = guarded(lambda: api().get(f"{BASE}/learning-report"), "learning report") or {}
    if lr:
        cons = lr.get("consensus") or {}
        cal = cons.get("calibration") or {}
        st.markdown(
            _md(f"**What the record says** — {lr.get('graded_calls', 0)} graded calls; consensus "
                f"{(cons.get('record') or {}).get('verdict', 'unproven')}, calibration {cal.get('status', 'unproven')}"
                + (f" (error {cal['ece']:.2f})" if cal.get("ece") is not None else ""))
        )  # fmt: skip
        agents = lr.get("agents") or {}
        if agents:
            st.dataframe(pd.DataFrame([
                {"agent": a, "verdict": v["record"]["verdict"], "hit rate": pct(v["record"].get("hit_rate"), 0),
                 "independent calls": v["record"]["n_effective"], "needs": v["needs"],
                 "calibration": v["calibration"].get("status"), "vs consensus": v["versus_consensus"]["status"]}
                for a, v in agents.items()
            ]).astype(str), hide_index=True, use_container_width=True)  # fmt: skip
        regimes = (lr.get("regimes") or {}).get("consensus") or {}
        if regimes:
            st.caption(_md("Consensus by regime: " + "; ".join(
                f"{b} {pct(c.get('hit_rate'), 0)} over {c['n_effective']} ({c['verdict']})" for b, c in regimes.items())))  # fmt: skip
    ideas = guarded(lambda: api().get(f"{BASE}/opportunity-outcomes"), "rejected ideas") or {}
    if ideas:
        st.markdown(_md(f"**Ideas considered** — {ideas['recorded']} recorded, {ideas['graded']} graded; taken vs "
                        f"rejected: {ideas['taken_vs_rejected']['status']}"))  # fmt: skip
        if ideas.get("by_reason"):
            st.dataframe(pd.DataFrame([
                {"rejected for": g["meaning"], "status": g["status"], "decisive": g["decisive"],
                 "avoided share": pct(g.get("avoided_share"), 0), "needs": g["needs"],
                 "protected control": "yes" if g["protected"] else ""}
                for g in ideas["by_reason"].values()
            ]).astype(str), hide_index=True, use_container_width=True)  # fmt: skip
    behaviour = guarded(lambda: api().get(f"{BASE}/behavior"), "behaviour") or {}
    if behaviour:
        st.markdown(_md(f"**Behaviour** (last {behaviour['window_days']} days): {behaviour['headline']}"))
        for f in behaviour["findings"]:
            icon = {"alert": "🔴", "warning": "🟠", "info": "·"}[f["severity"]]
            st.caption(_md(f"{icon} {f['code'].replace('_', ' ')}: {f['finding']} (sample {f['sample']})"))
    blockage = guarded(lambda: api().get(f"{BASE}/data-blockage"), "data blockage") or {}
    if blockage:
        st.markdown(_md(f"**When data stopped trading** — {blockage['headline']}"))
        if blockage.get("categories"):
            st.dataframe(pd.DataFrame([
                {"cause": k.replace("_", " "), "symbol-cycles": v["symbol_cycles"], "behind halts": v["behind_halts"],
                 "decisions stopped": v["decisions_stopped"], "what would address it": v["remedy"]}
                for k, v in blockage["categories"].items()
            ]).astype(str), hide_index=True, use_container_width=True)  # fmt: skip
        st.caption(_md(blockage.get("principle", "")))


def _sessions() -> None:
    data = guarded(lambda: api().get(f"{BASE}/sessions", limit=60), "sessions")
    rows = (data or {}).get("sessions") or []
    st.markdown(
        "**Trading days** (Brain-owned account): the pre-market check (paper endpoint, account, reconciliation, "
        "calendar, market data, overnight changes) and the close (equity, the day's return against the "
        "benchmark, orders, halts)."
    )
    if not rows:
        st.caption("No trading day recorded yet (recorded while the Brain owns the account).")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "day": r["day"],
                    "owner": r["owner"],
                    "return": pct(r.get("day_return"), 2),
                    "benchmark": pct(r.get("benchmark_return"), 2),
                    "excess": pct(r.get("excess_return"), 2),
                    "exposure": pct(r.get("exposure"), 0),
                    "positions": r.get("positions"),
                    "orders": f"{r['orders_sent']} sent / {r['orders_filled']} filled",
                    "cycles": r["cycles"],
                    "data-blocked": r["data_blocked_cycles"],
                    "pre-market": "ok"
                    if (r.get("premarket") or {}).get("ok")
                    else ("—" if not r.get("premarket") else "problems"),
                }
                for r in rows
            ]
        ).astype(str),
        hide_index=True,
        use_container_width=True,
    )
    pre = rows[0].get("premarket") or {}
    if pre.get("checks"):
        st.markdown(_md(f"**Latest pre-market check** ({rows[0]['day']})"))
        for c in pre["checks"]:
            mark = "✅" if c["ok"] else "❌" if c["ok"] is False else "ℹ️"
            st.caption(_md(f"{mark} {c['name']}: {c['detail']}"))


def _book() -> None:
    st.markdown(
        "**The Brain's paper book** — a hypothetical portfolio the Brain manages. Every trade the risk engine "
        "allows is simulated at the proposed price plus half the believed spread plus slippage, less fees. "
        "Nothing here reaches a broker. It is the Brain's portfolio only while it does not own the Alpaca paper "
        "account (QP_BRAIN_MODE other than paper_execution)."
    )
    book = guarded(lambda: api().get(f"{BASE}/book"), "paper book")
    if book is None:
        return
    perf = book.get("performance") or {}
    c = st.columns(5)
    c[0].metric("Equity", money(book["equity"]), help=f"started at {money(book['capital'])}")
    c[1].metric("Return", pct(perf.get("total_return"), 2))
    c[2].metric("Benchmark", pct(perf.get("benchmark_return"), 2))
    c[3].metric("Max drawdown", pct(perf.get("max_drawdown"), 2))
    c[4].metric("Sharpe", num(perf.get("sharpe")))
    if perf.get("too_short_to_judge"):
        st.info(
            f"{perf.get('sessions', 0)} session(s) recorded: too short to judge (needs 20). The numbers are "
            "reported, not trusted.",
            icon=":material/hourglass_empty:",
        )
    st.caption(
        f"Turnover {num(perf.get('turnover'))}× · slippage {money(perf.get('slippage'))} "
        f"({num(perf.get('slippage_bps'))}bp) · fees {money(perf.get('costs'))} · closed trades "
        f"{perf.get('closed_trades', 0)} (hit rate {pct(perf.get('closed_hit_rate'), 0)}) · realised "
        f"{money(perf.get('realized_pnl'))}"
    )
    curve = book.get("equity_curve") or []
    if len(curve) >= 2:
        frame = pd.DataFrame(curve).set_index("day")
        base = frame.iloc[0]
        rebased = pd.DataFrame({"paper book": frame["equity"] / base["equity"]})
        if frame["benchmark"].notna().all() and base["benchmark"]:
            rebased["benchmark"] = frame["benchmark"] / base["benchmark"]
        st.line_chart(rebased)
    if book["positions"]:
        st.markdown("**Positions** (entry, stop, the thesis and what would invalidate it)")
        st.dataframe(pd.DataFrame(book["positions"]).astype(str), hide_index=True, use_container_width=True)
    else:
        st.caption("No positions.")
    if book["trades"]:
        st.markdown("**Simulated fills** (proposed price vs fill price)")
        st.dataframe(pd.DataFrame(book["trades"]).astype(str), hide_index=True, use_container_width=True)
    a = book.get("assumptions") or {}
    st.caption(
        f"Assumptions: slippage {a.get('slippage_bps')}bp, fees {a.get('cost_bps')}bp, "
        f"{a.get('default_half_spread_bps')}bp half-spread when no bid/ask can be believed."
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
        "risk engine's answer. The Brain has no order access: an allowed trade is only *simulated* in its paper "
        "book, and nothing is ever sent to Alpaca."
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
            "④ execution": _execution(d),
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
            for line in (d.get("rationale") or {}).get("memory") or []:
                st.caption(_md(f"Memory — {line}"))


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
    patterns = guarded(
        lambda: api().get(f"{BASE}/memory", tier="long_term", kind="pattern", limit=100), "patterns"
    )
    if patterns:
        st.markdown(
            "**Recurring patterns** (counted across graded decisions; *tentative* until the sample is large "
            "and the interval excludes a coin flip). They are context for decisions and evidence for improvement "
            "proposals — they never change a rule by themselves."
        )
        for p in patterns:
            st.caption(_md(f"• {p['summary']}"))
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
        reasons = v.get("refutation") or []
        st.markdown("**Reasons it may not work**" + ("" if reasons else ": none found"))
        for r in reasons:
            st.caption(_md(f"• {r}"))
        sc = v.get("scrutiny") or {}
        if sc:
            c, cap, s = sc.get("costs") or {}, sc.get("capacity") or {}, sc.get("sensitivity") or {}
            st.caption(
                _md(
                    f"Break-even cost {c.get('break_even_bps')}bp per unit traded (assumed {c.get('assumed_cost_bps')}bp, "
                    f"turnover {c.get('annual_turnover')}×/yr) · capacity {money(cap.get('capacity_usd'))} at "
                    f"{pct(cap.get('participation'), 0)} of daily volume · {pct(s.get('positive_share'), 0)} of nearby "
                    "parameter sets still beat equal weight"
                )
            )
            if sc.get("regimes"):
                st.dataframe(
                    pd.DataFrame(sc["regimes"]).T.rename_axis("market").reset_index(),
                    hide_index=True,
                    use_container_width=True,
                )


def _improvements() -> None:
    st.markdown(
        "The Brain reviews its own record and writes **proposals** — problem, evidence, proposed change, expected "
        "improvement and validation plan. **Nothing is applied automatically**; changes are built as new versions and "
        "must pass PROPOSE → VERSION → TEST → BACKTEST → WALK-FORWARD → PAPER EVALUATION → COMPARE → PROMOTE ONLY IF "
        "VALIDATED. A proposal that would touch a protected control (loss, position and order limits, kill "
        "switches, paper-only settings, data freshness and spread, account checks) is only ever recorded as "
        "**protected_review** — for you to look at; the Brain never changes those controls."
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
    if chosen["status"] == "protected_review":
        st.warning(
            _md(p.get("protected") or "touches a protected control: for your review only"),
            icon=":material/shield:",
        )
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
        "weekends. When the Brain owns the Alpaca paper account (paper_execution) its cycles' decisions are "
        "executed by the trading service, as scheduled cycles (armed by hand once); otherwise it is analysis only "
        "— the Brain never sends orders."
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
    _sessions()
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
    status = guarded(lambda: api().get(f"{BASE}/status"), "brain status")
    if status is None:
        return
    ex = guarded(lambda: api().get(f"{BASE}/execution"), "Brain execution")
    if ex is not None:
        _execution_panel(ex)
    _layers_banner(bool(status.get("owns_account")))
    _status(status, ex)
    if status.get("limitations"):
        with st.expander("Known limitations"):
            for item in status["limitations"]:
                st.markdown(_md(f"- {item}"))
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
            "Execution",
            "Learning",
            "Strategy lab",
            "Improvements",
            "Supervisor & events",
            "Positions & theses",
            "Audit trail",
            "Market data & SIP",
            "Evaluation",
            "Experiment",
            "Paper book",
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
        _execution_tab()
    with tabs[6]:
        _learning()
    with tabs[7]:
        _lab()
    with tabs[8]:
        _improvements()
    with tabs[9]:
        _operations()
    with tabs[10]:
        _positions()
    with tabs[11]:
        _audit()
    with tabs[12]:
        _data_report()
    with tabs[13]:
        _evaluation()
    with tabs[14]:
        _experiment()
    with tabs[15]:
        _book()
    with tabs[16]:
        _memory()
    with tabs[17]:
        _history(cycles)
