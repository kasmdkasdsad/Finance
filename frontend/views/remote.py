"""Remote control — the phone page: is QuantPulse healthy, what is the Brain doing, and the stop button.

One narrow column that reads well on an iPhone: the health of every part, the Brain kill switch (one tap
stops new Brain orders and cancels its working orders; releasing it needs a second, deliberate step), the
supervisor's last and next cycle, the paper account's P&L, positions and orders, the last cycle's
agents and consensus, the opportunities taken and rejected, data-quality blocks, execution quality, the
20/40/60-session progress and the latest alerts. Nothing here shows a key, a token or a password.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.components import api, guarded, money, pct

BASE = "/brain"
BADGE = {"ok": ("green", ":material/check_circle:"), "warn": ("orange", ":material/warning:"),
         "fail": ("red", ":material/error:"), "standby": ("blue", ":material/pause_circle:"),
         "n/a": ("gray", ":material/remove:")}  # fmt: skip


def _md(text: str) -> str:
    return str(text).replace("$", "\\$")


def _when(ts: str | None) -> str:
    return (ts or "—")[:16].replace("T", " ") + (" UTC" if ts else "")


def _health(report: dict[str, Any]) -> None:
    color, icon = BADGE.get(report["status"], BADGE["n/a"])
    st.badge(f"Health: {report['status'].upper()}", icon=icon, color=color)
    for blocker in report.get("order_blockers") or []:
        st.error(_md(f"New Brain orders held: {blocker}"), icon=":material/block:")
    with st.expander("Every part", expanded=report["status"] == "fail"):
        for name, part in report["parts"].items():
            c, i = BADGE.get(part["status"], BADGE["n/a"])
            st.badge(f"{name.replace('_', ' ')} · {part['status']}", icon=i, color=c)
            st.caption(_md(part["detail"]))


def _kill_switch(ex: dict[str, Any]) -> None:
    kill = ex["brain_kill_switch"]
    if kill["active"]:
        st.error(
            _md(f"BRAIN TRADING STOPPED — {kill.get('reason') or kill['source']}"), icon=":material/block:"
        )
        if kill["source"] == "env":
            st.caption(
                "Set by QP_BRAIN_KILL_SWITCH=true on the server: change it there and restart to release."
            )
            return
        sure = st.checkbox("I have looked: let the Brain send orders again", key="remote_release_sure")
        if st.button("Allow Brain orders again", icon=":material/lock_open:", disabled=not sure,
                     use_container_width=True, key="remote_release") and guarded(
            lambda: api().post(f"{BASE}/kill-switch", {"active": False}), "Brain kill switch"
        ):  # fmt: skip
            st.rerun()
        return
    if not ex["owns_account"]:
        st.info(_md(f"QP_BRAIN_MODE={ex['mode']}: the Brain proposes only; it sends no orders."))
        return
    if st.button("STOP BRAIN TRADING", icon=":material/block:", type="primary", use_container_width=True,
                 key="remote_stop"):  # fmt: skip
        body = {"active": True, "reason": "stopped from the phone", "cancel_open_orders": True}
        if guarded(lambda: api().post(f"{BASE}/kill-switch", body), "Brain kill switch"):
            st.rerun()
    st.caption(
        "One tap: no new Brain orders, and its working orders are canceled. Positions stay as they are. The "
        "switch survives restarts."
    )
    if ex.get("blockers_scheduled"):
        st.warning(_md("Brain orders are not sent right now: " + "; ".join(ex["blockers_scheduled"])[:600]),
                   icon=":material/pause_circle:")  # fmt: skip


def _brain(sup: dict[str, Any]) -> None:
    state = (
        "paused" if sup.get("paused") else
        "standby (another server supervises)" if sup.get("standby") else
        "waiting for startup recovery" if sup.get("waiting") else
        "running" if sup.get("enabled") else "disabled"
    )  # fmt: skip
    c1, c2 = st.columns(2)
    c1.metric("Brain supervisor", state)
    c2.metric("Market", "open" if sup.get("market_open") else sup.get("session", "—").replace("_", " "))
    last = (sup.get("last") or {}).get("cycle")
    st.caption(
        _md(
            f"Last supervisor tick: {_when(sup.get('last_tick_at'))} · last full cycle: {_when(last)} · next "
            f"scheduled cycle: {_when(sup.get('next_cycle_at'))}"
        )
    )
    if sup.get("waiting"):
        st.warning(_md(f"Waiting: {sup['waiting']}"), icon=":material/hourglass_top:")


def _account(acct: dict[str, Any] | None, positions: dict[str, Any] | None, orders: list[Any] | None) -> None:
    if acct:
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", money(acct.get("equity"), 0), pct(acct.get("day_pl_pct"), 2, signed=True))
        c2.metric("Day P&L", money(acct.get("day_pl"), 0))
        c3.metric("Total P&L", money(acct.get("total_pl"), 0), pct(acct.get("total_pl_pct"), 2, signed=True))
    rows = (positions or {}).get("open") or []
    if rows:
        st.dataframe(
            pd.DataFrame(
                [{"symbol": p["symbol"], "qty": p.get("qty"), "value": money(p.get("market_value"), 0),
                  "P&L": money(p.get("unrealized_pnl"), 0), "return": pct(p.get("return_pct"), 1)} for p in rows]
            ).astype(str),
            hide_index=True, use_container_width=True,
        )  # fmt: skip
    else:
        st.caption("No open positions.")
    unexpected = (positions or {}).get("unexpected") or []
    if unexpected:
        st.warning(
            _md(f"Unexpected positions (not opened by the Brain): {unexpected}"), icon=":material/help:"
        )
    working = [
        o for o in orders or [] if o.get("status") in ("new", "accepted", "partially_filled", "pending_new")
    ]
    if working:
        st.caption(
            _md(
                "Working orders: "
                + ", ".join(f"{o['side']} {o['qty']:g} {o['symbol']}" for o in working[:10])
            )
        )


def _cycle(cycle: dict[str, Any]) -> None:
    summary = cycle.get("summary") or {}
    st.caption(
        _md(
            f"Cycle #{cycle['id']} ({cycle['kind']}, {_when(cycle.get('started_at'))}): "
            f"{summary.get('agents_run', 0)} agents, {summary.get('opinions', 0)} opinions, "
            f"{summary.get('trades_proposed', 0)} trade(s) proposed, {summary.get('risk_approved', 0)} risk-approved, "
            f"{summary.get('orders_sent', 0)} order(s) sent"
        )
    )
    halts = summary.get("entry_halts") or []
    if halts:
        st.warning(_md("New positions halted: " + ", ".join(map(str, halts))), icon=":material/front_hand:")
    rows = [
        {"symbol": c["subject"], "consensus": c.get("stance"), "score": f"{c.get('score') or 0:+.2f}",
         "confidence": pct(c.get("confidence"), 0)}
        for c in (cycle.get("consensus") or [])[:12]
    ]  # fmt: skip
    if rows:
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, use_container_width=True)
    decisions = [d for d in cycle.get("decisions") or [] if d.get("quantity")]
    for d in decisions[:8]:
        why = (d.get("execution") or {}).get("reason") or "; ".join(
            (d.get("rationale") or {}).get("reasons") or []
        )
        st.caption(_md(f"**{d['action']} {d['subject']}** ({d['status']}): {why}"[:300]))


def _experiment(cp: dict[str, Any] | None, opp: dict[str, Any] | None, data: dict[str, Any] | None) -> None:
    if cp:
        sessions = cp.get("sessions", 0)
        st.markdown(f"**20 / 40 / 60-session evaluation** — {sessions} session(s) recorded")
        st.progress(min(1.0, sessions / 60), text=" · ".join(
            f"{k}: {'reached' if sessions >= int(k) else f'{sessions}/{k}'}" for k in ("20", "40", "60")
        ))  # fmt: skip
    if opp:
        tv = opp.get("taken_vs_rejected") or {}
        st.caption(
            _md(
                f"Opportunities: {opp.get('recorded', 0)} recorded, {opp.get('graded', 0)} graded "
                f"({opp.get('open', 0)} still open). Taken vs rejected: {tv.get('status', '—')}"
            )
        )
    if data:
        st.caption(_md(f"Market data: {data.get('headline', '—')}"))


def _ago(seconds: int | None) -> str:
    if seconds is None:
        return "never"
    return (
        f"{seconds} s ago"
        if seconds < 90
        else f"{seconds // 60} min ago"
        if seconds < 5400
        else f"{seconds // 3600} h ago"
    )


def _cloud(cs: dict[str, Any]) -> None:
    """CLOUD and TRADING at a glance, from /brain/cloud-status (the existing gates, read-only)."""
    sv, sup, db = cs["service"], cs["supervisor"], cs["database"]
    ae = cs["autonomous_execution"]
    if ae["permitted"]:
        st.success("Autonomous PAPER execution: PERMITTED — the Brain may send paper orders on its own now.",
                   icon=":material/smart_toy:")  # fmt: skip
    else:
        st.warning(_md("Autonomous PAPER execution: not permitted now — " + "; ".join(ae["reasons"])[:700]),
                   icon=":material/pause_circle:")  # fmt: skip
    st.markdown("**Cloud**")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Service", "online", f"v{sv['version']} · {sv.get('short_commit') or 'local'}", delta_color="off"
    )
    c2.metric("Uptime", _ago(sv.get("uptime_seconds")).replace(" ago", ""))
    c3.metric("Database", db["status"], db.get("schema") or "", delta_color="off")
    leader = sup.get("leader") or {}
    recon = cs.get("reconciliation") or {}
    last = cs.get("last_cycle") or {}
    st.caption(
        _md(
            f"Supervisor: {'this instance leads' if sup.get('this_process_is_leader') else 'leader ' + str(leader.get('holder'))}"
            f" (lease {'live' if leader.get('live') else 'NOT live'}; {len(sup.get('standby_processes') or {})} "
            f"standing by) · last tick {_ago(sup.get('last_tick_age_seconds'))} · last reconciliation "
            f"{_ago(recon.get('last_success_age_seconds'))} · last Brain cycle: {last.get('detail', '—')}"
        )
    )
    st.markdown("**Trading**")
    sw, market, today = cs["switches"], cs["market"], cs.get("today") or {}
    t1, t2, t3 = st.columns(3)
    t1.metric("Paper endpoint", "verified" if cs["alpaca"]["paper_endpoint_verified"] else "NOT verified")
    t2.metric(
        "Market", "open" if market["open"] else "closed", market["new_york_time"] + " NY", delta_color="off"
    )
    t3.metric("Orders / fills today", f"{today.get('orders', 0)} / {today.get('fills', 0)}")
    bk = (sw.get("brain_kill_switch") or {}).get("active")
    st.caption(
        _md(
            f"Trading enabled {sw['trading_enabled']} · dry run {sw['dry_run']} · Brain mode {sw['brain_mode']} · "
            f"Brain kill switch {'ON' if bk else 'off'} · Alpaca {cs['alpaca']['connectivity']['status']} · "
            f"data: {(cs.get('data') or {}).get('detail', '—')}"
        )
    )
    for key, label in (("latest_decision", "Latest decision"), ("latest_rejection", "Latest rejection")):
        d = cs.get(key)
        if d:
            st.caption(_md(f"{label}: **{d['action']} {d['symbol']}** ({d['status']})" + (f" — {d['reason']}" if d.get("reason") else "")))  # fmt: skip


def render() -> None:
    st.title("QuantPulse · Remote", anchor=False)
    st.badge("ALPACA PAPER · simulated money", icon=":material/science:", color="orange")
    report = guarded(lambda: api().get("/system/health"), "health")
    if report:
        _health(report)
    ex = guarded(lambda: api().get(f"{BASE}/execution"), "Brain execution")
    if ex:
        _kill_switch(ex)
    cs = guarded(lambda: api().get(f"{BASE}/cloud-status"), "cloud status")
    if cs:
        _cloud(cs)
    sup = guarded(lambda: api().get(f"{BASE}/supervisor"), "supervisor")
    if sup:
        _brain(sup)
    st.subheader("Paper account", anchor=False)
    _account(
        guarded(lambda: api().get("/trading/account"), "account"),
        guarded(lambda: api().get(f"{BASE}/positions"), "positions"),
        guarded(lambda: api().get("/trading/orders"), "orders"),
    )
    st.subheader("Last cycle", anchor=False)
    cycles = guarded(lambda: api().get(f"{BASE}/cycles", limit=1), "cycles") or []
    if cycles:
        cycle = guarded(lambda: api().get(f"{BASE}/cycles/{cycles[0]['id']}"), "cycle")
        if cycle:
            _cycle(cycle)
    else:
        st.caption("No cycle yet.")
    learning = guarded(lambda: api().get(f"{BASE}/learning"), "learning") or {}
    if learning:
        preds = learning.get("predictions") or {}
        st.caption(
            _md(
                "Learning: "
                + ", ".join(f"{k} {v}" for k, v in preds.items() if isinstance(v, int | float))[:300]
            )
        )
    with st.expander("Execution, opportunities, experiment"):
        trades = guarded(lambda: api().get(f"{BASE}/trades"), "trades") or []
        sent = [t for t in trades if t.get("sent")]
        st.caption(_md(f"Recent trade decisions: {len(trades)}, sent to Alpaca: {len(sent)}"))
        for t in trades[:6]:
            st.caption(_md(f"{t['at'][:16]} {t['action']} {t['subject']}: {t['status']}" + (f" ({t['reason']})" if t.get("reason") else "")))  # fmt: skip
        _experiment(
            guarded(lambda: api().get(f"{BASE}/checkpoints"), "checkpoints"),
            guarded(lambda: api().get(f"{BASE}/opportunity-outcomes"), "opportunities"),
            guarded(lambda: api().get(f"{BASE}/data-blockage"), "data blockage"),
        )
    with st.expander("Recent alerts"):
        alerts = guarded(lambda: api().get("/system/alerts"), "alerts") or {}
        channels = ", ".join(alerts.get("channels") or []) or "none configured (recorded only)"
        st.caption(_md(f"Delivered to: {channels}; heartbeat {'on' if alerts.get('heartbeat') else 'off'}"))
        for a in (alerts.get("recent") or [])[:10]:
            st.caption(_md(f"{a['at'][:16]} [{a['severity']}] {a['title']}: {a['message']}"[:300]))
