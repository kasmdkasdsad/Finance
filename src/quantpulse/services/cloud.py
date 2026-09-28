"""The cloud status: is QuantPulse running, is the Brain supervising, and may it trade on its own right now?

``GET /api/v1/brain/cloud-status`` (API token required) answers in one place — service, database, Alpaca
(paper endpoint verified, reachable), the supervisor (leader, lease, last tick, recovery, processes standing
by), the last reconciliation and Brain cycle, the market, data freshness, the switches and kill switches,
today's orders and fills, the latest decision and rejection — and ends with ``autonomous_execution``:
``permitted`` or not, and every reason why not.

It adds no gate of its own. The reasons are the existing ones, read without side effects: the trading
service's own submission blockers for Brain orders (paper, keys, trading switches, dry run, Brain mode, both
kill switches, arming), the health checks that fail Brain orders closed, a failed reconciliation, the
supervisor's startup recovery, and the market clock. The supervisor's heartbeat and the reconciliation history
are read from the database, so any instance — the one serving the request during a deploy, too — tells the
truth about the one that supervises. Nothing here is a secret: no key, token, password or account number.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, or_, select

from quantpulse.core import runtime
from quantpulse.core.market_calendar import NEW_YORK, is_market_open, next_open, session_at
from quantpulse.db import migrate
from quantpulse.db.models import BrainCycleRow, BrainDecisionRow, BrokerOrderRow, TradingEventRow
from quantpulse.providers.alpaca_trading import PAPER_URL, BrokerError
from quantpulse.services import preflight

if TYPE_CHECKING:
    from quantpulse.services.container import Container

HEARTBEAT_STALE = timedelta(minutes=5)
FILLED = ("filled", "partially_filled")
LEADER_FIELDS = (
    "holder",
    "live",
    "acquired_at",
    "heartbeat_at",
    "heartbeat_age_seconds",
    "expires_at",
    "expires_in_seconds",
    "clock",
)


async def cloud_status(c: Container) -> dict[str, Any]:
    s, now = c.settings, c.clock.now()
    rt = runtime.current()
    health = c.health
    database = await health.database_part()
    db_ok = database["status"] == "ok"
    if db_ok:
        try:
            database["schema"] = await asyncio.to_thread(migrate.current_revision, s.database_url)
        except Exception as exc:  # reported, not raised
            database["schema"] = f"unknown ({type(exc).__name__})"
        database["schema_head"] = migrate.head_revision()
        database["schema_at_head"] = database["schema"] == database["schema_head"]

    # --- Alpaca: the paper endpoint (verified without a network call) and reachability ----------
    key = s.alpaca_api_key_id.get_secret_value() if s.alpaca_api_key_id is not None else ""
    try:
        endpoint = c.trading.verify_paper()
        verified = endpoint == PAPER_URL
    except BrokerError as exc:
        endpoint, verified = f"not verified: {type(exc).__name__}", False
    alpaca = {
        "paper_endpoint_verified": verified,
        "endpoint": endpoint,
        "paper_setting": s.alpaca_paper,
        "paper_key": key.startswith("PK"),
        # there is no live path at all; this says the running configuration is verified as paper
        "account": "PAPER" if verified and s.alpaca_paper and key.startswith("PK") else "NOT VERIFIED",
        "live_trading_possible": False,
        "connectivity": await health.alpaca_part(now),
    }

    # --- the supervisor (its heartbeat is in the database: true from any instance) ---------------
    sup = await c.brain.supervisor.status() if db_ok else {}
    beat = sup.get("heartbeat") or {}
    lease = sup.get("lease") or {}
    last_tick = datetime.fromisoformat(beat["last_tick_at"]) if beat.get("last_tick_at") else None
    supervisor = {
        "enabled": s.brain_supervisor_enabled,
        "paused": sup.get("paused"),
        "this_instance": rt.instance,
        "this_process_is_leader": bool(lease.get("mine")),
        "leader": {k: lease.get(k) for k in LEADER_FIELDS},
        "role": "leader" if lease.get("mine") else "standby" if lease.get("live") else "none",
        "standby_processes": sup.get("standby_processes") or {},
        "last_tick_at": beat.get("last_tick_at"),
        "last_tick_age_seconds": round((now - last_tick).total_seconds()) if last_tick else None,
        "last_result": beat.get("last_result"),
        "recovered": beat.get("recovered"),
        "waiting": beat.get("waiting"),
        "next_cycle_at": sup.get("next_cycle_at"),
        "session": sup.get("session"),
    }

    # --- reconciliation, the last cycle, today's orders, the latest decisions ---------------------
    recon = await _reconciliation(c, now) if db_ok else {}
    cycles = await health.cycles_part(now) if db_ok else {}
    activity = await _activity(c, now) if db_ok else {}

    # --- switches -------------------------------------------------------------------------------
    brain_kill = await c.trading.brain_kill_switch() if db_ok else None
    trading_kill = await c.trading.kill_switch() if db_ok else None
    switches = {
        "paper": s.alpaca_paper,
        "trading_enabled": s.alpaca_trading_enabled,
        "dry_run": s.trading_dry_run,
        "brain_mode": s.brain_mode,
        "brain_owns_account": s.brain_owns_account,
        "strategy_scheduler_enabled": s.trading_scheduler_enabled,
        "scheduled_orders_require_arming": s.trading_scheduler_requires_arming,
        "brain_kill_switch": brain_kill.model_dump(mode="json") if brain_kill else None,
        "trading_kill_switch": trading_kill.model_dump(mode="json") if trading_kill else None,
    }
    market = {
        "open": is_market_open(now),
        "session": session_at(now).value,
        "next_open": next_open(now).isoformat(),
        "new_york_time": now.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M"),
    }

    # --- may the Brain trade on its own right now? (the existing gates, read without side effects) ----
    reasons: list[str] = []
    if s.deployment == "cloud":
        reasons += [f"preflight: {f.name}: {f.detail}" for f in preflight.run(s).failures]
    if not db_ok:
        reasons.append(f"database: {database['detail']}")
    else:
        if not s.brain_supervisor_enabled:
            reasons.append("the Brain supervisor is disabled (QP_BRAIN_SUPERVISOR_ENABLED=false)")
        elif sup.get("paused"):
            reasons.append("the Brain supervisor is paused (dashboard/API)")
        if not lease.get("live"):
            reasons.append("no process holds the supervisor lease (starting, or stopped)")
        if last_tick is None or now - last_tick > HEARTBEAT_STALE:
            reasons.append(
                "the supervisor has not ticked "
                + (f"for {round((now - last_tick).total_seconds() / 60)} min" if last_tick else "yet")
            )
        if beat.get("waiting") or (beat and not beat.get("recovered")):
            reasons.append(f"startup recovery has not passed: {beat.get('waiting') or 'pending'}")
        kill = trading_kill
        assert kill is not None
        reasons += await c.trading.submit_blockers(kill, scheduled=True, owner="brain")
    if lease.get("mine"):  # this process supervises: its own fail-closed state applies
        reasons += await health.order_blockers()
        if c.trading.reconcile_error is not None:
            reasons.append(f"the last reconciliation failed: {c.trading.reconcile_error[1][:160]}")
    if recon.get("last_failed_after_success"):
        reasons.append(f"the last reconciliation attempt failed: {recon.get('last_error')}")
    if not market["open"]:
        reasons.append(f"the market is closed (next open {market['next_open']})")
    entry_reasons: list[str] = []
    md = cycles.get("market_data") or {}
    if md.get("status") == "warn":
        entry_reasons.append(f"market data: {md.get('detail')}")
    reasons = list(dict.fromkeys(reasons))
    return {
        "checked_at": now.isoformat(),
        "service": {
            "status": "ok",
            **rt.as_dict(),
            "deployment": s.deployment,
            "started_at": c.started_at.isoformat(),
            "uptime_seconds": round((now - c.started_at).total_seconds()),
        },
        "database": database,
        "alpaca": alpaca,
        "supervisor": supervisor,
        "reconciliation": recon,
        "last_cycle": cycles.get("last_cycle"),
        "market": market,
        "data": md,
        "switches": switches,
        "today": activity.get("today"),
        "latest_decision": activity.get("latest_decision"),
        "latest_rejection": activity.get("latest_rejection"),
        "autonomous_execution": {
            "permitted": not reasons,
            "reasons": reasons,
            "new_entries_permitted": not reasons and not entry_reasons,
            "entry_reasons": entry_reasons,
            "note": "exits, the kill switches and close-all are never blocked by these reasons",
        },
    }


async def _reconciliation(c: Container, now: datetime) -> dict[str, Any]:
    async with c.db.session() as session:
        rows = (
            await session.scalars(
                select(TradingEventRow)
                .where(TradingEventRow.kind.in_(("reconciliation_completed", "reconciliation_failed")))
                .order_by(TradingEventRow.id.desc())
                .limit(20)
            )
        ).all()
    ok = next((r for r in rows if r.kind == "reconciliation_completed"), None)
    bad = next((r for r in rows if r.kind == "reconciliation_failed"), None)
    return {
        "last_success_at": ok.created_at.isoformat() if ok else None,
        "last_success_age_seconds": round((now - ok.created_at).total_seconds()) if ok else None,
        "last_success": ok.message[:300] if ok else None,
        "last_error_at": bad.created_at.isoformat() if bad else None,
        "last_error": bad.message[:300] if bad else None,
        "last_failed_after_success": bool(bad and (ok is None or bad.id > ok.id)),
    }


async def _activity(c: Container, now: datetime) -> dict[str, Any]:
    local = now.astimezone(NEW_YORK)
    day_start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    async with c.db.session() as session:
        orders = await session.scalar(
            select(func.count()).select_from(BrokerOrderRow).where(BrokerOrderRow.created_at >= day_start)
        )
        brain_orders = await session.scalar(
            select(func.count())
            .select_from(BrokerOrderRow)
            .where(BrokerOrderRow.created_at >= day_start, BrokerOrderRow.strategy == "brain")
        )
        fills = await session.scalar(
            select(func.count())
            .select_from(BrokerOrderRow)
            .where(
                BrokerOrderRow.status.in_(FILLED),
                or_(BrokerOrderRow.filled_at >= day_start, BrokerOrderRow.updated_at >= day_start),
            )
        )
        decisions = (
            await session.scalars(
                select(BrainDecisionRow)
                .where(BrainDecisionRow.quantity.is_not(None))
                .order_by(BrainDecisionRow.id.desc())
                .limit(50)
            )
        ).all()
        last_cycle = (
            await session.scalars(select(BrainCycleRow).order_by(BrainCycleRow.id.desc()).limit(1))
        ).first()
    latest = decisions[0] if decisions else None
    rejected = next(
        (d for d in decisions if d.risk_approved is False or not (d.execution or {}).get("sent", False)), None
    )
    return {
        "today": {
            "date": local.date().isoformat(),
            "orders": int(orders or 0),
            "brain_orders": int(brain_orders or 0),
            "fills": int(fills or 0),
            "last_cycle_id": last_cycle.id if last_cycle else None,
        },
        "latest_decision": _decision(latest),
        "latest_rejection": _decision(rejected, why=True),
    }


def _decision(d: BrainDecisionRow | None, *, why: bool = False) -> dict[str, Any] | None:
    if d is None:
        return None
    ex, rationale = d.execution or {}, d.rationale or {}
    reason = ex.get("reason") or "; ".join(
        (rationale.get("blocked_by") or rationale.get("reasons") or [])[:3]
    )
    out = {
        "id": d.id,
        "cycle_id": d.cycle_id,
        "at": d.created_at.isoformat() if d.created_at else None,
        "symbol": d.subject,
        "action": d.action,
        "status": d.status,
        "quantity": d.quantity,
        "risk_approved": d.risk_approved,
        "sent": ex.get("sent"),
    }
    if why or not ex.get("sent"):
        out["reason"] = str(reason)[:300]
    return out
