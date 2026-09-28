"""Health of the 24/7 deployment — and the Brain failing closed when it is not healthy.

Once a minute (a poller job) the monitor checks each part and keeps the result:

==================  ======================================================================================
api                 the process: version, deployment, uptime
database            ``SELECT 1`` within a few seconds
supervisor          enabled, not paused, startup recovery passed, ticking (stalled after 5 minutes), the
                    single-supervisor lease held by this process — or *standby* while another one holds it
scheduler           the background poller runs, and the Brain's job ran recently
alpaca              the paper account can be read (Alpaca reachable, keys accepted)
market_data         the last in-session cycle's quote coverage and the data-quality veto, if any
reconciliation      the last reconciliation with Alpaca succeeded, and recently (in the session)
last_cycle          the last completed Brain cycle; a failed one; overdue in the session
kill_switches       the Brain and trading kill switches (on is not a failure, but it is shown)
==================  ======================================================================================

Each part is ``ok``, ``warn``, ``fail``, ``standby`` or ``n/a``; the overall status is the worst of them.

**Failing closed.** While the database or Alpaca is failing, the trading service refuses every new Brain
order at its last gate (:meth:`order_blockers`), and orders resume by themselves once the next check passes;
after a failed reconciliation, until one succeeds. Execution anomalies — ``QP_BRAIN_ANOMALY_ORDERS_PER_HOUR`` Brain orders
rejected, failed or left unknown within an hour — turn the Brain kill switch on (persisted; a person releases
it from the dashboard after looking). Exits, the trading kill switch and close-all are unaffected.

**Alerts** go out on changes, not on every check: Brain stopped or waiting (also seen from a standby: the
leader holds the lease but stopped ticking), the supervisor lease lost to another process, a takeover from a
supervisor that died holding it, reconciliation failed, an unexpected position, a kill switch turned on,
repeated data-quality halts, Alpaca unreachable, the database failing, an execution anomaly (repeated
rejections), Brain cycles failing in a row, autonomous execution blocked for ten minutes in the session —
and a note when things recover, and at every start. Normal skipped trades never alert.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text

from quantpulse import __version__
from quantpulse.core.market_calendar import is_market_open
from quantpulse.db.models import BrainCycleRow, TradingEventRow
from quantpulse.services.alerts import Alert
from quantpulse.services.order_manager import is_brain

if TYPE_CHECKING:
    from quantpulse.services.container import Container

logger = logging.getLogger(__name__)
ORDER = {"n/a": 0, "ok": 0, "standby": 0, "warn": 1, "fail": 2}
STALLED = timedelta(minutes=5)
DB_TIMEOUT = 5.0
ALPACA_TIMEOUT = 15.0
ALPACA_EVERY = timedelta(seconds=60)
ALPACA_EVERY_CLOSED = timedelta(minutes=5)
RECONCILE_STALE = timedelta(minutes=15)
GATE_FRESH = timedelta(minutes=3)  # a failing check older than this no longer blocks (it is re-checked)
# Parts whose failure stops new Brain orders at the last gate (a failed reconciliation stops them too, read
# live by the trading service itself, so a successful one releases them at once).
ORDER_CRITICAL = ("database", "alpaca")
ANOMALY_KINDS = ("order_rejected", "order_failed", "order_unknown")
FAILED_CYCLES_ALERT = 3  # Brain cycles failing in a row before an alert
BLOCKED_ALERT_AFTER = timedelta(minutes=10)  # autonomous execution blocked this long, in the session
BLOCKED_EVERY = timedelta(minutes=5)  # how often the leader re-reads the execution gates for that alert
ANOMALY_WINDOW = timedelta(hours=1)


def part(status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **extra}


class HealthMonitor:
    def __init__(self, container: Container) -> None:
        self._c = container
        self.last: dict[str, Any] | None = None
        self._alpaca: tuple[datetime, dict[str, Any]] | None = None
        self._previous: dict[str, str] = {}
        self._announced = False
        self._anomaly_seen_id = 0
        self._unexpected: set[str] = set()
        self._kills: dict[str, bool] = {}
        self._reported: set[str] = set()  # one-off events already alerted (a takeover, a lost lease)
        self._blocked_since: datetime | None = None
        self._blocked_checked: datetime | None = None
        self._blocked_alerted = False

    # ------------------------------------------------------------------ checks
    async def check(self) -> dict[str, Any]:
        c, now = self._c, self._c.clock.now()
        parts: dict[str, dict[str, Any]] = {
            "api": part(
                "ok",
                f"QuantPulse {__version__} ({c.settings.deployment}), up {_ago(now - c.started_at)}",
                version=__version__,
                deployment=c.settings.deployment,
                started_at=c.started_at.isoformat(),
            )
        }
        parts["database"] = await self.database_part()
        db_ok = parts["database"]["status"] == "ok"
        parts["supervisor"] = await self._supervisor(now) if db_ok else part("n/a", "database unavailable")
        parts["scheduler"] = self._scheduler(now)
        parts["alpaca"] = await self.alpaca_part(now)
        cycles = await self.cycles_part(now) if db_ok else {}
        parts["market_data"] = cycles.get("market_data") or part("n/a", "database unavailable")
        parts["reconciliation"] = self._reconciliation(now)
        parts["last_cycle"] = cycles.get("last_cycle") or part("n/a", "database unavailable")
        parts["kill_switches"] = await self._kill_switches() if db_ok else part("n/a", "database unavailable")
        worst = max(parts.values(), key=lambda p: ORDER.get(p["status"], 0))["status"]
        overall = {"fail": "fail", "warn": "warn"}.get(worst, "ok")
        report = {
            "status": overall,
            "checked_at": now.isoformat(),
            "parts": parts,
            "order_blockers": self._blockers_from(parts),
            "alerts": {"channels": c.alerts.channels, "heartbeat": c.settings.heartbeat_url is not None},
        }
        self.last = report
        return report

    async def database_part(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            async with self._c.db.session() as s:
                await asyncio.wait_for(s.execute(text("SELECT 1")), DB_TIMEOUT)
        except Exception as exc:
            return part("fail", f"the database is not answering ({type(exc).__name__})")
        ms = (time.perf_counter() - started) * 1000
        backend = self._c.settings.database_url.split(":", 1)[0].split("+", 1)[0]
        return part("ok", f"{backend} answered in {ms:.0f} ms", latency_ms=round(ms, 1))

    async def _supervisor(self, now: datetime) -> dict[str, Any]:
        s, sup = self._c.settings, self._c.brain.supervisor
        if not s.brain_supervisor_enabled:
            return part("warn", "the supervisor is disabled (QP_BRAIN_SUPERVISOR_ENABLED=false)")
        status = await sup.status()
        extra = {
            "last_tick_at": status["last_tick_at"],
            "next_cycle_at": status["next_cycle_at"],
            "last_result": status["last_result"],
            "lease": status["lease"],
            "session": status["session"],
        }
        if status["paused"]:
            return part("warn", "paused from the dashboard/API: no cycles run", **extra)
        if status["standby"]:
            beat = status.get("heartbeat") or {}
            leader_tick = beat.get("last_tick_at")
            age = now - datetime.fromisoformat(leader_tick) if leader_tick else None
            if age is not None and age > STALLED:  # the leader holds the lease but has stopped ticking
                return part("fail", f"supervisor lost: the leader ({beat.get('holder')}) has not ticked for "
                                    f"{_ago(age)}", **extra)  # fmt: skip
            return part("standby", status["standby"], **extra)
        if status["waiting"]:
            return part("fail", f"waiting: startup recovery has not passed ({status['waiting']})", **extra)
        last_tick = sup.last_tick_at
        if last_tick is None:
            if self._c.poller.running and now - self._c.started_at > STALLED:
                return part("fail", "the supervisor has not ticked since the start", **extra)
            return part("ok", "starting: the first tick is due within a minute", **extra)
        if self._c.poller.running and now - last_tick > STALLED:
            return part("fail", f"stalled: the last tick was {_ago(now - last_tick)} ago", **extra)
        lease = status["lease"] or {}
        if lease and not lease.get("mine"):
            return part("warn", "this process does not hold the supervisor lease right now", **extra)
        return part(
            "ok", f"ticking (last {_ago(now - last_tick)} ago): {status['last_result'] or '-'}", **extra
        )

    def _scheduler(self, now: datetime) -> dict[str, Any]:
        poller = self._c.poller
        if not self._c.settings.polling_enabled:
            return part("warn", "background jobs are off (QP_POLLING_ENABLED=false): nothing runs by itself")
        if not poller.running:
            return part("fail", "the background scheduler is not running")
        brain = poller.jobs.get("brain")
        if brain is not None and brain.last_run is not None and now - brain.last_run > STALLED:
            return part("fail", f"the Brain's job last ran {_ago(now - brain.last_run)} ago")
        failing = {
            n: j.last_error for n, j in poller.jobs.items() if j.last_error and n in ("brain", "trading")
        }
        if brain is not None and brain.failures and brain.last_error and brain.last_result is None:
            return part("warn", f"the Brain's job failed: {brain.last_error}"[:300], failing=failing)
        return part("ok", f"{len(poller.jobs)} background jobs running")

    async def alpaca_part(self, now: datetime) -> dict[str, Any]:
        broker = self._c.broker
        if not broker.configured():
            status = "fail" if self._c.settings.deployment == "cloud" else "n/a"
            return part(status, "Alpaca paper keys are not set")
        every = ALPACA_EVERY if is_market_open(now) else ALPACA_EVERY_CLOSED  # closed: no need to ask often
        if self._alpaca is not None and now - self._alpaca[0] < every:
            return self._alpaca[1]
        try:
            account = await asyncio.wait_for(broker.account(), ALPACA_TIMEOUT)
            result = part(
                "ok", f"the paper account answered (equity ${account.equity:,.0f})", endpoint=broker.base_url
            )
        except Exception as exc:
            result = part("fail", f"the Alpaca paper API is not answering ({type(exc).__name__})")
        self._alpaca = (now, result)
        return result

    def _reconciliation(self, now: datetime) -> dict[str, Any]:
        t = self._c.trading
        if not (self._c.settings.brain_owns_account and self._c.broker.configured()):
            return part("n/a", "the Brain does not manage an Alpaca paper account here")
        if t.reconcile_error is not None:
            at, error = t.reconcile_error
            return part("fail", f"the last reconciliation failed {_ago(now - at)} ago: {error}"[:300])
        last = t.last_reconciled_at
        if last is None:
            return part("warn", "not reconciled with Alpaca yet (startup recovery does it first)")
        if is_market_open(now) and now - last > RECONCILE_STALE:
            return part(
                "warn", f"the last reconciliation was {_ago(now - last)} ago", last_at=last.isoformat()
            )
        return part("ok", f"reconciled {_ago(now - last)} ago", last_at=last.isoformat())

    async def cycles_part(self, now: datetime) -> dict[str, dict[str, Any]]:
        s = self._c.settings
        async with self._c.db.session() as session:
            rows = (
                await session.scalars(
                    select(BrainCycleRow).order_by(BrainCycleRow.started_at.desc()).limit(20)
                )
            ).all()
        completed = [r for r in rows if r.status == "completed"]
        last = completed[0] if completed else None
        failed_streak = 0
        for r in rows:  # newest first: cycles that failed in a row (a running one is skipped)
            if r.status == "running":
                continue
            if r.status != "failed":
                break
            failed_streak += 1
        failed = next((r for r in rows if r.status == "failed"), None)
        if last is None:
            cycle = part("n/a", "no completed Brain cycle yet")
        else:
            age = now - (last.finished_at or last.started_at)
            overdue = is_market_open(now) and age > timedelta(minutes=2 * s.brain_cycle_minutes + 5)
            detail = f"#{last.id} ({last.kind}) finished {_ago(age)} ago"
            if failed is not None and failed.started_at > last.started_at:
                cycle = part("warn", f"{detail}; a later cycle #{failed.id} failed: {failed.error}"[:300])
            else:
                cycle = part("warn" if overdue else "ok", detail + ("; overdue" if overdue else ""))
            cycle["last_at"] = (last.finished_at or last.started_at).isoformat()
        cycle["failed_streak"] = failed_streak
        if failed_streak >= FAILED_CYCLES_ALERT and cycle["status"] != "fail":
            cycle = {**cycle, "status": "warn", "detail": f"{failed_streak} Brain cycles failed in a row; "
                                                          f"{cycle['detail']}"}  # fmt: skip
        if not is_market_open(now):
            data = part("n/a", "the market is closed: no executable quotes are expected")
        else:
            in_session = [r for r in completed if (r.market or {}).get("open")]
            if not in_session:
                data = part("n/a", "no in-session cycle yet")
            else:
                dq = in_session[0].data_quality or {}
                diag = dq.get("diagnosis") or {}
                usable = sum(1 for d in diag.values() if d.get("status") in ("fresh", "live"))
                veto = str((dq.get("market") or {}).get("veto") or "")
                halted = [
                    r for r in in_session if "data_quality" in ((r.summary or {}).get("entry_halts") or [])
                ]
                streak = 0
                for r in in_session:
                    if "data_quality" not in ((r.summary or {}).get("entry_halts") or []):
                        break
                    streak += 1
                detail = f"{usable}/{len(diag)} studied quotes usable in cycle #{in_session[0].id}"
                data = part(
                    "warn" if veto or streak else "ok",
                    detail + (f"; new positions halted: {veto}" if veto else ""),
                    halted_streak=streak,
                    halted_recent=len(halted),
                )
        return {"last_cycle": cycle, "market_data": data}

    async def _kill_switches(self) -> dict[str, Any]:
        t = self._c.trading
        brain, trading = await t.brain_kill_switch(), await t.kill_switch()
        on = [
            f"{name} ({k.reason or k.source})"
            for name, k in (("Brain", brain), ("trading", trading))
            if k.active
        ]
        return part(
            "warn" if on else "ok",
            "ON: " + "; ".join(on) if on else "both off",
            brain=brain.active,
            trading=trading.active,
            brain_reason=brain.reason,
            trading_reason=trading.reason,
        )

    # ------------------------------------------------------------------ failing closed
    def _blockers_from(self, parts: dict[str, dict[str, Any]]) -> list[str]:
        return [
            f"health check: {name} failing ({parts[name]['detail']})"[:240]
            for name in ORDER_CRITICAL
            if parts.get(name, {}).get("status") == "fail"
        ]

    async def order_blockers(self) -> list[str]:
        """Reasons the last health check gives not to send a Brain order (empty: none). Read by the trading
        service's last gate; a check older than a few minutes no longer blocks (it is due to be re-run)."""
        report = self.last
        if report is None:
            return []
        if self._c.clock.now() - datetime.fromisoformat(report["checked_at"]) > GATE_FRESH:
            return []
        return list(report["order_blockers"])

    async def anomalies(self) -> list[str]:
        """Brain orders rejected, failed or left unknown within the last hour (newest first)."""
        now = self._c.clock.now()
        async with self._c.db.session() as s:
            rows = (
                await s.scalars(
                    select(TradingEventRow)
                    .where(
                        TradingEventRow.kind.in_(ANOMALY_KINDS),
                        TradingEventRow.created_at >= now - ANOMALY_WINDOW,
                    )
                    .order_by(TradingEventRow.id.desc())
                    .limit(50)
                )
            ).all()
        return [
            f"{r.kind}: {r.message}"[:200] for r in rows if r.client_order_id and is_brain(r.client_order_id)
        ]

    # ------------------------------------------------------------------ the minute job
    async def run(self) -> str:
        """One health round: check, fail closed on execution anomalies, alert on changes, heartbeat."""
        c = self._c
        try:
            report = await self.check()
        except Exception as exc:  # the monitor itself failing is a failure to report, not to hide
            logger.exception("health check failed")
            await c.alerts.send(Alert("health_check", "Health check failed", f"{type(exc).__name__}: {exc}"[:300],
                                      "critical"))  # fmt: skip
            await c.alerts.heartbeat(False)
            return f"health check failed: {type(exc).__name__}"
        parts = report["parts"]
        if not self._announced:
            self._announced = True
            await c.alerts.send(
                Alert("started", "QuantPulse started",
                      f"{c.settings.deployment} deployment, health {report['status']}; startup recovery "
                      "(reconciliation, positions, working orders, safety checks) runs before any order.",
                      "info"),
                force=True,
            )  # fmt: skip
        # actions (not checks) belong to the leader alone: a standby during a deploy repeats none of them
        leader = parts["database"]["status"] == "ok" and (c.lease is None or await c.lease.held())
        if leader:
            await self._fail_closed_on_anomalies()
            await self._unexpected_positions()
            if c.broker.configured():
                try:  # a kill switch turned on during an Alpaca outage still owes its cancellations
                    await c.trading.retry_pending_cancels()
                except Exception:
                    logger.warning("retrying the kill switch's cancellations failed", exc_info=True)
        await self._transitions(parts)
        await self._supervisor_events()
        if leader:
            await self._execution_blocked()
        await c.alerts.heartbeat(report["status"] != "fail")
        return f"health {report['status']}"

    async def _supervisor_events(self) -> None:
        """One alert per event: this process lost the lease to another holder, or took it over from a
        supervisor that died holding it."""
        sup, alerts = self._c.brain.supervisor, self._c.alerts
        lost = sup.lost
        if lost and f"lost:{lost['at']}" not in self._reported:
            self._reported.add(f"lost:{lost['at']}")
            await alerts.send(Alert("supervisor_lost", "Brain supervisor lost its lease",
                                    f"another process ({lost['to']}) now supervises the Brain; this one stands by. "
                                    "Outside a deploy that means a second installation on the same database, or a "
                                    "long pause.", "warning"), force=True)  # fmt: skip
        takeover = sup.takeover
        if takeover and f"takeover:{takeover['at']}" not in self._reported:
            self._reported.add(f"takeover:{takeover['at']}")
            await alerts.send(Alert("supervisor_takeover", "Brain supervisor restarted after a crash",
                                    f"the previous supervisor ({takeover['previous_holder']}) stopped without handing "
                                    f"over (last heartbeat {takeover.get('previous_heartbeat_at')}); this process "
                                    "took over, reconciled with Alpaca and re-ran the safety audit before anything "
                                    "else.", "warning"), force=True)  # fmt: skip

    async def _execution_blocked(self) -> None:
        """The Brain owns the account and the market is open, yet autonomous execution has been refused for a
        while: say why (once per cool-down), and when it resumes. Read through the cloud status — the same
        gates the orders pass, never a second copy of them."""
        c = self._c
        now = c.clock.now()
        if not (
            c.settings.brain_owns_account and c.settings.brain_supervisor_enabled and is_market_open(now)
        ):
            self._blocked_since, self._blocked_alerted = None, False
            return
        if self._blocked_checked is not None and now - self._blocked_checked < BLOCKED_EVERY:
            return
        self._blocked_checked = now
        from quantpulse.services.cloud import cloud_status

        try:
            ae = (await cloud_status(c))["autonomous_execution"]
        except Exception:
            logger.warning("reading the execution gates for the health alert failed", exc_info=True)
            return
        reasons = [r for r in ae["reasons"] if "market is closed" not in r]
        if not reasons:
            if self._blocked_alerted:
                await c.alerts.send(Alert("execution_resumed", "Autonomous execution permitted again",
                                          "every execution gate passes", "info"), force=True)  # fmt: skip
            self._blocked_since, self._blocked_alerted = None, False
            return
        self._blocked_since = self._blocked_since or now
        if now - self._blocked_since >= BLOCKED_ALERT_AFTER:
            self._blocked_alerted = await c.alerts.send(
                Alert("execution_blocked", "Autonomous execution blocked",
                      f"for {_ago(now - self._blocked_since)} in the session: " + "; ".join(reasons[:4])[:600],
                      "warning")
            ) or self._blocked_alerted  # fmt: skip

    async def _fail_closed_on_anomalies(self) -> None:
        c = self._c
        found = await self.anomalies()
        limit = c.settings.brain_anomaly_orders_per_hour
        if len(found) < limit or not c.settings.brain_owns_account:
            return
        kill = await c.trading.brain_kill_switch()
        if kill.active:
            return
        reason = (
            f"automatic: {len(found)} Brain orders rejected/failed/unknown within an hour (limit {limit}); "
            "check the Execution tab, then release"
        )
        await c.trading.set_brain_kill_switch(True, reason, cancel_open_orders=False)
        await c.alerts.send(
            Alert(
                "execution_anomaly",
                "Brain stopped: execution anomaly",
                reason + ". Latest: " + found[0],
                "critical",
            ),
            force=True,
        )

    async def _unexpected_positions(self) -> None:
        from quantpulse.brain.store import BrainStore
        from quantpulse.brain.theses import STATE_KEY

        state = await BrainStore(self._c.db).get_state(STATE_KEY) or {}
        now_unexpected = {
            str(u.get("symbol") if isinstance(u, dict) else u) for u in state.get("unexpected") or []
        }
        new = sorted(now_unexpected - self._unexpected)
        self._unexpected = now_unexpected
        if new:
            await self._c.alerts.send(
                Alert("unexpected_position", "Unexpected position",
                      f"{', '.join(new)} held in the paper account without a Brain decision (bought outside the "
                      "Brain?). The Brain reports it; nothing is sold automatically.", "warning", ",".join(new)[:16])
            )  # fmt: skip

    async def _transitions(self, parts: dict[str, dict[str, Any]]) -> None:
        alerts = self._c.alerts
        for name, spec in {
            "database": ("Database failing", "critical"),
            "alpaca": ("Alpaca paper API unreachable", "critical"),
            "reconciliation": ("Reconciliation failed", "critical"),
            "supervisor": ("Brain stopped", "critical"),
            "scheduler": ("Scheduler stopped", "critical"),
        }.items():
            now_status, before = parts[name]["status"], self._previous.get(name)
            title, severity = spec
            if now_status == "fail" and before != "fail":
                await alerts.send(Alert(name, title, parts[name]["detail"], severity), force=True)  # type: ignore[arg-type]
            elif now_status != "fail" and before == "fail":
                await alerts.send(Alert(f"{name}_recovered", f"{title.split(' ')[0]} recovered",
                                        parts[name]["detail"], "info"), force=True)  # fmt: skip
            elif name == "supervisor" and now_status == "warn" and before not in ("warn", None):
                await alerts.send(
                    Alert("supervisor_warn", "Brain supervisor", parts[name]["detail"], "warning")
                )
            self._previous[name] = now_status
        lc = parts["last_cycle"]
        if lc.get("failed_streak", 0) >= FAILED_CYCLES_ALERT:
            await alerts.send(Alert("cycles_failing", "Brain cycles failing", lc["detail"], "warning"))
        md = parts["market_data"]
        if md.get("halted_streak", 0) >= self._c.settings.health_data_quality_alert_cycles:
            await alerts.send(Alert("data_quality", "Market data halting new positions",
                                    f"{md['halted_streak']} cycles in a row: {md['detail']}", "warning"))  # fmt: skip
        ks = parts["kill_switches"]
        for name in ("brain", "trading"):
            active = bool(ks.get(name))
            if active and self._kills.get(name) is False:
                await alerts.send(Alert(f"{name}_kill_switch", f"{name.capitalize()} kill switch ON",
                                        str(ks.get(f"{name}_reason") or "no reason given"), "warning"), force=True)  # fmt: skip
            if name in ks:
                self._kills[name] = active


def _ago(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{seconds // 60} min"
    if seconds < 172800:
        return f"{seconds // 3600} h"
    return f"{seconds // 86400} d"
