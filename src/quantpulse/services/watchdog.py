"""Is the Brain supervisor alive — not just the process, the supervisor?

The server's watchdog (``deploy/qpops.py watchdog``, every minute) reads this through
``GET /api/v1/system/watchdog`` and restarts the API container only when the verdict says a restart is the
cure. A process can answer HTTP while its supervisor is stuck: a tick that never returns (an await that never
completes), or a background scheduler that stopped asking for ticks. Those are ``stalled``.

Everything else is left alone, because a restart would not fix it or would undo a person's decision:

* ``blocked``: startup recovery has not passed (Alpaca unreachable, the audit failed, ...). That is the
  fail-closed state working as designed; a restart would only run the same recovery again.
* ``standby``: another process holds the supervisor lease. Restarting this one cannot help.
* ``paused`` / ``not_applicable``: a person paused it, it is disabled, background jobs are off, or the
  process is shutting down.

This module only reads: it never ticks the supervisor, never runs a cycle and never touches the broker. A
restart is a graceful stop (no new order, drain, cancel what still runs, final reconciliation, lease handed
over) followed by a fresh start whose first tick is the startup recovery — reconcile with Alpaca and the
execution audit before anything else. An order is never a watchdog's doing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from quantpulse.brain.supervisor import RUNTIME_KEY, STATE_KEY
from quantpulse.core import runtime
from quantpulse.core.market_calendar import is_market_open

if TYPE_CHECKING:
    from quantpulse.services.container import Container

# the poller asks for a tick about once a minute; ten minutes without one is not a slow minute
ATTEMPT_STALLED = timedelta(minutes=10)
# every agent, HTTP call and database command has its own timeout, so a tick that runs this long is stuck:
# in the session a full cycle takes minutes; off-hours ticks (learning, reviews, the weekend lab) take longer
TICK_HUNG_OPEN = timedelta(minutes=20)
TICK_HUNG_CLOSED = timedelta(minutes=90)
# the leader has not recorded a tick for this long: another process holds the lease but does nothing
LEADER_SILENT = timedelta(minutes=10)


@dataclass(frozen=True)
class SupervisorView:
    """What the verdict is decided from (read from the running process)."""

    now: datetime
    started_at: datetime
    enabled: bool
    polling_enabled: bool
    poller_running: bool
    stopping: bool
    paused: bool
    leader: bool
    standby: str | None
    waiting: str | None
    last_attempt_at: datetime | None
    tick_started_at: datetime | None
    last_tick_at: datetime | None
    leader_tick_at: datetime | None  # the leader's last tick, from the database (any process's)
    market_open: bool


@dataclass(frozen=True)
class Liveness:
    verdict: str  # ok | starting | stalled | blocked | standby | paused | not_applicable
    reason: str
    restart: bool  # a restart of this process is the cure (only ever with "stalled")


def _minutes(delta: timedelta) -> str:
    return f"{max(0, int(delta.total_seconds() // 60))} min"


def supervisor_liveness(v: SupervisorView) -> Liveness:
    if not v.enabled:
        return Liveness(
            "not_applicable", "the supervisor is disabled (QP_BRAIN_SUPERVISOR_ENABLED=false)", False
        )
    if v.stopping:
        return Liveness("not_applicable", "this process is shutting down", False)
    if not v.polling_enabled:
        return Liveness("not_applicable", "background jobs are off (QP_POLLING_ENABLED=false)", False)
    if not v.poller_running:
        return Liveness("stalled", "the background scheduler is not running: nothing asks the supervisor to tick",
                        True)  # fmt: skip
    if v.tick_started_at is not None:
        running = v.now - v.tick_started_at
        limit = TICK_HUNG_OPEN if v.market_open else TICK_HUNG_CLOSED
        if running > limit:
            return Liveness("stalled", f"hung: the current tick has run for {_minutes(running)} (limit "
                                       f"{_minutes(limit)})", True)  # fmt: skip
        return Liveness("ok", f"a tick is running (for {_minutes(running)})", False)
    if v.last_attempt_at is None:
        since = v.now - v.started_at
        if since > ATTEMPT_STALLED:
            return Liveness("stalled", f"no tick was asked for since the start {_minutes(since)} ago", True)
        return Liveness("starting", "the first tick is due within a minute", False)
    idle = v.now - v.last_attempt_at
    if idle > ATTEMPT_STALLED:
        return Liveness("stalled", f"the scheduler has not asked for a tick for {_minutes(idle)}", True)
    if v.paused:
        return Liveness("paused", "paused by a person (dashboard/API): no cycles run", False)
    if v.waiting:
        return Liveness("blocked", f"fail-closed: startup recovery has not passed ({v.waiting}); a restart does "
                                   "not fix this", False)  # fmt: skip
    if v.standby:
        silent = v.now - v.leader_tick_at if v.leader_tick_at else None
        if silent is not None and silent > LEADER_SILENT:
            return Liveness("standby", f"{v.standby}, which has not ticked for {_minutes(silent)}: stop or "
                                       "restart that process (restarting this one does not help)", False)  # fmt: skip
        return Liveness("standby", v.standby, False)
    if v.leader and v.last_tick_at is not None and v.now - v.last_tick_at > ATTEMPT_STALLED:
        return Liveness("blocked", f"ticks are asked for but fail before they start (last completed "
                                   f"{_minutes(v.now - v.last_tick_at)} ago): see /system/health", False)  # fmt: skip
    return Liveness("ok", "ticking", False)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


async def watchdog_report(c: Container) -> dict[str, Any]:
    """The verdict and what the server's status page shows (all read-only)."""
    sup = c.brain.supervisor
    now = c.clock.now()
    state = await c.brain.store.get_state(STATE_KEY) or {}
    beat = await c.brain.store.get_state(RUNTIME_KEY) or {}
    leader_tick = beat.get("last_tick_at")
    view = SupervisorView(
        now=now,
        started_at=c.started_at,
        enabled=c.settings.brain_supervisor_enabled,
        polling_enabled=c.settings.polling_enabled,
        poller_running=c.poller.running,
        stopping=sup.stopping,
        paused=bool(state.get("paused")),
        leader=sup.leader,
        standby=sup.standby,
        waiting=sup.waiting,
        last_attempt_at=sup.last_attempt_at,
        tick_started_at=sup.tick_started_at,
        last_tick_at=sup.last_tick_at,
        leader_tick_at=datetime.fromisoformat(leader_tick) if leader_tick else None,
        market_open=is_market_open(now),
    )
    cycles = await c.brain.store.cycles(1)
    last_cycle = cycles[0] if cycles else None
    return {
        **asdict(supervisor_liveness(view)),
        "now": now.isoformat(),
        "started_at": c.started_at.isoformat(),
        "leader": sup.leader,
        "last_attempt_at": _iso(sup.last_attempt_at),
        "tick_started_at": _iso(sup.tick_started_at),
        "last_tick_at": _iso(sup.last_tick_at),
        "last_result": sup.last_result,
        "supervisor_heartbeat": beat or None,
        "last_cycle": {k: last_cycle.get(k) for k in ("id", "kind", "status", "started_at", "finished_at")}
        if last_cycle
        else None,
        "alert_heartbeat": c.alerts.last_heartbeat,
        "paper": c.settings.alpaca_paper,
        "commit": runtime.current().commit,
    }
