"""The supervisor: keeps the brain working while the server runs — by market session and by event, never
by running every agent all the time.

It is ticked by the background poller (about once a minute) and decides what, if anything, to do:

=================  =========================================================================================
pre-market         once a day (from 08:30 New York): a research-only style *full* cycle to prepare the
                   session (the market is closed, so nothing it proposes is executable) and a learning pass
                   for anything that matured overnight
market open        a *full* cycle every ``QP_BRAIN_CYCLE_MINUTES``; a quote *monitor* every
                   ``QP_BRAIN_MONITOR_MINUTES`` (holdings and the last focus: large moves and stale quotes
                   become events); the trading service's audit trail is read for orders and risk limits;
                   event wake-ups run focused cycles, at most ``QP_BRAIN_MAX_EVENT_CYCLES_PER_HOUR``
after hours        once a day (from 16:40): a learning pass (grade the day's matured predictions and
                   ideas, reflect, update track records), trade lessons, a *portfolio* review of the
                   holdings, the strategy lab's paper (shadow) portfolios, a self-improvement review
                   (proposals only), the **daily review** — and after the week's last session the
                   **weekly review** (lessons; proposals only, never a change)
weekend, holiday   once a day: a learning pass; a *deep* research cycle; the strategy lab proposes
                   untried templates and validates up to two (it never promotes: that is a person's call)
=================  =========================================================================================

Events turn into wake-ups: a price move, a volume spike, a news item or earnings approaching for a
holding → an *event* cycle on that symbol; a position, portfolio or order change or a risk limit → a
*portfolio* review; a regime change → a *full* cycle. Wake-ups for the same thing are merged, and event
cycles only run while the market is open.

When the Brain owns the Alpaca paper account (``QP_BRAIN_MODE=paper_execution``) its cycles' decisions are
executed by the trading service — as *scheduled* cycles. The first order of each day (and of each process)
waits for the final execution audit (:meth:`BrainExecutor.audit`): paper endpoint, paper key, trading
switches, both kill switches, the environment, the account, reconciliation, the market clock and the data;
only when every check passes does it arm scheduled execution (``QP_TRADING_SCHEDULER_REQUIRES_ARMING``) and
send. While the market is open the account is reconciled every five minutes; from half an hour before the
close a *near-close* review de-risks what should not be held into an overnight earnings release and records
the day's decision state; after the close the day is reconciled and recorded, graded and learned from.

After a restart the first tick — never assuming the previous state was right — closes cycles the restart
interrupted, reconciles with Alpaca, brings the execution ledger up to date and runs the execution audit
before anything else; orders already sent keep their client ids, so nothing is sent twice. Ticks never
overlap: a tick that arrives while one is running (a duplicate scheduler event) does nothing.

``QP_BRAIN_SUPERVISOR_ENABLED`` turns it on or off at start-up; it can be paused and resumed at runtime
(``POST /brain/supervisor``), and the state survives restarts.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Any

from quantpulse.config import Settings
from quantpulse.core import runtime
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, is_market_open, next_open, regular_close
from quantpulse.logging_config import log_event

from .context import brain_session
from .events import Event, EventBus, EventType, TradingEventBridge
from .reviews import last_session_of_week
from .types import BrainSession

if TYPE_CHECKING:
    from .service import BrainService

logger = logging.getLogger(__name__)
PREMARKET_FROM = time(8, 30)
AFTER_HOURS_FROM = time(16, 40)
NEAR_CLOSE_MINUTES = 30  # the near-close review: overnight risk, hold or reduce, the day's decision state
STATE_KEY = "supervisor"
# startup checks that must pass before supervision resumes after a start (see _recover)
RESUME_REQUIRED = ("paper_setting", "paper_endpoint", "paper_key", "environment", "account", "reconciliation",
                   "clock_skew")  # fmt: skip
RECONCILE_EVERY = timedelta(minutes=5)
RUNTIME_KEY = "supervisor_runtime"  # the leader's heartbeat (last tick, result, recovery), for every instance
STANDBY_KEY = "supervisor_standby"  # processes that found the lease taken, recently
STANDBY_WINDOW = timedelta(minutes=10)
TICK_LOG_EVERY = timedelta(minutes=15)  # an idle supervisor still logs that it is alive

SYMBOL_EVENTS = {
    EventType.PRICE_MOVE_DETECTED,
    EventType.VOLUME_SPIKE_DETECTED,
    EventType.NEWS_EVENT_DETECTED,
    EventType.EARNINGS_APPROACHING,
}
PORTFOLIO_EVENTS = {
    EventType.POSITION_CHANGED,
    EventType.PORTFOLIO_CHANGED,
    EventType.ORDER_FILLED,
    EventType.ORDER_CANCELED,
    EventType.RISK_LIMIT_TRIGGERED,
}


@dataclass
class WakeUp:
    kind: str  # event | portfolio | full
    symbols: tuple[str, ...]
    reason: str
    queued_at: datetime


class Supervisor:
    def __init__(self, settings: Settings, clock: Clock, brain: BrainService, bus: EventBus) -> None:
        self._s = settings
        self._clock = clock
        self._brain = brain
        self._bus = bus
        self._bridge = TradingEventBridge(brain.db)
        self.queue: dict[tuple[str, tuple[str, ...]], WakeUp] = {}
        self._event_cycles: deque[datetime] = deque()
        self._daily_vol: dict[str, float] = {}
        self.log: deque[dict[str, Any]] = deque(maxlen=200)
        self._booted_at = clock.now()
        self._elected_at = self._booted_at  # when this process last became the leader
        self._recovered = False
        # the previous leader stopped without handing over (a crash, a kill, a lost connection): reported once
        self.takeover: dict[str, Any] | None = None
        # this process led and lost the lease to another holder without stopping (a pause, a partition, a second
        # installation on the same database): reported once
        self.lost: dict[str, Any] | None = None
        self.waiting: str | None = None  # why supervision has not resumed after a start
        self.standby: str | None = None  # another process holds the supervisor lease
        self.last_tick_at: datetime | None = None
        self.last_result: str | None = None
        # for the watchdog (read only): when a tick was last asked for, whatever the answer, and when the tick
        # running now began (``None``: none is running) — a hung tick and a scheduler that stopped asking
        self.last_attempt_at: datetime | None = None
        self.tick_started_at: datetime | None = None
        self.leader = False  # this process holds the supervisor lease
        self.stopping = False  # shutting down: no new tick
        self._last_tick_log: datetime | None = None
        self._tick_lock = asyncio.Lock()  # a duplicate or overlapping tick never runs the same work twice
        bus.subscribe([*SYMBOL_EVENTS, *PORTFOLIO_EVENTS, EventType.MARKET_REGIME_CHANGED], self._on_event)

    # ------------------------------------------------------------------ state
    async def _state(self) -> dict[str, Any]:
        return await self._brain.store.get_state(STATE_KEY) or {"paused": False, "last": {}}

    async def _save(self, state: dict[str, Any]) -> None:
        await self._brain.store.set_state(STATE_KEY, state, self._clock.now())

    async def set_paused(self, paused: bool) -> dict[str, Any]:
        state = await self._state()
        state["paused"] = paused
        await self._save(state)
        return await self.status()

    async def status(self) -> dict[str, Any]:
        state = await self._state()
        now = self._clock.now()
        return {
            "enabled": self._s.brain_supervisor_enabled,
            "paused": bool(state.get("paused")),
            "session": brain_session(now).value,
            "market_open": is_market_open(now),
            "last": state.get("last", {}),
            "queue": [
                {
                    "kind": w.kind,
                    "symbols": list(w.symbols),
                    "reason": w.reason,
                    "queued_at": w.queued_at.isoformat(),
                }
                for w in self.queue.values()
            ],
            "event_cycles_last_hour": len([t for t in self._event_cycles if now - t < timedelta(hours=1)]),
            "limits": {
                "cycle_minutes": self._s.brain_cycle_minutes,
                "monitor_minutes": self._s.brain_monitor_minutes,
                "max_event_cycles_per_hour": self._s.brain_max_event_cycles_per_hour,
            },
            "recent": list(self.log)[-20:],
            "recovered": self._recovered,
            "waiting": self.waiting,
            "standby": self.standby,
            "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
            "last_result": self.last_result,
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "tick_started_at": self.tick_started_at.isoformat() if self.tick_started_at else None,
            "next_cycle_at": self.next_cycle_at(state).isoformat(),
            "lease": await self._brain.trading.lease.info()
            if self._brain.trading.lease is not None
            else None,
            "leader": self.leader,
            "stopping": self.stopping,
            "heartbeat": await self._brain.store.get_state(RUNTIME_KEY),
            "standby_processes": await self.standby_processes(),
        }

    async def standby_processes(self) -> dict[str, Any]:
        now = self._clock.now()
        seen = await self._brain.store.get_state(STANDBY_KEY) or {}
        return {
            h: v for h, v in seen.items() if now - datetime.fromisoformat(v["last_seen"]) < STANDBY_WINDOW
        }

    def next_cycle_at(self, state: dict[str, Any]) -> datetime:
        """When the next scheduled full cycle is due: in the session, the last one plus the interval;
        otherwise the next open."""
        now = self._clock.now()
        if is_market_open(now):
            last = (state.get("last") or {}).get("cycle")
            if last:
                due = datetime.fromisoformat(last) + timedelta(minutes=self._s.brain_cycle_minutes)
                return max(due, now)
            return now
        return next_open(now)

    # ------------------------------------------------------------------ events → wake-ups
    async def _on_event(self, e: Event) -> None:
        """Queue a wake-up (merged with any already waiting for the same thing). Repeats of an event are
        already dropped by the bus, so a cycle's own events do not re-trigger it."""
        now = self._clock.now()
        if e.type in SYMBOL_EVENTS and e.subject and not e.subject.startswith("@"):
            self.queue.setdefault(("event", (e.subject,)), WakeUp("event", (e.subject,), e.type.value, now))
        elif e.type in PORTFOLIO_EVENTS:
            self.queue.setdefault(("portfolio", ()), WakeUp("portfolio", (), e.type.value, now))
        elif e.type is EventType.MARKET_REGIME_CHANGED:
            self.queue.setdefault(("full", ()), WakeUp("full", (), e.type.value, now))

    # ------------------------------------------------------------------ the tick
    async def tick(self) -> str:
        if not self._s.brain_supervisor_enabled:
            return "disabled"
        self.last_attempt_at = self._clock.now()
        if self.stopping:
            return "stopping: this process is shutting down; no new work"
        if self._tick_lock.locked():
            return "busy: the previous tick is still running"
        async with self._tick_lock:
            self.tick_started_at = self._clock.now()
            try:
                return await self._locked_tick()
            finally:
                self.tick_started_at = None

    async def _locked_tick(self) -> str:
        lease = self._brain.trading.lease
        if lease is not None:  # one supervisor at a time, across every process on this database
            before = None if self.leader else await lease.info()  # who held it, and how it ended
            if not await lease.acquire():
                info = await lease.info()
                standby = f"another process supervises the Brain ({info.get('holder')})"
                if self.leader:
                    log_event(logger, "supervisor.lost", "this process lost the supervisor lease: it stands by",
                              level=logging.WARNING, holder=info.get("holder"), this=lease.holder)  # fmt: skip
                    self.lost = {"at": self._clock.now().isoformat(), "to": info.get("holder")}
                elif self.standby != standby:
                    log_event(
                        logger,
                        "supervisor.standby",
                        standby,
                        holder=info.get("holder"),
                        this=lease.holder,
                    )
                self.leader, self.standby = False, standby
                await self._note_standby(lease.holder)
                return f"standby: {self.standby}"
            if not self.leader:
                self._on_elected(lease.holder, before or {})
            self.leader, self.standby = True, None
            lease.start_heartbeat()
        self.last_tick_at = self._clock.now()
        result = await self._tick()
        self.last_result = result
        await self._record_tick(result)
        return result

    # ------------------------------------------------------------------ running in the cloud
    def _on_elected(self, holder: str, before: dict[str, Any]) -> None:
        """This process has just become the leader. Nothing it remembers is trusted: another supervisor may have
        acted since it last led (or it never led), so the startup recovery runs again before any other work."""
        rt = runtime.current()
        now = self._clock.now()
        self._elected_at, self._recovered = now, False
        log_event(logger, "supervisor.elected", "this process supervises the Brain (lease acquired)",
                  holder=holder, commit=rt.short_commit, instance=rt.instance)  # fmt: skip
        previous = before.get("holder")
        if previous and previous != holder and not before.get("released"):
            self.takeover = {
                "at": now.isoformat(),
                "previous_holder": previous,
                "previous_heartbeat_at": before.get("heartbeat_at"),
                "previous_expires_at": before.get("expires_at"),
            }
            log_event(
                logger,
                "supervisor.takeover",
                f"the previous supervisor ({previous}) stopped without handing over (a crash, a kill or a lost "
                "connection): its lease lapsed; recovery runs before anything else",
                level=logging.WARNING,
                previous=previous,
                previous_heartbeat_at=before.get("heartbeat_at"),
            )

    def begin_stop(self) -> None:
        """No new tick from now on (the process is shutting down)."""
        self.stopping = True

    async def drain(self, timeout: float) -> bool:
        """Stop taking ticks and wait up to ``timeout`` seconds for the one running to finish; ``True`` when
        none is running any more."""
        self.begin_stop()
        try:  # the tick lock is free once the running tick (if any) is done; no new tick takes it
            await asyncio.wait_for(self._tick_lock.acquire(), max(0.0, timeout) or 0.001)
        except TimeoutError:
            return False
        self._tick_lock.release()
        return True

    async def _record_tick(self, result: str) -> None:
        """The leader's heartbeat, in the database: any instance (the one serving a request during a deploy, the
        dashboard) can tell when the supervisor last ticked and what it did."""
        now = self._clock.now()
        lease = self._brain.trading.lease
        rt = runtime.current()
        await self._brain.store.set_state(
            RUNTIME_KEY,
            {
                "holder": lease.holder if lease is not None else None,
                "instance": rt.instance,
                "commit": rt.short_commit,
                "last_tick_at": now.isoformat(),
                "last_result": result[:300],
                "recovered": self._recovered,
                "waiting": self.waiting,
                "elected_at": self._elected_at.isoformat(),
                "takeover": self.takeover,
            },
            now,
        )
        worked = not result.startswith(("idle", "paused"))
        if worked or self._last_tick_log is None or now - self._last_tick_log >= TICK_LOG_EVERY:
            self._last_tick_log = now
            log_event(logger, "supervisor.tick", f"supervisor tick: {result}"[:400], result=result[:300],
                      recovered=self._recovered)  # fmt: skip

    async def _note_standby(self, holder: str) -> None:
        """Processes standing by (another instance during a deploy, a second replica): shown in the status so
        a person can see there is exactly one leader."""
        now = self._clock.now()
        seen = await self._brain.store.get_state(STANDBY_KEY) or {}
        seen = {
            h: v for h, v in seen.items() if now - datetime.fromisoformat(v["last_seen"]) < STANDBY_WINDOW
        }
        seen[holder] = {"last_seen": now.isoformat(), "commit": runtime.current().short_commit}
        await self._brain.store.set_state(STANDBY_KEY, seen, now)

    async def _tick(self) -> str:
        state = await self._state()
        if state.get("paused"):
            return "paused"
        now = self._clock.now()
        local = now.astimezone(NEW_YORK)
        session = brain_session(now)
        done: list[str] = []
        last: dict[str, str] = state.setdefault("last", {})

        def due(task: str, every: timedelta | None = None, daily_from: time | None = None) -> bool:
            prev = datetime.fromisoformat(last[task]) if task in last else None
            if daily_from is not None:
                if local.time() < daily_from:
                    return False
                return prev is None or prev.astimezone(NEW_YORK).date() < local.date()
            return prev is None or (every is not None and now - prev >= every)

        async def run(task: str, coro: Any) -> None:
            try:
                result = await coro
                self.log.append({"at": now.isoformat(), "task": task, "result": _brief(result)})
                done.append(task)
            except Exception as exc:  # a failed task is reported and retried at its next slot
                logger.warning("supervisor task %s failed: %s", task, exc)
                self.log.append(
                    {"at": now.isoformat(), "task": task, "error": f"{type(exc).__name__}: {exc}"[:300]}
                )
            last[task] = now.isoformat()

        if not self._recovered:  # after a start: nothing else until recovery and its safety checks pass
            await run("startup_recovery", self._recover())
            self._recovered = "startup_recovery" in done
            if not self._recovered:
                state["last"] = last
                await self._save(state)
                self.waiting = self.log[-1].get("error") if self.log else "startup recovery failed"
                return f"waiting: startup recovery has not passed ({self.waiting}); no cycles, no orders"
            self.waiting = None

        trading_events = await self._bridge.poll()
        if trading_events:
            await self._bus.publish(trading_events)

        owns = self._s.brain_owns_account and self._brain.trading.broker.configured()
        if session is BrainSession.OPEN and is_market_open(now):
            if owns and due("reconcile", RECONCILE_EVERY):
                await run("reconcile", self._reconcile("brain periodic"))
            if (
                owns
                and self._s.brain_strategy_shadow
                and due("strategy_shadow", timedelta(minutes=self._s.trading_rebalance_interval_minutes))
            ):  # the replaced strategy against its own hypothetical portfolio (never an order)
                await run("strategy_shadow", self._brain.shadow.step())
            if due("monitor", timedelta(minutes=self._s.brain_monitor_minutes)):
                await run("monitor", self._monitor())
            if due("cycle", timedelta(minutes=self._s.brain_cycle_minutes)):
                self.queue.pop(("full", ()), None)
                await run("cycle", self._cycle("full", (), "scheduled"))
            await self._drain(run)
            closing = datetime.combine(local.date(), regular_close(local.date()), NEW_YORK)
            near = (closing - timedelta(minutes=NEAR_CLOSE_MINUTES)).time()
            if owns and due("near_close", daily_from=near):  # overnight risk: hold or reduce, recorded
                await run("near_close", self._near_close())
        elif session is BrainSession.PRE_MARKET:
            if due("premarket", daily_from=PREMARKET_FROM):
                if owns:  # verify the account, reconcile, calendar, data, overnight changes
                    await run("premarket_check", self._brain.sessions.premarket())
                await run("premarket_learn", self._brain.learn(wait=None))
                await run("premarket", self._cycle("full", (), "pre-market preparation"))
        elif session is BrainSession.AFTER_HOURS:
            if due("after_hours", daily_from=AFTER_HOURS_FROM):
                if owns:  # reconcile and record the day (what the 60-session evaluation reads)
                    await run("session_close", self._close())
                await run("learn", self._brain.learn(wait=None))
                await run("trade_lessons", self._brain.trade_lessons())
                await run("review", self._cycle("portfolio", (), "after-hours review"))
                await run("lab_paper", self._brain.lab.paper_update())
                await run("improve", self._improve())
                await run("daily_review", self._brain.reviewer.daily(local.date()))
                if last_session_of_week(local.date()):  # the week's last session: the weekly review too
                    await run("weekly_review", self._brain.reviewer.weekly(local.date()))
                last["after_hours"] = now.isoformat()
        else:  # weekend or holiday
            if due("offday_learn", daily_from=time(9, 0)):
                await run("offday_learn", self._brain.learn(wait=None))
            if due("deep", daily_from=time(10, 0)):
                await run("deep", self._cycle("deep", (), "weekend research"))
            if due("lab", daily_from=time(12, 0)):
                await run("lab", self._lab())
        state["last"] = last
        await self._save(state)
        return ", ".join(done) if done else f"idle ({session.value})"

    async def _drain(self, run: Any) -> None:
        now = self._clock.now()
        while self._event_cycles and now - self._event_cycles[0] >= timedelta(hours=1):
            self._event_cycles.popleft()
        for key in list(self.queue):
            if len(self._event_cycles) >= self._s.brain_max_event_cycles_per_hour:
                break  # the rest wait for the next hour (merged, not lost)
            wake = self.queue.pop(key)
            self._event_cycles.append(now)
            label = f"{wake.kind}:{','.join(wake.symbols) or '-'}"
            await run(label, self._cycle(wake.kind, wake.symbols, wake.reason))

    async def _recover(self) -> dict[str, Any]:
        """After a restart — never assuming the previous state was right: close the Brain cycles it
        interrupted; when the Brain owns the account, reconcile with Alpaca (which also closes interrupted
        trading cycles), bring the execution ledger up to date, and run the execution audit (the paper
        account, endpoint, key, environment, kill switches, clock and data). Orders resume only once a
        pre-trade audit passes (the executor runs one before the first order in every process)."""
        # cycles still marked running from before this process became the leader belong to a supervisor that
        # stopped (only the leader runs cycles): close them — even ones begun after this process started
        closed = await self._brain.store.close_interrupted(self._elected_at, self._clock.now())
        out: dict[str, Any] = {"interrupted_cycles": len(closed)}
        trading = self._brain.trading
        if self._s.brain_owns_account and trading.broker.configured():
            report = await trading.reconcile("startup")
            await self._brain.ledger.refresh()
            audit = await self._brain.executor.audit(purpose="startup")
            out.update(
                open_orders=report.open_orders,
                positions=report.positions,
                audit="passed" if audit["ok"] else "failed: " + "; ".join(audit["failed"])[:200],
            )
            # supervision resumes only if the account is the paper account, reachable, reconciled, the
            # environment is the one configured and the clock can be trusted (a kill switch or a disabled
            # trading switch does not stop the analysis — the order gates keep enforcing those)
            unsafe = [c for c in audit["checks"] if c["name"] in RESUME_REQUIRED and not c["ok"]]
            if unsafe:
                raise RuntimeError(
                    "startup safety checks failed: "
                    + "; ".join(f"{c['name']}: {c['detail']}" for c in unsafe)[:400]
                )
        return out

    async def _reconcile(self, trigger: str) -> dict[str, Any]:
        report = await self._brain.trading.reconcile(trigger)
        updated = await self._brain.ledger.refresh()
        return {"positions": report.positions, "open_orders": report.open_orders, "executions": updated}

    async def _close(self) -> dict[str, Any]:
        await self._brain.ledger.refresh()
        return await self._brain.sessions.close()

    async def _near_close(self) -> dict[str, Any]:
        """A portfolio review before the bell: the planner de-risks what should not be held into an earnings
        release overnight, the thesis checks run once more, and the day's decision state is recorded."""
        detail = await self._brain.run(trigger="supervisor: near close", kind="portfolio", wait=None)
        ctx = self._brain.orchestrator.last_ctx
        risk = (ctx.working.facts.get("event_risk") or {}) if ctx is not None else {}
        near = (ctx.working.facts.get("near_close") or {}) if ctx is not None else {}
        positions = await self._brain.theses.positions(closed=0)
        return await self._brain.sessions.near_close(
            detail, positions["open"], risk, derisked=near.get("derisked") or []
        )

    async def _improve(self) -> dict[str, Any]:
        """Look at the record and write improvement proposals (never applied automatically)."""
        found = await self._brain.improvements.review(self._clock.now())
        return {"proposals": len(found)}

    async def _lab(self) -> dict[str, Any]:
        """Weekend lab work: propose untried templates, validate up to two proposals (never promotes)."""
        proposed = await self._brain.lab.propose()
        validated = await self._brain.lab.validate_pending(limit=2)
        return {"proposed": len(proposed), "validated": len(validated),
                "verdicts": ", ".join(f"{v['key']} {v['verdict']}" for v in validated)}  # fmt: skip

    async def _cycle(self, kind: str, symbols: tuple[str, ...], reason: str) -> dict[str, Any]:
        detail = await self._brain.run(trigger=f"supervisor: {reason}", kind=kind, symbols=symbols, wait=None)
        ctx = self._brain.orchestrator.last_ctx
        if ctx is not None:
            rv = ctx.indicators.get("rv21") if "rv21" in ctx.indicators.columns else None
            if rv is not None:
                self._daily_vol = {
                    str(s): float(v) / math.sqrt(252) for s, v in rv.items() if v == v and float(v) > 0
                }
        return {"cycle": detail["id"], "status": detail["status"], "kind": kind}

    async def _monitor(self) -> dict[str, Any]:
        """Quotes for holdings and the last focus: big moves and stale quotes become events."""
        ctx = self._brain.orchestrator.last_ctx
        watch = list(dict.fromkeys([*(ctx.held if ctx else []), *(ctx.focus if ctx else [])]))
        if not watch:
            return {"watched": 0}
        quotes = await self._brain.data.live_quotes(watch)
        events: list[Event] = []
        max_age = self._s.trading_max_quote_age_seconds
        held = set(ctx.held) if ctx else set()
        for sym in watch:
            q = quotes.get(sym)
            if q is None or q.age_seconds > max_age:
                events.append(
                    Event(
                        EventType.QUOTE_BECAME_STALE,
                        sym,
                        {"age_seconds": q.age_seconds if q else None},
                        source="monitor",
                    )
                )
                continue
            vol = self._daily_vol.get(sym)
            prev = q.previous_close
            if vol and prev:
                move = (q.price / prev - 1) / vol
                if abs(move) >= 3:
                    events.append(
                        Event(
                            EventType.PRICE_MOVE_DETECTED,
                            sym,
                            {"move_sigma": round(move, 2), "held": sym in held},
                            source="monitor",
                        )
                    )
        events.append(
            Event(
                EventType.MARKET_DATA_UPDATED,
                None,
                {"quotes": len(quotes), "watched": len(watch)},
                source="monitor",
            )
        )
        kept = await self._bus.publish(events)
        return {"watched": len(watch), "quotes": len(quotes), "events": len(kept)}


def _brief(result: Any) -> Any:
    if isinstance(result, dict):
        return {k: result[k] for k in list(result)[:6] if not isinstance(result[k], list | dict)}
    return str(result)[:200]
