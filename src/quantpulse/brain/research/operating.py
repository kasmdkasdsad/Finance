"""The Brain's 24/7 operating model: what it does in each part of the day, and when execution may resume.

==============  ===============================================  =====================================================
mode            when (New York)                                  loop
==============  ===============================================  =====================================================
EXECUTION       the regular session                              EXECUTE → MONITOR → RECONCILE → LEARN
PRE_MARKET      trading days from 08:00 to the open              PRE-MARKET AUDIT → DATA HEALTH → PORTFOLIO
                                                                 RECONCILIATION → WATCHLIST → STRATEGY STATUS →
                                                                 EXECUTION READINESS
RESEARCH        everything else: after the close, overnight,     GRADE → ANALYZE → RESEARCH → TEST → LEARN → PREPARE
                weekends, holidays
==============  ===============================================  =====================================================

In EXECUTION the supervisor trades and learns from reality; research waits (execution and safety come first).
In RESEARCH the research scheduler works through its queue. In PRE_MARKET only light preparation jobs run, and
none starts after 09:15, so research has stopped before the bell.

**Execution readiness.** Before the first order of each session the readiness pipeline must pass: the pre-market
audit (paper endpoint, account, calendar), data health (live market data for the benchmark, no refused feed),
portfolio reconciliation with Alpaca, the watchlist (informational), and strategy status (nothing in production
without a person's promotion). The supervisor runs it from 08:30; the executor runs it itself before the first
order if it has not passed today (a process started mid-session), and retries at most every 5 minutes. It only
adds a gate in front of the existing ones — the pre-trade execution audit, the risk engine and every order gate
still apply unchanged.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Any

from quantpulse.brain.context import brain_session
from quantpulse.brain.types import BrainSession
from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.logging_config import log_event

from .lifecycle import Lifecycle

MODES = ("EXECUTION", "PRE_MARKET", "RESEARCH")
LOOPS = {
    "EXECUTION": ("EXECUTE", "MONITOR", "RECONCILE", "LEARN"),
    "PRE_MARKET": (
        "PRE-MARKET AUDIT",
        "DATA HEALTH",
        "PORTFOLIO RECONCILIATION",
        "WATCHLIST",
        "STRATEGY STATUS",
        "EXECUTION READINESS",
    ),
    "RESEARCH": ("GRADE", "ANALYZE", "RESEARCH", "TEST", "LEARN", "PREPARE"),
}
PREP_FROM = time(8, 0)
RESEARCH_STOP = time(9, 15)  # in pre-market, no research job starts after this
READINESS_KEY = "readiness"
OPERATING_KEY = "operating"
READINESS_RETRY = timedelta(minutes=5)
WATCHLIST_FRESH = timedelta(days=4)
logger = logging.getLogger(__name__)


def mode_at(now: datetime) -> str:
    session = brain_session(now)
    if session is BrainSession.OPEN:
        return "EXECUTION"
    if session is BrainSession.PRE_MARKET and now.astimezone(NEW_YORK).time() >= PREP_FROM:
        return "PRE_MARKET"
    return "RESEARCH"


def research_costs(now: datetime) -> tuple[str, ...] | None:
    """Which research jobs may start now: all in RESEARCH, light preparation before 09:15 in PRE_MARKET, none
    in EXECUTION."""
    mode = mode_at(now)
    if mode == "RESEARCH":
        return ("light", "medium", "heavy")
    if mode == "PRE_MARKET" and now.astimezone(NEW_YORK).time() < RESEARCH_STOP:
        return ("light",)
    return None


class OperatingModel:
    def __init__(self, settings: Settings, clock: Clock, brain: Any, lifecycle: Lifecycle) -> None:
        self._s = settings
        self._clock = clock
        self._brain = brain
        self._lifecycle = lifecycle
        self._last_attempt: datetime | None = None

    # ------------------------------------------------------------------ the mode
    async def observe(self, now: datetime | None = None) -> dict[str, Any]:
        """Record the current mode; a change is logged and kept (the last 50 transitions)."""
        now = now or self._clock.now()
        mode = mode_at(now)
        state = await self._brain.store.get_state(OPERATING_KEY) or {"mode": None, "history": []}
        if state.get("mode") != mode:
            log_event(
                logger,
                "brain.mode",
                f"the Brain enters {mode}: {' → '.join(LOOPS[mode])}",
                previous=state.get("mode"),
                mode=mode,
            )
            history = [
                *state.get("history", []),
                {"mode": mode, "at": now.isoformat(), "from": state.get("mode")},
            ]
            state = {"mode": mode, "since": now.isoformat(), "history": history[-50:]}
            await self._brain.store.set_state(OPERATING_KEY, state, now)
        return state

    # ------------------------------------------------------------------ execution readiness
    def _day(self, now: datetime) -> str:
        return now.astimezone(NEW_YORK).date().isoformat()

    async def readiness(self, now: datetime | None = None) -> dict[str, Any]:
        """PRE-MARKET AUDIT → DATA HEALTH → PORTFOLIO RECONCILIATION → WATCHLIST → STRATEGY STATUS → EXECUTION
        READINESS, recorded for the day."""
        now = now or self._clock.now()
        self._last_attempt = now
        brain, s = self._brain, self._s
        steps: list[dict[str, Any]] = []

        def step(name: str, ok: bool | None, required: bool, detail: str) -> None:
            steps.append({"step": name, "ok": ok, "required": required, "detail": detail})

        owns = s.brain_owns_account and brain.trading.broker.configured()
        if owns:
            try:
                pm = await brain.sessions.premarket()
                failed = set(pm.get("failed") or [])
            except Exception as exc:  # fail closed
                pm, failed = {"ok": False}, {"premarket"}
                step("PRE-MARKET AUDIT", False, True, f"{type(exc).__name__}: {exc}"[:200])
            if "premarket" not in failed:
                audit_failed = sorted(failed & {"paper_endpoint", "account", "calendar"})
                step(
                    "PRE-MARKET AUDIT",
                    not audit_failed,
                    True,
                    "paper endpoint, account and calendar verified"
                    if not audit_failed
                    else "failed: " + ", ".join(audit_failed),
                )
                step(
                    "DATA HEALTH",
                    "market_data" not in failed,
                    True,
                    "live market data for the benchmark"
                    if "market_data" not in failed
                    else "no usable live market data (or a refused feed)",
                )
                step(
                    "PORTFOLIO RECONCILIATION",
                    "reconciliation" not in failed,
                    True,
                    "the record matches the Alpaca paper account"
                    if "reconciliation" not in failed
                    else "reconciliation with Alpaca failed",
                )
        else:
            step(
                "PRE-MARKET AUDIT",
                None,
                False,
                "the Brain does not own the Alpaca paper account: nothing executes",
            )
        watch = await brain.store.get_state("research_watchlist") or {}
        fresh = bool(watch.get("at")) and now - datetime.fromisoformat(watch["at"]) <= WATCHLIST_FRESH
        step(
            "WATCHLIST",
            fresh or None,
            False,
            f"{len(watch.get('symbols', []))} symbol(s) prepared {watch.get('day')}"
            if fresh
            else "none prepared recently: the cycle builds its own focus",
        )
        unapproved = await self._lifecycle.unapproved_in_production()
        promoted = await brain.lab.strategies("promoted")
        step(
            "STRATEGY STATUS",
            not unapproved,
            True,
            f"{len(promoted)} promoted strateg{'y' if len(promoted) == 1 else 'ies'}; nothing in production without a "
            "person's promotion"
            if not unapproved
            else f"in production without a person's promotion: {unapproved}",
        )
        passed = all(st["ok"] is True for st in steps if st["required"])
        step(
            "EXECUTION READINESS",
            passed,
            True,
            "every required gate passed: execution may resume"
            if passed
            else "held: " + "; ".join(st["step"] for st in steps if st["required"] and st["ok"] is not True),
        )
        failed_steps: list[str] = [str(st["step"]) for st in steps if st["required"] and st["ok"] is not True]
        report = {
            "day": self._day(now),
            "at": now.isoformat(),
            "passed": passed,
            "owns_account": owns,
            "steps": steps,
            "failed": failed_steps,
        }
        await brain.store.set_state(READINESS_KEY, report, now)
        log_event(
            logger,
            "brain.readiness",
            "execution readiness " + ("passed" if passed else "HELD"),
            level=logging.INFO if passed else logging.WARNING,
            failed=",".join(failed_steps),
        )
        return report

    async def today(self, now: datetime | None = None) -> dict[str, Any] | None:
        now = now or self._clock.now()
        report = await self._brain.store.get_state(READINESS_KEY)
        return report if report and report.get("day") == self._day(now) else None

    async def ready_today(self, now: datetime | None = None) -> bool:
        report = await self.today(now)
        return bool(report and report.get("passed"))

    async def ensure_ready(self) -> dict[str, Any]:
        """For the executor, before an order: today's readiness, run now if it has not passed (at most every
        5 minutes)."""
        now = self._clock.now()
        report = await self.today(now)
        if report and report.get("passed"):
            return report
        if (
            report is not None
            and self._last_attempt is not None
            and now - self._last_attempt < READINESS_RETRY
        ):
            return report
        return await self.readiness(now)

    async def status(self) -> dict[str, Any]:
        now = self._clock.now()
        mode = mode_at(now)
        state = await self._brain.store.get_state(OPERATING_KEY) or {}
        return {
            "mode": mode,
            "loop": list(LOOPS[mode]),
            "since": state.get("since") if state.get("mode") == mode else None,
            "research_allowed": list(research_costs(now) or ()),
            "readiness": await self.today(now),
            "transitions": list(reversed(state.get("history", [])))[:10],
            "modes": {m: list(v) for m, v in LOOPS.items()},
        }
