"""Asynchronous background polling.

Keeps hot data warm so user requests are served from cache instead of hitting rate-limited APIs:

* watchlist quotes — every ``poll_quotes_market_hours_seconds`` during the NYSE regular session, and
  every ``poll_quotes_off_hours_seconds`` otherwise;
* the Treasury curve — every ``poll_rates_seconds`` (the feed is slow, so this matters);
* NFL / college-football scoreboards — every ``poll_sports_seconds``;
* the daily picks email — once per trading day at ``picks_send_time`` (America/New_York) when enabled;
* the paper-trading sandbox — auto-trading agents rebalance once per trading day after
  ``sandbox_trade_time`` and every account is marked to market after ``sandbox_mark_time``;
* the prediction ledger — forecasts and model predictions are logged after ``predictions_log_time`` on
  trading days and graded when their target date's close is in;
* the stock model — the run for the latest close is started in the background (every few minutes it
  checks whether one is cached or running), so pages and the ledger rarely wait for it;
* Alpaca paper trading — reconciliation at startup and every few minutes in the session, and one strategy
  cycle per ``trading_rebalance_interval_minutes`` slot from ``trading_time`` (dry runs included; orders are
  only sent when paper execution is enabled).
* the brain's supervisor — once a minute it decides, by market session and by event, whether to run a
  cycle, a quote monitor, a reconciliation or a learning pass. When the Brain owns the Alpaca paper account
  (``QP_BRAIN_MODE=paper_execution``) its cycles' decisions are executed by the trading service — the Brain
  itself never talks to the broker's order endpoint.
* health — once a minute: the database, the supervisor, the scheduler, Alpaca, market data, reconciliation
  and the last cycle; alerts on changes, the heartbeat, and Brain orders failing closed while unhealthy.
* options research — once per trading day after ``options_research_time``: the lab's budgeted research run
  (background), then the Options Brain's learning pass (weights, lessons, missed opportunities graded);
* market evolution — once per trading day after ``evolution_time``: the day's metrics, the scan for
  structural change, competing hypotheses, relationships and re-validation.
  Both run only in the process holding the supervisor lease, and never place an order.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, time
from typing import TYPE_CHECKING, Any

from quantpulse.core.market_calendar import NEW_YORK, is_market_open, is_trading_day
from quantpulse.db import repositories as repo
from quantpulse.services.notifications import SyntheticDataRefused

if TYPE_CHECKING:
    from quantpulse.services.container import Container

logger = logging.getLogger(__name__)
PICKS_CHECK_SECONDS = 60.0
SANDBOX_CHECK_SECONDS = 60.0
PREDICTIONS_CHECK_SECONDS = 60.0
MODEL_WARM_SECONDS = 300.0
TRADING_CHECK_SECONDS = 60.0
BRAIN_CHECK_SECONDS = 60.0
HEALTH_CHECK_SECONDS = 60.0
RESEARCH_CHECK_SECONDS = 300.0


@dataclass
class JobStatus:
    name: str
    runs: int = 0
    failures: int = 0
    last_run: datetime | None = None
    last_error: str | None = None
    last_interval: float | None = None
    last_result: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "failures": self.failures,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_error": self.last_error,
            "interval_sec": self.last_interval,
            "last_result": self.last_result,
        }


class Poller:
    def __init__(self, container: Container) -> None:
        self._c = container
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self.jobs: dict[str, JobStatus] = {}
        self._picks_sent_on: str | None = None
        self._picks_failures: dict[str, int] = {}

    @property
    def running(self) -> bool:
        return any(not t.done() for t in self._tasks)

    def start(self) -> None:
        if self.running:
            return
        self._stop = asyncio.Event()
        s = self._c.settings
        self._tasks = [
            asyncio.create_task(self._loop("quotes", self._quote_interval, self.poll_quotes)),
            asyncio.create_task(self._loop("yield_curve", lambda: s.poll_rates_seconds, self.poll_curve)),
            asyncio.create_task(self._loop("sports", lambda: s.poll_sports_seconds, self.poll_sports)),
            asyncio.create_task(
                self._loop("picks_email", lambda: PICKS_CHECK_SECONDS, self.maybe_send_picks)
            ),
            asyncio.create_task(self._loop("sandbox", lambda: SANDBOX_CHECK_SECONDS, self.run_sandbox)),
            asyncio.create_task(
                self._loop("predictions", lambda: PREDICTIONS_CHECK_SECONDS, self.run_predictions)
            ),
            asyncio.create_task(self._loop("model", lambda: MODEL_WARM_SECONDS, self.warm_model)),
            asyncio.create_task(self._loop("trading", lambda: TRADING_CHECK_SECONDS, self.run_trading)),
            asyncio.create_task(self._loop("brain", lambda: BRAIN_CHECK_SECONDS, self.run_brain)),
            asyncio.create_task(self._loop("health", lambda: HEALTH_CHECK_SECONDS, self.run_health)),
            asyncio.create_task(
                self._loop("options_research", lambda: RESEARCH_CHECK_SECONDS, self.run_research)
            ),
            asyncio.create_task(self._loop("evolution", lambda: RESEARCH_CHECK_SECONDS, self.run_evolution)),
        ]
        logger.info("poller started (%d jobs)", len(self._tasks))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []

    def _quote_interval(self) -> float:
        s = self._c.settings
        if is_market_open(self._c.clock.now()):
            return s.poll_quotes_market_hours_seconds
        return s.poll_quotes_off_hours_seconds

    async def _loop(
        self, name: str, interval: Callable[[], float], job: Callable[[], Awaitable[str | None]]
    ) -> None:
        status = self.jobs.setdefault(name, JobStatus(name))
        while not self._stop.is_set():
            try:
                status.last_result = await job()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                status.failures += 1
                status.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("poller job %s failed: %s", name, status.last_error)
            status.runs += 1
            status.last_run = self._c.clock.now()
            status.last_interval = interval()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=status.last_interval)
            except TimeoutError:
                continue

    # ------------------------------------------------------------------ jobs
    async def poll_quotes(self) -> str:
        symbols = self._c.settings.watchlist
        results = await self._c.market.quotes(symbols, force_refresh=True)
        return ", ".join(f"{s}:{r.status.value}" for s, r in results.items())

    async def poll_curve(self) -> str:
        res = await self._c.rates.curve(force_refresh=True)
        return f"{res.value.as_of} {res.status.value}"

    async def poll_sports(self) -> str:
        out = []
        for league in ("nfl", "college-football"):
            res = await self._c.sports.games(league, force_refresh=True)
            out.append(f"{league}:{res.status.value}:{len(res.value[0])}")
        return ", ".join(out)

    async def maybe_send_picks(self) -> str | None:
        s = self._c.settings
        if not s.picks_email_enabled:
            return "disabled"
        if not (self._c.notifier.configured and s.picks_recipients):
            return "not configured (SMTP host/from/recipients)"
        now = self._c.clock.now().astimezone(NEW_YORK)
        today = now.date()
        send_at = time.fromisoformat(s.picks_send_time)
        if not is_trading_day(today) or now.time() < send_at:
            return "waiting"
        key = today.isoformat()
        if self._picks_sent_on == key or await self._already_sent(key):
            self._picks_sent_on = key
            return f"sent for {key}"
        if self._picks_failures.get(key, 0) >= 3:
            return f"gave up for {key} after 3 delivery failures"
        try:
            result = await self._c.picks.email_digest(self._c.notifier)
        except SyntheticDataRefused as exc:
            self._picks_sent_on = key  # do not retry every minute; tomorrow is a new attempt
            logger.warning("daily picks email skipped: %s", exc)
            return f"skipped for {key}: synthetic data"
        except Exception:
            self._picks_failures[key] = self._picks_failures.get(key, 0) + 1
            raise
        async with self._c.db.session() as session:
            await repo.record_ingestion(session, "picks_email", key, "smtp", len(result.sent_to))
        self._picks_sent_on = key
        return f"sent for {key} to {len(result.sent_to)} recipient(s)"

    async def run_sandbox(self) -> str:
        return await self._c.sandbox.run_scheduled()

    async def run_predictions(self) -> str:
        return await self._c.predictions.run_scheduled()

    async def run_trading(self) -> str:
        return await self._c.trading.run_scheduled()

    async def run_brain(self) -> str:
        """The brain's supervisor: session- and event-driven cycles (orders only via the trading service)."""
        return await self._c.brain.supervisor.tick()

    async def run_health(self) -> str:
        """Health of every part, alerts on changes, the heartbeat, and failing closed when unhealthy."""
        return await self._c.health.run()

    async def _daily(self, key: str, at: str) -> str | None:
        """``None`` when a once-per-trading-day job is due now (else why not); marks it done for today."""
        now = self._c.clock.now().astimezone(NEW_YORK)
        today = now.date()
        if not is_trading_day(today) or now.time() < time.fromisoformat(at):
            return "waiting"
        lease = getattr(self._c, "lease", None)
        if lease is not None and not await lease.acquire():
            return "standby (another process holds the supervisor lease)"
        from quantpulse.db.models import BrainStateRow

        async with self._c.db.session() as session:
            row = await session.get(BrainStateRow, key)
            if row is not None and (row.value or {}).get("day") == today.isoformat():
                return f"done for {today}"
            if row is None:
                session.add(
                    BrainStateRow(key=key, value={"day": today.isoformat()}, updated_at=self._c.clock.now())
                )
            else:
                row.value, row.updated_at = {"day": today.isoformat()}, self._c.clock.now()
        return None

    async def run_research(self) -> str:
        """The options research lab and the Options Brain's learning pass, once a trading day."""
        s = self._c.settings
        if not s.options_enabled:
            return "disabled"
        why = await self._daily("options_research_day", s.options_research_time)
        if why:
            return why
        await self._c.registry.bootstrap()
        job = self._c.options_lab.start_research()
        learned = await self._c.options_brain.learn()
        return f"research job {job.id} started; learned: {learned.get('weights')} weight(s), {learned.get('lessons')} lesson(s)"

    async def run_evolution(self) -> str:
        """The market evolution monitor, once a trading day."""
        s = self._c.settings
        if not s.evolution_enabled:
            return "disabled"
        why = await self._daily("evolution_day", s.evolution_time)
        if why:
            return why
        job = self._c.evolution.start()
        return f"evolution job {job.id} started"

    async def warm_model(self) -> str:
        """Keep the latest close's model run computed, so pages and the ledger never wait for it."""
        if not self._c.settings.model_warmup:
            return "disabled"
        return await self._c.model.warm()

    async def _already_sent(self, key: str) -> bool:
        async with self._c.db.session() as session:
            events = await repo.recent_ingestions(session, 200)
        return any(e.dataset == "picks_email" and e.key == key for e in events)

    def snapshot(self) -> dict[str, Any]:
        return {"running": self.running, "jobs": {n: j.snapshot() for n, j in self.jobs.items()}}
