"""The brain as a service: builds the registry, perception and orchestrator from QuantPulse's container and
exposes cycles, agents, memory and predictions to the API. Cycles run in the background job registry so a
slow cycle answers 202 with its progress instead of blocking a request."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError, NotFoundError
from quantpulse.core.jobs import Job, JobPending, JobRegistry
from quantpulse.db.session import Database
from quantpulse.providers.alpaca_trading import AlpacaPaperBroker
from quantpulse.services.market import MarketService
from quantpulse.services.model import ModelService
from quantpulse.services.options import OptionsService
from quantpulse.services.reference import ReferenceService
from quantpulse.services.trading import TradingService
from quantpulse.services.trading_data import TradingDataLoader

from .agents import default_agents
from .context import BrokerView
from .evaluation import MarketPrices
from .events import Event, EventBus, EventType
from .improvement import ImprovementEngine
from .lab.service import StrategyLab
from .learning import Learner, PredictionRecorder
from .llm import ModelRouter
from .memory import MemoryStore
from .orchestrator import Orchestrator
from .perception import Perception
from .reflection import consensus_calibration
from .registry import AgentRegistry
from .store import BrainStore
from .supervisor import Supervisor
from .types import BrainMode

JOB_KEY = "brain-cycle"
LEARN_KEY = "brain-learn"


class BrainCycleRunning(DomainError):
    """A brain cycle is still running; the API answers 202 with its progress."""

    def __init__(self, job: Job) -> None:
        super().__init__(f"{job.description}: {job.progress:.0%} ({job.stage})")
        self.job = job


class BrainService:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        db: Database,
        jobs: JobRegistry,
        broker: AlpacaPaperBroker,
        trading: TradingService,
        data: TradingDataLoader,
        reference: ReferenceService | None = None,
        model: ModelService | None = None,
        options: OptionsService | None = None,
        market: MarketService | None = None,
    ) -> None:
        self._s = settings
        self._clock = clock
        self._jobs = jobs
        self._db = db
        self.db = db
        self.data = data
        self.bus = EventBus(db, clock)
        self.registry = AgentRegistry(default_agents())
        self.store = BrainStore(db)
        self.memory = MemoryStore(db)
        self.models = ModelRouter(settings, clock, store=self.store)  # 'none' unless a provider is configured
        self.perception = Perception(
            settings, clock, data, BrokerView(broker), trading, reference, model, options, market
        )
        self.orchestrator = Orchestrator(
            settings,
            clock,
            self.perception,
            self.registry,
            self.store,
            self.memory,
            PredictionRecorder(db),
            self.bus,
            self.models,
        )
        self._synced = False
        self.learner = (
            Learner(
                db,
                MarketPrices(market, clock),
                clock,
                self.memory,
                settings.benchmark_symbol,
                settings.brain_min_reliability_observations,
            )
            if market is not None
            else None
        )
        self.lab = StrategyLab(settings, clock, db, market, data, self.store)
        self.improvements = ImprovementEngine(
            db, settings.brain_min_reliability_observations, settings.brain_min_confidence
        )
        self.supervisor = Supervisor(settings, clock, self, self.bus)

    async def _sync(self) -> None:
        if self._synced:
            return
        enabled = await self.store.sync_agents(self.registry.all(), self._clock.now())
        for agent_id, on in enabled.items():
            self.registry.set_enabled(agent_id, on)
        self._synced = True

    # ------------------------------------------------------------------ cycles
    async def run(
        self,
        *,
        trigger: str = "manual",
        kind: str = "full",
        symbols: Sequence[str] = (),
        wait: float | None = None,
    ) -> dict[str, Any]:
        await self._sync()

        async def work(job: Job) -> dict[str, Any]:
            job.reporter(0.0, 1.0)(0.1, "perceiving the market and the portfolio")
            result = await self.orchestrator.run(trigger=trigger, kind=kind, symbols=symbols)
            detail = await self.store.cycle(result.cycle_id)
            assert detail is not None
            return detail

        job = self._jobs.start("brain", JOB_KEY, f"Brain cycle ({kind}, {trigger})", work)
        try:
            return await self._jobs.wait(job, wait)
        except JobPending:
            raise BrainCycleRunning(job) from None

    # ------------------------------------------------------------------ learning
    async def learn(self, *, wait: float | None = None) -> dict[str, Any]:
        """Grade matured predictions and learn from them (a background job; 202 while it runs)."""
        if self.learner is None:
            raise DomainError("learning needs the market service (price history) and is not configured")
        learner = self.learner

        async def work(job: Job) -> dict[str, Any]:
            job.reporter(0.0, 1.0)(0.1, "grading matured predictions")
            summary = await learner.learn()
            await self.store.set_state("learning", summary, self._clock.now())
            events = [Event(EventType.TRADE_OUTCOME_AVAILABLE, o["subject"], o) for o in summary["outcomes"]]
            if summary["evaluated"]:
                events.insert(0, Event(EventType.PREDICTION_MATURED, None, {"graded": summary["evaluated"]}))
            await self.bus.publish(events)
            return summary

        job = self._jobs.start("brain", LEARN_KEY, "Brain learning pass", work)
        try:
            return await self._jobs.wait(job, wait)
        except JobPending:
            raise BrainCycleRunning(job) from None

    async def validate_strategy(
        self, strategy_id: str, version: int, *, wait: float | None = None
    ) -> dict[str, Any]:
        """Run the lab's full validation for one version (a background job; 202 while it runs)."""

        async def work(job: Job) -> dict[str, Any]:
            job.reporter(0.0, 1.0)(0.1, f"validating {strategy_id}@v{version}")
            return await self.lab.validate(strategy_id, version)

        job = self._jobs.start(
            "brain", f"brain-lab:{strategy_id}@v{version}", f"Lab validation {strategy_id}@v{version}", work
        )
        try:
            return await self._jobs.wait(job, wait)
        except JobPending:
            raise BrainCycleRunning(job) from None

    async def learning(self) -> dict[str, Any]:
        return {
            "predictions": await self.store.prediction_summary(),
            "last_run": await self.store.get_state("learning"),
            "calibration": await consensus_calibration(self._db),
            "measured_agents": sorted(
                {
                    r["agent_id"]
                    for r in await self.store.performance("all")
                    if r["reliability"] is not None and r["agent_id"] != "consensus"
                }
            ),
            "min_observations": self._s.brain_min_reliability_observations,
        }

    async def cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self.store.cycles(limit)

    async def cycle(self, cycle_id: int) -> dict[str, Any]:
        found = await self.store.cycle(cycle_id)
        if found is None:
            raise NotFoundError(f"brain cycle {cycle_id} not found")
        return found

    # ------------------------------------------------------------------ agents and memory
    async def agents(self) -> list[dict[str, Any]]:
        await self._sync()
        return await self.store.agents()

    async def set_agent_enabled(self, agent_id: str, enabled: bool) -> dict[str, Any]:
        await self._sync()
        try:
            self.registry.set_enabled(agent_id, enabled)
        except KeyError:
            raise NotFoundError(f"agent {agent_id!r} not found") from None
        await self.store.set_agent_enabled(agent_id, enabled, self._clock.now())
        return next(a for a in await self.store.agents() if a["id"] == agent_id)

    async def memories(self, **filters: Any) -> list[dict[str, Any]]:
        return await self.memory.recall(now=self._clock.now(), **filters)

    def _models_line(self) -> str:
        m = self.models.status()
        if not m["available"]:
            return f"not in use — {m['reason']}; every analysis is deterministic"
        u = m["usage"]
        return (
            f"{m['provider']} (fast: {m['models']['fast'] or '—'}, strong: {m['models']['strong'] or '—'}); "
            f"{u['tokens'] + u['estimated']:,} of {m['daily_token_budget']:,} tokens used today, "
            f"{u['calls']} calls, {u['cached']} answered from cache"
        )

    async def status(self) -> dict[str, Any]:
        await self._sync()
        recent = await self.store.cycles(1)
        job = self._jobs.latest(JOB_KEY)
        preds = await self.store.prediction_summary()
        measured = {
            r["agent_id"] for r in await self.store.performance("all") if r["reliability"] is not None
        }
        return {
            "paper_only": True,
            "mode": BrainMode(self._s.brain_mode).value,
            "orders": "never sent by the brain: proposals go to the deterministic risk engine only",
            "agents": {
                "registered": len(self.registry.all()),
                "enabled": sum(1 for a in self.registry.all() if self.registry.enabled(a.spec.id)),
            },
            "running": bool(job is not None and job.task is not None and not job.task.done()),
            "language_models": self._models_line(),
            "last_cycle": recent[0] if recent else None,
            "open_predictions": preds["open"],
            "learning": (
                f"{preds['evaluated']} predictions graded against real prices"
                + (f" (hit rate {preds['hit_rate']:.0%})" if preds["hit_rate"] is not None else "")
                + f"; {len(measured - {'consensus'})} agents have a measured record"
                + f" (≥{self._s.brain_min_reliability_observations} graded calls); "
                + (f"next due {preds['next_due']}" if preds["next_due"] else "nothing open")
            ),
        }
