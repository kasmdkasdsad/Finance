"""The brain as a service: builds the registry, perception and orchestrator from QuantPulse's container and
exposes cycles, agents, memory and predictions to the API. Cycles run in the background job registry so a
slow cycle answers 202 with its progress instead of blocking a request."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError, NotFoundError
from quantpulse.core.jobs import Job, JobPending, JobRegistry
from quantpulse.db.session import Database
from quantpulse.logging_config import log_event
from quantpulse.providers.alpaca_trading import AlpacaPaperBroker
from quantpulse.services.market import MarketService
from quantpulse.services.model import ModelService
from quantpulse.services.options import OptionsService
from quantpulse.services.reference import ReferenceService
from quantpulse.services.trading import TradingService
from quantpulse.services.trading_data import TradingDataLoader

from .agents import default_agents
from .book import PaperBook
from .context import BrokerView
from .evaluation import MarketPrices
from .events import Event, EventBus, EventType
from .execution import BrainExecutor
from .improvement import ImprovementEngine
from .lab.service import StrategyLab
from .learning import Learner, PredictionRecorder
from .ledger import ExecutionLedger
from .llm import ModelRouter
from .memory import MemoryStore
from .orchestrator import Orchestrator
from .perception import Perception
from .reflection import consensus_calibration
from .registry import AgentRegistry
from .reviews import Reviewer
from .sessions import SessionKeeper
from .shadow import StrategyShadow
from .store import BrainStore
from .supervisor import Supervisor
from .theses import ThesisBook
from .types import BrainMode

# Shown on the Brain page and in /brain/status: what the Brain cannot (yet) do or know.
LIMITATIONS = (
    "Alpaca PAPER only: simulated money. There is no live mode.",
    "Track records start empty: every agent is unproven until enough of its calls are graded (weeks).",
    "No news provider: no news event is ever produced or invented.",
    "No historical macro dataset: macro context is today's regime, VIX and breadth only.",
    "The strategy lab backtests today's liquid universe: its results carry survivorship bias.",
    "Paper-book and strategy-shadow fills are modelled (spread, slippage, fees); the Brain's own are Alpaca's "
    "paper fills.",
    "With the free IEX feed a quiet name's price can still go stale (its IEX book stops moving) and IEX "
    "spreads can be wider than the national best: the data checks can block trading (real-time SIP is the "
    "fix, and your decision).",
    "It may go days without a trade: NO TRADE whenever the evidence, the data or any gate is not there.",
    "The overnight earnings rule does not know a release's time of day: any release before the next "
    "session halves the position in the last half hour (the server must be running then).",
    "Take-profit needs a calibrated target: until the consensus is calibrated there are none.",
    "Execution quality is unproven below 10 fills; Alpaca's paper fills can be kinder than real ones.",
    "Event days are detected from prices (a large benchmark move or VIX ≥ 30); there is no macro calendar.",
    "Ideas are graded at one horizon per kind of idea, from their first detection that day.",
    "Past checkpoint windows have no unrealised P&L: positions are not re-marked historically.",
    "Stops and thesis checks run at each cycle, not as resting stop orders: a gap can fill beyond the stop.",
    "Agent performance and the 60-session evaluation are not established until enough observations exist.",
)

JOB_KEY = "brain-cycle"
_events_log = logging.getLogger("quantpulse.events")
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
        self.trading = trading  # the only way a Brain decision reaches the Alpaca paper account
        self.bus = EventBus(db, clock)
        self.registry = AgentRegistry(default_agents())
        self.store = BrainStore(db)
        self.memory = MemoryStore(db)
        self.models = ModelRouter(settings, clock, store=self.store)  # 'none' unless a provider is configured
        self.book = PaperBook(settings, db, clock)  # the Brain's own hypothetical portfolio
        # the Alpaca account's position theses (paper_execution)
        self.theses = ThesisBook(settings, db, clock)
        self.ledger = ExecutionLedger(db, clock)  # every Brain order, decision to final state
        self.executor = BrainExecutor(  # the only way a Brain decision becomes an order (via trading)
            settings,
            clock,
            trading,
            data,
            self.store,
            lambda: {
                "registered": len(self.registry.all()),
                "enabled": sum(1 for a in self.registry.all() if self.registry.enabled(a.spec.id)),
            },
        )
        self.perception = Perception(
            settings, clock, data, BrokerView(broker), trading, reference, model, options, market, self.book
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
            self.book,
            self.executor,
            self.theses,
            self.ledger,
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
        # daily and weekly reviews of its own results (lessons; proposals only, never a change)
        self.reviewer = Reviewer(db, settings, clock, self.memory, self.improvements)
        self.shadow = StrategyShadow(settings, clock, db, trading)  # the replaced strategy, for comparison
        self.sessions = SessionKeeper(
            settings,
            clock,
            db,
            trading,
            data,
            MarketPrices(market, clock) if market is not None else None,
            market.feed_status if market is not None else None,
            self.shadow,
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
            _log_cycle(detail)
            return detail

        job = self._jobs.start("brain", JOB_KEY, f"Brain cycle ({kind}, {trigger})", work)
        try:
            return await self._jobs.wait(job, wait)
        except JobPending:
            raise BrainCycleRunning(job) from None

    # ------------------------------------------------------------------ learning
    async def trade_lessons(self) -> dict[str, Any]:
        """Structured lessons from the positions closed since the last pass (after each close)."""
        from .trade_lessons import learn_from_trades

        return await learn_from_trades(self._db, self.memory, self._clock)

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

    async def execution_status(self) -> dict[str, Any]:
        """Ownership, the Brain kill switch and every reason a Brain order would not be sent right now (the
        entry halts are the last cycle's: they depend on what it saw)."""
        trading = self.trading
        kill = await trading.kill_switch()
        recent = await self.store.cycles(1)
        last = (await self.store.cycle(recent[0]["id"])) if recent else None
        executed = ((last or {}).get("portfolio") or {}).get("execution") or {}
        return {
            "paper_only": True,
            "endpoint": trading.broker.base_url,
            "owner": trading.owner,
            "mode": self._s.brain_mode,
            "owns_account": self._s.brain_owns_account,
            "brain_kill_switch": (await trading.brain_kill_switch()).model_dump(mode="json"),
            "trading_kill_switch": kill.model_dump(mode="json"),
            "blockers_manual": await trading.submit_blockers(kill, owner="brain"),
            "blockers_scheduled": await trading.submit_blockers(kill, scheduled=True, owner="brain"),
            "last_cycle": {
                "id": last["id"],
                "started_at": last["started_at"],
                "orders_sent": executed.get("orders_sent", 0),
                "entries_allowed": executed.get("entries_allowed"),
                "entry_halts": executed.get("entry_halts") or [],
                "blockers": executed.get("blockers") or [],
                "trading_cycle_id": executed.get("trading_cycle_id"),
            }
            if last
            else None,
        }

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
            "owns_account": self._s.brain_owns_account,
            "orders": (
                "the Brain owns the Alpaca paper account: its decisions are executed by the trading service "
                "(reconciliation, fresh quotes, the risk engine, the order manager, every trading switch)"
                if self._s.brain_owns_account
                else "proposals only: managed in the Brain's simulated paper book; nothing is sent to Alpaca"
            ),
            "agents": {
                "registered": len(self.registry.all()),
                "enabled": sum(1 for a in self.registry.all() if self.registry.enabled(a.spec.id)),
            },
            "running": bool(job is not None and job.task is not None and not job.task.done()),
            "language_models": self._models_line(),
            "limitations": list(LIMITATIONS),
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


def _log_cycle(detail: dict[str, Any]) -> None:
    """A Brain cycle, its trade decisions and any data-quality halt as structured log lines."""
    summary = detail.get("summary") or {}
    halts = summary.get("entry_halts") or []
    failed = detail.get("status") == "failed"
    log_event(_events_log, "brain.cycle", f"Brain cycle #{detail['id']} {detail.get('status')} ({detail.get('kind')})",
              level=logging.WARNING if failed else logging.INFO, cycle_id=detail["id"], kind=detail.get("kind"),
              trigger=str(detail.get("trigger"))[:96], status=detail.get("status"),
              duration_ms=detail.get("duration_ms"), trades_proposed=summary.get("trades_proposed"),
              risk_approved=summary.get("risk_approved"), orders_sent=summary.get("orders_sent"),
              entry_halts=",".join(map(str, halts)) or None, error=detail.get("error"))  # fmt: skip
    if "data_quality" in halts:
        veto = ((detail.get("data_quality") or {}).get("market") or {}).get(
            "veto"
        ) or "market data not usable"
        log_event(_events_log, "brain.data_quality_halt", f"new positions halted by market data: {veto}"[:400],
                  level=logging.WARNING, cycle_id=detail["id"])  # fmt: skip
    for d in detail.get("decisions") or []:
        if not d.get("quantity"):
            continue
        ex = d.get("execution") or {}
        why = ex.get("reason") or "; ".join(((d.get("rationale") or {}).get("reasons") or [])[:2])
        event = "brain.risk_rejection" if d.get("risk_approved") is False else "brain.decision"
        log_event(_events_log, event, f"{d.get('action')} {d.get('subject')}: {d.get('status')}"[:200],
                  cycle_id=detail["id"], decision_id=d.get("id"), symbol=d.get("subject"), action=d.get("action"),
                  status=d.get("status"), quantity=d.get("quantity"), sent=ex.get("sent"), reason=str(why)[:300])  # fmt: skip
