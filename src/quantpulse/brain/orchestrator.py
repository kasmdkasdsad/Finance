"""The orchestrator (chief intelligence): one brain cycle, end to end.

1. **Perceive** — the market, the portfolio and data quality (:mod:`~quantpulse.brain.perception`).
2. **Select** — which agents are relevant and able to run, and on which subjects (each choice explained).
3. **Run** — agents concurrently by dependency level; failures and timeouts are recorded, not fatal.
4. **Consensus** — per subject from the forecasting agents; vetoes from constraint agents stay attached;
   disagreement is measured and kept; "unknown" when the evidence does not support a view.
5. **Decide** — proposed portfolio actions (HOLD / REDUCE / CLOSE / INCREASE / BUY / WATCH / NO_ACTION).
6. **Risk preview** — every proposed trade through the existing deterministic risk engine. Nothing is sent
   to Alpaca: the brain has no broker access beyond a read-only view.
7. **Remember** — the cycle, agent runs, opinions, consensus, decisions, gradeable predictions and memory.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock

from .consensus import Consensus, ReliabilityBook, build_consensus
from .context import BrainContext
from .debate import Debate, review
from .decisions import Proposal, plan, risk_preview
from .events import EventBus, from_cycle
from .learning import PredictionRecorder
from .llm import ModelRouter
from .memory import LONG_TERM, SHORT_TERM, WORKING, MemoryStore
from .opportunities import trace
from .perception import Perception
from .registry import AgentRegistry, AgentRun, Skip
from .routing import route
from .store import BrainStore
from .types import MARKET, PORTFOLIO, BrainMode

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    cycle_id: int
    status: str
    ctx: BrainContext | None = None
    runs: list[AgentRun] = field(default_factory=list)
    skips: list[Skip] = field(default_factory=list)
    consensus: dict[str, Consensus] = field(default_factory=dict)
    proposals: list[Proposal] = field(default_factory=list)
    debates: dict[str, Debate] = field(default_factory=dict)
    predictions: int = 0
    error: str | None = None


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        perception: Perception,
        registry: AgentRegistry,
        store: BrainStore,
        memory: MemoryStore,
        recorder: PredictionRecorder,
        bus: EventBus | None = None,
        models: ModelRouter | None = None,
    ) -> None:
        self._s = settings
        self._clock = clock
        self._perception = perception
        self.registry = registry
        self._store = store
        self._memory = memory
        self._recorder = recorder
        self._bus = bus
        self._models = models
        self._lock = asyncio.Lock()
        self.last_ctx: BrainContext | None = None  # the latest completed cycle's picture (for the monitor)

    @property
    def mode(self) -> BrainMode:
        return BrainMode(self._s.brain_mode)

    async def run(
        self,
        *,
        trigger: str = "manual",
        kind: str = "full",
        symbols: Sequence[str] = (),
        only: Sequence[str] | None = None,
    ) -> CycleResult:
        async with self._lock:  # one cycle at a time
            return await self._run(trigger, kind, symbols, only)

    async def _run(
        self, trigger: str, kind: str, symbols: Sequence[str], only: Sequence[str] | None
    ) -> CycleResult:
        from .context import brain_session

        mode = self.mode
        started = self._clock.now()
        cycle_id = await self._store.start_cycle(
            kind=kind, trigger=trigger, session=brain_session(started).value, mode=mode.value, now=started
        )
        result = CycleResult(cycle_id=cycle_id, status="running")
        try:
            last_regime = await self._memory.latest(LONG_TERM, "regime_change", MARKET)
            previous_regime = last_regime["data"].get("to") if last_regime else None
            plan_route = route(kind)
            ctx = await self._perception.perceive(
                mode,
                symbols,
                previous_regime=previous_regime,
                pre_screen=plan_route.pre_screen,
                opportunity_budget=plan_route.opportunities,
            )
            ctx.strategy_signals = await self._store.get_state("promoted_signals") or {}
            ctx.llm = self._models
            result.ctx = ctx
            wanted = only if only is not None else plan_route.agents
            selections, skips = self.registry.select(ctx, wanted)
            if wanted is not None:
                skips.extend(
                    Skip(a.spec.id, a.spec.version, f"not needed for a {kind} cycle")
                    for a in self.registry.all()
                    if a.spec.id not in wanted
                )
            runs = await self.registry.run(ctx, selections, self._s.brain_agent_timeout_seconds)
            result.runs, result.skips = runs, skips

            reliability = ReliabilityBook(
                await self._store.reliability_rows(), self._s.brain_min_reliability_observations
            )
            result.consensus = self._consensus(ctx, reliability)
            result.debates = review(ctx, result.consensus)  # bull, bear, devil's advocate
            if mode is not BrainMode.RESEARCH_ONLY:
                result.proposals = plan(
                    ctx,
                    result.consensus,
                    min_confidence=self._s.brain_min_confidence,
                    max_new=self._s.brain_max_new_positions_per_cycle,
                    vol_budget=self._s.trading_position_vol_budget,
                    vol_floor=self._s.trading_vol_floor,
                    earnings_caution_days=max(
                        self._s.brain_earnings_caution_days, self._s.trading_earnings_blackout_days
                    ),
                    debates=result.debates,
                )
                risk_preview(ctx, result.proposals, mode)
            trace(
                ctx.opportunities,
                focus=ctx.focus,
                states=ctx.data_states,
                market_open=ctx.market_open,
                opinions=ctx.working.opinions,
                consensus=result.consensus,
                debates=result.debates,
                proposals={p.subject: p for p in result.proposals},
            )

            now = self._clock.now()
            await self._store.save_runs(cycle_id, runs, skips, now)
            consensus_ids = await self._store.save_consensus(cycle_id, result.consensus, now)
            await self._store.save_decisions(cycle_id, result.proposals, consensus_ids, mode.value, now)
            await self._store.save_debates(cycle_id, result.debates, now)
            await self._store.save_opportunities(cycle_id, ctx.opportunities, now)
            forecasts = [
                o for r in runs for o in r.opinions if self.registry.get(r.agent_id).role == "forecast"
            ]
            result.predictions = await self._recorder.record(ctx, cycle_id, forecasts, result.consensus)
            await self._remember(ctx, cycle_id, result)
            result.status = "completed"
            await self._store.finish_cycle(
                cycle_id, self._clock.now(), status="completed", **self._summary(ctx, result)
            )
            self.last_ctx = ctx
            await self._publish(ctx, cycle_id, result, previous_regime)
        except Exception as exc:
            logger.exception("brain cycle %s failed", cycle_id)
            result.status, result.error = "failed", f"{type(exc).__name__}: {exc}"
            await self._store.finish_cycle(cycle_id, self._clock.now(), status="failed", error=result.error)
        return result

    # ------------------------------------------------------------------ events
    async def _publish(
        self, ctx: BrainContext, cycle_id: int, result: CycleResult, previous_regime: str | None
    ) -> None:
        """What this cycle noticed, as events (never fatal: the cycle is already recorded)."""
        if self._bus is None:
            return
        try:
            before = await self._store.get_state("last_portfolio")
            portfolio = ctx.portfolio.summary()
            moves = {}
            if "move_z" in ctx.indicators.columns and ctx.market_open:
                moves = {str(s): float(v) for s, v in ctx.indicators["move_z"].items() if v == v}
            events = from_cycle(
                cycle_id,
                regime=ctx.regime.label if ctx.regime else None,
                previous_regime=previous_regime,
                states={s: st.value for s, st in ctx.data_states.items()},
                held=ctx.held,
                focus=ctx.focus,
                moves=moves,
                opportunities=ctx.opportunities,
                event_risk=ctx.working.facts.get("event_risk") or {},
                portfolio=portfolio,
                previous_portfolio=before,
                runs=result.runs,
            )
            if portfolio.get("available"):
                await self._store.set_state("last_portfolio", portfolio, self._clock.now())
            await self._bus.publish(events)
        except Exception:
            logger.exception("publishing events for brain cycle %s failed", cycle_id)

    # ------------------------------------------------------------------ consensus
    def _consensus(self, ctx: BrainContext, reliability: ReliabilityBook) -> dict[str, Consensus]:
        out: dict[str, Consensus] = {}
        horizons: dict[str, int] = {}
        regime = ctx.working.facts.get("regime")
        for subject, opinions in ctx.working.opinions.items():
            if subject == PORTFOLIO:
                continue
            forecasts = [o for o in opinions if self.registry.get(o.agent_id).role == "forecast"]
            constraints = [o for o in opinions if self.registry.get(o.agent_id).role == "constraint"]
            if not forecasts:
                continue
            out[subject] = build_consensus(subject, forecasts, constraints, reliability, regime)
            voting = [o for o in forecasts if o.directional]
            if voting:
                horizons[subject] = round(
                    sum(o.horizon_days * o.confidence for o in voting)
                    / max(sum(o.confidence for o in voting), 1e-9)
                )
        ctx.working.post("horizons", horizons)
        return out

    # ------------------------------------------------------------------ memory
    async def _remember(self, ctx: BrainContext, cycle_id: int, result: CycleResult) -> None:
        now = self._clock.now()
        regime = ctx.regime
        label = regime.label if regime else None
        market = result.consensus.get(MARKET)
        await self._memory.remember(
            SHORT_TERM,
            "market_state",
            MARKET,
            f"{label or 'unknown'} regime; market view {market.to_dict()['stance'] if market else 'n/a'}",
            now,
            key="market_state",
            data={
                "regime": label,
                "trend_score": regime.trend_score if regime else None,
                **ctx.market_stats,
                "vix": ctx.vix,
                "market_open": ctx.market_open,
            },
            tags=["market", label or "unknown"],
            ttl=timedelta(days=1),
            cycle_id=cycle_id,
        )
        await self._memory.remember(
            SHORT_TERM,
            "portfolio_state",
            PORTFOLIO,
            f"{len(ctx.held)} positions, equity {ctx.portfolio.equity:,.0f}"
            if ctx.portfolio.available
            else "paper account unavailable",
            now,
            key="portfolio_state",
            data={**ctx.portfolio.summary(), "constraints": ctx.working.facts.get("portfolio_constraints")},
            tags=["portfolio"],
            ttl=timedelta(days=1),
            cycle_id=cycle_id,
        )
        disputes = {
            s: c.primary_disagreement["summary"]
            for s, c in result.consensus.items()
            if c.primary_disagreement
        }
        await self._memory.remember(
            WORKING,
            "investigation",
            MARKET,
            f"cycle {cycle_id}: {len(ctx.focus)} focus symbols, {len(disputes)} disagreements, "
            f"{sum(1 for p in result.proposals if p.is_trade)} proposed trades",
            now,
            key=f"cycle:{cycle_id}",
            data={
                "focus": ctx.focus_reasons,
                "disagreements": disputes,
                "questions": ctx.working.questions,
                "unknown": [s for s, c in result.consensus.items() if c.unknown],
            },
            tags=["cycle"],
            ttl=timedelta(days=3),
            cycle_id=cycle_id,
        )
        if label is not None:
            last = await self._memory.latest(LONG_TERM, "regime_change", MARKET)
            if last is None or last["data"].get("to") != label:
                await self._memory.remember(
                    LONG_TERM,
                    "regime_change",
                    MARKET,
                    f"regime {last['data'].get('to') if last else 'first observed'} → {label}",
                    now,
                    data={
                        "from": last["data"].get("to") if last else None,
                        "to": label,
                        "trend_score": regime.trend_score if regime else None,
                    },
                    tags=["regime", label],
                    importance=0.8,
                    cycle_id=cycle_id,
                )
        for p in result.proposals:
            if p.is_trade:
                await self._memory.remember(
                    LONG_TERM,
                    "decision",
                    p.subject,
                    f"{p.action.value} {p.quantity:g} {p.subject}: {p.status}"
                    + (f" — {p.risk.get('summary')}" if p.risk else ""),
                    now,
                    data=p.to_dict() | {"consensus": p.consensus.to_dict() if p.consensus else None},
                    tags=["decision", p.action.value, p.status],
                    importance=0.7,
                    cycle_id=cycle_id,
                )

    # ------------------------------------------------------------------ summary
    def _summary(self, ctx: BrainContext, result: CycleResult) -> dict[str, Any]:
        regime = ctx.regime
        runs = result.runs
        dq = [o for o in ctx.working.opinions.get(MARKET, []) if o.agent_id == "data_quality"]
        by_status: dict[str, int] = {}
        for p in result.proposals:
            by_status[p.status] = by_status.get(p.status, 0) + 1
        return {
            "regime": {
                "label": regime.label,
                "trend_score": regime.trend_score,
                "stressed": regime.stressed,
                "description": regime.description,
                "reasons": regime.reasons,
            }
            if regime
            else {},
            "market": {
                "open": ctx.market_open,
                "clock": ctx.clock_source,
                "stats": ctx.market_stats,
                "vix": ctx.vix,
                "price_status": ctx.price_status.value,
                "situation": ctx.working.facts.get("situation") or {},
            },
            "portfolio": {
                **ctx.portfolio.summary(),
                "constraints": ctx.working.facts.get("portfolio_constraints"),
                "trading_controls": {
                    "orders_would_reach_alpaca": not ctx.trading_blockers,
                    "blockers": ctx.trading_blockers,
                    "note": "read only: the Brain never submits orders",
                },
            },
            "data_quality": {
                "market": dq[0].to_dict() if dq else None,
                "states": {s: ctx.state(s).value for s in ctx.focus},
                "diagnosis": {s: ctx.data_health[s].to_dict() for s in ctx.focus if s in ctx.data_health},
                "feed": ctx.feed,
                "provider_errors": ctx.provider_errors,
            },
            "focus": [{"symbol": s, "reason": ctx.focus_reasons.get(s, "")} for s in ctx.focus],
            "agents": [
                *[
                    {
                        "agent_id": r.agent_id,
                        "status": r.status,
                        "duration_ms": round(r.duration_ms, 1),
                        "opinions": len(r.opinions),
                        "error": r.error,
                    }
                    for r in runs
                ],
                *[{"agent_id": s.agent_id, "status": "skipped", "reason": s.reason} for s in result.skips],
            ],
            "summary": {
                "agents_run": sum(1 for r in runs if r.status == "ok"),
                "agents_failed": sum(1 for r in runs if r.status != "ok"),
                "agents_skipped": len(result.skips),
                "opinions": sum(len(r.opinions) for r in runs),
                "subjects": len(result.consensus),
                "unknown": sum(1 for c in result.consensus.values() if c.unknown),
                "disagreements": sum(1 for c in result.consensus.values() if c.primary_disagreement),
                "proposals": by_status,
                "trades_proposed": sum(1 for p in result.proposals if p.is_trade),
                "risk_approved": sum(1 for p in result.proposals if p.risk_approved),
                "predictions_recorded": result.predictions,
                "orders_sent": 0,
                "opportunities": _count(o.status for o in ctx.opportunities),
                "debates": _count(d.verdict for d in result.debates.values()),
                "posture": (ctx.working.facts.get("situation") or {}).get("posture"),
            },
            "notes": ctx.notes,
        }


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
