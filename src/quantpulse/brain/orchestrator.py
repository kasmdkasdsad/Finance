"""The orchestrator (chief intelligence): one brain cycle, end to end.

1. **Perceive** — the market, the portfolio and data quality (:mod:`~quantpulse.brain.perception`).
2. **Select** — which agents are relevant and able to run, and on which subjects (each choice explained).
3. **Run** — agents concurrently by dependency level; failures and timeouts are recorded, not fatal.
4. **Consensus** — per subject from the forecasting agents; vetoes from constraint agents stay attached;
   disagreement is measured and kept; "unknown" when the evidence does not support a view.
5. **Decide** — proposed portfolio actions (HOLD / REDUCE / CLOSE / INCREASE / BUY / WATCH / NO_ACTION).
6. **Risk preview** — every proposed trade through the existing deterministic risk engine.
7. **Execute** — ``paper_execution`` only (the Brain owns the Alpaca paper account): the decisions that may
   go are handed to the trading service, which alone sends orders — after reconciling, fresh quotes and
   the same risk engine and order manager as every order (:mod:`~quantpulse.brain.execution`). In the
   other modes the Brain's simulated paper book trades instead; nothing reaches Alpaca.
8. **Remember** — the cycle, agent runs, opinions, consensus, decisions, gradeable predictions and memory.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock

from . import opportunity_outcomes
from .book import Fill, PaperBook, expected_by_subject
from .consensus import CONSENSUS_VERSION, Consensus, ReliabilityBook, build_consensus
from .context import BrainContext
from .debate import Debate, review
from .decisions import Proposal, plan, risk_preview
from .events import EventBus, from_cycle
from .execution import BrainExecutor
from .learning import PredictionRecorder
from .ledger import ExecutionLedger
from .llm import ModelRouter
from .memory import LONG_TERM, SHORT_TERM, WORKING, MemoryStore
from .opportunities import trace
from .patterns import recall
from .perception import Perception
from .registry import AgentRegistry, AgentRun, Skip
from .routing import route
from .store import BrainStore
from .theses import ThesisBook, attach_entries
from .types import MARKET, PORTFOLIO, SELLING, Action, BrainMode, Opinion, Stance

logger = logging.getLogger(__name__)
NEAR_CLOSE_MINUTES = 30  # the last half hour: holdings are reviewed for the overnight (earnings releases)


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
    fills: list[Fill] = field(default_factory=list)  # simulated in the Brain's paper book
    mark: dict[str, Any] = field(default_factory=dict)
    # paper_execution: what went to the trading service, and the positions' theses reconciled with Alpaca
    execution: dict[str, Any] = field(default_factory=dict)
    theses: dict[str, Any] = field(default_factory=dict)
    ideas: int = 0  # new ideas recorded for outcome grading (one per kind, symbol, direction and day)
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
        book: PaperBook | None = None,
        executor: BrainExecutor | None = None,
        theses: ThesisBook | None = None,
        ledger: ExecutionLedger | None = None,
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
        self._book = book
        self._executor = executor
        self._theses = theses
        self._ledger = ledger
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
            owns = mode is BrainMode.PAPER_EXECUTION and self._theses is not None and ctx.account.available
            if (
                owns and self._theses is not None
            ):  # the account's positions and theses, reconciled with Alpaca
                result.theses = await self._theses.sync(ctx, cycle_id)
            await self._context(ctx)
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
            self._fail_safe(ctx, runs, skips)
            result.consensus = self._consensus(ctx, reliability, runs, skips)
            result.debates = review(ctx, result.consensus)  # bull, bear, devil's advocate
            if owns and self._theses is not None:  # each thesis checked against this cycle's evidence
                checks = await self._theses.review(ctx, result.consensus, self._s.brain_min_confidence)
                result.theses["checks"] = _count(c["status"] for c in checks.values())
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
                attach_entries(
                    ctx,
                    result.proposals,
                    expected_by_subject(result.consensus, reliability, CONSENSUS_VERSION),
                )
                await self._recall(ctx, result)
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
            decision_ids = await self._store.save_decisions(
                cycle_id, result.proposals, consensus_ids, mode.value, now
            )
            if mode is BrainMode.PAPER_EXECUTION and self._executor is not None:
                # the Brain owns the Alpaca paper account: the trading service executes (or says why not)
                result.execution = await self._executor.execute(
                    ctx, result.proposals, cycle_id=cycle_id, scheduled=trigger != "manual"
                )
                await self._store.record_execution(decision_ids, result.proposals)
                if self._ledger is not None:  # every order sent, from the decision to its final state
                    await self._ledger.record(cycle_id, decision_ids, result.proposals)
            elif self._book is not None:  # the Brain's paper book: simulated fills, never a broker order
                result.fills = await self._book.execute(
                    ctx,
                    result.proposals,
                    cycle_id=cycle_id,
                    decision_ids=decision_ids,
                    expected=expected_by_subject(result.consensus, reliability, CONSENSUS_VERSION),
                )
                await self._store.record_book_fills(decision_ids, result.fills)
                result.mark = await self._book.mark(ctx, cycle_id)
            await self._store.save_debates(cycle_id, result.debates, now)
            opportunity_ids = await self._store.save_opportunities(cycle_id, ctx.opportunities, now)
            try:  # every idea considered, taken or not and why not — graded later (never fatal)
                result.ideas = await opportunity_outcomes.record(
                    self._store.db,
                    ctx,
                    ctx.opportunities,
                    {p.subject: p for p in result.proposals},
                    cycle_id,
                    opportunity_ids,
                )
            except Exception:
                logger.exception("recording the opportunity outcomes of brain cycle %s failed", cycle_id)
            forecasts = [
                o for r in runs for o in r.opinions if self.registry.get(r.agent_id).role == "forecast"
            ]
            result.predictions = await self._recorder.record(
                ctx, cycle_id, forecasts, result.consensus, reliability, result.debates
            )
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

    async def _context(self, ctx: BrainContext) -> None:
        """What the context agents read: the Brain's measured execution and track record (from its own
        records — nothing estimated), and whether the close is near (the overnight review)."""
        from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, regular_close

        from .scorecard import execution_quality, scorecard

        db = self._store.db
        try:
            ctx.execution_quality = await execution_quality(db, ctx.as_of - timedelta(days=30))
            ctx.track_record = await scorecard(db, self._s)
        except Exception as exc:  # context only: the agents that read it abstain
            ctx.provider_errors["track_record"] = f"{type(exc).__name__}: {exc}"[:200]
        local = ctx.as_of.astimezone(NEW_YORK)
        if ctx.market_open and is_trading_day(local.date()):
            close = datetime.combine(local.date(), regular_close(local.date()), NEW_YORK)
            left = (close - local).total_seconds() / 60
            if 0 <= left <= NEAR_CLOSE_MINUTES:
                # holdings already halved for tonight (from the ledger, so a restart does not repeat it)
                derisked: list[str] = []
                if self._ledger is not None:
                    start = datetime.combine(local.date(), time(0, 0), NEW_YORK)
                    derisked = sorted(
                        {
                            r["symbol"]
                            for r in await self._ledger.rows(limit=200, since=start)
                            if r["side"] == "sell" and (r["reason"] or "").startswith("overnight:")
                        }
                    )
                ctx.working.post("near_close", {"minutes_to_close": round(left, 1), "derisked": derisked})

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
    @staticmethod
    def _fail_safe(ctx: BrainContext, runs: list[AgentRun], skips: list[Skip]) -> None:
        """Checks that must hold even when the agent responsible did not run: fail closed."""
        why = {r.agent_id: f"{r.status}: {r.error}" for r in runs if r.status != "ok"}
        why.update({s.agent_id: s.reason for s in skips})
        if not any(o.agent_id == "data_quality" for o in ctx.working.opinions.get(MARKET, [])):
            reason = why.get("data_quality", "it produced no market view")
            ctx.working.post(
                "system_vetoes", [f"the data-quality check did not run ({reason}): nothing is executable"]
            )
        if "situation" not in ctx.working.facts:
            reason = why.get("situational_awareness", "it produced no posture")
            ctx.working.post(
                "situation",
                {
                    "posture": "cautious",
                    "risk_scale": 0.6,
                    "reasons": [f"situational awareness did not run ({reason}): cautious by default"],
                    "session": ctx.session.value,
                },
            )

    def _missing(
        self, subject: str, opinions: list[Opinion], runs: list[AgentRun], skips: list[Skip]
    ) -> list[dict[str, str]]:
        """The forecasting agents that could have had a view on ``subject`` and did not, with the reason."""
        kind = "market" if subject == MARKET else "symbol"
        ran = {r.agent_id: r for r in runs}
        skipped = {s.agent_id: s.reason for s in skips}
        said = {o.agent_id: o for o in opinions}
        out: list[dict[str, str]] = []
        for agent in self.registry.all():
            spec = agent.spec
            if agent.role != "forecast" or kind not in spec.subjects:
                continue
            o = said.get(spec.id)
            if o is not None:
                if o.stance is Stance.ABSTAIN:
                    out.append(self._gap(spec.id, spec.source, "abstained", o.thesis))
                continue
            if spec.id in skipped:
                out.append(self._gap(spec.id, spec.source, "skipped", skipped[spec.id]))
            elif spec.id in ran and ran[spec.id].status != "ok":
                r = ran[spec.id]
                out.append(self._gap(spec.id, spec.source, r.status, r.error or r.status))
            elif spec.id in ran:
                out.append(self._gap(spec.id, spec.source, "not asked", "not asked about this subject"))
        return out

    @staticmethod
    def _gap(agent_id: str, source: str, kind: str, reason: str) -> dict[str, str]:
        return {"agent_id": agent_id, "source": source or agent_id, "kind": kind, "reason": reason[:200]}

    def _consensus(
        self, ctx: BrainContext, reliability: ReliabilityBook, runs: list[AgentRun], skips: list[Skip]
    ) -> dict[str, Consensus]:
        out: dict[str, Consensus] = {}
        horizons: dict[str, int] = {}
        regime = ctx.working.facts.get("regime")
        sources = {a.spec.id: a.spec.source for a in self.registry.all() if a.spec.source}
        for subject, opinions in ctx.working.opinions.items():
            if subject == PORTFOLIO:
                continue
            forecasts = [o for o in opinions if self.registry.get(o.agent_id).role == "forecast"]
            constraints = [o for o in opinions if self.registry.get(o.agent_id).role == "constraint"]
            if not forecasts:
                continue
            out[subject] = build_consensus(
                subject,
                forecasts,
                constraints,
                reliability,
                regime,
                sources=sources,
                missing=self._missing(subject, opinions, runs, skips),
            )
            voting = [o for o in forecasts if o.directional]
            if voting:
                horizons[subject] = round(
                    sum(o.horizon_days * o.confidence for o in voting)
                    / max(sum(o.confidence for o in voting), 1e-9)
                )
        ctx.working.post("horizons", horizons)
        return out

    # ------------------------------------------------------------------ memory
    async def _recall(self, ctx: BrainContext, result: CycleResult) -> None:
        """Attach what memory says to each decision: lessons on the subject and the recurring patterns for
        the objections raised, the kinds of opportunity involved and the regime (context, not a vote)."""
        patterns = await self._memory.recall(tier=LONG_TERM, kind="pattern", limit=300)
        lessons = await self._memory.recall(tier=LONG_TERM, kind="lesson", limit=300)
        if not patterns and not lessons:
            return
        kinds: dict[str, set[str]] = {}
        for o in ctx.opportunities:
            for sym in o.symbols[:1]:
                kinds.setdefault(sym, set()).add(o.kind)
        regime = ctx.working.facts.get("regime")
        for p in result.proposals:
            debate = result.debates.get(p.subject)
            p.memory = recall(
                p.subject,
                patterns=patterns,
                lessons=lessons,
                objections=[o.code for o in debate.objections] if debate else [],
                kinds=sorted(kinds.get(p.subject, set())),
                regime=regime,
            )

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
        # what the Brain did (its book's fills), compactly — proposals that were not acted on stay in the
        # cycle record only, so memory is not filled with the same idea every half hour
        for f in result.fills:
            p = next((x for x in result.proposals if x.subject == f.symbol), None)
            await self._memory.remember(
                LONG_TERM,
                "trade",
                f.symbol,
                f"paper book {f.side} {f.qty:g} {f.symbol} at ${f.fill_price:,.2f} ({f.action})"
                + (f", realised ${f.realized_pnl:,.0f}" if f.realized_pnl is not None else ""),
                now,
                data={
                    **f.to_dict(),
                    "reasons": (p.reasons if p else [])[:4],
                    "confidence": round(p.confidence, 3) if p else None,
                    "regime": label,
                    "posture": (ctx.working.facts.get("situation") or {}).get("posture"),
                },
                tags=["trade", f.action, f.side],
                importance=0.7,
                cycle_id=cycle_id,
            )
        # orders the trading service sent for the Brain (paper_execution) that Alpaca filled
        for p in result.proposals:
            ex = p.execution or {}
            if not ex.get("filled_qty"):
                continue
            side = "sell" if p.action in SELLING else "buy"
            await self._memory.remember(
                LONG_TERM,
                "trade",
                p.subject,
                f"Alpaca paper {side} {ex['filled_qty']:g} {p.subject} at ${ex.get('filled_avg_price') or 0:,.2f} "
                f"({p.action.value}; order {ex.get('client_order_id')})",
                now,
                data={
                    **ex,
                    "action": p.action.value,
                    "reasons": p.reasons[:4],
                    "confidence": round(p.confidence, 3),
                    "regime": label,
                    "posture": (ctx.working.facts.get("situation") or {}).get("posture"),
                },
                tags=["trade", p.action.value, side, "alpaca_paper"],
                importance=0.8,
                cycle_id=cycle_id,
            )

    # ------------------------------------------------------------------ summary
    def _summary(self, ctx: BrainContext, result: CycleResult) -> dict[str, Any]:
        regime = ctx.regime
        owns = ctx.mode is BrainMode.PAPER_EXECUTION
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
                "owner": "the Alpaca paper account (owned by the Brain)"
                if owns
                else "the Brain's paper book (hypothetical)"
                if self._book is not None
                else "Alpaca account",
                **ctx.portfolio.summary(),
                "constraints": ctx.working.facts.get("portfolio_constraints"),
                "book": {"fills": [f.to_dict() for f in result.fills], "mark": result.mark},
                "alpaca_account": {
                    "owner": "the Brain (QP_BRAIN_MODE=paper_execution)"
                    if owns
                    else "the trading strategy (the Brain only reads it)",
                    **ctx.account.summary(),
                },
                "trading_controls": {
                    "orders_would_reach_alpaca": not ctx.trading_blockers,
                    "blockers": ctx.trading_blockers,
                    "note": "every Brain order goes through the trading service (reconciliation, fresh quotes, "
                    "the risk engine, the order manager)"
                    if owns
                    else "proposals only: nothing is sent to Alpaca in this mode",
                },
                "execution": result.execution,
                "theses": result.theses,
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
                "ideas_recorded": result.ideas,
                "orders_sent": int(result.execution.get("orders_sent") or 0),
                "entry_halts": [h["code"] for h in result.execution.get("entry_halts") or []],
                "book_fills": len(result.fills),
                "opportunities": _count(o.status for o in ctx.opportunities),
                "debates": _count(d.verdict for d in result.debates.values()),
                "posture": (ctx.working.facts.get("situation") or {}).get("posture"),
                "decision": explain(ctx, result),  # why it traded, or why it did not
            },
            "notes": ctx.notes,
        }


def explain(ctx: BrainContext, result: CycleResult) -> dict[str, Any]:
    """Why the Brain traded, or why it did not — in plain words, from what this cycle recorded."""
    mode = ctx.mode
    proposals = result.proposals
    trades = [p for p in proposals if p.is_trade]
    sent = [p for p in trades if (p.execution or {}).get("sent")]
    ex = result.execution or {}
    why: list[str] = []
    if mode is BrainMode.RESEARCH_ONLY:
        return {"outcome": "research_only", "headline": "research only: no decisions are made in this mode",
                "orders": [], "reasons": []}  # fmt: skip
    if sent:
        orders = [
            {
                "subject": p.subject,
                "action": p.action.value,
                "qty": (p.execution or {}).get("qty") or p.quantity,
                "status": p.status,
                "client_order_id": (p.execution or {}).get("client_order_id"),
                "why": p.reasons[:3],
            }
            for p in sent
        ]
        headline = f"{len(sent)} Alpaca paper order(s) sent: " + ", ".join(
            f"{o['action']} {o['qty']:g} {o['subject']}" for o in orders
        )
        for p in trades:
            if p not in sent:
                why.append(f"{p.action.value} {p.subject} not sent: {_not_sent(p)}")
        return {"outcome": "traded", "headline": headline, "orders": orders, "reasons": why}
    if trades:
        for p in trades:
            why.append(f"{p.action.value} {p.subject}: {_not_sent(p)}")
        if mode is BrainMode.PAPER_EXECUTION:
            headline = "no trade: the proposed trade(s) did not pass every gate"
        else:
            headline = f"no order: QP_BRAIN_MODE={mode.value} proposes only"
        return {"outcome": "no_trade", "headline": headline, "orders": [], "reasons": why[:12]}
    unknown = [p.subject for p in proposals if p.action is Action.NO_ACTION and p.reasons
               and p.reasons[0].startswith("I do not know")]  # fmt: skip
    watch = [p for p in proposals if p.action is Action.WATCH]
    holds = [p for p in proposals if p.action is Action.HOLD]
    if not ctx.market_open:
        why.append("the market is closed")
    why += [f"data: {v}" for v in (ctx.working.facts.get("system_vetoes") or [])]
    market = next((o for o in ctx.working.opinions.get(MARKET, []) if o.agent_id == "data_quality"), None)
    if market is not None and market.veto:
        why.append(f"data quality: {market.veto}")
    why += [f"halt: {h['reason']}" for h in ex.get("entry_halts") or []]
    if not any(c.stance is Stance.BULLISH and c.actionable_view for s, c in result.consensus.items()
               if s != MARKET):  # fmt: skip
        why.append("no symbol has a clear bullish consensus")
    if unknown:
        why.append(f"not enough evidence for a view on {len(unknown)} symbol(s): " + ", ".join(unknown[:6]))
    for p in watch[:6]:
        why.append(f"watch {p.subject}: " + "; ".join(p.reasons)[:200])
    if holds:
        why.append(f"{len(holds)} holding(s) kept: nothing calls for a change")
    return {
        "outcome": "no_trade",
        "headline": "no trade: no opportunity met every requirement — doing nothing is the decision",
        "orders": [],
        "reasons": why[:14],
    }


def _not_sent(p: Proposal) -> str:
    ex = p.execution or {}
    if ex.get("reason"):
        return str(ex["reason"])
    if p.risk_approved is False:
        return "the risk engine: " + str((p.risk or {}).get("summary") or "rejected")
    if p.blocked_by:
        return "blocked: " + "; ".join(p.blocked_by)
    return p.status


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
