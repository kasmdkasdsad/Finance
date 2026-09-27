"""The learning loop: predictions are written when they are made and graded against reality later.

::

    opinion / consensus ──► prediction (status "open")
        ──► horizon passes ──► evaluation against real closes (hit or miss, relative return)
        ──► decision outcome ──► reflection (decision quality vs outcome quality, lessons)
        ──► agent / consensus performance (hit rate, Brier, IC, calibration, reliability)
        ──► failure analysis ──► lessons in memory ──► consensus weights (and improvement proposals)

What is recorded, and why it can be graded honestly:

* only **gradeable forecasts** — directional opinions of *forecast* agents (not data-quality, portfolio or
  research context), and directional consensus that was not "unknown";
* each with its horizon (in sessions), the benchmark it is measured against, the regime it was made in,
  the entry price and benchmark level, and the due date;
* symbols are graded on their return **relative to the benchmark**; ``@market`` views on the benchmark's
  own return.

No agent gets a reliability number until its own predictions have matured and been graded
(:mod:`~quantpulse.brain.performance`); decisions are judged on process and outcome separately
(:mod:`~quantpulse.brain.reflection`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, sessions_after
from quantpulse.db.models import BrainPredictionRow
from quantpulse.db.session import Database

from . import performance as perf
from . import reflection
from .consensus import CONSENSUS_VERSION, Consensus
from .context import BrainContext
from .evaluation import PriceSource, evaluate_due
from .memory import AGENT, LONG_TERM, MemoryStore
from .reflection import reflect_on_decisions
from .types import MARKET, Opinion


def _due(made_at: datetime, horizon: int) -> object:
    return sessions_after(made_at.astimezone(NEW_YORK).date(), max(horizon, 1))


class PredictionRecorder:
    def __init__(self, db: Database) -> None:
        self._db = db

    def _row(
        self,
        ctx: BrainContext,
        cycle_id: int,
        *,
        source_type: str,
        source_id: str,
        version: str,
        subject: str,
        direction: int,
        score: float,
        confidence: float,
        horizon: int,
        extra: dict,
    ) -> BrainPredictionRow | None:
        bench_price = ctx.price(ctx.benchmark_symbol)
        entry = bench_price if subject == MARKET else ctx.price(subject)
        if entry is None or bench_price is None:
            return None  # cannot be graded without an entry price
        return BrainPredictionRow(
            cycle_id=cycle_id,
            source_type=source_type,
            source_id=source_id,
            source_version=version,
            subject=subject,
            direction=direction,
            score=round(score, 4),
            confidence=round(confidence, 4),
            horizon_days=horizon,
            benchmark="absolute" if subject == MARKET else ctx.benchmark_symbol,
            regime=ctx.working.facts.get("regime"),
            made_at=ctx.as_of,
            due_date=_due(ctx.as_of, horizon),
            entry_price=entry,
            entry_benchmark=bench_price,
            status="open",
            context={
                "session": ctx.session.value,
                "data_state": ctx.state(subject).value if subject != MARKET else None,
                **extra,
            },
        )

    async def record(
        self,
        ctx: BrainContext,
        cycle_id: int,
        forecasts: Sequence[Opinion],
        consensus: dict[str, Consensus],
    ) -> int:
        rows: list[BrainPredictionRow] = []
        for o in forecasts:
            if not o.directional or o.horizon_days <= 0 or o.meta.get("gradeable") is False:
                continue
            row = self._row(
                ctx,
                cycle_id,
                source_type="agent",
                source_id=o.agent_id,
                version=o.agent_version,
                subject=o.subject,
                direction=1 if o.score > 0 else -1,
                score=o.score,
                confidence=o.confidence,
                horizon=o.horizon_days,
                extra={"thesis": o.thesis[:300], "invalidation": o.invalidation},
            )
            if row is not None:
                rows.append(row)
        for subject, c in consensus.items():
            if not c.actionable_view:
                continue
            horizon = max((ctx.working.facts.get("horizons") or {}).get(subject, 5), 1)
            row = self._row(
                ctx,
                cycle_id,
                source_type="consensus",
                source_id="consensus",
                version=CONSENSUS_VERSION,
                subject=subject,
                direction=1 if c.score > 0 else -1,
                score=c.score,
                confidence=c.confidence,
                horizon=horizon,
                extra={
                    "supporting": c.supporting,
                    "opposing": c.opposing,
                    "disagreement": round(c.disagreement, 3),
                    "independent_sources": c.independent,
                    "uncertainty": c.uncertainty[:6],
                },
            )
            if row is not None:
                rows.append(row)
        if rows:
            async with self._db.session() as s:
                s.add_all(rows)
        return len(rows)


class Learner:
    """One learning pass: evaluate due predictions, give decisions their outcomes, reflect, recompute
    track records, analyse failures and remember the lessons. Safe to run at any time: nothing is graded
    before its due date's close exists."""

    def __init__(self, db: Database, prices: PriceSource, clock: Clock, memory: MemoryStore, benchmark: str,
                 min_observations: int) -> None:  # fmt: skip
        self._db = db
        self._prices = prices
        self._clock = clock
        self._memory = memory
        self._benchmark = benchmark
        self._min = min_observations

    async def learn(self) -> dict[str, Any]:
        now = self._clock.now()
        evaluated = await evaluate_due(self._db, self._prices, self._clock, self._benchmark)
        reflections = await reflect_on_decisions(self._db, now)
        rows = await perf.recompute(self._db, now, self._min)
        findings = reflection.failure_analysis(await perf.graded(self._db), self._min)
        notable = await reflection.record_failure_analysis(self._db, findings, now)
        for r in reflections:
            if r["category"] in ("process_failure", "lucky", "block_saved_money", "block_cost_opportunity"):
                await self._memory.remember(
                    LONG_TERM,
                    "lesson",
                    r["subject"],
                    f"{r['category'].replace('_', ' ')}: {r['lessons'][0]}",
                    now,
                    data=r,
                    tags=["lesson", r["category"], r["decision_quality"], r["outcome_quality"]],
                    importance=0.8 if r["category"] == "process_failure" else 0.6,
                )
        for f in findings:
            await self._memory.remember(
                AGENT,
                "performance",
                f["agent_id"],
                f"{f['agent_id']} v{f['version']}: {f['hit_rate']:.0%} over {f['n']} graded calls"
                + (f"; {'; '.join(f['notes'])}" if f["notes"] else ""),
                now,
                key=f"performance:{f['agent_id']}:{f['version']}",
                data=f,
                tags=["performance", f["agent_id"]],
                importance=0.7 if f["notes"] else 0.4,
            )
        summary = {
            "at": now.isoformat(),
            "evaluated": evaluated.evaluated,
            "voided": evaluated.voided,
            "pending": evaluated.pending,
            "decision_outcomes": evaluated.decisions,
            "reflections": len(reflections),
            "by_category": _tally(r["category"] for r in reflections),
            "performance_rows": rows,
            "agents_with_findings": notable,
            "outcomes": [
                {"decision_id": r["decision_id"], "subject": r["subject"], "category": r["category"]}
                for r in reflections
            ],
        }
        return summary


def _tally(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
