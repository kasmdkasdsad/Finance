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
from datetime import datetime, time
from typing import Any

from sqlalchemy import select

from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, sessions_after
from quantpulse.db.models import BrainPredictionRow
from quantpulse.db.session import Database

from . import performance as perf
from . import reflection
from .consensus import CONSENSUS_VERSION, Consensus, ReliabilityBook
from .context import BrainContext
from .debate import Debate
from .evaluation import PriceSource, evaluate_due
from .memory import AGENT, LONG_TERM, MemoryStore
from .patterns import consolidate
from .reflection import reflect_on_decisions
from .types import MARKET, Opinion


def _due(made_at: datetime, horizon: int) -> object:
    return sessions_after(made_at.astimezone(NEW_YORK).date(), max(horizon, 1))


class PredictionRecorder:
    """Writes gradeable claims. A claim is recorded once a day: while an open prediction from the same
    source, on the same subject, horizon and direction was already made today, repeating the view in a
    later cycle adds nothing (a changed view is a new claim). Each record carries what is needed to judge it
    later — expected return (once the source is calibrated), thesis, evidence, invalidation, the agents and
    sources involved, the consensus and disagreement, the portfolio context, the regime and the data state —
    and nothing that is only known afterwards."""

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
        expected: float | None = None,
    ) -> BrainPredictionRow | None:
        bench_price = ctx.price(ctx.benchmark_symbol)
        entry = bench_price if subject == MARKET else ctx.price(subject)
        if entry is None or bench_price is None:
            return None  # cannot be graded without an entry price
        diag = ctx.data_health.get(subject)
        situation = ctx.working.facts.get("situation") or {}
        vol = ctx.market_stats.get("benchmark_rv21") if subject == MARKET else ctx.ind(subject, "risk_vol")
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
            expected_return=expected,
            context={
                "session": ctx.session.value,
                "data_state": ctx.state(subject).value if subject != MARKET else None,
                "data_status": diag.status.value if diag is not None else None,
                "quote_age_s": round(diag.trade_age_s, 1) if diag and diag.trade_age_s is not None else None,
                "vol": round(float(vol), 5) if vol else None,
                "market_vol": ctx.market_stats.get("benchmark_rv21"),  # the volatility environment
                "portfolio": {
                    "held": subject in ctx.held,
                    "weight": round(ctx.portfolio.weight(subject), 4) if subject != MARKET else None,
                    "posture": situation.get("posture"),
                    "positions": len(ctx.held),
                },
                **extra,
            },
        )

    @staticmethod
    def _key(row: BrainPredictionRow) -> tuple[str, str, str, str, int, int]:
        return (
            row.source_type,
            row.source_id,
            row.source_version,
            row.subject,
            row.horizon_days,
            row.direction,
        )

    async def record(
        self,
        ctx: BrainContext,
        cycle_id: int,
        forecasts: Sequence[Opinion],
        consensus: dict[str, Consensus],
        reliability: ReliabilityBook | None = None,
        debates: dict[str, Debate] | None = None,
    ) -> int:
        book = reliability or ReliabilityBook()
        rows: list[BrainPredictionRow] = []
        for o in forecasts:
            if not o.directional or o.horizon_days <= 0 or o.meta.get("gradeable") is False:
                continue
            direction = 1 if o.score > 0 else -1
            c = consensus.get(o.subject)
            row = self._row(
                ctx,
                cycle_id,
                source_type="agent",
                source_id=o.agent_id,
                version=o.agent_version,
                subject=o.subject,
                direction=direction,
                score=o.score,
                confidence=o.confidence,
                horizon=o.horizon_days,
                expected=book.expected_return(o.agent_id, o.agent_version, o.confidence, direction),
                extra={
                    "thesis": o.thesis[:300],
                    "invalidation": o.invalidation,
                    "evidence": [
                        {"name": e.name, "detail": e.detail[:160], "direction": e.direction}
                        for e in sorted(o.evidence, key=lambda e: -e.strength)[:3]
                    ],
                    "consensus": (
                        {
                            "stance": "unknown" if c.unknown else c.stance.value,
                            "score": round(c.score, 4),
                            "disagreement": round(c.disagreement, 3),
                            "independent_sources": c.independent,
                        }
                        if c is not None
                        else None
                    ),
                },
            )
            if row is not None:
                rows.append(row)
        for subject, c in consensus.items():
            if not c.actionable_view:
                continue
            horizon = max((ctx.working.facts.get("horizons") or {}).get(subject, 5), 1)
            direction = 1 if c.score > 0 else -1
            lead = [v for v in c.votes if v.score * direction >= 0.15]
            debate = (debates or {}).get(subject)
            row = self._row(
                ctx,
                cycle_id,
                source_type="consensus",
                source_id="consensus",
                version=CONSENSUS_VERSION,
                subject=subject,
                direction=direction,
                score=c.score,
                confidence=c.confidence,
                horizon=horizon,
                expected=book.expected_return("consensus", CONSENSUS_VERSION, c.confidence, direction),
                extra={
                    "thesis": "; ".join(v.thesis for v in sorted(lead, key=lambda v: -v.weight)[:2])[:300],
                    "invalidation": "; ".join(debate.change_our_mind)[:300] if debate else None,
                    "agents": {
                        "supporting": [v.agent_id for v in lead],
                        "opposing": [v.agent_id for v in c.votes if v.score * direction <= -0.15],
                    },
                    "sources": {k: v["score"] for k, v in c.sources.items()},
                    "supporting": c.supporting,
                    "opposing": c.opposing,
                    "disagreement": round(c.disagreement, 3),
                    "independent_sources": c.independent,
                    "uncertainty": c.uncertainty[:6],
                    "debate": debate.verdict if debate else None,
                },
            )
            if row is not None:
                rows.append(row)
        if not rows:
            return 0
        day_start = datetime.combine(ctx.as_of.astimezone(NEW_YORK).date(), time(0), NEW_YORK)
        async with self._db.session() as s:
            today = (
                await s.scalars(
                    select(BrainPredictionRow).where(
                        BrainPredictionRow.status == "open", BrainPredictionRow.made_at >= day_start
                    )
                )
            ).all()
            seen = {self._key(r) for r in today}
            fresh = []
            for r in rows:
                if self._key(r) not in seen:
                    seen.add(self._key(r))
                    fresh.append(r)
            s.add_all(fresh)
        return len(fresh)


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
        patterns = await consolidate(self._db, self._memory, now, self._min)
        purged = await self._memory.purge_expired(now)
        summary = {
            "at": now.isoformat(),
            "patterns": patterns,
            "memory_purged": purged,
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
