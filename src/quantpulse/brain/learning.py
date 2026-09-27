"""The learning ledger: predictions written when they are made, so they can be graded against reality later.

Lifecycle (the arrows after ``open`` are the future outcome engine — not built yet)::

    opinion / consensus / decision ──► prediction (status "open", outcome columns empty)
        ──► horizon passes ──► evaluation (realised return vs the benchmark, hit or miss)
        ──► reflection (what was wrong: thesis, timing, regime, data, confidence)
        ──► agent / strategy performance (only from evaluated predictions)
        ──► improvement proposals

What is recorded now, and why it can be graded honestly later:

* only **gradeable forecasts** — directional opinions of *forecast* agents (not data-quality or portfolio
  constraints), and directional consensus that was not "unknown";
* each with its horizon (in sessions), the benchmark it is measured against, the regime it was made in,
  the entry price and benchmark level, and the due date;
* symbols are graded on their return **relative to the benchmark**; ``@market`` views on the benchmark's
  own return.

Nothing is scored here and no agent gets a reliability number from this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from quantpulse.core.market_calendar import NEW_YORK, sessions_after
from quantpulse.db.models import BrainPredictionRow
from quantpulse.db.session import Database

from .consensus import Consensus
from .context import BrainContext
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
                version="1",
                subject=subject,
                direction=1 if c.score > 0 else -1,
                score=c.score,
                confidence=c.confidence,
                horizon=horizon,
                extra={
                    "supporting": c.supporting,
                    "opposing": c.opposing,
                    "disagreement": round(c.disagreement, 3),
                },
            )
            if row is not None:
                rows.append(row)
        if rows:
            async with self._db.session() as s:
                s.add_all(rows)
        return len(rows)
