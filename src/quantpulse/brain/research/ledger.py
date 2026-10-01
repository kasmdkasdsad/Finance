"""The learning ledger: what the Brain concluded, and on what evidence.

No fake learning. A conclusion is recorded with its sample size, period, market regime, benchmark, method,
statistics, confidence and limitations, and its status is *computed* from that evidence — a caller cannot
assert it:

* ``UNPROVEN`` — fewer independent observations than the minimum, or no statistical test was possible;
* ``SUPPORTED`` — the test rejects "no effect" (p < 0.05) in the claim's direction;
* ``REFUTED`` — the test rejects "no effect" in the opposite direction;
* ``INCONCLUSIVE`` — enough data, no detectable effect.

Confidence is the strength of that evidence (``1 - p``, scaled down until the sample is twice the minimum) and is
0 for anything UNPROVEN or INCONCLUSIVE. A new conclusion on the same topic supersedes the older one; both are
kept (the history of what the Brain believed and when).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import select

from quantpulse.db.research_models import BrainLearningRow
from quantpulse.db.session import Database

STATUSES = ("UNPROVEN", "SUPPORTED", "REFUTED", "INCONCLUSIVE")
ALPHA = 0.05


@dataclass
class Finding:
    """A conclusion and its evidence, as a research job states it (before the ledger judges it)."""

    topic: str  # what it is about: "agent:momentum", "feature:rv21", "rejection:spread", ...
    claim: str  # the statement tested, in plain words ("momentum's calls beat a coin flip")
    sample_size: int  # independent observations
    regime: str  # "all", or the slice ("vol:high", "regime:bull")
    benchmark: str  # what it is compared against ("a coin flip", "SPY", "random portfolios", "the consensus")
    method: str  # the test ("block binomial test", "Welch t-test", "rank IC t-test", ...)
    limitations: list[str]
    min_sample: int
    p_value: float | None = None  # None: no test was possible
    effect: float | None = None  # signed, positive in the claim's direction
    period_start: date | None = None
    period_end: date | None = None
    statistics: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


def judge(f: Finding) -> tuple[str, float]:
    """The status and confidence the evidence supports — the only way a status is decided."""
    if f.sample_size < f.min_sample or f.p_value is None or f.effect is None or not math.isfinite(f.p_value):
        return "UNPROVEN", 0.0
    if f.p_value >= ALPHA or f.effect == 0:
        return "INCONCLUSIVE", 0.0
    adequacy = min(1.0, f.sample_size / (2 * f.min_sample))
    confidence = round(max(0.0, min(0.99, (1 - f.p_value) * adequacy)), 3)
    return ("SUPPORTED" if f.effect > 0 else "REFUTED"), confidence


def check(f: Finding) -> None:
    """Every conclusion carries its evidence; a finding that does not is refused, not recorded."""
    missing = [name for name in ("topic", "claim", "regime", "benchmark", "method") if not getattr(f, name)]
    if missing:
        raise ValueError(f"a finding needs {', '.join(missing)}")
    if not f.limitations:
        raise ValueError("a finding needs its limitations (every conclusion has some)")
    if f.sample_size < 0 or f.min_sample < 1:
        raise ValueError("sample sizes must be counts")
    if f.period_start and f.period_end and f.period_end < f.period_start:
        raise ValueError("the period ends before it starts")


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _out(r: BrainLearningRow) -> dict[str, Any]:
    return {
        "id": r.id,
        "job_id": r.job_id,
        "kind": r.kind,
        "topic": r.topic,
        "claim": r.claim,
        "status": r.status,
        "sample_size": r.sample_size,
        "min_sample": r.min_sample,
        "period": {
            "start": r.period_start.isoformat() if r.period_start else None,
            "end": r.period_end.isoformat() if r.period_end else None,
        },
        "regime": r.regime,
        "benchmark": r.benchmark,
        "method": r.method,
        "statistics": r.statistics,
        "confidence": r.confidence,
        "limitations": r.limitations,
        "evidence": r.evidence,
        "created_at": r.created_at.isoformat(),
        "supersedes_id": r.supersedes_id,
    }


class LearningLedger:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self, kind: str, finding: Finding, now: datetime, job_id: int | None = None
    ) -> dict[str, Any]:
        check(finding)
        status, confidence = judge(finding)
        stats = {**finding.statistics, "p_value": finding.p_value, "effect": finding.effect}
        async with self._db.session() as s:
            previous = await s.scalar(
                select(BrainLearningRow)
                .where(BrainLearningRow.topic == _clip(finding.topic, 160))
                .order_by(BrainLearningRow.id.desc())
                .limit(1)
            )
            row = BrainLearningRow(
                job_id=job_id,
                kind=_clip(kind, 40),
                topic=_clip(finding.topic, 160),
                claim=finding.claim,
                status=status,
                sample_size=int(finding.sample_size),
                min_sample=int(finding.min_sample),
                period_start=finding.period_start,
                period_end=finding.period_end,
                regime=_clip(finding.regime, 48),
                benchmark=_clip(finding.benchmark, 64),
                method=_clip(finding.method, 96),
                statistics=_jsonable(stats),
                confidence=confidence,
                limitations=list(finding.limitations),
                evidence=_jsonable(finding.evidence),
                created_at=now,
                supersedes_id=previous.id if previous is not None else None,
            )
            s.add(row)
            await s.flush()
            return _out(row)

    async def learnings(
        self,
        *,
        topic: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        limit: int = 100,
        current_only: bool = False,
    ) -> list[dict[str, Any]]:
        stmt = select(BrainLearningRow).order_by(BrainLearningRow.id.desc())
        if topic:
            stmt = stmt.where(BrainLearningRow.topic == topic)
        if status:
            stmt = stmt.where(BrainLearningRow.status == status)
        if kind:
            stmt = stmt.where(BrainLearningRow.kind == kind)
        async with self._db.session() as s:
            rows = list((await s.scalars(stmt.limit(limit if not current_only else 5000))).all())
        if current_only:  # the latest conclusion per topic
            latest: dict[str, BrainLearningRow] = {}
            for r in rows:
                latest.setdefault(r.topic, r)
            rows = list(latest.values())[:limit]
        return [_out(r) for r in rows]

    async def latest(self, topic: str) -> dict[str, Any] | None:
        found = await self.learnings(topic=topic, limit=1)
        return found[0] if found else None

    async def summary(self) -> dict[str, Any]:
        current = await self.learnings(current_only=True, limit=5000)
        return {
            "topics": len(current),
            "by_status": dict(Counter(r["status"] for r in current)),
            "rule": "a conclusion is UNPROVEN until its sample reaches the minimum and a test supports it",
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, date | datetime):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalars
        return _jsonable(value.item())
    return value


def p_two_sided(z: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    return math.erfc(abs(z) / math.sqrt(2))


def t_test_mean(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    """Mean, t statistic and two-sided p-value (normal approximation) of "the mean is 0"."""
    n = len(values)
    if n < 3:
        return (sum(values) / n if n else None), None, None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if var <= 0:
        return mean, None, None
    t = mean / math.sqrt(var / n)
    return mean, t, p_two_sided(t)


def binomial_vs_half(k: float, n: int) -> tuple[float | None, float | None]:
    """Hit rate and two-sided p-value against a coin flip (normal approximation)."""
    if n <= 0:
        return None, None
    z = (k - n / 2) / math.sqrt(n / 4)
    return k / n, p_two_sided(z)
