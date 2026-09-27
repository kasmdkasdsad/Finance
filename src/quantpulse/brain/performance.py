"""Measured track records — computed only from evaluated predictions, never assigned.

For every source (each agent version, the consensus) and slice (all regimes and each regime; all time and
the last 90 days) the metrics are:

* **hit rate** with its binomial z-score against a coin flip;
* **Brier score** of the implied probability ``0.5 + 0.5 × score × confidence`` that the call is right
  (0.25 is a coin flip; lower is better);
* **rank IC** — Spearman correlation between the score and the realised relative return (≥ 10 calls);
* **calibration** — hit rate by confidence bucket (does higher confidence mean more hits?), plus a summary
  entry with the z-score and the share of bullish calls;
* **reliability** — the vote weight used by the consensus. It stays ``None`` (*unproven*, weight 1.0) until
  the slice has ``min_observations`` graded calls; then it is ``1 + 4 × (shrunk hit rate − 0.5)``, bounded
  to 0.25…1.75, where the hit rate is shrunk toward 50% by 20 pseudo-observations so a lucky streak on few
  calls cannot dominate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import delete, select

from quantpulse.db.models import BrainAgentPerformanceRow, BrainPredictionRow
from quantpulse.db.session import Database

BUCKETS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))
PRIOR_N = 20
RECENT = timedelta(days=90)


@dataclass(frozen=True)
class Graded:
    source: str
    version: str
    regime: str
    horizon: int
    score: float
    confidence: float
    relative: float
    hit: bool
    made_at: datetime
    direction: int


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 10:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def metrics(rows: Sequence[Graded], min_observations: int) -> dict[str, Any]:
    n = len(rows)
    hits = sum(1 for r in rows if r.hit)
    p_right = [min(max(0.5 + 0.5 * abs(r.score) * r.confidence, 0.0), 1.0) for r in rows]
    brier = (
        sum((p - (1.0 if r.hit else 0.0)) ** 2 for p, r in zip(p_right, rows, strict=True)) / n if n else None
    )
    calibration = []
    for lo, hi in BUCKETS:
        inside = [r for r in rows if lo <= r.confidence < hi]
        if inside:
            calibration.append(
                {
                    "kind": "bucket",
                    "confidence": f"{lo:.1f}–{min(hi, 1.0):.1f}",
                    "n": len(inside),
                    "hit_rate": round(sum(1 for r in inside if r.hit) / len(inside), 4),
                    "mean_relative": round(sum(r.relative for r in inside) / len(inside), 5),
                }
            )
    hit_rate = hits / n if n else None
    z = (hits - n / 2) / math.sqrt(n / 4) if n else None
    shrunk = (hits + PRIOR_N / 2) / (n + PRIOR_N) if n else None
    reliability = (
        round(max(0.25, min(1.75, 1 + 4 * (shrunk - 0.5))), 4)
        if shrunk is not None and n >= min_observations
        else None
    )
    bullish = sum(1 for r in rows if r.direction > 0)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "brier": round(brier, 5) if brier is not None else None,
        "ic": _spearman([r.score for r in rows], [r.relative for r in rows]),
        "calibration": [
            *calibration,
            {
                "kind": "summary",
                "z": round(z, 2) if z is not None else None,
                "bullish_share": round(bullish / n, 3) if n else None,
            },
        ],
        "reliability": reliability,
    }


async def graded(db: Database) -> list[Graded]:
    async with db.session() as s:
        rows = (
            await s.scalars(select(BrainPredictionRow).where(BrainPredictionRow.status == "evaluated"))
        ).all()
    return [
        Graded(
            source=r.source_id,
            version=r.source_version,
            regime=r.regime or "unknown",
            horizon=r.horizon_days,
            score=r.score,
            confidence=r.confidence,
            relative=float(r.realized_relative or 0.0),
            hit=bool(r.hit),
            made_at=r.made_at,
            direction=r.direction,
        )
        for r in rows
    ]


async def recompute(db: Database, now: datetime, min_observations: int) -> int:
    """Rebuild ``brain_agent_performance`` from every evaluated prediction; returns the rows written."""
    rows = await graded(db)
    groups: dict[tuple[str, str, str, str], list[Graded]] = defaultdict(list)
    for r in rows:
        for regime in ("all", r.regime):
            groups[(r.source, r.version, regime, "all")].append(r)
            if now - r.made_at <= RECENT:
                groups[(r.source, r.version, regime, "90d")].append(r)
    async with db.session() as s:
        await s.execute(delete(BrainAgentPerformanceRow))
        for (source, version, regime, window), items in groups.items():
            m = metrics(items, min_observations)
            horizon = round(sum(i.horizon for i in items) / len(items))
            s.add(
                BrainAgentPerformanceRow(
                    agent_id=source,
                    agent_version=version,
                    regime=regime,
                    horizon_days=horizon,
                    window=window,
                    n=m["n"],
                    hits=m["hits"],
                    hit_rate=m["hit_rate"],
                    brier=m["brier"],
                    ic=m["ic"],
                    calibration=m["calibration"],
                    reliability=m["reliability"],
                    computed_at=now,
                )
            )
    return len(groups)
