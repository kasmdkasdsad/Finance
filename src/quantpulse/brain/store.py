"""Persistence of brain cycles: the cycle, every agent run (including failures), every opinion, the
consensus per subject and the proposed actions — enough to replay what the brain saw and why it concluded
what it did, and to grade it later."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import case, func, select

from quantpulse.db.models import (
    BrainAgentPerformanceRow,
    BrainAgentRow,
    BrainAgentRunRow,
    BrainConsensusRow,
    BrainCycleRow,
    BrainDecisionRow,
    BrainOpinionRow,
    BrainPredictionRow,
)
from quantpulse.db.session import Database

from .agents.base import Agent
from .consensus import Consensus
from .decisions import Proposal
from .registry import AgentRun, Skip


def _cols(row: Any, names: Sequence[str]) -> dict[str, Any]:
    return {n: getattr(row, n) for n in names}


CYCLE_COLS = (
    "id", "kind", "trigger", "session", "mode", "status", "started_at", "finished_at", "duration_ms",
    "regime", "market", "portfolio", "data_quality", "focus", "agents", "summary", "notes", "error",
)  # fmt: skip
RUN_COLS = (
    "id",
    "agent_id",
    "agent_version",
    "status",
    "reason",
    "started_at",
    "duration_ms",
    "subjects",
    "opinions",
    "model_tier",
    "cost",
)
OPINION_COLS = (
    "id", "run_id", "agent_id", "agent_version", "subject", "stance", "score", "confidence", "horizon_days",
    "thesis", "evidence", "data_missing", "data_quality", "invalidation", "veto", "meta", "created_at",
)  # fmt: skip
CONSENSUS_COLS = (
    "id", "subject", "stance", "score", "confidence", "unknown", "supporting", "neutral", "opposing",
    "abstaining", "disagreement", "detail", "vetoes", "data_quality", "reasons", "created_at",
)  # fmt: skip
DECISION_COLS = (
    "id", "consensus_id", "subject", "action", "mode", "status", "confidence", "quantity", "est_price",
    "notional", "current_weight", "target_weight", "rationale", "risk_approved", "risk", "execution",
    "outcome", "created_at", "evaluated_at",
)  # fmt: skip


class BrainStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------ agents
    async def sync_agents(self, agents: Sequence[Agent], now: datetime) -> dict[str, bool]:
        """Mirror the code-registered agents into ``brain_agents``; returns the stored enabled flags."""
        enabled: dict[str, bool] = {}
        async with self._db.session() as s:
            for a in agents:
                spec = a.spec
                row = await s.get(BrainAgentRow, spec.id)
                if row is None:
                    row = BrainAgentRow(id=spec.id, registered_at=now, enabled=True)
                    s.add(row)
                row.name, row.family, row.version = spec.name, spec.family.value, spec.version
                row.spec = {**spec.to_dict(), "role": a.role}
                row.updated_at = now
                enabled[spec.id] = row.enabled
        return enabled

    async def set_agent_enabled(self, agent_id: str, enabled: bool, now: datetime) -> None:
        async with self._db.session() as s:
            row = await s.get(BrainAgentRow, agent_id)
            if row is None:
                raise KeyError(agent_id)
            row.enabled, row.updated_at = enabled, now

    async def agents(self) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = list((await s.scalars(select(BrainAgentRow).order_by(BrainAgentRow.id))).all())
            stats = {
                r[0]: {"runs": r[1], "failures": r[2], "avg_ms": r[3], "last_run": r[4]}
                for r in (
                    await s.execute(
                        select(
                            BrainAgentRunRow.agent_id,
                            func.count(),
                            func.sum(case((BrainAgentRunRow.status.in_(("failed", "timeout")), 1), else_=0)),
                            func.avg(BrainAgentRunRow.duration_ms),
                            func.max(BrainAgentRunRow.started_at),
                        ).group_by(BrainAgentRunRow.agent_id)
                    )
                ).all()
            }
            perf = list((await s.scalars(select(BrainAgentPerformanceRow))).all())
        measured: dict[str, list[dict[str, Any]]] = {}
        for p in perf:
            measured.setdefault(p.agent_id, []).append(
                {
                    "version": p.agent_version,
                    "regime": p.regime,
                    "horizon_days": p.horizon_days,
                    "n": p.n,
                    "hit_rate": p.hit_rate,
                    "reliability": p.reliability,
                }
            )
        return [
            {
                "id": r.id,
                "name": r.name,
                "family": r.family,
                "version": r.version,
                "enabled": r.enabled,
                "spec": r.spec,
                "runs": stats.get(r.id, {}),
                "performance": measured.get(r.id, []),
            }
            for r in rows
        ]

    async def reliability_rows(self) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = list(
                (
                    await s.scalars(
                        select(BrainAgentPerformanceRow).where(BrainAgentPerformanceRow.window == "all")
                    )
                ).all()
            )
        return [
            {
                "agent_id": r.agent_id,
                "agent_version": r.agent_version,
                "regime": r.regime,
                "n": r.n,
                "reliability": r.reliability,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ cycles
    async def start_cycle(self, *, kind: str, trigger: str, session: str, mode: str, now: datetime) -> int:
        async with self._db.session() as s:
            row = BrainCycleRow(
                kind=kind, trigger=trigger, session=session, mode=mode, status="running", started_at=now
            )
            s.add(row)
            await s.flush()
            return row.id

    async def finish_cycle(self, cycle_id: int, now: datetime, **fields: Any) -> None:
        async with self._db.session() as s:
            row = await s.get(BrainCycleRow, cycle_id)
            assert row is not None
            for k, v in fields.items():
                setattr(row, k, v)
            row.finished_at = now
            row.duration_ms = (now - row.started_at).total_seconds() * 1000

    async def save_runs(
        self, cycle_id: int, runs: Sequence[AgentRun], skips: Sequence[Skip], now: datetime
    ) -> dict[str, int]:
        """Agent runs (and skips) and every opinion; returns run ids by agent."""
        ids: dict[str, int] = {}
        async with self._db.session() as s:
            for run in runs:
                row = BrainAgentRunRow(
                    cycle_id=cycle_id,
                    agent_id=run.agent_id,
                    agent_version=run.version,
                    status=run.status,
                    reason=run.error,
                    started_at=datetime.fromtimestamp(run.started, tz=now.tzinfo) if run.started else now,
                    duration_ms=round(run.duration_ms, 2),
                    subjects=run.subjects,
                    opinions=len(run.opinions),
                    model_tier=run.model_tier,
                    cost=run.cost,
                )
                s.add(row)
                await s.flush()
                ids[run.agent_id] = row.id
                for o in run.opinions:
                    d = o.to_dict()
                    s.add(
                        BrainOpinionRow(
                            cycle_id=cycle_id,
                            run_id=row.id,
                            agent_id=o.agent_id,
                            agent_version=o.agent_version,
                            subject=o.subject,
                            stance=o.stance.value,
                            score=d["score"],
                            confidence=d["confidence"],
                            horizon_days=o.horizon_days,
                            thesis=o.thesis,
                            evidence=d["evidence"],
                            data_missing=o.data_missing,
                            data_quality=o.data_quality.value,
                            invalidation=o.invalidation,
                            veto=o.veto,
                            meta=d["meta"],
                            created_at=now,
                        )
                    )
            for skip in skips:
                s.add(
                    BrainAgentRunRow(
                        cycle_id=cycle_id,
                        agent_id=skip.agent_id,
                        agent_version=skip.version,
                        status="skipped",
                        reason=skip.reason,
                        started_at=now,
                        duration_ms=0.0,
                        subjects=0,
                        opinions=0,
                        model_tier="deterministic",
                        cost=0.0,
                    )
                )
        return ids

    async def save_consensus(
        self, cycle_id: int, items: dict[str, Consensus], now: datetime
    ) -> dict[str, int]:
        ids: dict[str, int] = {}
        async with self._db.session() as s:
            for subject, c in items.items():
                d = c.to_dict()
                row = BrainConsensusRow(
                    cycle_id=cycle_id,
                    subject=subject,
                    stance=d["stance"],
                    score=d["score"],
                    confidence=d["confidence"],
                    unknown=c.unknown,
                    supporting=c.supporting,
                    neutral=c.neutral,
                    opposing=c.opposing,
                    abstaining=c.abstaining,
                    disagreement=d["disagreement"],
                    detail={"votes": d["votes"], "primary_disagreement": c.primary_disagreement},
                    vetoes=c.vetoes,
                    data_quality=c.data_quality.value,
                    reasons=c.reasons,
                    created_at=now,
                )
                s.add(row)
                await s.flush()
                ids[subject] = row.id
        return ids

    async def save_decisions(
        self,
        cycle_id: int,
        proposals: Sequence[Proposal],
        consensus_ids: dict[str, int],
        mode: str,
        now: datetime,
    ) -> None:
        async with self._db.session() as s:
            for p in proposals:
                d = p.to_dict()
                s.add(
                    BrainDecisionRow(
                        cycle_id=cycle_id,
                        consensus_id=consensus_ids.get(p.subject),
                        subject=p.subject,
                        action=p.action.value,
                        mode=mode,
                        status=p.status,
                        confidence=d["confidence"],
                        quantity=p.quantity,
                        est_price=p.est_price,
                        notional=d["notional"],
                        current_weight=p.current_weight,
                        target_weight=p.target_weight,
                        rationale={"reasons": p.reasons, "blocked_by": p.blocked_by},
                        risk_approved=p.risk_approved,
                        risk=p.risk,
                        execution={"sent": False, "reason": "the brain never sends orders itself"},
                        outcome={},
                        created_at=now,
                    )
                )

    # ------------------------------------------------------------------ reads
    async def cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = list(
                (await s.scalars(select(BrainCycleRow).order_by(BrainCycleRow.id.desc()).limit(limit))).all()
            )
        return [_cols(r, CYCLE_COLS) for r in rows]

    async def cycle(self, cycle_id: int) -> dict[str, Any] | None:
        async with self._db.session() as s:
            row = await s.get(BrainCycleRow, cycle_id)
            if row is None:
                return None
            runs = (
                await s.scalars(
                    select(BrainAgentRunRow)
                    .where(BrainAgentRunRow.cycle_id == cycle_id)
                    .order_by(BrainAgentRunRow.id)
                )
            ).all()
            opinions = (
                await s.scalars(
                    select(BrainOpinionRow)
                    .where(BrainOpinionRow.cycle_id == cycle_id)
                    .order_by(BrainOpinionRow.id)
                )
            ).all()
            consensus = (
                await s.scalars(
                    select(BrainConsensusRow)
                    .where(BrainConsensusRow.cycle_id == cycle_id)
                    .order_by(BrainConsensusRow.id)
                )
            ).all()
            decisions = (
                await s.scalars(
                    select(BrainDecisionRow)
                    .where(BrainDecisionRow.cycle_id == cycle_id)
                    .order_by(BrainDecisionRow.id)
                )
            ).all()
            predictions = await s.scalar(
                select(func.count())
                .select_from(BrainPredictionRow)
                .where(BrainPredictionRow.cycle_id == cycle_id)
            )
        return {
            **_cols(row, CYCLE_COLS),
            "runs": [_cols(r, RUN_COLS) for r in runs],
            "opinions": [_cols(o, OPINION_COLS) for o in opinions],
            "consensus": [_cols(c, CONSENSUS_COLS) for c in consensus],
            "decisions": [_cols(d, DECISION_COLS) for d in decisions],
            "predictions_recorded": int(predictions or 0),
        }

    async def predictions(self, status: str | None = None, limit: int = 200) -> list[BrainPredictionRow]:
        stmt = select(BrainPredictionRow).order_by(BrainPredictionRow.id.desc()).limit(limit)
        if status:
            stmt = stmt.where(BrainPredictionRow.status == status)
        async with self._db.session() as s:
            return list((await s.scalars(stmt)).all())
