"""The improvement lifecycle: how a discovery could ever reach production — one tested stage at a time.

    DISCOVERED → HYPOTHESIS → BACKTEST → WALK_FORWARD → STRESS_TEST → PAPER_SHADOW → EVALUATION
               → (a person promotes) → PRODUCTION

* :meth:`Lifecycle.advance` moves a hypothesis exactly one stage forward, and only with evidence that the stage
  passed; a failed stage rejects it. There is no way to set a stage directly, and none to skip one.
* EVALUATION → PRODUCTION is :meth:`Lifecycle.promote`, which refuses the Brain's own actors: only a person,
  through the authenticated API, promotes. Even then nothing in trading changes by itself — a promoted strategy
  goes through the strategy lab's own promotion gates; any other kind is an approved change for a person to make.
* A hypothesis that would touch a protected control (risk limits, kill switches, paper-only, data requirements,
  execution safeguards — the improvement engine's list) is parked as PROTECTED_REVIEW and never advances.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from quantpulse.brain.improvement import withheld
from quantpulse.db.research_models import BrainHypothesisRow
from quantpulse.db.session import Database

STAGES = (
    "DISCOVERED",
    "HYPOTHESIS",
    "BACKTEST",
    "WALK_FORWARD",
    "STRESS_TEST",
    "PAPER_SHADOW",
    "EVALUATION",
    "PRODUCTION",
)
TERMINAL = ("REJECTED", "PROTECTED_REVIEW", "RETIRED")
# the Brain's own actors: none of them may promote anything
SYSTEM_ACTORS = frozenset({"brain", "research", "system", "supervisor", "lab", "automatic", ""})
KINDS = ("strategy", "feature", "agent_combination", "process")


class LifecycleError(ValueError):
    pass


def promotion_refusal(stage: str, history: list[dict[str, Any]], key: str, by: str, note: str) -> str | None:
    """Why a promotion is refused (None: it may go ahead) — checked in full before anything changes."""
    if by.strip().lower() in SYSTEM_ACTORS:
        return "only a person promotes: the Brain and its research never do"
    if not note.strip():
        return "a promotion needs a note (why the evidence is enough)"
    if stage != "EVALUATION":
        return f"{key} is at {stage}: only an evaluated hypothesis can be promoted"
    last = history[-1] if history else {}
    if last.get("to") != "EVALUATION" or last.get("passed") is not True:
        return f"{key}: its evaluation did not pass"
    return None


def _out(r: BrainHypothesisRow) -> dict[str, Any]:
    nxt = STAGES[STAGES.index(r.stage) + 1] if r.stage in STAGES[:-1] else None
    return {
        "id": r.id,
        "key": r.key,
        "kind": r.kind,
        "title": r.title,
        "source": r.source,
        "source_ref": r.source_ref,
        "stage": r.stage,
        "next_stage": nxt,
        "awaiting_person": r.stage == "EVALUATION",
        "protected_control": r.protected_control,
        "detail": r.detail,
        "history": r.history,
        "next_step_at": r.next_step_at.isoformat() if r.next_step_at else None,
        "decided_by": r.decided_by,
        "promoted_at": r.promoted_at.isoformat() if r.promoted_at else None,
        "created_at": r.created_at.isoformat(),
        "updated_at": r.updated_at.isoformat(),
    }


class Lifecycle:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def discover(
        self,
        *,
        kind: str,
        key: str,
        title: str,
        source: str,
        detail: dict[str, Any],
        now: datetime,
        source_ref: str | None = None,
    ) -> dict[str, Any]:
        """Record a discovery (once per key). One that names a protected control is parked for a person."""
        if kind not in KINDS:
            raise LifecycleError(f"unknown kind {kind!r}")
        control = withheld({"title": title, "detail": detail})
        async with self._db.session() as s:
            row = await s.scalar(select(BrainHypothesisRow).where(BrainHypothesisRow.key == key[:160]))
            if row is not None:
                return _out(row)
            stage = "PROTECTED_REVIEW" if control else "DISCOVERED"
            row = BrainHypothesisRow(
                key=key[:160],
                kind=kind,
                title=title,
                source=source[:24],
                source_ref=(source_ref or None),
                stage=stage,
                protected_control=control,
                detail=detail,
                history=[
                    {
                        "to": stage,
                        "at": now.isoformat(),
                        "by": source,
                        "passed": None,
                        "evidence": {"protected_control": control} if control else {"discovered": True},
                    }
                ],
                created_at=now,
                updated_at=now,
            )
            s.add(row)
            await s.flush()
            return _out(row)

    async def advance(
        self,
        hypothesis_id: int,
        *,
        passed: bool,
        evidence: dict[str, Any],
        by: str,
        now: datetime,
        wait_until: datetime | None = None,
    ) -> dict[str, Any]:
        """One stage forward when ``passed`` (with its evidence); REJECTED when not. Never to PRODUCTION."""
        if not evidence:
            raise LifecycleError("a stage is decided on evidence: none given")
        async with self._db.session() as s:
            row = await s.get(BrainHypothesisRow, hypothesis_id)
            if row is None:
                raise LifecycleError(f"no hypothesis {hypothesis_id}")
            if row.stage in TERMINAL:
                raise LifecycleError(f"{row.key} is {row.stage}: it does not advance")
            if row.stage in ("EVALUATION", "PRODUCTION"):
                raise LifecycleError(f"{row.key} is at {row.stage}: only a person promotes it")
            target = STAGES[STAGES.index(row.stage) + 1] if passed else "REJECTED"
            row.history = [
                *row.history,
                {
                    "from": row.stage,
                    "to": target,
                    "at": now.isoformat(),
                    "by": by,
                    "passed": passed,
                    "evidence": evidence,
                },
            ]
            row.stage, row.updated_at, row.next_step_at = target, now, wait_until
            await s.flush()
            return _out(row)

    async def promote(self, hypothesis_id: int, *, by: str, note: str, now: datetime) -> dict[str, Any]:
        """EVALUATION → PRODUCTION: a person's explicit decision, recorded with their note."""
        async with self._db.session() as s:
            row = await s.get(BrainHypothesisRow, hypothesis_id)
            if row is None:
                raise LifecycleError(f"no hypothesis {hypothesis_id}")
            refused = promotion_refusal(row.stage, row.history, row.key, by, note)
            if refused:
                raise LifecycleError(refused)
            row.history = [
                *row.history,
                {
                    "from": "EVALUATION",
                    "to": "PRODUCTION",
                    "at": now.isoformat(),
                    "by": by,
                    "passed": True,
                    "evidence": {"note": note},
                },
            ]
            row.stage, row.decided_by, row.promoted_at, row.updated_at = "PRODUCTION", by[:48], now, now
            await s.flush()
            return _out(row)

    async def reject(self, hypothesis_id: int, *, by: str, note: str, now: datetime) -> dict[str, Any]:
        async with self._db.session() as s:
            row = await s.get(BrainHypothesisRow, hypothesis_id)
            if row is None:
                raise LifecycleError(f"no hypothesis {hypothesis_id}")
            if row.stage in ("PRODUCTION", "REJECTED"):
                raise LifecycleError(f"{row.key} is {row.stage}")
            row.history = [
                *row.history,
                {
                    "from": row.stage,
                    "to": "REJECTED",
                    "at": now.isoformat(),
                    "by": by,
                    "passed": False,
                    "evidence": {"note": note},
                },
            ]
            row.stage, row.decided_by, row.updated_at = "REJECTED", by[:48], now
            await s.flush()
            return _out(row)

    async def get(self, hypothesis_id: int) -> dict[str, Any] | None:
        async with self._db.session() as s:
            row = await s.get(BrainHypothesisRow, hypothesis_id)
            return _out(row) if row is not None else None

    async def by_key(self, key: str) -> dict[str, Any] | None:
        async with self._db.session() as s:
            row = await s.scalar(select(BrainHypothesisRow).where(BrainHypothesisRow.key == key[:160]))
            return _out(row) if row is not None else None

    async def hypotheses(
        self, *, stage: str | None = None, kind: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        stmt = select(BrainHypothesisRow).order_by(BrainHypothesisRow.updated_at.desc())
        if stage:
            stmt = stmt.where(BrainHypothesisRow.stage == stage)
        if kind:
            stmt = stmt.where(BrainHypothesisRow.kind == kind)
        async with self._db.session() as s:
            return [_out(r) for r in (await s.scalars(stmt.limit(limit))).all()]

    async def counts(self) -> dict[str, int]:
        rows = await self.hypotheses(limit=10_000)
        out: dict[str, int] = {}
        for r in rows:
            out[r["stage"]] = out.get(r["stage"], 0) + 1
        return out

    async def unapproved_in_production(self) -> list[str]:
        """An invariant: everything in PRODUCTION was promoted by a person (checked before every session)."""
        rows = await self.hypotheses(stage="PRODUCTION", limit=10_000)
        return [
            r["key"]
            for r in rows
            if (r["decided_by"] or "").strip().lower() in SYSTEM_ACTORS or not r["promoted_at"]
        ]
