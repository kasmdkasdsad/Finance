"""The versioned model registry: every model (statistical, machine-learned, AI or rule) lives in a *slot* and
advances through stages one gate at a time —

    CANDIDATE → OOS_VALIDATED → WALK_FORWARD_VALIDATED → STRESS_VALIDATED → PAPER_SHADOW → AUTHORITATIVE

— and **no model becomes authoritative because it performed well in-sample**: in-sample results are recorded
(``in_sample``) and never consulted by a gate. Becoming authoritative needs out-of-sample, walk-forward and
stress evidence and a live shadow record that beats the current champion; an AI model also needs a person's
approval, which the system can never give itself. The champion it replaces keeps its whole history (role
``none``); nothing is overwritten.

Models that predate the registry are registered as *legacy champions* (their own validation lives where it
always did: the model lab, the Brain's scorecard) so that any new candidate has an incumbent to beat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from quantpulse.core.clock import Clock
from quantpulse.db.evolution_models import ModelRegistryRow
from quantpulse.db.session import Database
from quantpulse.evolution.registry import KINDS, ModelEvidence, ModelStage, advance, gate
from quantpulse.services.options_lab import jsonable

LEGACY: tuple[tuple[str, str, str, str], ...] = (
    ("stock_alpha", "ml", "Walk-forward stock model (gradient boosting / ensemble)",
     "The cross-sectional stock model; validated by its own walk-forward in the model lab."),
    ("brain_consensus", "rule", "Brain agent consensus (reliability-weighted)",
     "The agents' weighted consensus; its calibration is graded in the Brain's scorecard."),
    ("options_candidate_score", "rule", "Option candidate score (expected value per dollar at risk)",
     "Deterministic: payoff over the market-implied and empirical distributions, costs included."),
)  # fmt: skip


class ModelRegistryService:
    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    async def bootstrap(self) -> int:
        """Register the models that predate the registry as legacy champions (once)."""
        now = self._clock.now()
        added = 0
        async with self._db.session() as s:
            for slot, kind, name, desc in LEGACY:
                if await s.scalar(select(ModelRegistryRow).where(ModelRegistryRow.slot == slot)) is not None:
                    continue
                s.add(ModelRegistryRow(slot=slot, version=1, kind=kind, name=name, description=desc, params={},
                                       data={}, stage=ModelStage.AUTHORITATIVE.value,
                                       stage_history=[{"stage": ModelStage.AUTHORITATIVE.value, "at": now.isoformat(),
                                                       "reason": "legacy champion: predates the registry (validated "
                                                       "where it always was); new candidates must beat it"}],
                                       evidence={}, in_sample={}, role="champion", created_at=now, updated_at=now))  # fmt: skip
                added += 1
        return added

    async def register(self, slot: str, kind: str, name: str, *, description: str = "",
                       params: dict[str, Any] | None = None, data: dict[str, Any] | None = None,
                       in_sample: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        now = self._clock.now()
        async with self._db.session() as s:
            last = await s.scalar(
                select(func.max(ModelRegistryRow.version)).where(ModelRegistryRow.slot == slot)
            )
            row = ModelRegistryRow(slot=slot[:48], version=(last or 0) + 1, kind=kind, name=name[:120],
                                   description=description, params=jsonable(params or {}), data=jsonable(data or {}),
                                   stage=ModelStage.CANDIDATE.value,
                                   stage_history=[{"stage": ModelStage.CANDIDATE.value, "at": now.isoformat(),
                                                   "reason": "registered"}],
                                   evidence={}, in_sample=jsonable(in_sample or {}), role="challenger",
                                   created_at=now, updated_at=now)  # fmt: skip
            s.add(row)
            await s.flush()
            return _out(row)

    async def record(self, model_id: int, **evidence: Any) -> dict[str, Any]:
        """Add out-of-sample, walk-forward, stress or shadow evidence (``oos=``, ``walk_forward=``,
        ``stress=``, ``shadow=``); anything else (in-sample) is refused here — it goes to ``in_sample`` at
        registration and is never evidence."""
        allowed = {"oos", "walk_forward", "stress", "shadow"}
        bad = set(evidence) - allowed
        if bad:
            raise ValueError(
                f"not evidence a gate may use: {sorted(bad)} (in-sample results are recorded, never used)"
            )
        async with self._db.session() as s:
            row = await s.get(ModelRegistryRow, model_id)
            if row is None:
                raise KeyError(model_id)
            row.evidence = jsonable({**(row.evidence or {}), **evidence})
            row.updated_at = self._clock.now()
            return _out(row)

    async def approve(self, model_id: int, by: str) -> dict[str, Any]:
        """A person's approval (needed for an AI model to become authoritative); recorded with their name."""
        async with self._db.session() as s:
            row = await s.get(ModelRegistryRow, model_id)
            if row is None:
                raise KeyError(model_id)
            row.approved_by = by[:64]
            row.stage_history = [*row.stage_history, {"stage": row.stage, "at": self._clock.now().isoformat(),
                                                      "reason": f"approved by {by[:64]}"}]  # fmt: skip
            return _out(row)

    async def advance(self, model_id: int) -> dict[str, Any]:
        """One gate at a time, as far as the evidence allows; the champion is replaced only at AUTHORITATIVE."""
        now = self._clock.now()
        async with self._db.session() as s:
            row = await s.get(ModelRegistryRow, model_id)
            if row is None:
                raise KeyError(model_id)
            champion = await s.scalar(select(ModelRegistryRow).where(ModelRegistryRow.slot == row.slot,
                                                                     ModelRegistryRow.role == "champion",
                                                                     ModelRegistryRow.id != row.id))  # fmt: skip
            ev = _evidence(row, champion)
            stage = ModelStage(row.stage)
            moved = []
            while True:
                new, why = advance(stage, ev, kind=row.kind)
                if new == stage:
                    row.evidence = jsonable({**(row.evidence or {}), "next_gate": why})
                    break
                moved.append(new.value)
                row.stage_history = [*row.stage_history, {"stage": new.value, "at": now.isoformat(),
                                                          "reason": f"gate for {new.value} passed"}]  # fmt: skip
                stage = new
            row.stage, row.updated_at = stage.value, now
            if stage is ModelStage.AUTHORITATIVE and row.role != "champion":
                if champion is not None:
                    champion.role = "none"
                    champion.stage_history = [*champion.stage_history, {"stage": champion.stage, "at": now.isoformat(),
                                                                        "reason": f"replaced by v{row.version}"}]  # fmt: skip
                row.role = "champion"
            return {**_out(row), "moved": moved}

    async def models(self, slot: str | None = None) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            q = select(ModelRegistryRow).order_by(ModelRegistryRow.slot, ModelRegistryRow.version)
            if slot:
                q = q.where(ModelRegistryRow.slot == slot)
            return [_out(r) for r in (await s.scalars(q)).all()]


def _evidence(row: ModelRegistryRow, champion: ModelRegistryRow | None) -> ModelEvidence:
    ev = row.evidence or {}
    shadow = dict(ev.get("shadow") or {})
    if champion is not None and "champion_score" not in shadow:
        shadow["champion_score"] = ((champion.evidence or {}).get("shadow") or {}).get("score")
    return ModelEvidence(oos=dict(ev.get("oos") or {}), walk_forward=dict(ev.get("walk_forward") or {}),
                         stress=dict(ev.get("stress") or {}), shadow=shadow, human_approved=bool(row.approved_by),
                         in_sample=dict(row.in_sample or {}))  # fmt: skip


def _out(r: ModelRegistryRow) -> dict[str, Any]:
    next_gate = None
    if r.stage not in (ModelStage.AUTHORITATIVE.value, ModelStage.RETIRED.value):
        from quantpulse.evolution.registry import ORDER

        nxt = ORDER[ORDER.index(ModelStage(r.stage)) + 1]
        next_gate = gate(nxt, _evidence(r, None), kind=r.kind)
    return {"id": r.id, "slot": r.slot, "version": r.version, "kind": r.kind, "name": r.name,
            "description": r.description, "stage": r.stage, "role": r.role, "approved_by": r.approved_by,
            "evidence": {k: v for k, v in (r.evidence or {}).items() if k != "next_gate"},
            "in_sample": r.in_sample, "in_sample_note": "recorded for reference; never used by any gate",
            "stage_history": r.stage_history, "next_gate": next_gate,
            "created_at": r.created_at.isoformat(), "updated_at": r.updated_at.isoformat()}  # fmt: skip


def _now(clock: Clock) -> datetime:
    return clock.now()
