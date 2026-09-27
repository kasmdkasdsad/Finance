"""The brain's API shapes. Nested findings (evidence, votes, risk checks) are kept as structured JSON exactly
as stored, so what the API shows is what the brain recorded."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import StrictModel


class BrainRunIn(StrictModel):
    symbols: list[str] = Field(
        default_factory=list,
        max_length=25,
        description="Symbols to study in addition to holdings and the pre-screen",
    )
    kind: Literal["full", "portfolio", "deep"] = "full"


class AgentToggleIn(StrictModel):
    enabled: bool


class BrainAgentOut(StrictModel):
    id: str
    name: str
    family: str
    version: str
    enabled: bool
    spec: dict[str, Any]
    runs: dict[str, Any] = Field(description="Run count, failures, average duration, last run")
    performance: list[dict[str, Any]] = Field(
        description="Measured from evaluated predictions only (empty until there are observations)"
    )


class BrainCycleSummaryOut(StrictModel):
    id: int
    kind: str
    trigger: str
    session: str
    mode: str
    status: str
    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    duration_ms: float | None
    regime: dict[str, Any]
    market: dict[str, Any]
    portfolio: dict[str, Any]
    data_quality: dict[str, Any]
    focus: list[Any]
    agents: list[Any]
    summary: dict[str, Any]
    notes: list[Any]
    error: str | None


class BrainCycleOut(BrainCycleSummaryOut):
    runs: list[dict[str, Any]]
    opinions: list[dict[str, Any]]
    consensus: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    predictions_recorded: int


class BrainStatusOut(StrictModel):
    paper_only: bool
    mode: str
    orders: str
    agents: dict[str, int]
    running: bool
    last_cycle: BrainCycleSummaryOut | None
    open_predictions: int
    learning: str


class BrainMemoryOut(StrictModel):
    id: int
    tier: str
    kind: str
    subject: str
    key: str | None
    summary: str
    data: dict[str, Any]
    tags: list[Any]
    importance: float
    cycle_id: int | None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    expires_at: AwareDatetime | None
