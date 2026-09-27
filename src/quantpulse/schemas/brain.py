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
    kind: Literal["full", "portfolio", "deep", "event"] = "full"


class AgentToggleIn(StrictModel):
    enabled: bool


class SupervisorIn(StrictModel):
    paused: bool


class StrategyIn(StrictModel):
    template: str = Field(max_length=48)
    top_n: int | None = Field(default=None, ge=1, le=100)
    rebalance_days: int | None = Field(default=None, ge=1, le=252)
    weighting: Literal["equal", "inverse_vol"] | None = None
    cost_bps: float | None = Field(default=None, ge=0, le=200)
    search_grid: bool = Field(default=True, description="Let walk-forward choose top_n and rebalance_days")

    def overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            k: v
            for k, v in {
                "top_n": self.top_n,
                "rebalance_days": self.rebalance_days,
                "weighting": self.weighting,
                "cost_bps": self.cost_bps,
            }.items()
            if v is not None
        }
        if not self.search_grid:
            out["grid"] = {}
        return out


class StrategyStatusIn(StrictModel):
    status: Literal["paper", "promoted", "retired"]


class ImprovementDecisionIn(StrictModel):
    status: Literal["testing", "validated", "rejected", "applied"]
    note: str | None = Field(default=None, max_length=1000)


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
    opportunities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ideas detected this cycle and how far each got through the pipeline",
    )
    debates: list[dict[str, Any]] = Field(
        default_factory=list, description="Bull case, bear case and devil's advocate per subject"
    )


class BrainStatusOut(StrictModel):
    paper_only: bool
    mode: str
    orders: str
    agents: dict[str, int]
    running: bool
    last_cycle: BrainCycleSummaryOut | None
    open_predictions: int
    learning: str
    language_models: str


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


class BrainOpportunityOut(StrictModel):
    id: int
    cycle_id: int
    kind: str
    subject: str
    symbols: list[str]
    direction: int
    strength: float
    headline: str
    evidence: dict[str, Any]
    status: str
    stages: list[dict[str, Any]]
    created_at: AwareDatetime
