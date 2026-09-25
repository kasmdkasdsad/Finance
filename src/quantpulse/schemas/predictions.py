"""The prediction ledger: every logged forecast, how it turned out, and the running scorecard."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import DataStatus, StrictModel
from quantpulse.schemas.jobs import JobOut

PredictionSource = Literal["forecast", "model"]
PredictionStatus = Literal["open", "resolved", "void"]
PredictionOrigin = Literal["live", "backfill"]


class PredictionOut(StrictModel):
    id: int
    created_at: AwareDatetime
    made_on: date
    target_date: date
    symbol: str
    source: PredictionSource
    horizon_days: int
    reference_price: float
    benchmark: str
    prob_up: float | None
    prob_outperform: float | None
    expected_return: float | None
    q05: float | None
    q25: float | None
    q50: float | None
    q75: float | None
    q95: float | None
    rank: int | None
    model_version: str
    data_status: DataStatus
    origin: PredictionOrigin = "live"
    status: PredictionStatus
    resolved_at: AwareDatetime | None
    realized_price: float | None
    realized_return: float | None
    benchmark_return: float | None
    outcome_up: bool | None
    outcome_outperform: bool | None
    in_50: bool | None
    in_90: bool | None


class CalibrationBucket(StrictModel):
    lower: float
    upper: float
    n: int
    mean_predicted: float | None
    observed: float | None


class SourceScore(StrictModel):
    source: PredictionSource
    horizon_days: int | None
    resolved: int
    open: int
    brier: float | None
    brier_base_rate: float | None = Field(
        description="Brier score of always predicting the observed base rate."
    )
    brier_skill: float | None
    hit_rate: float | None
    base_rate: float | None
    coverage_50: float | None = None
    coverage_90: float | None = None
    top_ranked_excess: float | None = Field(
        default=None, description="Mean excess return of top-5 ranked names."
    )
    others_excess: float | None = None
    calibration: list[CalibrationBucket]


class Scorecard(StrictModel):
    computed_at: AwareDatetime
    symbol: str | None
    origin: Literal["live", "backfill", "all"] = "all"
    sources: list[SourceScore]
    recent: list[PredictionOut]
    note: str = (
        "Predictions are logged after the close on each trading day from live data only, then graded "
        "automatically on their target date. Brier score: 0 is perfect, 0.25 is a coin flip; skill > 0 means "
        "better than always guessing the base rate. A few weeks of results are mostly noise."
    )


class LogResult(StrictModel):
    made_on: date
    logged: int
    skipped: dict[str, str] = Field(default_factory=dict)
    data_status: DataStatus


class ResolveResult(StrictModel):
    resolved: int
    voided: int
    pending: int


class BackfillOut(StrictModel):
    forecast_rows: int = Field(description="Backfilled forecast predictions inserted (already graded)")
    model_rows: int = Field(description="Backfilled model rankings inserted (already graded)")
    replaced: int = Field(description="Earlier backfilled rows removed first")
    first_date: date | None
    last_date: date | None
    skipped: dict[str, str] = Field(default_factory=dict)


class LedgerCounts(StrictModel):
    origin: PredictionOrigin
    source: PredictionSource
    open: int
    resolved: int
    void: int


class BackfillStatus(StrictModel):
    job: JobOut | None
    result: BackfillOut | None
    counts: list[LedgerCounts]
    note: str = (
        "Backfilled predictions replay history point-in-time (each uses only data available on its date) and "
        "are kept apart from the live record. They use today's risk-free rate and dividend yield and no options."
    )
