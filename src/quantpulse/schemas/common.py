"""Shared Pydantic v2 schemas: data provenance, envelopes and common types."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Generic, TypeVar

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field

T = TypeVar("T")


def _normalise_symbol(value: str) -> str:
    value = value.strip().upper()
    if not value or len(value) > 15:
        raise ValueError("symbol must be 1-15 characters")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=")
    if any(ch not in allowed for ch in value):
        raise ValueError("symbol contains invalid characters")
    return value


Symbol = Annotated[str, AfterValidator(_normalise_symbol)]


class StrictModel(BaseModel):
    """Base for request/response schemas: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DataStatus(StrEnum):
    """How fresh/authentic a piece of data is. Ordered from best to worst."""

    LIVE = "live"
    CACHED = "cached"
    STALE = "stale"
    SYNTHETIC = "synthetic"

    @property
    def rank(self) -> int:
        return _STATUS_RANK[self]

    @classmethod
    def worst(cls, statuses: list[DataStatus]) -> DataStatus:
        if not statuses:
            return cls.SYNTHETIC
        return max(statuses, key=lambda s: s.rank)


_STATUS_RANK = {DataStatus.LIVE: 0, DataStatus.CACHED: 1, DataStatus.STALE: 2, DataStatus.SYNTHETIC: 3}


class ProviderAttempt(StrictModel):
    provider: str
    ok: bool
    error: str | None = None
    latency_ms: float | None = None


class Provenance(StrictModel):
    """Where a payload came from; drives the status badge in the UI."""

    status: DataStatus
    provider: str = Field(description="Provider that produced the data (e.g. 'yahoo', 'synthetic').")
    as_of: AwareDatetime = Field(description="Timestamp of the underlying observation.")
    fetched_at: AwareDatetime = Field(description="When QuantPulse obtained the data from the provider.")
    latency_ms: float | None = None
    message: str | None = None
    attempts: list[ProviderAttempt] = Field(default_factory=list)


class Envelope(StrictModel, Generic[T]):
    """Standard response wrapper for single-source data."""

    data: T
    meta: Provenance


class CompositeMeta(StrictModel):
    """Provenance for analytics that combine several sources (e.g. DCF = fundamentals + price + rates)."""

    status: DataStatus
    computed_at: AwareDatetime
    sources: dict[str, Provenance]

    @classmethod
    def from_sources(cls, sources: dict[str, Provenance], computed_at: datetime) -> CompositeMeta:
        return cls(
            status=DataStatus.worst([p.status for p in sources.values()]),
            computed_at=computed_at,
            sources=sources,
        )


class CompositeEnvelope(StrictModel, Generic[T]):
    data: T
    meta: CompositeMeta


class ErrorResponse(StrictModel):
    error: str
    detail: str | list[dict[str, object]] | None = None
    request_id: str | None = None
