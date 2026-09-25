"""System / health schemas."""

from __future__ import annotations

from typing import Any

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import StrictModel


class Health(StrictModel):
    status: str
    version: str


class SystemStatus(StrictModel):
    version: str
    environment: str
    started_at: str
    now: str
    market_session: str
    next_market_open: str
    live_data_enabled: bool
    market_provider_order: list[str]
    credentials: dict[str, bool]
    database: dict[str, Any]
    cache: dict[str, Any]
    gateway: dict[str, Any]
    rate_limiters: dict[str, Any]
    odds_api_quota: dict[str, Any]
    poller: dict[str, Any]
    trading: dict[str, Any] = Field(
        default_factory=dict, description="Alpaca paper trading: paper-only endpoint, enabled, dry run"
    )


class IngestionEvent(StrictModel):
    dataset: str
    key: str
    provider: str
    rows: int
    created_at: AwareDatetime


class MarketSessionOut(StrictModel):
    now: AwareDatetime
    session: str
    is_trading_day: bool
    next_open: AwareDatetime
