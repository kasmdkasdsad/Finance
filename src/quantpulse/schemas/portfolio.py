"""Portfolio management and risk-report schemas."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from quantpulse.schemas.common import StrictModel, Symbol


class HoldingIn(StrictModel):
    symbol: Symbol
    quantity: float = Field(gt=0, le=1e9, description="Shares held (long-only).")
    cost_basis: float | None = Field(default=None, ge=0, description="Average cost per share.")


class PortfolioIn(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    holdings: list[HoldingIn] = Field(min_length=1, max_length=50)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @model_validator(mode="after")
    def _unique_symbols(self) -> PortfolioIn:
        symbols = [h.symbol for h in self.holdings]
        dupes = sorted({s for s in symbols if symbols.count(s) > 1})
        if dupes:
            raise ValueError(f"duplicate symbols: {', '.join(dupes)}")
        return self


class HoldingOut(StrictModel):
    symbol: str
    quantity: float
    cost_basis: float | None


class PortfolioOut(StrictModel):
    id: int
    name: str
    created_at: AwareDatetime
    updated_at: AwareDatetime
    holdings: list[HoldingOut]


class RiskRequest(StrictModel):
    lookback_days: int = Field(default=730, ge=90, le=3650, description="Calendar days of daily history.")
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    horizon_days: int = Field(default=1, ge=1, le=30)
    monte_carlo_paths: int = Field(default=20_000, ge=1000, le=200_000)
    seed: int | None = Field(default=None, ge=0)
    benchmark: Symbol | None = None
    covariance: Literal["sample", "ledoit_wolf"] = "ledoit_wolf"
    frontier_points: int = Field(default=30, ge=5, le=100)
    max_weight: float = Field(default=1.0, gt=0, le=1.0)


class AdHocRiskRequest(RiskRequest):
    holdings: list[HoldingIn] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _unique(self) -> AdHocRiskRequest:
        symbols = [h.symbol for h in self.holdings]
        if len(symbols) != len(set(symbols)):
            raise ValueError("duplicate symbols in holdings")
        return self


class PositionRisk(StrictModel):
    symbol: str
    quantity: float
    price: float
    market_value: float
    weight: float
    cost_basis: float | None
    unrealized_pnl: float | None
    annual_return: float
    annual_volatility: float
    beta: float | None
    risk_contribution: float


class VaRResult(StrictModel):
    method: Literal["historical", "parametric", "cornish_fisher", "monte_carlo"]
    confidence: float
    horizon_days: int
    var_pct: float
    var_amount: float
    cvar_pct: float | None
    cvar_amount: float | None


class PerformanceMetrics(StrictModel):
    annual_return: float
    annual_volatility: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    beta: float | None
    risk_free_rate: float
    benchmark: str
    observations: int
    start: date
    end: date


class FrontierPoint(StrictModel):
    expected_return: float
    volatility: float
    sharpe: float | None
    weights: dict[str, float]


class AssetStats(StrictModel):
    symbol: str
    expected_return: float
    volatility: float


class EfficientFrontier(StrictModel):
    covariance_method: str
    shrinkage: float | None
    risk_free_rate: float
    assets: list[AssetStats]
    points: list[FrontierPoint]
    min_variance: FrontierPoint
    max_sharpe: FrontierPoint
    current: FrontierPoint
    equal_weight: FrontierPoint
    random_portfolios: list[tuple[float, float]] = Field(description="(volatility, expected_return) pairs")


class Correlation(StrictModel):
    symbols: list[str]
    matrix: list[list[float]]


class ValuePoint(StrictModel):
    on: date
    value: float


class PortfolioRiskReport(StrictModel):
    portfolio_value: float
    positions: list[PositionRisk]
    var: list[VaRResult]
    metrics: PerformanceMetrics
    frontier: EfficientFrontier | None
    correlation: Correlation
    value_history: list[ValuePoint]
    warnings: list[str] = Field(default_factory=list)
