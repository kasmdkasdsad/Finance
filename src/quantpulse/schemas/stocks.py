"""One-page stock intelligence report: technicals, forecast, model view, valuation and track record."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import DataStatus, StrictModel
from quantpulse.schemas.forecast import StockForecast
from quantpulse.schemas.predictions import PredictionOut

REPORT_DISCLAIMER = (
    "This report combines a volatility forecast, a statistical stock model, options prices and a DCF. Each "
    "has its own assumptions and error; none predicts prices reliably. Use the ranges and track record to size "
    "risk, not as a promise of returns. Not investment advice."
)


class Technicals(StrictModel):
    price: float
    change_percent: float | None
    sma20: float | None
    sma50: float | None
    sma200: float | None
    rsi_14: float | None
    high_52w: float | None
    low_52w: float | None
    from_high_52w: float | None
    return_1m: float | None
    return_3m: float | None
    return_6m: float | None
    return_1y: float | None
    volatility_3m: float | None
    max_drawdown_1y: float | None
    beta: float | None
    avg_volume_20d: float | None


class PricePoint(StrictModel):
    date: date
    close: float
    sma50: float | None
    sma200: float | None


class ModelView(StrictModel):
    in_universe: bool
    rank: int
    universe_size: int
    z: float
    rating: int
    prob_outperform: float
    expected_excess_return: float
    horizon: int
    benchmark: str
    has_skill: bool
    verdict: str
    as_of: date
    data_status: DataStatus


class ValuationView(StrictModel):
    value_per_share: float
    current_price: float | None
    upside: float | None
    wacc: float
    terminal_value_share: float
    warnings: list[str]
    data_status: DataStatus


class TrackRecord(StrictModel):
    resolved: int
    open: int
    forecast_brier: float | None
    forecast_coverage_90: float | None
    model_hit_rate: float | None
    recent: list[PredictionOut]


class StockReport(StrictModel):
    symbol: str
    name: str | None
    as_of: AwareDatetime
    technicals: Technicals
    chart: list[PricePoint]
    forecast: StockForecast
    model: ModelView | None
    valuation: ValuationView | None
    track_record: TrackRecord
    summary: list[str] = Field(description="Plain-English takeaways, each tied to a number above.")
    notes: list[str] = Field(default_factory=list)
    data_status: DataStatus
    disclaimer: str = REPORT_DISCLAIMER
