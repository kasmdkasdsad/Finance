"""Probabilistic price forecasts, options-implied views and forecast calibration."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import DataStatus, StrictModel

FORECAST_DISCLAIMER = (
    "Forecasts are probability ranges from a volatility model (GARCH-t with bootstrapped residuals) and a "
    "CAPM drift. They describe uncertainty; they are not price targets. Stock direction over weeks is close to "
    "a coin flip, so the useful part is the width of the range, not the midpoint. Not investment advice."
)


class Band(StrictModel):
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float


class ImpliedView(StrictModel):
    """What the options market prices for one expiry (risk-neutral: includes risk premia)."""

    expiration: date
    days_to_expiry: float
    atm_iv: float
    move_1sd: float = Field(description="One-standard-deviation move to expiry, as a fraction of the price.")
    expected_abs_move: float = Field(description="Expected absolute move (≈ ATM straddle / price).")
    prob_up: float = Field(description="Risk-neutral probability the price finishes above today's.")
    prob_above_target: float | None = None
    band: Band
    data_status: DataStatus


class HorizonOut(StrictModel):
    days: int
    target_date: date
    expected_price: float
    median_price: float
    band: Band
    prob_up: float
    prob_above_target: float | None = None
    expected_return: float
    volatility: float = Field(description="Standard deviation of the log return over the horizon.")
    var_95: float = Field(description="5% worst-case loss (fraction of price).")
    expected_shortfall_95: float
    implied: ImpliedView | None = None


class ConePoint(StrictModel):
    day: int
    date: date
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float


class VolModelOut(StrictModel):
    method: str
    alpha: float
    beta: float
    nu: float | None
    persistence: float
    half_life_days: float | None
    current_vol_annual: float
    long_run_vol_annual: float
    forecast_vol_annual_21d: float
    n_obs: int


class DriftOut(StrictModel):
    annual_expected_return: float
    risk_free: float
    beta: float | None
    equity_risk_premium: float
    dividend_yield: float
    model_alpha: float | None = Field(
        default=None, description="Annualised tilt from the stock model, if used."
    )
    method: str


class CalibrationOut(StrictModel):
    horizon: int
    n: int
    effective_n: float
    coverage_50: float
    coverage_90: float
    pit_histogram: list[float]
    brier: float
    brier_climatology: float
    brier_skill: float | None
    direction_hit_rate: float | None
    volatility_ratio: float
    start: date
    end: date


class StockForecast(StrictModel):
    symbol: str
    as_of: AwareDatetime
    spot: float
    target: float | None
    horizons: list[HorizonOut]
    cone: list[ConePoint]
    volatility: VolModelOut
    drift: DriftOut
    realized_vol: dict[str, float | None]
    calibration: CalibrationOut | None = None
    notes: list[str] = Field(default_factory=list)
    data_status: DataStatus
    disclaimer: str = FORECAST_DISCLAIMER
