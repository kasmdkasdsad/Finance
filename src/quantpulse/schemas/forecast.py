"""Probabilistic price forecasts, options-implied views and forecast calibration."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import DataStatus, StrictModel

FORECAST_DISCLAIMER = (
    "Forecasts are probability ranges from a volatility model (GARCH-t with bootstrapped residuals, blended "
    "with options-implied volatility when available), scheduled earnings jumps and a CAPM drift. They describe "
    "uncertainty; they are not price targets. Stock direction over weeks is close to a coin flip, so the useful "
    "part is the width of the range, not the midpoint. Not investment advice."
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
    earnings_days_excluded: int = Field(default=0, description="Earnings days left out of the GARCH fit")
    garch_vol_annual_21d: float | None = Field(
        default=None, description="21-day volatility from GARCH alone (ex earnings)"
    )
    implied_vol_annual_21d: float | None = Field(
        default=None, description="Options-implied ATM volatility near 21 days (includes any earnings jump)"
    )
    blended_vol_annual_21d: float | None = Field(
        default=None, description="Diffusive 21-day volatility used by the simulation after the blend"
    )
    iv_weight: float = 0.0
    variance_premium: float | None = None


class EarningsForecastOut(StrictModel):
    next_date: date | None = Field(description="Next expected earnings reaction day")
    source: str | None = Field(description="'scheduled' (vendor calendar) or 'estimated' (quarterly cadence)")
    sessions_ahead: int | None = Field(description="Trading days until the reaction day (1 = next session)")
    in_horizons: list[int] = Field(description="Forecast horizons that include the reaction day")
    typical_move: float | None = Field(description="RMS of past earnings-day returns")
    events_used: int = Field(description="Past reactions the jump size is drawn from")
    modelled: bool = Field(description="Whether an earnings jump was simulated")


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
    earnings: EarningsForecastOut | None = None
    calibration: CalibrationOut | None = None
    notes: list[str] = Field(default_factory=list)
    data_status: DataStatus
    disclaimer: str = FORECAST_DISCLAIMER
