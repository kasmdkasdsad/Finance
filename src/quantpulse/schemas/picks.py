"""Daily stock-picks (factor screen) and email digest schemas."""

from __future__ import annotations

from pydantic import AwareDatetime, EmailStr, Field

from quantpulse.schemas.common import DataStatus, StrictModel

DISCLAIMER = (
    "QuantPulse daily picks are a systematic, backward-looking factor screen (momentum, trend, risk-adjusted "
    "return, volatility and short-term reversal) ranked relative to the screened universe. They are not "
    "investment advice or a forecast of returns; past price behaviour does not guarantee future results."
)


class FactorScores(StrictModel):
    momentum_12_1: float | None = Field(description="Return from 12 months to 1 month ago")
    momentum_3m: float | None
    trend: float | None = Field(description="Blend of price vs 50-day SMA and 50-day vs 200-day SMA")
    risk_adjusted: float | None = Field(description="Annualised 6-month return / volatility")
    volatility_3m: float | None = Field(description="Annualised 3-month volatility (lower scores higher)")
    rsi_14: float | None


class StockPick(StrictModel):
    rank: int
    symbol: str
    name: str | None
    price: float
    change_percent: float | None
    rating: int = Field(ge=1, le=10, description="1 (weakest) … 10 (strongest) within the screened universe")
    score: float = Field(description="Composite cross-sectional z-score")
    factors: FactorScores
    factor_z: dict[str, float]
    drivers: list[str] = Field(description="Factors contributing most to the score")
    data_status: DataStatus
    provider: str


class DailyPicks(StrictModel):
    as_of: AwareDatetime
    trading_day: str
    universe_size: int
    screened: int
    picks: list[StockPick]
    top_pick: StockPick | None
    data_status: DataStatus
    methodology: str
    disclaimer: str = DISCLAIMER
    skipped: dict[str, str] = Field(default_factory=dict)


class PicksEmailRequest(StrictModel):
    recipients: list[EmailStr] | None = Field(
        default=None, min_length=1, max_length=20, description="Defaults to QP_PICKS_RECIPIENTS."
    )
    top_n: int = Field(default=10, ge=1, le=50)
    allow_synthetic: bool = Field(
        default=False,
        description="Send even if prices are synthetic (the email is then clearly marked). Off by default so "
        "fabricated numbers are never mailed as real recommendations.",
    )


class PicksEmailResult(StrictModel):
    sent_to: list[str]
    subject: str
    data_status: DataStatus
    top_pick: str | None
