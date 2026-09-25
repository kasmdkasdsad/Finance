"""Options, yield-curve and volatility-surface schemas."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from quantpulse.schemas.common import StrictModel

OptionKind = Literal["call", "put"]


class OptionContract(StrictModel):
    contract_symbol: str
    kind: OptionKind
    strike: float = Field(gt=0)
    expiration: date
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    last: float | None = Field(default=None, ge=0)
    volume: float | None = Field(default=None, ge=0)
    open_interest: float | None = Field(default=None, ge=0)
    implied_volatility: float | None = Field(default=None, ge=0, description="Vendor-reported IV (decimal).")
    in_the_money: bool | None = None
    last_trade: AwareDatetime | None = None
    # Populated by the analytics layer:
    mid: float | None = None
    model_iv: float | None = Field(default=None, description="IV solved by QuantPulse from the mid price.")
    delta: float | None = None
    gamma: float | None = None
    theta_per_day: float | None = None
    vega_per_pct: float | None = None

    @property
    def reference_price(self) -> float | None:
        """Mid when the market is two-sided and sane, otherwise last trade."""
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask >= self.bid:
            return 0.5 * (self.bid + self.ask)
        if self.last is not None and self.last > 0:
            return self.last
        return None


class OptionChain(StrictModel):
    underlying: str
    underlying_price: float = Field(gt=0)
    as_of: AwareDatetime
    expirations: list[date]
    contracts: list[OptionContract]


class YieldPoint(StrictModel):
    tenor: str
    years: float = Field(gt=0)
    rate: float = Field(description="Bond-equivalent yield, decimal (0.0425 = 4.25%).")


class YieldCurve(StrictModel):
    as_of: date
    points: list[YieldPoint] = Field(min_length=1)

    @model_validator(mode="after")
    def _sort(self) -> YieldCurve:
        self.points = sorted(self.points, key=lambda p: p.years)
        return self

    @property
    def tenors(self) -> list[float]:
        return [p.years for p in self.points]

    @property
    def rates(self) -> list[float]:
        return [p.rate for p in self.points]


class RateQuery(StrictModel):
    years: float = Field(gt=0, le=50)


class RateAtTenor(StrictModel):
    years: float
    bey_rate: float
    continuous_rate: float
    curve_date: date


class BSMRequest(StrictModel):
    """Price an option. Omit ``spot``/``rate``/``volatility``/``dividend_yield`` to use live market inputs."""

    symbol: str | None = Field(default=None, description="Underlying used to source live inputs.")
    kind: OptionKind = "call"
    strike: float = Field(gt=0)
    expiration: date | None = None
    days_to_expiry: float | None = Field(default=None, gt=0, le=3650)
    spot: float | None = Field(default=None, gt=0)
    volatility: float | None = Field(default=None, gt=0, le=5)
    rate: float | None = Field(default=None, ge=-0.05, le=0.5, description="Continuously compounded.")
    dividend_yield: float | None = Field(default=None, ge=0, le=0.5)
    market_price: float | None = Field(default=None, gt=0, description="If given, implied vol is solved.")

    @model_validator(mode="after")
    def _one_maturity(self) -> BSMRequest:
        if (self.expiration is None) == (self.days_to_expiry is None):
            raise ValueError("provide exactly one of 'expiration' or 'days_to_expiry'")
        if self.symbol is None and (
            self.spot is None or (self.volatility is None and self.market_price is None)
        ):
            raise ValueError(
                "without 'symbol', both 'spot' and 'volatility' (or 'market_price') are required"
            )
        return self


class GreeksOut(StrictModel):
    price: float
    delta: float
    gamma: float
    vega_per_pct: float
    theta_per_day: float
    rho_per_pct: float
    vanna: float
    vomma: float
    charm_per_day: float
    d1: float | None
    d2: float | None


class BSMInputsUsed(StrictModel):
    spot: float
    strike: float
    years_to_expiry: float
    rate: float
    volatility: float
    dividend_yield: float
    kind: OptionKind
    volatility_source: str
    rate_source: str
    spot_source: str


class BSMResult(StrictModel):
    inputs: BSMInputsUsed
    greeks: GreeksOut
    implied_volatility: float | None = None
    counterpart_price: float = Field(description="Price of the opposite option type (same strike/expiry).")


class SurfacePoint(StrictModel):
    expiration: date
    years: float
    strike: float
    moneyness: float = Field(description="K / S")
    log_moneyness: float = Field(description="ln(K / F)")
    iv: float
    kind: OptionKind
    iv_source: Literal["model", "vendor"]
    delta: float | None = None


class SmileFit(StrictModel):
    a: float
    b: float
    c: float


class Smile(StrictModel):
    expiration: date
    years: float
    forward: float
    rate: float
    atm_iv: float | None
    skew_90_110: float | None = Field(default=None, description="IV(0.9·S) − IV(1.1·S)")
    fit: SmileFit | None = Field(default=None, description="iv ≈ a + b·k + c·k², k = ln(K/F)")
    points: list[SurfacePoint]


class VolSurface(StrictModel):
    underlying: str
    spot: float
    as_of: AwareDatetime
    dividend_yield: float
    moneyness_grid: list[float]
    expirations: list[date]
    years: list[float]
    iv_grid: list[list[float | None]] = Field(description="rows = expirations, cols = moneyness_grid")
    smiles: list[Smile]
    points_used: int
    points_rejected: int
