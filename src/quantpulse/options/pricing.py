"""Option values, Greeks, implied volatility and probabilities — deterministic, per share.

* :func:`greeks` — Black-Scholes-Merton (the market's quoting convention) with theta per calendar day and vega
  and rho per 1 point, as brokers show them;
* :func:`american_price` — a Cox-Ross-Rubinstein tree for the early-exercise value US equity options have
  (a deep in-the-money put, a call before an ex-dividend date); used to judge assignment risk;
* :func:`implied_vol` — from a price, or ``None`` when no volatility reproduces it (never a guess);
* :func:`prob_itm`, :func:`prob_touch` — risk-neutral probabilities under the model. They are the *model's*
  probabilities; :mod:`quantpulse.options.analytics` compares them with what actually happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantpulse.quant.black_scholes import OptionType, bsm_greeks, implied_volatility, norm_cdf


@dataclass(frozen=True, slots=True)
class Valuation:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float  # per volatility point
    rho: float  # per 1% rate
    iv: float
    intrinsic: float
    extrinsic: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(getattr(self, k), 6) for k in self.__dataclass_fields__}


def intrinsic(kind: OptionType, spot: float, strike: float) -> float:
    return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)


def extrinsic(kind: OptionType, price: float, spot: float, strike: float) -> float:
    """Time value: what is left above intrinsic value (never negative)."""
    return max(price - intrinsic(kind, spot, strike), 0.0)


def greeks(
    kind: OptionType,
    spot: float,
    strike: float,
    years: float,
    vol: float,
    rate: float = 0.04,
    div: float = 0.0,
) -> Valuation:
    g = bsm_greeks(spot, strike, max(years, 0.0), rate, max(vol, 0.0), div, kind)
    return Valuation(
        price=g.price,
        delta=g.delta,
        gamma=g.gamma,
        theta=g.theta / 365.0,
        vega=g.vega / 100.0,
        rho=g.rho / 100.0,
        iv=vol,
        intrinsic=intrinsic(kind, spot, strike),
        extrinsic=extrinsic(kind, g.price, spot, strike),
    )


def implied_vol(
    kind: OptionType,
    price: float,
    spot: float,
    strike: float,
    years: float,
    rate: float = 0.04,
    div: float = 0.0,
) -> float | None:
    if years <= 0 or price <= 0:
        return None
    return implied_volatility(price, spot, strike, years, rate, div, kind)


def american_price(
    kind: OptionType,
    spot: float,
    strike: float,
    years: float,
    vol: float,
    rate: float = 0.04,
    div: float = 0.0,
    steps: int = 200,
) -> float:
    """Cox-Ross-Rubinstein binomial value with early exercise (continuous dividend yield)."""
    if years <= 0 or vol <= 0:
        return intrinsic(kind, spot, strike)
    dt = years / steps
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - div) * dt)
    p = (growth - down) / (up - down)
    if not 0.0 < p < 1.0:  # too few steps for these inputs: fall back to more steps
        return (
            american_price(kind, spot, strike, years, vol, rate, div, steps * 4) if steps < 5000 else math.nan
        )
    disc = math.exp(-rate * dt)
    j = np.arange(steps + 1)
    prices = spot * up ** (steps - 2 * j)
    values = np.maximum(prices - strike, 0.0) if kind == "call" else np.maximum(strike - prices, 0.0)
    for _ in range(steps):
        prices = prices[:-1] * down  # the node prices one step earlier
        cont = disc * (p * values[:-1] + (1 - p) * values[1:])
        exercise = prices - strike if kind == "call" else strike - prices
        values = np.maximum(cont, exercise)
    return float(values[0])


def early_exercise_premium(
    kind: OptionType,
    spot: float,
    strike: float,
    years: float,
    vol: float,
    rate: float = 0.04,
    div: float = 0.0,
) -> float:
    """American value minus European value: what early exercise is worth (≥ 0 up to numerical noise)."""
    euro = greeks(kind, spot, strike, years, vol, rate, div).price
    return max(american_price(kind, spot, strike, years, vol, rate, div) - euro, 0.0)


def _d2(spot: float, strike: float, years: float, vol: float, rate: float, div: float) -> float:
    return (math.log(spot / strike) + (rate - div - 0.5 * vol * vol) * years) / (vol * math.sqrt(years))


def prob_itm(
    kind: OptionType,
    spot: float,
    strike: float,
    years: float,
    vol: float,
    rate: float = 0.04,
    div: float = 0.0,
) -> float:
    """Risk-neutral probability of finishing in the money: N(d2) for a call, N(-d2) for a put."""
    if years <= 0 or vol <= 0:
        return 1.0 if intrinsic(kind, spot, strike) > 0 else 0.0
    d2 = _d2(spot, strike, years, vol, rate, div)
    return norm_cdf(d2) if kind == "call" else norm_cdf(-d2)


def prob_touch(
    spot: float, barrier: float, years: float, vol: float, rate: float = 0.04, div: float = 0.0
) -> float:
    """Probability the price touches ``barrier`` before ``years`` (geometric Brownian motion, reflection
    principle with drift). About twice the probability of finishing beyond it."""
    if barrier == spot:
        return 1.0
    if years <= 0 or vol <= 0:
        return 0.0
    mu = rate - div - 0.5 * vol * vol
    b = math.log(barrier / spot)
    s = vol * math.sqrt(years)
    if b > 0:  # an upper barrier
        return min(
            1.0,
            norm_cdf((-b + mu * years) / s)
            + math.exp(2 * mu * b / (vol * vol)) * norm_cdf((-b - mu * years) / s),
        )
    return min(
        1.0,
        norm_cdf((b - mu * years) / s) + math.exp(2 * mu * b / (vol * vol)) * norm_cdf((b + mu * years) / s),
    )


def expected_move(spot: float, vol: float, years: float) -> float:
    """One standard deviation of the price over the horizon, in dollars (the "expected move")."""
    return spot * vol * math.sqrt(max(years, 0.0))
