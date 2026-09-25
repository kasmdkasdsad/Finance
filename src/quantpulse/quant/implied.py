"""What option prices imply about a stock's future: implied moves and risk-neutral probabilities.

For one expiry with forward ``F``, time ``T`` and discount factor ``DF = e^{−rT}``, the smile gives an
implied volatility ``σ(K)`` for every strike. Call prices ``C(K)`` follow from Black-Scholes at ``σ(K)``;
Breeden & Litzenberger (1978) then give, without assuming lognormality,

* the risk-neutral probability of finishing above ``K``:  ``P(S_T > K) = −(1/DF)·∂C/∂K``
  (this includes the skew term, so a downside-skewed smile raises the probability of a crash);
* the risk-neutral density:  ``f(K) = (1/DF)·∂²C/∂K²``.

Inside the quoted strikes the smile is interpolated linearly in log-moneyness ``k = ln(K/F)``; beyond
them it is held flat. These are **risk-neutral** probabilities: they embed investors' risk premia (for
example the price of crash insurance), so they are the market's hedging-adjusted view, not an unbiased
forecast of real-world odds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quantpulse.core.errors import DomainError

Array = NDArray[np.float64]
SMILE_WIDTH = 6.0  # grid spans ±6 ATM standard deviations in log-moneyness
GRID_POINTS = 801


def implied_move(atm_iv: float, years: float) -> tuple[float, float]:
    """(one-standard-deviation move, expected absolute move) as fractions of the price.

    The expected absolute move ``σ√T·√(2/π)`` is what an at-the-money straddle costs, as a fraction of spot."""
    if atm_iv <= 0 or years <= 0:
        raise DomainError("atm_iv and years must be positive")
    sd = atm_iv * math.sqrt(years)
    return sd, sd * math.sqrt(2.0 / math.pi)


def smile_function(log_moneyness: Sequence[float], ivs: Sequence[float]) -> Callable[[Array], Array]:
    """σ(k): linear inside the quotes, flat outside."""
    k = np.asarray(log_moneyness, dtype=float)
    v = np.asarray(ivs, dtype=float)
    if k.size == 0 or k.size != v.size or np.any(v <= 0) or not np.all(np.isfinite(v)):
        raise DomainError("smile needs matching positive implied volatilities")
    order = np.argsort(k)
    k, v = k[order], v[order]

    def fn(x: Array) -> Array:
        return np.interp(np.asarray(x, dtype=float), k, v)

    return fn


def _calls(forward: float, strikes: Array, years: float, df: float, sigma: Array) -> Array:
    sd = sigma * math.sqrt(years)
    d1 = (np.log(forward / strikes) + 0.5 * sd**2) / sd
    return df * (forward * norm.cdf(d1) - strikes * norm.cdf(d1 - sd))


@dataclass(frozen=True, slots=True)
class RiskNeutralDistribution:
    forward: float
    years: float
    strikes: Array
    cdf: Array  # P(S_T <= K)
    pdf: Array

    def prob_above(self, price: float) -> float:
        if price <= 0:
            raise DomainError("price must be positive")
        return float(1.0 - np.interp(price, self.strikes, self.cdf, left=0.0, right=1.0))

    def quantile(self, p: float) -> float:
        if not 0 < p < 1:
            raise DomainError("p must be in (0, 1)")
        return float(np.interp(p, self.cdf, self.strikes))

    def mean(self) -> float:
        return float(np.trapezoid(self.strikes * self.pdf, self.strikes))


def risk_neutral_distribution(
    forward: float, years: float, rate: float, iv_of_k: Callable[[Array], Array], atm_iv: float
) -> RiskNeutralDistribution:
    """Breeden-Litzenberger distribution of ``S_T`` from a smile ``iv_of_k(k)`` with ``k = ln(K/F)``."""
    if forward <= 0 or years <= 0 or atm_iv <= 0:
        raise DomainError("forward, years and atm_iv must be positive")
    df = math.exp(-rate * years)
    width = SMILE_WIDTH * atm_iv * math.sqrt(years)
    k = np.linspace(-width, width, GRID_POINTS)
    strikes = forward * np.exp(k)
    h = 1e-4 * forward

    def call(kk: Array) -> Array:
        return _calls(forward, kk, years, df, iv_of_k(np.log(kk / forward)))

    digital = -(call(strikes + h) - call(strikes - h)) / (2 * h) / df  # P(S_T > K)
    density = (call(strikes + h) - 2 * call(strikes) + call(strikes - h)) / h**2 / df
    cdf = np.clip(1.0 - digital, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)  # a noisy smile must not produce a decreasing CDF
    return RiskNeutralDistribution(
        forward=forward, years=years, strikes=strikes, cdf=cdf, pdf=np.clip(density, 0.0, None)
    )
