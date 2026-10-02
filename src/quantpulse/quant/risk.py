"""Portfolio risk statistics on daily simple returns.

Conventions: VaR and CVaR are returned as *positive loss fractions* of portfolio value (0.021 = 2.1% loss);
``max_drawdown`` is returned as a negative fraction; annualisation uses 252 trading days.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quantpulse.core.errors import DomainError

TRADING_DAYS = 252


def _check_confidence(confidence: float) -> None:
    if not 0.5 < confidence < 1.0:
        raise DomainError("confidence must be in (0.5, 1)")


def simple_returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Simple returns along axis 0 (rows are dates)."""
    p = np.asarray(prices, dtype=float)
    if p.shape[0] < 2:
        raise DomainError("need at least two prices to compute returns")
    return p[1:] / p[:-1] - 1.0


def horizon_returns(returns: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    """Overlapping compounded ``horizon``-day returns from daily returns."""
    r = np.asarray(returns, dtype=float)
    if horizon <= 1:
        return r
    if r.size < horizon:
        raise DomainError("not enough observations for the requested horizon")
    log_r = np.log1p(r)
    csum = np.concatenate([[0.0], np.cumsum(log_r)])
    return np.expm1(csum[horizon:] - csum[:-horizon])


def historical_var_cvar(
    returns: NDArray[np.float64], confidence: float = 0.95, horizon: int = 1
) -> tuple[float, float]:
    _check_confidence(confidence)
    r = horizon_returns(returns, horizon)
    if r.size < 20:
        raise DomainError("historical VaR needs at least 20 observations")
    cutoff = float(np.quantile(r, 1.0 - confidence))
    tail = r[r <= cutoff]
    return -cutoff, -float(tail.mean())


def parametric_var_cvar(
    mu: float, sigma: float, confidence: float = 0.95, horizon: int = 1
) -> tuple[float, float]:
    """Gaussian (variance-covariance) VaR/CVaR with square-root-of-time scaling."""
    _check_confidence(confidence)
    if sigma < 0:
        raise DomainError("sigma must be non-negative")
    mu_h = mu * horizon
    sigma_h = sigma * math.sqrt(horizon)
    z = float(norm.ppf(confidence))
    var = -(mu_h - z * sigma_h)
    cvar = -(mu_h - sigma_h * float(norm.pdf(z)) / (1.0 - confidence))
    return var, cvar


def cornish_fisher_var(returns: NDArray[np.float64], confidence: float = 0.95, horizon: int = 1) -> float:
    """Modified VaR adjusting the Gaussian quantile for sample skewness and excess kurtosis."""
    _check_confidence(confidence)
    r = np.asarray(returns, dtype=float)
    if r.size < 30:
        raise DomainError("Cornish-Fisher VaR needs at least 30 observations")
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    if sigma == 0:
        return -mu * horizon
    z_scores = (r - mu) / sigma
    skew = float(np.mean(z_scores**3))
    kurt = float(np.mean(z_scores**4)) - 3.0
    z = float(norm.ppf(1.0 - confidence))
    z_cf = z + (z * z - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skew * skew / 36
    return -(mu * horizon + z_cf * sigma * math.sqrt(horizon))


def monte_carlo_var_cvar(
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
    weights: NDArray[np.float64],
    confidence: float = 0.95,
    horizon: int = 1,
    paths: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Simulate correlated asset returns from a multivariate normal fitted to history."""
    _check_confidence(confidence)
    rng = np.random.default_rng(seed)
    cov_arr = np.asarray(cov, dtype=float)
    method: Literal["cholesky", "eigh"] = "cholesky" if _is_pd(cov_arr) else "eigh"
    sims = rng.multivariate_normal(np.asarray(mean) * horizon, cov_arr * horizon, size=paths, method=method)
    port = sims @ np.asarray(weights)
    cutoff = float(np.quantile(port, 1.0 - confidence))
    return -cutoff, -float(port[port <= cutoff].mean())


def _is_pd(matrix: NDArray[np.float64]) -> bool:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def annualized_return(returns: NDArray[np.float64]) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        raise DomainError("no returns")
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS / r.size) - 1.0


def annualized_volatility(returns: NDArray[np.float64]) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise DomainError("need at least two returns")
    return float(r.std(ddof=1)) * math.sqrt(TRADING_DAYS)


def daily_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS) - 1.0


def sharpe_ratio(returns: NDArray[np.float64], risk_free_annual: float = 0.0) -> float | None:
    r = np.asarray(returns, dtype=float)
    excess = r - daily_rate(risk_free_annual)
    sd = float(excess.std(ddof=1)) if excess.size > 1 else 0.0
    if sd == 0:
        return None
    return float(excess.mean()) / sd * math.sqrt(TRADING_DAYS)


def sortino_ratio(returns: NDArray[np.float64], risk_free_annual: float = 0.0) -> float | None:
    """Sortino ratio with the risk-free rate as the minimum acceptable return."""
    r = np.asarray(returns, dtype=float)
    excess = r - daily_rate(risk_free_annual)
    downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    if downside == 0:
        return None
    return float(excess.mean()) / downside * math.sqrt(TRADING_DAYS)


def max_drawdown(returns: NDArray[np.float64]) -> float:
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    wealth = np.concatenate([[1.0], wealth])
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def beta(asset: NDArray[np.float64], benchmark: NDArray[np.float64]) -> float | None:
    a = np.asarray(asset, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    if a.size != b.size or a.size < 2:
        raise DomainError("beta requires aligned series with at least two observations")
    var_b = float(np.var(b, ddof=1))
    if var_b == 0:
        return None
    return float(np.cov(a, b, ddof=1)[0, 1]) / var_b


def risk_contributions(weights: NDArray[np.float64], cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fraction of total portfolio volatility contributed by each asset (sums to 1)."""
    w = np.asarray(weights, dtype=float)
    sigma_w = np.asarray(cov) @ w
    total_var = float(w @ sigma_w)
    if total_var <= 0:
        return np.zeros_like(w)
    return w * sigma_w / total_var


def ledoit_wolf(returns: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Ledoit–Wolf (2004) shrinkage of the sample covariance towards a scaled identity.

    Returns ``(covariance, shrinkage_intensity)``. The sample covariance here uses the 1/n (MLE)
    normalisation, as in the original paper.
    """
    x = np.asarray(returns, dtype=float)
    n, p = x.shape
    if n < 2:
        raise DomainError("need at least two observations")
    x = x - x.mean(axis=0)
    sample = x.T @ x / n
    mu = float(np.trace(sample)) / p
    x2 = x**2
    beta_raw = float(np.sum(x2.T @ x2)) / n
    delta_raw = float(np.sum(sample**2))
    beta_ = (beta_raw / n - delta_raw / n) / p
    delta = (delta_raw - 2.0 * mu * float(np.trace(sample)) + p * mu * mu) / p
    beta_ = min(beta_, delta)
    shrinkage = 0.0 if delta == 0 else max(0.0, beta_ / delta)
    shrunk = (1.0 - shrinkage) * sample + shrinkage * mu * np.eye(p)
    return shrunk, shrinkage
