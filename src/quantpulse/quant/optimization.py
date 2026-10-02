"""Mean-variance (Markowitz) optimisation: minimum variance, maximum Sharpe and the efficient frontier.

All inputs are annualised: ``mu`` is the vector of expected returns and ``cov`` the covariance matrix.
Weights sum to one and respect per-asset bounds (long-only by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize

from quantpulse.core.errors import DomainError

Bounds = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PortfolioPoint:
    weights: NDArray[np.float64]
    expected_return: float
    volatility: float
    sharpe: float | None


def stats(
    weights: NDArray[np.float64], mu: NDArray[np.float64], cov: NDArray[np.float64], rf: float
) -> PortfolioPoint:
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    vol = math.sqrt(max(float(w @ cov @ w), 0.0))
    sharpe = (ret - rf) / vol if vol > 0 else None
    return PortfolioPoint(w, ret, vol, sharpe)


def _check(mu: NDArray[np.float64], cov: NDArray[np.float64], bounds: Bounds) -> int:
    n = mu.shape[0]
    if n < 1 or cov.shape != (n, n):
        raise DomainError("mu and cov dimensions do not match")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise DomainError("covariance matrix must be symmetric")
    lo, hi = bounds
    if lo > hi:
        raise DomainError("lower weight bound exceeds upper bound")
    if n * hi < 1.0 - 1e-9 or n * lo > 1.0 + 1e-9:
        raise DomainError(f"weights cannot sum to 1 with {n} assets and bounds [{lo}, {hi}]")
    return n


def _feasible_start(n: int, bounds: Bounds) -> NDArray[np.float64]:
    lo, hi = bounds
    return np.full(n, min(max(1.0 / n, lo), hi))


def _solve(
    objective,
    jac,
    n: int,
    bounds: Bounds,
    extra: list[dict] | None = None,
    x0: NDArray[np.float64] | None = None,
) -> NDArray[np.float64] | None:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]
    constraints += extra or []
    start = _feasible_start(n, bounds) if x0 is None else x0
    result = minimize(
        objective,
        start,
        jac=jac,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return None
    w = np.clip(result.x, bounds[0], bounds[1])
    total = w.sum()
    return w / total if total != 0 else None


def min_variance(
    mu: NDArray[np.float64], cov: NDArray[np.float64], rf: float, bounds: Bounds = (0.0, 1.0)
) -> PortfolioPoint:
    n = _check(mu, cov, bounds)
    w = _solve(lambda w: float(w @ cov @ w), lambda w: 2.0 * cov @ w, n, bounds)
    if w is None:
        raise DomainError("minimum-variance optimisation did not converge")
    return stats(w, mu, cov, rf)


def max_sharpe(
    mu: NDArray[np.float64], cov: NDArray[np.float64], rf: float, bounds: Bounds = (0.0, 1.0)
) -> PortfolioPoint:
    n = _check(mu, cov, bounds)

    def neg_sharpe(w: NDArray[np.float64]) -> float:
        vol = math.sqrt(max(float(w @ cov @ w), 1e-18))
        return -(float(w @ mu) - rf) / vol

    def neg_sharpe_grad(w: NDArray[np.float64]) -> NDArray[np.float64]:
        var = max(float(w @ cov @ w), 1e-18)
        vol = math.sqrt(var)
        excess = float(w @ mu) - rf
        return -(mu * vol - excess * (cov @ w) / vol) / var

    best: PortfolioPoint | None = None
    starts = [_feasible_start(n, bounds)]
    # Additional starts tilted towards each asset guard against poor local optima.
    for i in range(n):
        tilt = np.full(n, bounds[0]) if bounds[0] > 0 else np.zeros(n)
        tilt[i] = 1.0
        tilt = 0.5 * tilt / tilt.sum() + 0.5 * starts[0]
        starts.append(np.clip(tilt, bounds[0], bounds[1]))
    for x0 in starts:
        w = _solve(neg_sharpe, neg_sharpe_grad, n, bounds, x0=x0)
        if w is None:
            continue
        point = stats(w, mu, cov, rf)
        if point.sharpe is not None and (best is None or point.sharpe > (best.sharpe or -np.inf)):
            best = point
    if best is None:
        raise DomainError("maximum-Sharpe optimisation did not converge")
    return best


def max_return(mu: NDArray[np.float64], bounds: Bounds) -> float:
    n = mu.shape[0]
    res = linprog(-mu, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[bounds] * n, method="highs")
    if not res.success:
        raise DomainError("could not determine the maximum attainable return")
    return float(mu @ res.x)


def efficient_frontier(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    rf: float,
    bounds: Bounds = (0.0, 1.0),
    points: int = 30,
) -> list[PortfolioPoint]:
    """Minimum-variance portfolios for evenly spaced target returns from the global minimum-variance
    return up to the highest attainable return."""
    n = _check(mu, cov, bounds)
    gmv = min_variance(mu, cov, rf, bounds)
    top = max_return(mu, bounds)
    if top - gmv.expected_return < 1e-9:
        return [gmv]
    frontier: list[PortfolioPoint] = [gmv]
    x0 = gmv.weights
    for target in np.linspace(gmv.expected_return, top, points)[1:]:
        extra = [
            {
                "type": "eq",
                "fun": lambda w, t=target: float(w @ mu) - t,
                "jac": lambda w: mu,
            }
        ]
        w = _solve(lambda w: float(w @ cov @ w), lambda w: 2.0 * cov @ w, n, bounds, extra=extra, x0=x0)
        if w is None:
            continue
        point = stats(w, mu, cov, rf)
        if abs(point.expected_return - target) > 1e-4:
            continue
        frontier.append(point)
        x0 = w
    return frontier


def random_portfolios(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    rf: float,
    count: int = 1500,
    max_weight: float = 1.0,
    seed: int = 7,
) -> list[PortfolioPoint]:
    """Long-only Dirichlet-sampled portfolios for visualising the feasible region."""
    n = mu.shape[0]
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n), size=count * 3)
    weights = weights[np.all(weights <= max_weight + 1e-12, axis=1)][:count]
    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    return [
        PortfolioPoint(w, float(r), float(v), float((r - rf) / v) if v > 0 else None)
        for w, r, v in zip(weights, rets, vols, strict=True)
    ]
