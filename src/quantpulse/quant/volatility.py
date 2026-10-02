"""Volatility models: EWMA, GARCH(1,1) with Student-t innovations, and range-based realised estimators.

GARCH(1,1)
    ``ε_t = r_t − μ``,  ``σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}``,  ``ε_t = σ_t·z_t`` with ``z_t`` a
    unit-variance Student-t (ν > 2 degrees of freedom), so fat tails are modelled explicitly.

    The fit uses *variance targeting*: ``ω = s²·(1 − α − β)`` with ``s²`` the sample variance. This pins the
    long-run variance to the data and leaves three parameters (α, β, ν) for maximum likelihood, which
    makes the estimate stable on the few hundred observations a single stock provides. The parameters are
    optimised in an unconstrained space (``α + β = 0.999·sigmoid(a)``, ``α/(α+β) = sigmoid(b)``,
    ``ν = 2.1 + exp(c)``) so every candidate is a valid, stationary model.

    Multi-step forecasts: ``E[σ²_{t+k}] = V + (α+β)^{k−1}·(σ²_{t+1} − V)`` where ``V`` is the long-run
    variance; the variance of a ``h``-day return is the sum over ``k = 1…h``.

Earnings days (jumps)
    A stock's earnings reactions are scheduled jumps, not volatility news: a 9% move on results day says
    little about the next quiet week. Days flagged in ``jumps`` therefore feed the recursion with the
    typical variance of normal days instead of their squared return, are left out of the likelihood and
    out of the bootstrap residuals. Forecasts add earnings jumps back explicitly on the days they are due.

Returns are daily log returns. Internally they are scaled by 100 (percent) for numerical conditioning;
everything returned is in decimal units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

from quantpulse.core.errors import DomainError

TRADING_DAYS = 252
MIN_GARCH_OBS = 250
EWMA_LAMBDA = 0.94  # RiskMetrics daily decay
SCALE = 100.0
Array = NDArray[np.float64]
Flags = NDArray[Any]  # boolean flags, one per return


def log_returns(closes: Array) -> Array:
    c = np.asarray(closes, dtype=float)
    if c.ndim != 1 or c.size < 2 or np.any(c <= 0) or not np.all(np.isfinite(c)):
        raise DomainError("need at least two positive, finite closes")
    return np.diff(np.log(c))


# ----------------------------------------------------------------------------- EWMA
def jump_mask(n: int, jumps: Flags | None) -> NDArray[np.bool_]:
    mask = np.zeros(n, dtype=bool) if jumps is None else np.asarray(jumps, dtype=bool)
    if mask.shape != (n,):
        raise DomainError("jumps must flag every return")
    return mask


def _squared(eps: Array, mask: NDArray[np.bool_]) -> tuple[Array, float]:
    """Squared innovations with jump days replaced by the variance of normal days, and that variance."""
    normal = eps[~mask]
    if normal.size < 2:
        raise DomainError("need at least two returns outside earnings days")
    s2 = float(normal.var(ddof=1))
    return np.where(mask, s2, eps**2), s2


def ewma_variance(returns: Array, lam: float = EWMA_LAMBDA, jumps: Flags | None = None) -> Array:
    """RiskMetrics variance path ``σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1}``; element ``t`` uses data before ``t``.

    Returns ``n + 1`` values: the last one is the one-step-ahead forecast."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise DomainError("need at least two returns")
    if not 0 < lam < 1:
        raise DomainError("lambda must be in (0, 1)")
    mask = jump_mask(r.size, jumps)
    sq, _ = _squared(r, mask)
    head = r[: min(r.size, 30)][~mask[: min(r.size, 30)]]
    seed = float(np.var(head, ddof=1)) if head.size > 2 else float(np.mean(sq[:2]))
    x = (1 - lam) * sq
    path, _ = lfilter([1.0], [1.0, -lam], x, zi=[lam * seed])
    return np.concatenate([[seed], path])


# ----------------------------------------------------------------------------- GARCH(1,1)-t
@dataclass(frozen=True, slots=True)
class GarchFit:
    """A fitted GARCH(1,1)-t model. Variances are daily, in decimal units."""

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float
    log_likelihood: float
    n_obs: int
    last_variance: float  # σ²_t at the last observation
    next_variance: float  # σ²_{t+1}, the one-step-ahead forecast
    std_residuals: Array  # z_t = ε_t / σ_t on normal (non-earnings) days
    method: str = "garch"
    jump_days: int = 0  # earnings days excluded from the fit

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        p = self.persistence
        return self.omega / (1.0 - p) if p < 1 else self.next_variance

    @property
    def half_life_days(self) -> float | None:
        p = self.persistence
        return math.log(0.5) / math.log(p) if 0 < p < 1 else None

    def variance_term_structure(self, horizon: int) -> Array:
        """``E[σ²_{t+k}]`` for ``k = 1…horizon``."""
        if horizon < 1:
            raise DomainError("horizon must be >= 1")
        k = np.arange(horizon, dtype=float)
        v = self.long_run_variance
        return v + self.persistence**k * (self.next_variance - v)

    def horizon_volatility(self, horizon: int) -> float:
        """Standard deviation of the ``horizon``-day log return."""
        return float(math.sqrt(self.variance_term_structure(horizon).sum()))

    def annualized_volatility(self, horizon: int = 21) -> float:
        """Average forecast volatility over the next ``horizon`` days, annualised."""
        return float(math.sqrt(self.variance_term_structure(horizon).mean() * TRADING_DAYS))

    def summary(self) -> dict[str, float | int | str | None]:
        return {
            "method": self.method,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "nu": self.nu,
            "persistence": self.persistence,
            "half_life_days": self.half_life_days,
            "long_run_vol_annual": math.sqrt(self.long_run_variance * TRADING_DAYS),
            "current_vol_annual": math.sqrt(self.next_variance * TRADING_DAYS),
            "n_obs": self.n_obs,
            "log_likelihood": self.log_likelihood,
        }


def _t_loglik(eps: Array, var: Array, nu: float) -> float:
    """Log-likelihood of ``eps`` under a unit-variance Student-t scaled by ``sqrt(var)``."""
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return float(np.sum(const - 0.5 * np.log(var) - (nu + 1) / 2 * np.log1p(eps**2 / (var * (nu - 2)))))


def variance_path(
    eps: Array, omega: float, alpha: float, beta: float, seed_var: float, squared: Array | None = None
) -> Array:
    """σ²_t for t = 0…n (n + 1 values; the last is the one-step-ahead forecast). ``squared`` overrides
    ``eps²`` (e.g. with earnings days neutralised)."""
    x = omega + alpha * (eps**2 if squared is None else squared)
    path, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * seed_var])
    return np.concatenate([[seed_var], path])


def _unpack(theta: Array) -> tuple[float, float, float]:
    persistence = 0.999 / (1.0 + math.exp(-theta[0]))
    share = 1.0 / (1.0 + math.exp(-theta[1]))
    nu = 2.1 + math.exp(min(theta[2], 6.0))
    return persistence * share, persistence * (1.0 - share), nu


def _pack(fit: GarchFit) -> Array:
    """Inverse of :func:`_unpack`: the optimiser coordinates of an existing fit (for warm starts)."""
    p = min(max(fit.persistence / 0.999, 1e-6), 1 - 1e-6)
    share = min(max(fit.alpha / fit.persistence if fit.persistence > 0 else 0.1, 1e-6), 1 - 1e-6)
    nu = fit.nu if math.isfinite(fit.nu) else 30.0
    return np.array([math.log(p / (1 - p)), math.log(share / (1 - share)), math.log(max(nu - 2.1, 1e-3))])


def fit_garch(
    returns: Array, *, demean: bool = True, jumps: Flags | None = None, warm: GarchFit | None = None
) -> GarchFit:
    """Maximum-likelihood GARCH(1,1)-t on daily log returns (≥ 250 observations); ``jumps`` flags
    earnings days, which are neutralised in the recursion and left out of the likelihood. ``warm``
    starts the optimiser from an earlier fit on overlapping data (walk-forward refits), instead of
    from three generic starting points."""
    r = np.asarray(returns, dtype=float)
    if r.size < MIN_GARCH_OBS:
        raise DomainError(f"GARCH needs at least {MIN_GARCH_OBS} returns, got {r.size}")
    if not np.all(np.isfinite(r)):
        raise DomainError("returns must be finite")
    mask = jump_mask(r.size, jumps)
    x = r * SCALE
    mu = float(x[~mask].mean()) if demean else 0.0
    eps = x - mu
    sq, s2 = _squared(eps, mask)
    if s2 <= 0:
        raise DomainError("returns have zero variance")
    keep = ~mask

    def nll(theta: Array) -> float:
        alpha, beta, nu = _unpack(theta)
        omega = s2 * (1.0 - alpha - beta)
        var = variance_path(eps, omega, alpha, beta, s2, sq)[:-1]
        if np.any(var <= 0) or not np.all(np.isfinite(var)):
            return 1e12
        return -_t_loglik(eps[keep], var[keep], nu)

    starts = [np.array([3.0, -2.5, 1.5]), np.array([2.0, -1.5, 2.5]), np.array([4.5, -3.0, 1.0])]
    if warm is not None and warm.method == "garch":
        starts = [_pack(warm)]
    best = None
    for x0 in starts:
        res = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    alpha, beta, nu = _unpack(best.x)
    omega = s2 * (1.0 - alpha - beta)
    path = variance_path(eps, omega, alpha, beta, s2, sq)
    scale2 = SCALE**2
    return GarchFit(
        mu=mu / SCALE,
        omega=omega / scale2,
        alpha=alpha,
        beta=beta,
        nu=nu,
        log_likelihood=float(-best.fun + int(keep.sum()) * math.log(SCALE)),
        n_obs=int(r.size),
        last_variance=float(path[-2] / scale2),
        next_variance=float(path[-1] / scale2),
        std_residuals=(eps / np.sqrt(path[:-1]))[keep],
        jump_days=int(mask.sum()),
    )


def fit_ewma(returns: Array, lam: float = EWMA_LAMBDA, jumps: Flags | None = None) -> GarchFit:
    """EWMA expressed as an integrated GARCH (ω = 0, α = 1 − λ, β = λ) with Gaussian residuals.

    Used when there is too little history for GARCH. Its variance forecast is flat."""
    r = np.asarray(returns, dtype=float)
    mask = jump_mask(r.size, jumps)
    path = ewma_variance(r, lam, mask)
    return GarchFit(
        mu=0.0,
        omega=0.0,
        alpha=1.0 - lam,
        beta=lam,
        nu=math.inf,
        log_likelihood=float("nan"),
        n_obs=int(r.size),
        last_variance=float(path[-2]),
        next_variance=float(path[-1]),
        std_residuals=(r / np.sqrt(path[:-1]))[~mask],
        method="ewma",
        jump_days=int(mask.sum()),
    )


def fit_best(returns: Array, jumps: Flags | None = None, warm: GarchFit | None = None) -> GarchFit:
    """GARCH(1,1)-t when there is enough data, otherwise EWMA (earnings days neutralised in both)."""
    r = np.asarray(returns, dtype=float)
    if r.size >= MIN_GARCH_OBS:
        try:
            return fit_garch(r, jumps=jumps, warm=warm)
        except DomainError:
            pass
    return fit_ewma(r, jumps=jumps)


# ----------------------------------------------------------------------------- realised estimators
def close_to_close(closes: Array, window: int = 21) -> float:
    r = log_returns(closes)[-window:]
    if r.size < 2:
        raise DomainError("not enough data")
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def parkinson(high: Array, low: Array, window: int = 21) -> float:
    """High-low range estimator (Parkinson, 1980): ``σ² = mean(ln(H/L)²) / (4 ln 2)``."""
    h = np.asarray(high, dtype=float)[-window:]
    lo = np.asarray(low, dtype=float)[-window:]
    if h.size < 2 or h.shape != lo.shape or np.any(lo <= 0) or np.any(h < lo):
        raise DomainError("need matching positive highs >= lows")
    var = float(np.mean(np.log(h / lo) ** 2) / (4 * math.log(2)))
    return math.sqrt(var * TRADING_DAYS)


def garman_klass(open_: Array, high: Array, low: Array, close: Array, window: int = 21) -> float:
    """Garman-Klass (1980): ``σ² = mean(½ln(H/L)² − (2ln2 − 1)·ln(C/O)²)``."""
    o, h, lo, c = (np.asarray(a, dtype=float)[-window:] for a in (open_, high, low, close))
    if o.size < 2 or np.any(np.array([o, h, lo, c]) <= 0):
        raise DomainError("need positive OHLC")
    var = float(np.mean(0.5 * np.log(h / lo) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2))
    return math.sqrt(max(var, 0.0) * TRADING_DAYS)
