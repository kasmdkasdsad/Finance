"""Probabilistic price forecasts and their walk-forward calibration.

Forecast (filtered historical simulation, FHS)
    1. Fit GARCH(1,1)-t to daily log returns (EWMA when history is short).
    2. Simulate ``n_paths`` future paths: each day draws a *historical standardised residual* (bootstrap,
       so skew and fat tails come from the stock's own history), scales it by the GARCH volatility, and
       updates the variance recursively, so volatility clustering carries into the forecast.
    3. Drift: the caller supplies an expected annual return (the platform uses CAPM,
       ``r_f + β·ERP − dividend yield``, optionally tilted by a model alpha). Each horizon is shifted so the
       *mean* simulated price equals ``S₀·exp(μ·h/252)`` exactly; the shape of the distribution is untouched.

    Outputs per horizon: quantile prices (a "cone"), mean and median, P(price up), P(price > target),
    and the 5% value-at-risk / expected shortfall.

Calibration (``evaluate_forecasts``)
    Replays history: at each origin the model is fitted on data up to that day only, a forecast is made,
    and the realised ``h``-day return is scored. A calibrated forecaster has ~50% of outcomes inside its
    25-75% band, ~90% inside its 5-95% band, and a flat PIT histogram. Direction probabilities are scored
    with the Brier score against a climatology baseline (the historical frequency of up moves known at
    each origin), so "skill" means beating the naive base rate, not merely being above 50%.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantpulse.core.errors import DomainError
from quantpulse.quant import volatility as vol

Array = NDArray[np.float64]
QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
TRADING_DAYS = 252


def simulate_log_returns(
    fit: vol.GarchFit,
    horizon: int,
    annual_drift: float,
    *,
    n_paths: int = 5000,
    seed: int = 7,
) -> Array:
    """Cumulative log returns, shape ``(n_paths, horizon)``; column ``k`` is the return after ``k+1`` days."""
    if horizon < 1 or n_paths < 100:
        raise DomainError("horizon must be >= 1 and n_paths >= 100")
    z = np.asarray(fit.std_residuals, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 20:
        raise DomainError("not enough residuals to bootstrap")
    z = (z - z.mean()) / z.std(ddof=0)
    rng = np.random.default_rng(seed)
    draws = z[rng.integers(0, z.size, size=(n_paths, horizon))]
    var = np.full(n_paths, fit.next_variance)
    steps = np.empty((n_paths, horizon))
    for k in range(horizon):
        eps = np.sqrt(var) * draws[:, k]
        steps[:, k] = eps
        var = fit.omega + fit.alpha * eps**2 + fit.beta * var
    cum = np.cumsum(steps, axis=1)
    # Martingale-style correction: E[S_k / S_0] = exp(μ·k/252) exactly at every horizon.
    days = np.arange(1, horizon + 1)
    target = annual_drift * days / TRADING_DAYS
    shift = target - np.log(np.mean(np.exp(cum), axis=0))
    return cum + shift


@dataclass(frozen=True, slots=True)
class HorizonForecast:
    days: int
    expected_price: float
    median_price: float
    quantiles: dict[float, float]
    prob_up: float
    volatility: float  # s.d. of the h-day log return
    var_95: float  # 5% worst-case loss as a positive fraction
    expected_shortfall_95: float


@dataclass
class PriceForecast:
    spot: float
    annual_drift: float
    fit: vol.GarchFit
    horizons: list[HorizonForecast]
    cone: dict[float, list[float]]  # quantile -> price for days 1..max horizon
    _terminal: dict[int, Array] = field(default_factory=dict, repr=False)

    def prob_above(self, price: float, days: int) -> float:
        """Probability that the price after ``days`` trading days exceeds ``price``."""
        if price <= 0:
            raise DomainError("price must be positive")
        if days not in self._terminal:
            raise DomainError(f"no simulation for {days} days; available: {sorted(self._terminal)}")
        return float(np.mean(self._terminal[days] > math.log(price / self.spot)))


def forecast_prices(
    closes: Sequence[float] | Array,
    spot: float,
    horizons: Sequence[int],
    annual_drift: float,
    *,
    n_paths: int = 5000,
    seed: int = 7,
    fit: vol.GarchFit | None = None,
) -> PriceForecast:
    """Forecast the price distribution ``h`` trading days ahead for each ``h`` in ``horizons``."""
    if spot <= 0:
        raise DomainError("spot must be positive")
    hs = sorted({int(h) for h in horizons})
    if not hs or hs[0] < 1 or hs[-1] > 504:
        raise DomainError("horizons must be between 1 and 504 trading days")
    if fit is None:
        rets = vol.log_returns(np.asarray(closes, dtype=float))
        if rets.size < 60:
            raise DomainError(f"need at least 61 closes to forecast, got {rets.size + 1}")
        fit = vol.fit_best(rets)
    cum = simulate_log_returns(fit, hs[-1], annual_drift, n_paths=n_paths, seed=seed)
    prices = spot * np.exp(cum)
    cone = {q: [float(v) for v in np.quantile(prices, q, axis=0)] for q in QUANTILES}
    out: list[HorizonForecast] = []
    terminal: dict[int, Array] = {}
    for h in hs:
        lr = cum[:, h - 1]
        terminal[h] = lr
        p = prices[:, h - 1]
        rets = p / spot - 1.0
        cutoff = float(np.quantile(rets, 0.05))
        out.append(
            HorizonForecast(
                days=h,
                expected_price=float(p.mean()),
                median_price=float(np.median(p)),
                quantiles={q: float(np.quantile(p, q)) for q in QUANTILES},
                prob_up=float(np.mean(lr > 0)),
                volatility=float(lr.std(ddof=1)),
                var_95=max(0.0, -cutoff),
                expected_shortfall_95=max(0.0, -float(rets[rets <= cutoff].mean())),
            )
        )
    return PriceForecast(
        spot=spot, annual_drift=annual_drift, fit=fit, horizons=out, cone=cone, _terminal=terminal
    )


# ----------------------------------------------------------------------------- calibration backtest
@dataclass(frozen=True, slots=True)
class ForecastRecord:
    origin: int
    prob_up: float
    climatology: float
    outcome_up: bool
    pit: float
    in_50: bool
    in_90: bool
    predicted_sd: float
    realized: float


@dataclass
class CalibrationReport:
    horizon: int
    records: list[ForecastRecord]
    step: int

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def effective_n(self) -> float:
        """Forecasts overlap when ``step < horizon``; this is the equivalent number of independent ones."""
        return self.n * min(1.0, self.step / self.horizon)

    def coverage(self, band: int) -> float:
        hits = [r.in_50 if band == 50 else r.in_90 for r in self.records]
        return float(np.mean(hits))

    def pit_histogram(self, bins: int = 10) -> list[float]:
        counts, _ = np.histogram([r.pit for r in self.records], bins=bins, range=(0.0, 1.0))
        return [float(c / max(1, self.n)) for c in counts]

    def brier(self) -> float:
        return float(np.mean([(r.prob_up - r.outcome_up) ** 2 for r in self.records]))

    def brier_climatology(self) -> float:
        return float(np.mean([(r.climatology - r.outcome_up) ** 2 for r in self.records]))

    def brier_skill(self) -> float | None:
        ref = self.brier_climatology()
        return None if ref == 0 else 1.0 - self.brier() / ref

    def volatility_ratio(self) -> float:
        """RMS realised h-day log return / RMS predicted s.d. (1.0 = volatility forecasts are unbiased)."""
        realized = math.sqrt(float(np.mean([r.realized**2 for r in self.records])))
        predicted = math.sqrt(float(np.mean([r.predicted_sd**2 for r in self.records])))
        return realized / predicted if predicted > 0 else float("nan")

    def direction_hit_rate(self) -> float | None:
        called = [r for r in self.records if r.prob_up != 0.5]
        if not called:
            return None
        return float(np.mean([(r.prob_up > 0.5) == r.outcome_up for r in called]))


def evaluate_forecasts(
    closes: Sequence[float] | Array,
    horizon: int = 21,
    *,
    step: int = 5,
    min_obs: int = 500,
    refit_every: int = 63,
    annual_drift: float = 0.0,
    n_paths: int = 2000,
    seed: int = 11,
) -> CalibrationReport:
    """Walk-forward replay of :func:`forecast_prices` with no look-ahead."""
    c = np.asarray(closes, dtype=float)
    rets = vol.log_returns(c)
    n = rets.size
    if horizon < 1 or step < 1:
        raise DomainError("horizon and step must be >= 1")
    if n < min_obs + horizon + step:
        raise DomainError(f"need at least {min_obs + horizon + step + 1} closes, got {c.size}")
    records: list[ForecastRecord] = []
    fit: vol.GarchFit | None = None
    fitted_at = -(10**9)
    for t in range(min_obs, n - horizon + 1, step):
        history = rets[:t]  # returns strictly before the origin close c[t]
        if t - fitted_at >= refit_every or fit is None:
            fit = vol.fit_best(history)
            fitted_at = t
        else:
            fit = refilter(fit, history)
        cum = simulate_log_returns(fit, horizon, annual_drift, n_paths=n_paths, seed=seed + t)[:, -1]
        realized = float(np.log(c[t + horizon] / c[t]))
        q05, q25, q75, q95 = np.quantile(cum, [0.05, 0.25, 0.75, 0.95])
        past = np.log(c[horizon : t + 1] / c[: t + 1 - horizon])  # h-day moves fully known at t
        clim = float(np.mean(past > 0)) if past.size else 0.5
        records.append(
            ForecastRecord(
                origin=t,
                prob_up=float(np.mean(cum > 0)),
                climatology=clim,
                outcome_up=realized > 0,
                pit=float(np.mean(cum <= realized)),
                in_50=bool(q25 <= realized <= q75),
                in_90=bool(q05 <= realized <= q95),
                predicted_sd=float(cum.std(ddof=1)),
                realized=realized,
            )
        )
    return CalibrationReport(horizon=horizon, records=records, step=step)


def refilter(fit: vol.GarchFit, returns: Array) -> vol.GarchFit:
    """Re-run the variance recursion of an existing fit over ``returns`` (parameters unchanged)."""
    r = np.asarray(returns, dtype=float)
    if fit.method == "ewma":
        return vol.fit_ewma(r, lam=fit.beta)
    eps = r - fit.mu
    seed = float(eps.var(ddof=1))
    path = vol.variance_path(eps, fit.omega, fit.alpha, fit.beta, seed)
    return vol.GarchFit(
        mu=fit.mu,
        omega=fit.omega,
        alpha=fit.alpha,
        beta=fit.beta,
        nu=fit.nu,
        log_likelihood=fit.log_likelihood,
        n_obs=int(r.size),
        last_variance=float(path[-2]),
        next_variance=float(path[-1]),
        std_residuals=eps / np.sqrt(path[:-1]),
        method=fit.method,
    )
