"""Realized-volatility forecasts and the forward-looking volatility risk premium.

**HAR-RV** (Corsi, 2009): tomorrow's variance is explained by today's, the past week's and the past month's —
three horizons of traders. With daily closes only, the daily variance proxy is the squared log return
(annualised). The model is fitted point in time, on logs (variance is skewed; logs keep the fit stable), to the
average variance over the next ``h`` days directly (no iteration), with Duan's smearing correction to undo the
log's bias. With too little history it falls back to an EWMA (RiskMetrics, λ = 0.94) and says so.

The **volatility risk premium** a structure faces is the implied volatility at its own expiration minus the
volatility forecast over the same horizon: option sellers are paid it on average; option buyers pay it. Its
sign and size are among the best-documented predictors of option returns (Goyal and Saretto, 2009; Carr and Wu;
Bali et al., 2023) — here they are features, never a rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ANNUAL = 252.0
EPS = 1e-8
MIN_FIT = 120  # daily observations to fit HAR; fewer: EWMA
WINDOW = 750  # at most three years of history in a fit


@dataclass(frozen=True, slots=True)
class VolForecast:
    vol: float  # annualised volatility expected over the horizon
    horizon: int  # trading days
    method: str  # "har" | "ewma"
    n: int  # observations behind it

    def as_dict(self) -> dict[str, float | int | str]:
        return {"vol": round(self.vol, 6), "horizon": self.horizon, "method": self.method, "n": self.n}


def _log_returns(closes: Sequence[float]) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) < 3:
        return np.empty(0)
    return np.diff(np.log(c))


def ewma_vol(closes: Sequence[float], lam: float = 0.94) -> float | None:
    r = _log_returns(closes)
    if len(r) < 10:
        return None
    var = float(np.mean(r[:10] ** 2))
    for x in r[10:]:
        var = lam * var + (1 - lam) * float(x * x)
    return math.sqrt(max(var, 0.0) * ANNUAL)


def _har_design(rv: np.ndarray) -> np.ndarray:
    """[1, log RV_d, log RV_w, log RV_m] for every day with a month of history (row i ↔ day i + 21)."""
    n = len(rv)
    csum = np.concatenate([[0.0], np.cumsum(rv)])
    idx = np.arange(21, n)
    d = rv[idx]
    w = (csum[idx + 1] - csum[idx - 4]) / 5
    m = (csum[idx + 1] - csum[idx - 21]) / 22
    return np.column_stack([np.ones(len(idx)), np.log(d + EPS), np.log(w + EPS), np.log(m + EPS)])


def har_forecast(closes: Sequence[float], horizon: int) -> VolForecast | None:
    """The annualised volatility expected over the next ``horizon`` trading days, from ``closes`` (oldest
    first, up to and including today) only."""
    h = max(1, int(horizon))
    r = _log_returns(closes)[-WINDOW:]
    if len(r) < 25:
        return None
    rv = r * r * ANNUAL
    n = len(rv)
    if n - 21 - h < MIN_FIT:
        v = ewma_vol(closes)
        return None if v is None else VolForecast(v, h, "ewma", n)
    x_all = _har_design(rv)  # row j is day j + 21
    csum = np.concatenate([[0.0], np.cumsum(rv)])
    days = np.arange(21, n)
    fit_rows = days + h < n  # the future average over (day, day + h] must be known
    future = (csum[np.minimum(days + h + 1, n)] - csum[days + 1]) / h
    x, y = x_all[fit_rows], np.log(future[fit_rows] + EPS)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    smear = float(np.mean(np.exp(resid)))
    var = float(np.exp(x_all[-1] @ beta)) * smear
    if not math.isfinite(var) or var <= 0:
        v = ewma_vol(closes)
        return None if v is None else VolForecast(v, h, "ewma", n)
    return VolForecast(math.sqrt(var), h, "har", int(fit_rows.sum()))


def vrp(implied_vol: float | None, forecast: VolForecast | None) -> tuple[float | None, float | None]:
    """(implied − forecast, implied / forecast): positive means options price more movement than expected."""
    if implied_vol is None or forecast is None or forecast.vol <= 0:
        return None, None
    return implied_vol - forecast.vol, implied_vol / forecast.vol
