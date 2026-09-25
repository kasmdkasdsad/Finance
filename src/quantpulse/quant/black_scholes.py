"""Black-Scholes-Merton pricing with continuous dividend yield, analytic Greeks and implied volatility.

Conventions
-----------
* ``T`` is in years, ``r`` and ``q`` are continuously compounded annual rates, ``sigma`` is annualised.
* ``vega`` and ``rho`` are reported per 1.00 change (divide by 100 for "per vol point / per 1%").
* ``theta`` is per year (divide by 365 for calendar-day theta).

US single-stock options are American-style; BSM is the market-standard approximation used for implied
volatility quoting (early exercise premium for calls on non-dividend payers is zero).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import ndtr

from quantpulse.core.errors import DomainError

OptionType = Literal["call", "put"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
IV_LOWER = 1e-4
IV_UPPER = 5.0


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


@dataclass(frozen=True, slots=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    vomma: float
    charm: float
    d1: float | None
    d2: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _validate(spot: float, strike: float, t: float, sigma: float) -> None:
    if not (spot > 0 and math.isfinite(spot)):
        raise DomainError("spot must be a positive finite number")
    if not (strike > 0 and math.isfinite(strike)):
        raise DomainError("strike must be a positive finite number")
    if t < 0 or not math.isfinite(t):
        raise DomainError("time to expiry must be >= 0")
    if sigma < 0 or not math.isfinite(sigma):
        raise DomainError("volatility must be >= 0")


def _degenerate(spot: float, strike: float, t: float, r: float, q: float, kind: OptionType) -> Greeks:
    """Expiry (T=0) or zero-vol limit: the option is worth its discounted forward intrinsic value."""
    df_q = math.exp(-q * t)
    df_r = math.exp(-r * t)
    fwd_spot = spot * df_q
    pv_strike = strike * df_r
    if kind == "call":
        itm = fwd_spot > pv_strike
        price = max(fwd_spot - pv_strike, 0.0)
        delta = df_q if itm else 0.0
        rho = strike * t * df_r if itm else 0.0
        theta = (-r * pv_strike + q * fwd_spot) if itm else 0.0
    else:
        itm = pv_strike > fwd_spot
        price = max(pv_strike - fwd_spot, 0.0)
        delta = -df_q if itm else 0.0
        rho = -strike * t * df_r if itm else 0.0
        theta = (r * pv_strike - q * fwd_spot) if itm else 0.0
    return Greeks(price, delta, 0.0, 0.0, theta, rho, 0.0, 0.0, 0.0, None, None)


def bsm_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionType = "call",
) -> Greeks:
    """Price and full first/second-order Greeks for a European option under BSM."""
    if kind not in ("call", "put"):
        raise DomainError("kind must be 'call' or 'put'")
    _validate(spot, strike, t, sigma)
    if t == 0.0 or sigma == 0.0:
        return _degenerate(spot, strike, t, r, q, kind)

    sqrt_t = math.sqrt(t)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    df_q = math.exp(-q * t)
    df_r = math.exp(-r * t)
    pdf_d1 = norm_pdf(d1)

    gamma = df_q * pdf_d1 / (spot * vol_sqrt_t)
    vega = spot * df_q * pdf_d1 * sqrt_t
    vanna = -df_q * pdf_d1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma
    charm_common = df_q * pdf_d1 * (2.0 * (r - q) * t - d2 * vol_sqrt_t) / (2.0 * t * vol_sqrt_t)
    decay = -spot * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)

    if kind == "call":
        n_d1, n_d2 = norm_cdf(d1), norm_cdf(d2)
        price = spot * df_q * n_d1 - strike * df_r * n_d2
        delta = df_q * n_d1
        theta = decay - r * strike * df_r * n_d2 + q * spot * df_q * n_d1
        rho = strike * t * df_r * n_d2
        charm = q * df_q * n_d1 - charm_common
    else:
        n_md1, n_md2 = norm_cdf(-d1), norm_cdf(-d2)
        price = strike * df_r * n_md2 - spot * df_q * n_md1
        delta = -df_q * n_md1
        theta = decay + r * strike * df_r * n_md2 - q * spot * df_q * n_md1
        rho = -strike * t * df_r * n_md2
        charm = -q * df_q * n_md1 - charm_common

    return Greeks(price, delta, gamma, vega, theta, rho, vanna, vomma, charm, d1, d2)


def bsm_price(
    spot: float, strike: float, t: float, r: float, sigma: float, q: float = 0.0, kind: OptionType = "call"
) -> float:
    return bsm_greeks(spot, strike, t, r, sigma, q, kind).price


def no_arbitrage_bounds(
    spot: float, strike: float, t: float, r: float, q: float, kind: OptionType
) -> tuple[float, float]:
    fwd_spot = spot * math.exp(-q * t)
    pv_strike = strike * math.exp(-r * t)
    if kind == "call":
        return max(fwd_spot - pv_strike, 0.0), fwd_spot
    return max(pv_strike - fwd_spot, 0.0), pv_strike


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    t: float,
    r: float,
    q: float = 0.0,
    kind: OptionType = "call",
    tol: float = 1e-10,
) -> float | None:
    """Solve BSM for sigma with Brent's method. Returns ``None`` when no volatility reproduces ``price``
    (price outside no-arbitrage bounds or beyond the search bracket)."""
    _validate(spot, strike, t, 0.0)
    if t <= 0 or not (price > 0 and math.isfinite(price)):
        return None
    lower, upper = no_arbitrage_bounds(spot, strike, t, r, q, kind)
    if price <= lower + 1e-12 or price >= upper - 1e-12:
        return None

    def objective(sigma: float) -> float:
        return bsm_price(spot, strike, t, r, sigma, q, kind) - price

    f_lo, f_hi = objective(IV_LOWER), objective(IV_UPPER)
    if f_lo > 0 or f_hi < 0:
        return None
    if f_lo == 0:
        return IV_LOWER
    return float(brentq(objective, IV_LOWER, IV_UPPER, xtol=tol, maxiter=200))


# --------------------------------------------------------------------------- vectorised versions
def bsm_price_vec(
    spot: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
) -> NDArray[np.float64]:
    """Vectorised BSM price. Inputs broadcast; requires ``t > 0`` and ``sigma > 0`` elementwise."""
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strike, dtype=float)
    tt = np.asarray(t, dtype=float)
    rr = np.asarray(r, dtype=float)
    vol = np.asarray(sigma, dtype=float)
    qq = np.asarray(q, dtype=float)
    call = np.asarray(is_call, dtype=bool)
    vst = vol * np.sqrt(tt)
    d1 = (np.log(s / k) + (rr - qq + 0.5 * vol * vol) * tt) / vst
    d2 = d1 - vst
    fwd_spot = s * np.exp(-qq * tt)
    pv_k = k * np.exp(-rr * tt)
    call_px = fwd_spot * ndtr(d1) - pv_k * ndtr(d2)
    put_px = pv_k * ndtr(-d2) - fwd_spot * ndtr(-d1)
    return np.where(call, call_px, put_px)


def bsm_delta_vec(
    spot: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
) -> NDArray[np.float64]:
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strike, dtype=float)
    tt = np.asarray(t, dtype=float)
    vol = np.asarray(sigma, dtype=float)
    qq = np.asarray(q, dtype=float)
    rr = np.asarray(r, dtype=float)
    vst = vol * np.sqrt(tt)
    d1 = (np.log(s / k) + (rr - qq + 0.5 * vol * vol) * tt) / vst
    df_q = np.exp(-qq * tt)
    return np.where(np.asarray(is_call, dtype=bool), df_q * ndtr(d1), -df_q * ndtr(-d1))


def implied_volatility_vec(
    price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> NDArray[np.float64]:
    """Vectorised implied volatility by bisection (BSM price is strictly increasing in sigma).

    Elements without a solution (non-positive price, price outside no-arbitrage bounds, ``t <= 0``) are NaN.
    """
    px, s, k, tt, rr, qq, call = np.broadcast_arrays(
        np.asarray(price, dtype=float),
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(q, dtype=float),
        np.asarray(is_call, dtype=bool),
    )
    out = np.full(px.shape, np.nan)
    fwd_spot = s * np.exp(-qq * np.clip(tt, 0, None))
    pv_k = k * np.exp(-rr * np.clip(tt, 0, None))
    lower = np.where(call, np.maximum(fwd_spot - pv_k, 0.0), np.maximum(pv_k - fwd_spot, 0.0))
    upper = np.where(call, fwd_spot, pv_k)
    valid = (
        np.isfinite(px)
        & (px > 0)
        & (tt > 0)
        & (s > 0)
        & (k > 0)
        & (px > lower + 1e-12)
        & (px < upper - 1e-12)
    )
    if not valid.any():
        return out
    idx = np.nonzero(valid)
    p_v, s_v, k_v, t_v, r_v, q_v, c_v = (a[idx] for a in (px, s, k, tt, rr, qq, call))
    lo = np.full(p_v.shape, IV_LOWER)
    hi = np.full(p_v.shape, IV_UPPER)
    price_hi = bsm_price_vec(s_v, k_v, t_v, r_v, hi, q_v, c_v)
    price_lo = bsm_price_vec(s_v, k_v, t_v, r_v, lo, q_v, c_v)
    solvable = (price_lo <= p_v) & (price_hi >= p_v)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        above = bsm_price_vec(s_v, k_v, t_v, r_v, mid, q_v, c_v) > p_v
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
        if float(np.max(hi - lo)) < tol:
            break
    solved = np.where(solvable, 0.5 * (lo + hi), np.nan)
    out[idx] = solved
    return out
