"""Implied-volatility smile and surface construction from a live option chain.

Method
------
1. Each contract's reference price is its bid/ask mid (last trade if the market is one-sided).
2. Time to expiry uses the 16:00 America/New_York expiration cut-off, in ACT/365 years.
3. The forward is ``F = S·exp((r − q)·T)`` with ``r`` read from the live Treasury curve at ``T``.
4. Only out-of-the-money options are used (puts for K < F, calls for K ≥ F): they are the liquid,
   early-exercise-free side of each strike and are the market standard for surface building.
5. IV is solved by vectorised bisection on BSM; if no solution exists the vendor IV is used when
   plausible, otherwise the point is rejected.
6. The surface grid holds, for each listed expiry, the smile linearly interpolated in log-moneyness
   onto a common K/S grid (no extrapolation beyond quoted strikes).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date, datetime, time

import numpy as np

from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.quant.black_scholes import bsm_delta_vec, implied_volatility_vec
from quantpulse.schemas.options import OptionContract, Smile, SmileFit, SurfacePoint, VolSurface

SECONDS_PER_YEAR = 365.0 * 86400.0
EXPIRY_CUTOFF = time(16, 0)


def expiry_datetime(expiration: date) -> datetime:
    return datetime.combine(expiration, EXPIRY_CUTOFF, NEW_YORK)


def years_to_expiry(expiration: date, as_of: datetime) -> float:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return (expiry_datetime(expiration) - as_of).total_seconds() / SECONDS_PER_YEAR


def build_vol_surface(
    underlying: str,
    contracts: Sequence[OptionContract],
    spot: float,
    as_of: datetime,
    rate_at: Callable[[float], float],
    dividend_yield: float = 0.0,
    *,
    min_days: float = 1.0,
    moneyness_range: tuple[float, float] = (0.7, 1.3),
    grid_points: int = 25,
    max_relative_spread: float = 1.0,
    otm_only: bool = True,
) -> VolSurface:
    """Build smiles and a moneyness × expiry IV grid. ``rate_at(T)`` returns a continuous rate."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    lo_m, hi_m = moneyness_range
    if not (0 < lo_m < 1 < hi_m):
        raise ValueError("moneyness_range must bracket 1.0")

    rows: list[tuple[OptionContract, float, float, float]] = []  # contract, T, price, r
    rejected = 0
    rate_cache: dict[date, float] = {}
    for c in contracts:
        t = years_to_expiry(c.expiration, as_of)
        if t * 365.0 < min_days:
            rejected += 1
            continue
        price = c.reference_price
        if price is None:
            rejected += 1
            continue
        if c.bid and c.ask and c.ask >= c.bid > 0:
            mid = 0.5 * (c.bid + c.ask)
            if (c.ask - c.bid) / mid > max_relative_spread:
                rejected += 1
                continue
        moneyness = c.strike / spot
        if not (lo_m * 0.8 <= moneyness <= hi_m * 1.2):
            rejected += 1
            continue
        if c.expiration not in rate_cache:
            rate_cache[c.expiration] = rate_at(t)
        r = rate_cache[c.expiration]
        if otm_only:
            forward = spot * math.exp((r - dividend_yield) * t)
            if (c.kind == "call" and c.strike < forward) or (c.kind == "put" and c.strike >= forward):
                continue
        rows.append((c, t, price, r))

    points: list[SurfacePoint] = []
    if rows:
        price_arr = np.array([p for _, _, p, _ in rows])
        strike_arr = np.array([c.strike for c, _, _, _ in rows])
        t_arr = np.array([t for _, t, _, _ in rows])
        r_arr = np.array([r for _, _, _, r in rows])
        call_arr = np.array([c.kind == "call" for c, _, _, _ in rows])
        iv_arr = implied_volatility_vec(price_arr, spot, strike_arr, t_arr, r_arr, dividend_yield, call_arr)
        for i, (c, t, _price, r) in enumerate(rows):
            iv = float(iv_arr[i])
            source = "model"
            if not math.isfinite(iv):
                vendor = c.implied_volatility
                if vendor is not None and 0.01 <= vendor <= 5.0:
                    iv, source = float(vendor), "vendor"
                else:
                    rejected += 1
                    continue
            forward = spot * math.exp((r - dividend_yield) * t)
            delta = float(bsm_delta_vec(spot, c.strike, t, r, iv, dividend_yield, c.kind == "call"))
            points.append(
                SurfacePoint(
                    expiration=c.expiration,
                    years=t,
                    strike=c.strike,
                    moneyness=c.strike / spot,
                    log_moneyness=math.log(c.strike / forward),
                    iv=iv,
                    kind=c.kind,
                    iv_source=source,
                    delta=delta,
                )
            )

    by_expiry: dict[date, list[SurfacePoint]] = defaultdict(list)
    for p in points:
        by_expiry[p.expiration].append(p)

    grid = np.linspace(lo_m, hi_m, grid_points)
    smiles: list[Smile] = []
    iv_grid: list[list[float | None]] = []
    expirations: list[date] = []
    years: list[float] = []
    for expiration in sorted(by_expiry):
        pts = sorted(by_expiry[expiration], key=lambda p: p.strike)
        t = pts[0].years
        r = rate_cache[expiration]
        forward = spot * math.exp((r - dividend_yield) * t)
        # A strike can appear twice only at the call/put boundary when otm_only=False; average them.
        ks: dict[float, list[float]] = defaultdict(list)
        for p in pts:
            ks[p.strike].append(p.iv)
        strikes = np.array(sorted(ks))
        ivs = np.array([float(np.mean(ks[k])) for k in strikes])
        log_m = np.log(strikes / forward)

        atm_iv = _interp_inside(0.0, log_m, ivs)
        iv_90 = _interp_inside(math.log(0.9 * spot / forward), log_m, ivs)
        iv_110 = _interp_inside(math.log(1.1 * spot / forward), log_m, ivs)
        skew = iv_90 - iv_110 if iv_90 is not None and iv_110 is not None else None
        fit = None
        if len(strikes) >= 3:
            c2, c1, c0 = np.polyfit(log_m, ivs, 2)
            fit = SmileFit(a=float(c0), b=float(c1), c=float(c2))

        row: list[float | None] = []
        for m in grid:
            val = _interp_inside(math.log(m * spot / forward), log_m, ivs)
            row.append(None if val is None else round(val, 6))
        smiles.append(
            Smile(
                expiration=expiration,
                years=t,
                forward=forward,
                rate=r,
                atm_iv=atm_iv,
                skew_90_110=skew,
                fit=fit,
                points=pts,
            )
        )
        iv_grid.append(row)
        expirations.append(expiration)
        years.append(t)

    return VolSurface(
        underlying=underlying,
        spot=spot,
        as_of=as_of,
        dividend_yield=dividend_yield,
        moneyness_grid=[round(float(m), 6) for m in grid],
        expirations=expirations,
        years=years,
        iv_grid=iv_grid,
        smiles=smiles,
        points_used=len(points),
        points_rejected=rejected,
    )


def _interp_inside(x: float, xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Linear interpolation that refuses to extrapolate (returns ``None`` outside the data)."""
    if xs.size == 0:
        return None
    if xs.size == 1:
        return float(ys[0]) if math.isclose(x, float(xs[0]), abs_tol=1e-9) else None
    if x < xs[0] - 1e-12 or x > xs[-1] + 1e-12:
        return None
    return float(np.interp(x, xs, ys))
