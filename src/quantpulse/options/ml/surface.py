"""The implied-volatility surface: an SVI smile per expiration, its arbitrage checks and what it says.

For one expiration with time to expiry ``T`` and log-moneyness ``k = ln(K / F)``, Gatheral's *raw* SVI gives
the total implied variance ``w = σ²T`` as

    w(k) = a + b · (ρ · (k − m) + √((k − m)² + s²))

with ``b ≥ 0``, ``|ρ| < 1``, ``s > 0`` and ``a + b·s·√(1 − ρ²) ≥ 0`` (the variance never goes negative). It is
fitted to the out-of-the-money side of the listed chain (puts below the forward, calls above), each quote
weighted by how tight its market is. A fitted slice is checked for **butterfly arbitrage** (Gatheral's density
condition ``g(k) ≥ 0``) and consecutive slices for **calendar arbitrage** (total variance never falls with
maturity at a fixed moneyness). Violations are counted, never hidden: they are a data-quality feature.

From the fitted surface come the features the edge model reads — the at-the-money level at a constant maturity,
the skew and curvature of the smile, a one-standard-deviation risk reversal and butterfly, the term slope — and
each contract's **residual**: how many volatility points its own quote sits above (rich) or below (cheap) the
smooth surface. A structure's residual edge is what it would gain, at its legs' vegas, if every leg came back to
the surface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from quantpulse.options.analytics import quote_iv
from quantpulse.options.quotes import OptionQuote

RATE = 0.04
MIN_POINTS = 5  # quotes needed to fit a slice
MIN_DTE, MAX_DTE = 3, 400
GRID = np.linspace(-1.0, 1.0, 81)  # log-moneyness grid for the arbitrage checks


@dataclass(frozen=True, slots=True)
class SVISlice:
    expiration: date
    years: float
    forward: float
    a: float
    b: float
    rho: float
    m: float
    s: float
    rmse_vol: float  # fit error, in volatility (not variance) units
    points: int

    def w(self, k: float | np.ndarray) -> float | np.ndarray:
        """Total implied variance at log-moneyness ``k``."""
        d = np.asarray(k, dtype=float) - self.m
        out = self.a + self.b * (self.rho * d + np.sqrt(d * d + self.s * self.s))
        return float(out) if np.ndim(out) == 0 else out

    def vol(self, k: float | np.ndarray) -> float | np.ndarray:
        w = np.maximum(np.asarray(self.w(k), dtype=float), 1e-12)
        out = np.sqrt(w / max(self.years, 1e-9))
        return float(out) if np.ndim(out) == 0 else out

    def vol_at_strike(self, strike: float) -> float:
        return float(self.vol(math.log(strike / self.forward)))

    def density_ok(self, k: np.ndarray = GRID) -> np.ndarray:
        """Gatheral's butterfly condition g(k) ≥ 0 at each grid point (``True``: no arbitrage there)."""
        d = k - self.m
        root = np.sqrt(d * d + self.s * self.s)
        w = np.maximum(self.a + self.b * (self.rho * d + root), 1e-12)
        w1 = self.b * (self.rho + d / root)
        w2 = self.b * self.s * self.s / root**3
        g = (1 - k * w1 / (2 * w)) ** 2 - (w1 * w1 / 4) * (1 / w + 0.25) + w2 / 2
        return np.asarray(g >= -1e-9)

    def as_dict(self) -> dict[str, float | int | str]:
        return {"expiration": self.expiration.isoformat(), "years": round(self.years, 5), "a": self.a, "b": self.b,
                "rho": self.rho, "m": self.m, "s": self.s, "rmse_vol": round(self.rmse_vol, 5), "points": self.points}  # fmt: skip


@dataclass
class Surface:
    spot: float
    as_of: datetime
    slices: list[SVISlice] = field(default_factory=list)
    butterfly_violations: int = 0
    calendar_violations: int = 0

    @property
    def usable(self) -> bool:
        return bool(self.slices)

    def slice_for(self, expiration: date) -> SVISlice | None:
        return next((s for s in self.slices if s.expiration == expiration), None)

    def total_variance(self, k: float, years: float) -> float | None:
        """Total variance at (k, T): the slice itself, or linear in T between the two around it (flat in
        volatility beyond the ends)."""
        if not self.slices:
            return None
        ts = [s.years for s in self.slices]
        if years <= ts[0]:
            s0 = self.slices[0]
            return float(s0.w(k)) * years / s0.years
        if years >= ts[-1]:
            s1 = self.slices[-1]
            return float(s1.w(k)) * years / s1.years
        i = int(np.searchsorted(ts, years))
        lo, hi = self.slices[i - 1], self.slices[i]
        t = (years - lo.years) / (hi.years - lo.years)
        return float((1 - t) * lo.w(k) + t * hi.w(k))

    def vol(self, k: float, years: float) -> float | None:
        w = self.total_variance(k, years)
        if w is None or years <= 0:
            return None
        return math.sqrt(max(w, 1e-12) / years)

    def vol_at(self, strike: float, expiration: date) -> float | None:
        s = self.slice_for(expiration)
        if s is not None:
            return s.vol_at_strike(strike)
        years = _years(expiration, self.as_of)
        if years <= 0:
            return None
        fwd = self.spot * math.exp(RATE * years)
        return self.vol(math.log(strike / fwd), years)

    def residual(self, q: OptionQuote) -> float | None:
        """The quote's implied volatility minus the surface's, in volatility (decimal) units: positive = rich."""
        iv = q.iv if q.iv and q.iv > 0 else quote_iv(q, self.as_of)
        fair = self.vol_at(q.contract.strike, q.contract.expiration)
        if iv is None or fair is None:
            return None
        return float(iv - fair)

    def features(self, days: int = 30) -> dict[str, float | None]:
        """The surface at a constant maturity: level, skew (dσ/dk), curvature (d²σ/dk²), the one-standard-
        deviation risk reversal and butterfly, and the term slope (60-day minus 30-day ATM volatility)."""
        out: dict[str, float | None] = {"atm_iv": None, "iv_skew": None, "iv_curv": None, "rr": None, "bf": None,
                                        "term_slope": None, "svi_rmse": None, "arb_violations": None}  # fmt: skip
        if not self.slices:
            return out
        t = days / 365.0
        atm = self.vol(0.0, t)
        if atm is None:
            return out
        h = 0.02
        up, dn = self.vol(h, t), self.vol(-h, t)
        sd = atm * math.sqrt(t)
        call_wing, put_wing = self.vol(sd, t), self.vol(-sd, t)
        far = self.vol(0.0, 60 / 365.0)
        out.update(
            atm_iv=atm,
            iv_skew=None if up is None or dn is None else (up - dn) / (2 * h),
            iv_curv=None if up is None or dn is None else (up - 2 * atm + dn) / (h * h),
            rr=None if call_wing is None or put_wing is None else call_wing - put_wing,
            bf=None if call_wing is None or put_wing is None else 0.5 * (call_wing + put_wing) - atm,
            term_slope=None if far is None or self.slices[-1].years < 45 / 365 else far - atm,
            svi_rmse=float(np.mean([s.rmse_vol for s in self.slices])),
            arb_violations=float(self.butterfly_violations + self.calendar_violations),
        )
        return out

    def summary(self) -> dict[str, object]:
        return {"spot": self.spot, "as_of": self.as_of.isoformat(), "slices": [s.as_dict() for s in self.slices],
                "butterfly_violations": self.butterfly_violations, "calendar_violations": self.calendar_violations,
                "features": self.features()}  # fmt: skip


def _years(expiration: date, now: datetime) -> float:
    from quantpulse.options.contracts import expiration_time

    return max((expiration_time(expiration) - now).total_seconds(), 0.0) / (365.0 * 86400)


def fit_slice(
    ks: Sequence[float],
    vols: Sequence[float],
    weights: Sequence[float],
    years: float,
    warm: Sequence[float] | None = None,
) -> tuple[float, float, float, float, float, float] | None:
    """Raw SVI parameters (a, b, ρ, m, s) and the fit's RMSE in volatility units, or ``None`` when the slice
    cannot be fitted (too few points, or no admissible solution). Weighted least squares on total variance with
    the analytic Jacobian, from two standard starts and ``warm`` (the neighbouring slice's solution)."""
    from scipy.optimize import least_squares

    k = np.asarray(ks, dtype=float)
    v = np.asarray(vols, dtype=float)
    wts = np.sqrt(np.asarray(weights, dtype=float))
    ok = np.isfinite(k) & np.isfinite(v) & (v > 0) & np.isfinite(wts)
    k, v, wts = k[ok], v[ok], wts[ok]
    if len(k) < MIN_POINTS or years <= 0:
        return None
    target = v * v * years
    wmax = float(target.max())
    pen = 50.0

    def resid(p: np.ndarray) -> np.ndarray:
        a, b, rho, m, s = p
        d = k - m
        model = a + b * (rho * d + np.sqrt(d * d + s * s))
        floor = a + b * s * math.sqrt(max(1 - rho * rho, 0.0))
        out = np.empty(len(k) + 1)
        out[:-1] = wts * (model - target)
        out[-1] = pen * min(floor, 0.0)
        return out

    def jac(p: np.ndarray) -> np.ndarray:
        a, b, rho, m, s = p
        d = k - m
        r = np.sqrt(d * d + s * s)
        j = np.empty((len(k) + 1, 5))
        j[:-1, 0] = wts
        j[:-1, 1] = wts * (rho * d + r)
        j[:-1, 2] = wts * b * d
        j[:-1, 3] = wts * b * (-rho - d / r)
        j[:-1, 4] = wts * b * s / r
        q = math.sqrt(max(1 - rho * rho, 1e-12))
        if a + b * s * q < 0:
            j[-1] = [pen, pen * s * q, -pen * b * s * rho / q, 0.0, pen * b * q]
        else:
            j[-1] = 0.0
        return j

    lo = [-wmax, 0.0, -0.999, float(k.min()) - 0.5, 1e-4]
    hi = [wmax, 10.0, 0.999, float(k.max()) + 0.5, 2.0]
    starts = [
        [float(target.min()) * 0.8, 0.1, -0.4, 0.0, 0.1],
        [float(target.min()) * 0.8, 0.1, 0.0, float(np.median(k)), 0.2],
    ]
    if warm is not None and len(warm) == 5:
        starts.insert(0, list(warm))
    best: tuple[float, np.ndarray] | None = None
    good = 1e-4 * float(np.sum(wts * wts * target * target))  # a fit this close needs no other start
    for guess in starts:
        x0 = np.clip(guess, lo, hi)
        try:
            r = least_squares(resid, x0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=80,
                              ftol=1e-9, xtol=1e-9)  # fmt: skip
        except (ValueError, FloatingPointError):
            continue
        if best is None or float(r.cost) < best[0]:
            best = (float(r.cost), r.x)
        if best[0] <= good:
            break
    if best is None:
        return None
    a, b, rho, m, s = (float(x) for x in best[1])
    if a + b * s * math.sqrt(max(1 - rho * rho, 0.0)) < -1e-6:
        return None
    d = k - m
    fitted = np.sqrt(np.maximum(a + b * (rho * d + np.sqrt(d * d + s * s)), 1e-12) / years)
    rmse = float(np.sqrt(np.mean((fitted - v) ** 2)))
    return a, b, rho, m, s, rmse


def fit_surface(quotes: Sequence[OptionQuote], spot: float, now: datetime, *, rate: float = RATE,
                max_slices: int = 12) -> Surface:  # fmt: skip
    """The SVI surface of a chain: one slice per expiration with enough two-sided out-of-the-money quotes."""
    surf = Surface(spot, now)
    if not spot or spot <= 0:
        return surf
    by_exp: dict[date, list[OptionQuote]] = {}
    for q in quotes:
        dte = q.contract.dte(now)
        if MIN_DTE <= dte <= MAX_DTE and q.two_sided:
            by_exp.setdefault(q.contract.expiration, []).append(q)
    for exp in sorted(by_exp)[:max_slices]:
        years = _years(exp, now)
        if years <= 0:
            continue
        fwd = spot * math.exp(rate * years)
        ks, vols, wts = [], [], []
        for q in by_exp[exp]:
            c = q.contract
            if (c.is_call and c.strike < fwd) or (not c.is_call and c.strike >= fwd):
                continue  # out of the money only: the liquid side, and no put–call duplication
            iv = q.iv if q.iv and q.iv > 0 else quote_iv(q, now, rate)
            mid = q.mid
            if iv is None or not (0.01 < iv < 5.0) or not mid:
                continue
            k = math.log(c.strike / fwd)
            if abs(k) > 1.0:
                continue
            rel = (q.ask - q.bid) / mid  # type: ignore[operator]
            ks.append(k)
            vols.append(iv)
            wts.append(1.0 / max(rel, 0.02))
        prev = surf.slices[-1] if surf.slices else None
        warm = None if prev is None else [prev.a * years / prev.years, prev.b, prev.rho, prev.m, prev.s]
        fit = fit_slice(ks, vols, wts, years, warm)
        if fit is None:
            continue
        a, b, rho, m, s, rmse = fit
        sl = SVISlice(exp, years, fwd, a, b, rho, m, s, rmse, len(ks))
        surf.slices.append(sl)
        surf.butterfly_violations += int((~sl.density_ok()).sum() > 0)
    for lo, hi in zip(surf.slices, surf.slices[1:], strict=False):
        if np.any(np.asarray(hi.w(GRID)) < np.asarray(lo.w(GRID)) - 1e-6):
            surf.calendar_violations += 1
    return surf
