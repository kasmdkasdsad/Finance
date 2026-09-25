"""Risk-free yield-curve utilities.

U.S. Treasury par yields are quoted as bond-equivalent (semi-annually compounded) annual percentages.
Option pricing needs continuously compounded rates, so :func:`bey_to_continuous` converts
``r_c = 2 * ln(1 + y / 2)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from quantpulse.core.errors import DomainError

_TENOR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mo|month|months|yr|year|years|wk|week|weeks)\s*$", re.I)


def parse_tenor_label(label: str) -> float:
    """Convert Treasury column labels such as ``'1 Mo'``, ``'1.5 Month'``, ``'10 Yr'`` to years."""
    match = _TENOR_RE.match(label)
    if not match:
        raise ValueError(f"unrecognised tenor label: {label!r}")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("mo"):
        return amount / 12.0
    if unit.startswith("w"):
        return amount * 7.0 / 365.0
    return amount


def bey_to_continuous(rate: float) -> float:
    """Bond-equivalent (semi-annual) yield -> continuously compounded rate (both decimals)."""
    if rate <= -2.0:
        raise DomainError("rate must be greater than -200%")
    return 2.0 * math.log1p(rate / 2.0)


def continuous_to_bey(rate: float) -> float:
    return 2.0 * math.expm1(rate / 2.0)


def interpolate_rate(tenors: Sequence[float], rates: Sequence[float], t: float) -> float:
    """Linear interpolation in tenor with flat extrapolation at both ends."""
    if len(tenors) == 0 or len(tenors) != len(rates):
        raise DomainError("curve requires matching, non-empty tenors and rates")
    x = np.asarray(tenors, dtype=float)
    y = np.asarray(rates, dtype=float)
    order = np.argsort(x)
    return float(np.interp(t, x[order], y[order]))


def continuous_rate_at(tenors: Sequence[float], bey_rates: Sequence[float], t: float) -> float:
    """Continuously compounded zero-ish rate for maturity ``t`` from a par BEY curve (approximation:
    par yields are used directly as zero yields, which is accurate for the short maturities that
    dominate listed options)."""
    return bey_to_continuous(interpolate_rate(tenors, bey_rates, t))


def discount_factor(rate_continuous: float, t: float) -> float:
    return math.exp(-rate_continuous * t)
