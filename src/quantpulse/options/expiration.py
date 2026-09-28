"""The expiration state machine — no option is ever carried through expiration without explicit handling.

States::

    OPEN → NEAR_EXPIRATION (≤ near_days) → EXPIRATION_RISK (≤ risk_days, or in/near the money late)
         → EXPIRING_TODAY → EXPIRED | EXERCISED | ASSIGNED        (terminal)
    any open state → CLOSED                                       (terminal: we closed it)

QuantPulse never exercises an option by choice and never lets one expire by accident: from
``EXPIRATION_RISK`` the required action is to close (or roll) before the cut-off, and the position manager
treats it as mandatory. What happens at expiration regardless is modelled explicitly in
:func:`settlement` (OCC exercises long options in the money by $0.01 or more; short ones are then assigned).
Early assignment risk — a short call in the money before an ex-dividend date with little time value left, a
deep in-the-money short put with none — is assessed in :func:`assignment_risk`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.options.contracts import OptionContract, expiration_time


class ExpiryState(StrEnum):
    OPEN = "OPEN"
    NEAR_EXPIRATION = "NEAR_EXPIRATION"
    EXPIRATION_RISK = "EXPIRATION_RISK"
    EXPIRING_TODAY = "EXPIRING_TODAY"
    EXPIRED = "EXPIRED"
    EXERCISED = "EXERCISED"
    ASSIGNED = "ASSIGNED"
    CLOSED = "CLOSED"


TERMINAL = frozenset({ExpiryState.EXPIRED, ExpiryState.EXERCISED, ExpiryState.ASSIGNED, ExpiryState.CLOSED})
ORDER = [
    ExpiryState.OPEN,
    ExpiryState.NEAR_EXPIRATION,
    ExpiryState.EXPIRATION_RISK,
    ExpiryState.EXPIRING_TODAY,
]
ALLOWED: dict[ExpiryState, frozenset[ExpiryState]] = {
    ExpiryState.OPEN: frozenset({ExpiryState.NEAR_EXPIRATION, ExpiryState.EXPIRATION_RISK, ExpiryState.EXPIRING_TODAY,
                                 ExpiryState.CLOSED, ExpiryState.ASSIGNED, ExpiryState.EXPIRED, ExpiryState.EXERCISED}),
    ExpiryState.NEAR_EXPIRATION: frozenset({ExpiryState.EXPIRATION_RISK, ExpiryState.EXPIRING_TODAY, ExpiryState.CLOSED,
                                            ExpiryState.ASSIGNED, ExpiryState.EXPIRED, ExpiryState.EXERCISED}),
    ExpiryState.EXPIRATION_RISK: frozenset({ExpiryState.EXPIRING_TODAY, ExpiryState.CLOSED, ExpiryState.ASSIGNED,
                                            ExpiryState.EXPIRED, ExpiryState.EXERCISED}),
    ExpiryState.EXPIRING_TODAY: frozenset({ExpiryState.CLOSED, ExpiryState.EXPIRED, ExpiryState.EXERCISED,
                                           ExpiryState.ASSIGNED}),
    **{s: frozenset() for s in TERMINAL},
}  # fmt: skip


class TransitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExpiryRules:
    near_days: int = 7
    risk_days: int = 2
    near_money_pct: float = 0.02  # within 2% of a strike counts as "near the money" late in life
    close_by_minutes_before: int = 30  # the latest moment to close on the expiration day


@dataclass(frozen=True, slots=True)
class ExpiryAssessment:
    state: ExpiryState
    dte: int
    must_close: bool
    reason: str
    close_by: datetime | None

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "dte": self.dte, "must_close": self.must_close, "reason": self.reason,
                "close_by": self.close_by.isoformat() if self.close_by else None}  # fmt: skip


def assess(
    contracts: list[OptionContract], spot: float, now: datetime, rules: ExpiryRules | None = None
) -> ExpiryAssessment:
    """The state of a position (its nearest expiration governs) and whether it must be closed now."""
    from datetime import timedelta

    r = rules or ExpiryRules()
    if not contracts:
        return ExpiryAssessment(ExpiryState.OPEN, 10_000, False, "no option legs", None)
    first = min(contracts, key=lambda c: c.expiration)
    exp_at = expiration_time(first.expiration)
    close_by = exp_at - timedelta(minutes=r.close_by_minutes_before)
    dte = (first.expiration - now.astimezone(NEW_YORK).date()).days
    near_money = any(abs(c.strike / spot - 1) <= r.near_money_pct or c.intrinsic(spot) > 0
                     for c in contracts if c.expiration == first.expiration)  # fmt: skip
    if now >= exp_at:
        return ExpiryAssessment(
            ExpiryState.EXPIRED, dte, False, "past the last trading moment: settle it", None
        )
    if dte <= 0:
        return ExpiryAssessment(ExpiryState.EXPIRING_TODAY, dte, True,
                                f"expires today: close before {close_by.astimezone(NEW_YORK):%H:%M} New York", close_by)  # fmt: skip
    if dte <= r.risk_days or (dte <= r.near_days and near_money and dte <= r.risk_days + 1):
        return ExpiryAssessment(ExpiryState.EXPIRATION_RISK, dte, True,
                                f"{dte} day(s) to expiration{' and near/in the money' if near_money else ''}: close or roll",
                                close_by)  # fmt: skip
    if dte <= r.near_days:
        return ExpiryAssessment(
            ExpiryState.NEAR_EXPIRATION, dte, False, f"{dte} days to expiration: watch", close_by
        )
    return ExpiryAssessment(ExpiryState.OPEN, dte, False, f"{dte} days to expiration", close_by)


def transition(current: ExpiryState, new: ExpiryState) -> ExpiryState:
    """Move to ``new`` if the machine allows it (staying put is always allowed)."""
    if new == current:
        return current
    if new not in ALLOWED[current]:
        raise TransitionError(f"{current} → {new} is not a valid expiration transition")
    return new


def settlement(c: OptionContract, side: str, contracts: int, close_price: float) -> dict[str, Any]:
    """What expiration does to an option still held at the close: OCC exercises a long option in the money by
    $0.01 or more (and the short side is assigned); anything else expires worthless. The resulting share
    delivery is spelled out — QuantPulse's policy is never to get here, but if it does it is not a surprise."""
    itm = c.intrinsic(close_price)
    exercised = itm >= 0.01
    shares = contracts * c.multiplier
    if not exercised:
        state = ExpiryState.EXPIRED
        delivery = 0
    elif side == "long":
        state = ExpiryState.EXERCISED
        delivery = shares if c.is_call else -shares  # a long call buys shares; a long put sells them
    else:
        state = ExpiryState.ASSIGNED
        delivery = -shares if c.is_call else shares  # a short call delivers shares; a short put takes them
    return {
        "symbol": c.symbol,
        "state": state.value,
        "intrinsic": round(itm, 4),
        "share_delivery": delivery,
        "cash_flow": round(
            -delivery * c.strike, 2
        ),  # buying shares at the strike costs cash, selling raises it
    }


def assignment_risk(
    c: OptionContract,
    spot: float,
    option_price: float | None,
    now: datetime,
    *,
    ex_dividend: date | None = None,
    dividend: float | None = None,
) -> dict[str, Any]:
    """Early-assignment risk of a *short* American option (for a long one there is none)."""
    itm = c.intrinsic(spot)
    extrinsic = max((option_price or 0.0) - itm, 0.0) if option_price is not None else None
    reasons: list[str] = []
    level = "low"
    if itm <= 0:
        return {"level": "none", "reasons": ["out of the money"], "extrinsic": extrinsic}
    before_dividend = ex_dividend is not None and dividend and now.date() < ex_dividend <= c.expiration
    if c.is_call and before_dividend and extrinsic is not None and dividend and extrinsic < dividend:
        level = "high"
        reasons.append(f"in the money before the ex-dividend date {ex_dividend} with time value "
                           f"${extrinsic:.2f} below the ${dividend:.2f} dividend")  # fmt: skip
    if not c.is_call and extrinsic is not None and extrinsic < 0.05 and itm / c.strike > 0.05:
        level = "high"
        reasons.append(f"deep in the money put with almost no time value (${extrinsic:.2f})")
    if level == "low" and extrinsic is not None and extrinsic < 0.10:
        level = "medium"
        reasons.append(f"in the money with little time value (${extrinsic:.2f})")
    if not reasons:
        reasons.append("in the money; time value still protects against early exercise")
    return {"level": level, "reasons": reasons, "extrinsic": extrinsic}
