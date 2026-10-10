"""Triple-barrier outcomes: what a candidate actually returned per dollar at risk under one standard exit policy.

A strategy's own exits differ from genome to genome; to learn *which candidates are good* the model needs one
yardstick for all of them. Each candidate is opened at the REALISTIC fill (a quarter of the spread paid on every
leg, fees), marked at the mid each following day, and closed at the first of three barriers (López de Prado):

* **take profit** — the mark-to-mid P&L reaches ``take_profit`` × the reference (the debit paid, or the credit
  received);
* **stop** — it falls to −``stop_loss`` × the reference;
* **time** — ``horizon`` trading days have passed, or ``exit_dte`` days are left before the first expiration.

The exit is filled at REALISTIC prices again (longs sold below the mid, shorts bought back above it); a leg past
its expiration is worth its intrinsic value. The label is that net P&L divided by the structure's maximum loss
(spread included) — the same unit as the rule's expected value per dollar at risk. Each label keeps its time
span ``[t0, t1]``, which the purged cross-validation needs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from quantpulse.options.fills import ExecutionModel, quote_fill
from quantpulse.options.lab.chains import ChainSource
from quantpulse.options.quotes import OptionQuote
from quantpulse.options.selection import Candidate


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    horizon: int = 10  # trading days
    take_profit: float = 0.5
    stop_loss: float = 1.0  # debit structures; credit structures use ``credit_stop``
    credit_stop: float = 2.0
    exit_dte: int = 3
    model: ExecutionModel = ExecutionModel.REALISTIC
    fee_per_contract: float = 0.05


@dataclass(frozen=True, slots=True)
class Label:
    t0: date
    t1: date
    ror: float  # net P&L per dollar of maximum loss
    pnl: float  # dollars per unit
    barrier: str  # take_profit | stop | time | expiry | no_data
    days_held: int

    @property
    def win(self) -> bool:
        return self.ror > 0


def _leg_value(sign: int, units: int, q: OptionQuote | None, intrinsic: float, expired: bool, last: float
               ) -> tuple[float, float]:  # fmt: skip
    """(mark at the mid, value if closed now at REALISTIC prices) for one leg, per unit."""
    if expired:
        v = sign * units * intrinsic
        return v, v
    if q is not None and q.two_sided:
        mid = q.mid or 0.0
        fill = quote_fill(-sign, q, ExecutionModel.REALISTIC)  # closing: sell a long, buy back a short
        return sign * units * mid, sign * units * (fill if fill is not None else mid)
    if q is not None and q.ask is not None and sign < 0:
        return sign * units * float(q.ask), sign * units * float(q.ask)
    return sign * units * last, sign * units * (0.0 if sign > 0 else last)


def triple_barrier(
    cand: Candidate,
    source: ChainSource,
    underlying: str,
    entry_day: date,
    days: Sequence[date],
    policy: ExitPolicy = ExitPolicy(),
) -> Label | None:
    """The outcome of ``cand`` opened at the close of ``entry_day`` (``days``: the underlying's trading days,
    sorted). ``None`` when it cannot be opened (a leg without a two-sided market) or nothing follows."""
    st = cand.structure
    legs = st.option_legs
    if not legs or len(cand.quotes) != len(legs):
        return None
    entry = 0.0
    for leg, q in zip(legs, cand.quotes, strict=True):
        fill = quote_fill(leg.sign, q, policy.model)
        if fill is None:
            return None
        entry += leg.sign * leg.units * fill
    risk = cand.metrics.get("max_loss")
    if not risk or not math.isfinite(risk) or risk <= 0:
        return None
    contracts = sum(leg.ratio for leg in legs)
    fees = 2 * contracts * policy.fee_per_contract
    ref = abs(entry) if abs(entry) > 1e-9 else risk
    stop = policy.stop_loss if entry > 0 else policy.credit_stop
    first_exp = st.first_expiration
    last_mid = {leg.contract.symbol: (q.mid or 0.0) for leg, q in zip(legs, cand.quotes, strict=True)
                if leg.contract is not None}  # fmt: skip
    try:
        i0 = days.index(entry_day)
    except ValueError:
        return None
    future = list(days[i0 + 1 : i0 + 1 + policy.horizon])
    if not future:
        return None
    complete = (
        len(future) == policy.horizon
    )  # at the end of the data only a barrier, never the clock, may close
    held = 0
    for d in future:
        held += 1
        spot = source.spot(underlying, d)
        if spot is None:
            continue
        mark = close = 0.0
        for leg in legs:
            c = leg.contract
            assert c is not None
            lq = source.quote(c.symbol, underlying, d)
            if lq is not None and lq.mid:
                last_mid[c.symbol] = lq.mid
            m, v = _leg_value(
                leg.sign, leg.units, lq, c.intrinsic(spot), d >= c.expiration, last_mid[c.symbol]
            )
            mark += m
            close += v
        pnl_mid = mark - entry
        barrier = None
        if pnl_mid >= policy.take_profit * ref:
            barrier = "take_profit"
        elif pnl_mid <= -stop * ref:
            barrier = "stop"
        elif first_exp is not None and (first_exp - d).days <= policy.exit_dte:
            barrier = "expiry"
        elif d == future[-1] and complete:
            barrier = "time"
        if barrier is not None:
            pnl = close - entry - fees
            return Label(entry_day, d, max(pnl / risk, -1.5), pnl, barrier, held)
    return None
