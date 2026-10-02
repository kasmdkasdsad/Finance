"""How an option order fills — five explicit assumptions, always named, never mixed up.

==============  ==========================================================================================
OPTIMISTIC      every leg at the mid, no fees, nothing missed — the most flattering case (an upper bound)
MIDPOINT        every leg at the mid, fees paid — labelled: a mid fill is an assumption, not a fact
REALISTIC       every leg a quarter of the spread worse than the mid (half the half-spread), fees, 2% missed
PESSIMISTIC     every leg at the far side (buy at the ask, sell at the bid), fees, 5% missed
STRESS          spreads twice as wide and crossed in full, fees × 1.5, 15% missed, exits a day late
==============  ==========================================================================================

A backtest is run under all five; a strategy whose edge exists only under OPTIMISTIC or MIDPOINT has no
edge. Live paper orders are limit orders priced at the REALISTIC level: the price QuantPulse is willing
to pay, never a market order into a wide option market.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from quantpulse.options.quotes import OptionQuote


class ExecutionModel(StrEnum):
    OPTIMISTIC = "OPTIMISTIC"
    MIDPOINT = "MIDPOINT"
    REALISTIC = "REALISTIC"
    PESSIMISTIC = "PESSIMISTIC"
    STRESS = "STRESS"


@dataclass(frozen=True, slots=True)
class FillAssumption:
    spread_fraction: float  # of the half-spread paid beyond the mid on every leg
    spread_multiplier: float  # spreads widened by this much first (stress)
    fee_multiplier: float
    miss_rate: float  # share of orders that never fill
    exit_delay_days: int
    label: str


ASSUMPTIONS: dict[ExecutionModel, FillAssumption] = {
    ExecutionModel.OPTIMISTIC: FillAssumption(0.0, 1.0, 0.0, 0.0, 0, "mid fills, no fees, nothing missed (flattering upper bound)"),
    ExecutionModel.MIDPOINT: FillAssumption(0.0, 1.0, 1.0, 0.0, 0, "mid fills with fees (an assumption, not market evidence)"),
    ExecutionModel.REALISTIC: FillAssumption(0.5, 1.0, 1.0, 0.02, 0, "a quarter of the spread paid per leg, fees, 2% missed"),
    ExecutionModel.PESSIMISTIC: FillAssumption(1.0, 1.0, 1.0, 0.05, 0, "full spread crossed on every leg, fees, 5% missed"),
    ExecutionModel.STRESS: FillAssumption(1.0, 2.0, 1.5, 0.15, 1, "spreads doubled and crossed, fees ×1.5, 15% missed, exits a day late"),
}  # fmt: skip


def leg_fill(side_sign: int, bid: float, ask: float, model: ExecutionModel) -> float:
    """The per-share fill price of one leg: a buy (+1) above the mid, a sell (−1) below it."""
    a = ASSUMPTIONS[model]
    mid = 0.5 * (bid + ask)
    half = 0.5 * (ask - bid) * a.spread_multiplier
    return max(0.0, mid + side_sign * a.spread_fraction * half)


def quote_fill(side_sign: int, q: OptionQuote, model: ExecutionModel) -> float | None:
    if not q.two_sided:
        return None
    return leg_fill(side_sign, q.bid, q.ask, model)  # type: ignore[arg-type]


def fees(contracts: int, per_contract: float, model: ExecutionModel) -> float:
    return contracts * per_contract * ASSUMPTIONS[model].fee_multiplier


def missed(model: ExecutionModel, rng: random.Random) -> bool:
    return rng.random() < ASSUMPTIONS[model].miss_rate


def limit_price(
    side_sign: int,
    quotes: list[tuple[int, int, OptionQuote]],
    model: ExecutionModel = ExecutionModel.REALISTIC,
) -> float | None:
    """The net limit price per share for a structure (``[(side_sign, ratio, quote), …]``): positive is a
    debit paid, negative a credit received — each leg priced under ``model``, rounded to the cent."""
    total = 0.0
    for sign, ratio, q in quotes:
        px = quote_fill(sign, q, model)
        if px is None:
            return None
        total += sign * ratio * px
    return round(total, 2)
