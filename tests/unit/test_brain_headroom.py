"""The Brain sizes its buys within what the risk engine will allow, so it stops proposing buys that are bound
to be refused every cycle: long exposure under the 95% cap (pending buys included), room under the position
cap, the cash left once the working buys are paid for, and never a second order while one is still working.
A refusal that does happen names the checks that failed."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from quantpulse.brain.decisions import Headroom, Proposal, plan
from quantpulse.brain.execution import refused
from quantpulse.brain.types import Action
from quantpulse.providers.alpaca_trading import BrokerOrder
from quantpulse.services.trading_risk import RiskBook
from tests.unit.test_brain_agents import path
from tests.unit.test_brain_research import account, bullish, planning_ctx

EQUITY = 100_000.0
HELD_PRICE = float(path(0.0005, 0.01, 13)[-1])


def invested(fraction: float, open_orders=()):
    """A book ``fraction`` invested in HOLD, the rest cash."""
    value = fraction * EQUITY
    qty = round(value / HELD_PRICE)
    ctx = planning_ctx(held={"HOLD": (qty, HELD_PRICE)})
    held = qty * HELD_PRICE
    ctx.portfolio.account = replace(account(EQUITY, cash=EQUITY - held), long_market_value=held)
    ctx.portfolio.open_orders = list(open_orders)
    return ctx


def planned(ctx):
    views = {"NEW": bullish("NEW"), "HOLD": bullish("HOLD")}
    out = plan(ctx, views, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15)
    return {p.subject: p for p in out}


def working_buy(symbol: str, qty: float, price: float) -> BrokerOrder:
    now = datetime(2026, 10, 2, 13, 55, tzinfo=UTC)
    return BrokerOrder("id-1", f"qp-brain-x-{symbol}-b", symbol, "buy", "limit", "day", "new", qty, None, 0.0,
                       None, price, now, now, now, None, None, None, None)  # fmt: skip


def test_a_new_buy_is_sized_within_the_exposure_cap_and_the_risk_engine_accepts_it():
    ctx = invested(0.93)  # 7% cash: sized to the cash it would take exposure to ~98%, over the 95% cap
    p = planned(ctx)["NEW"]
    assert p.action is Action.BUY
    notional = p.quantity * p.est_price
    assert 0 < notional <= 0.95 * EQUITY - ctx.portfolio.account.long_market_value
    book = RiskBook(ctx.limits, ctx.portfolio.account, ctx.portfolio.positions, [], True, False, {})
    after = book.exposure + notional
    assert after <= ctx.limits.max_total_exposure_pct * EQUITY  # the total_exposure check passes


def test_a_full_book_watches_instead_of_proposing_a_refused_buy():
    p = planned(invested(0.95))["NEW"]
    assert p.action is Action.WATCH and "exposure limit" in p.reasons[0]


def test_no_second_order_while_one_is_still_working():
    price = float(path(0.0008, 0.012, 12)[-1])
    free = planned(invested(0.02))
    assert free["NEW"].action is Action.BUY and free["HOLD"].action is Action.INCREASE  # without the orders
    ctx = invested(0.02, open_orders=[working_buy("NEW", 10, price), working_buy("HOLD", 5, HELD_PRICE)])
    out = planned(ctx)
    assert out["NEW"].action is Action.WATCH and "still working" in out["NEW"].reasons[0]
    assert out["HOLD"].action is Action.HOLD and "still working" in out["HOLD"].reasons[0]


def test_working_buys_use_up_cash_and_exposure():
    ctx = invested(0.80, open_orders=[working_buy("OTHER", 100, 100.0)])  # $10,000 still to be paid
    a = ctx.portfolio.account
    room = Headroom.of(ctx)
    assert room.working == {"OTHER"}
    assert room.cash == pytest.approx(a.cash - 10_000 - 0.02 * EQUITY)  # the 2% reserve stays
    assert room.exposure == pytest.approx((0.95 * EQUITY - a.long_market_value - 10_000) * Headroom.MARGIN)


def test_a_position_near_its_cap_is_topped_up_only_to_the_cap():
    ctx = invested(0.29)  # HOLD is 29% of equity: 1 point under the 30% position cap
    room = Headroom.of(ctx)
    assert room.for_symbol("HOLD") <= 0.01 * EQUITY
    assert room.for_symbol("NEW") > room.for_symbol("HOLD")


def test_a_refusal_names_the_checks_that_failed():
    p = Proposal(subject="NEW", action=Action.BUY, confidence=0.5, reasons=["bullish"])
    p.risk = {
        "approved": False,
        "summary": "total_exposure: 98.1% long exposure after the order (limit 95%)",
        "checks": [
            {"name": "market_open", "passed": True, "detail": "market open"},
            {
                "name": "total_exposure",
                "passed": False,
                "detail": "98.1% long exposure after the order (limit 95%)",
            },
            {"name": "liquidity", "passed": False, "detail": "spread 45bp > 30bp"},
        ],
    }
    assert refused(p) == (
        "the risk preview did not approve it: total exposure: 98.1% long exposure after the order (limit 95%); "
        "liquidity: spread 45bp > 30bp"
    )
    p.risk = {"approved": False, "summary": "the paper account could not be read: no risk check possible"}
    assert refused(p).endswith("the paper account could not be read: no risk check possible")
    p.risk = None
    assert refused(p).endswith("no reason was recorded")
