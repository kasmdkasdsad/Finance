"""Option and multi-leg orders through the real alpaca-py SDK against the fake paper API: whole contracts,
limit orders only, position intents, net debit/credit prices, legs reported back, no naked short leg can
even be expressed, and expiration settles like the OCC (never an automatic exercise by QuantPulse)."""

from datetime import date

import pytest

from quantpulse.providers.alpaca_trading import (
    AlpacaPaperBroker,
    OrderLeg,
    OrderRejected,
    OrderSpec,
    mleg_symbol,
    naked_short_legs,
    order_request,
)
from tests.fakes.alpaca_paper import FakeAlpacaPaper

KEY, SECRET = "PKFAKEKEYID", "fake-secret-value"
C200 = "AAPL261016C00200000"
C210 = "AAPL261016C00210000"
P190 = "AAPL261016P00190000"
P180 = "AAPL261016P00180000"


@pytest.fixture
def fake():
    f = FakeAlpacaPaper(equity=100_000.0)
    f.prices.update({"AAPL": 200.0, C200: 6.0, C210: 2.5, P190: 3.0, P180: 1.2})
    return f


@pytest.fixture
def broker(fake):
    return AlpacaPaperBroker(KEY, SECRET, transport=fake)


def leg(symbol, side, intent, ratio=1):
    return OrderLeg(symbol=symbol, side=side, ratio_qty=ratio, position_intent=intent)


def test_option_order_specs_fail_closed():
    ok = OrderSpec(C200, "buy", 2, "limit", "qp-x-1", limit_price=6.1, asset_class="us_option",
                   position_intent="buy_to_open")  # fmt: skip
    assert ok.qty == 2.0
    bad = [
        dict(order_type="market", limit_price=None),  # never a market order for options
        dict(qty=1.5),  # whole contracts
        dict(qty=None, notional=500.0),  # never sized in dollars
        dict(position_intent=None),  # the intent must be explicit
        dict(position_intent="sell_to_open"),  # a buy cannot sell to open
        dict(limit_price=-1.0),
    ]
    base = dict(symbol=C200, side="buy", qty=2, order_type="limit", client_order_id="qp-x-1",
                limit_price=6.1, asset_class="us_option", position_intent="buy_to_open")  # fmt: skip
    for change in bad:
        with pytest.raises(ValueError):
            OrderSpec(**{**base, **change})
    with pytest.raises(ValueError, match="option contract"):
        OrderSpec("AAPL", "buy", 1, "limit", "qp-x-2", limit_price=1.0, asset_class="us_option",
                  position_intent="buy_to_open")  # fmt: skip
    with pytest.raises(ValueError, match="send it as an option order"):
        OrderSpec(C200, "buy", 1, "limit", "qp-x-3", limit_price=6.0)  # an OCC symbol as a stock


def test_multi_leg_specs_refuse_naked_short_legs():
    spread = (leg(C200, "buy", "buy_to_open"), leg(C210, "sell", "sell_to_open"))
    assert naked_short_legs(spread) == []
    ratio = (leg(C200, "buy", "buy_to_open"), leg(C210, "sell", "sell_to_open", ratio=2))
    assert naked_short_legs(ratio) == ["call"]
    with pytest.raises(ValueError, match="naked short calls"):
        OrderSpec(
            "AAPL:MLEG", "buy", 1, "limit", "qp-x-4", limit_price=1.0, asset_class="us_option", legs=ratio
        )
    short_put_long_call = (leg(C200, "buy", "buy_to_open"), leg(P190, "sell", "sell_to_open"))
    with pytest.raises(ValueError, match="naked short puts"):
        OrderSpec("AAPL:MLEG", "buy", 1, "limit", "qp-x-5", limit_price=1.0, asset_class="us_option",
                  legs=short_put_long_call)  # fmt: skip
    with pytest.raises(ValueError, match="different contracts"):
        OrderSpec("AAPL:MLEG", "buy", 1, "limit", "qp-x-6", limit_price=1.0, asset_class="us_option",
                  legs=(leg(C200, "buy", "buy_to_open"), leg(C200, "sell", "sell_to_close")))  # fmt: skip
    with pytest.raises(ValueError):
        leg(C200, "buy", "sell_to_open")
    with pytest.raises(ValueError):
        leg("AAPL", "buy", "buy_to_open")
    assert mleg_symbol([C200, C210]) == "AAPL:MLEG"


def test_the_sdk_request_carries_legs_intents_and_a_signed_net_price():
    spec = OrderSpec("AAPL:MLEG", "sell", 3, "limit", "qp-x-7", limit_price=-1.25, asset_class="us_option",
                     legs=(leg(P190, "sell", "sell_to_open"), leg(P180, "buy", "buy_to_open")))  # fmt: skip
    body = order_request(spec).to_request_fields()
    assert body["order_class"] == "mleg" and body["qty"] == 3 and body["limit_price"] == -1.25
    assert "symbol" not in body and "side" not in body
    assert [(x["symbol"], x["position_intent"], x["ratio_qty"]) for x in body["legs"]] == [
        (P190, "sell_to_open", 1),
        (P180, "buy_to_open", 1),
    ]


async def test_single_leg_round_trip(broker, fake):
    spec = OrderSpec(C200, "buy", 2, "limit", "qp-brain-t-1", limit_price=6.1, asset_class="us_option",
                     position_intent="buy_to_open")  # fmt: skip
    o = await broker.submit(spec)
    assert o.status == "filled" and o.is_option and o.asset_class == "us_option"
    assert o.position_intent == "buy_to_open" and o.filled_avg_price == pytest.approx(6.1)
    assert fake.bodies[-1]["position_intent"] == "buy_to_open" and fake.bodies[-1]["type"] == "limit"
    assert fake.cash == pytest.approx(100_000 - 2 * 6.1 * 100)
    pos = {p.symbol: p for p in await broker.positions()}
    assert (
        pos[C200].is_option and pos[C200].qty == 2 and pos[C200].market_value == pytest.approx(2 * 6.0 * 100)
    )


async def test_multi_leg_round_trip_reports_legs_and_a_record_symbol(broker, fake):
    spec = OrderSpec("AAPL:MLEG", "buy", 2, "limit", "qp-brain-t-2", limit_price=3.4, asset_class="us_option",
                     legs=(leg(C200, "buy", "buy_to_open"), leg(C210, "sell", "sell_to_open")))  # fmt: skip
    o = await broker.submit(spec)
    assert o.order_class == "mleg" and o.symbol == "AAPL:MLEG" and o.side == "buy" and o.status == "filled"
    assert {x.symbol for x in o.legs} == {C200, C210} and o.symbols == (C200, C210)
    assert o.filled_avg_price == pytest.approx(3.4)  # the net debit, as limited
    assert fake.cash == pytest.approx(100_000 - 2 * 3.4 * 100)
    pos = {p.symbol: p for p in await broker.positions()}
    assert pos[C200].qty == 2 and pos[C210].qty == -2 and pos[C210].side == "short"
    listed = await broker.orders("all")
    assert [x.client_order_id for x in listed] == ["qp-brain-t-2"]  # legs never appear as orders of their own
    again = await broker.order_by_client_id("qp-brain-t-2")
    assert again is not None and len(again.legs) == 2
    # a credit spread: the net price is negative
    credit = OrderSpec("AAPL:MLEG", "sell", 1, "limit", "qp-brain-t-3", limit_price=-1.7,
                       asset_class="us_option",
                       legs=(leg(P190, "sell", "sell_to_open"), leg(P180, "buy", "buy_to_open")))  # fmt: skip
    c = await broker.submit(credit)
    assert c.side == "sell" and c.filled_avg_price == pytest.approx(-1.7)


async def test_the_fake_broker_enforces_coverage_and_levels(broker, fake):
    naked = OrderSpec(C210, "sell", 1, "limit", "qp-t-4", limit_price=2.4, asset_class="us_option",
                      position_intent="sell_to_open")  # fmt: skip
    with pytest.raises(OrderRejected, match="uncovered"):
        await broker.submit(naked)
    fake.hold("AAPL", 100, 190.0, 200.0)
    covered = await broker.submit(naked)
    assert covered.status == "filled"
    fake.options_level = 2
    with pytest.raises(OrderRejected, match="not approved"):
        await broker.submit(OrderSpec("AAPL:MLEG", "buy", 1, "limit", "qp-t-5", limit_price=3.4,
                                      asset_class="us_option",
                                      legs=(leg(C200, "buy", "buy_to_open"), leg(C210, "sell", "sell_to_open"))))  # fmt: skip
    with pytest.raises(OrderRejected, match="insufficient qty"):
        await broker.submit(OrderSpec(C200, "sell", 1, "limit", "qp-t-6", limit_price=6.0,
                                      asset_class="us_option", position_intent="sell_to_close"))  # fmt: skip


async def test_expiration_settles_like_the_occ(broker, fake):
    fake.hold(C200, 1, 5.0)
    fake.hold(C210, -1, 2.0)
    fake.hold(P180, 1, 1.0)
    cash = fake.cash
    events = {e["symbol"]: e for e in fake.expire(date(2026, 10, 16), {"AAPL": 205.0})}
    assert events[C200]["kind"] == "exercised" and events[C200]["shares"] == 100
    assert events[C210]["kind"] == "expired_worthless" and events[P180]["kind"] == "expired_worthless"
    pos = {p.symbol: p for p in await broker.positions()}
    assert set(pos) == {"AAPL"} and pos["AAPL"].qty == 100 and not pos["AAPL"].is_option
    assert fake.cash == pytest.approx(cash - 200.0 * 100)


async def test_close_position_buys_back_a_short_option(broker, fake):
    fake.hold("AAPL", 100, 190.0, 200.0)
    fake.hold(C210, -1, 2.0, 2.5)
    o = await broker.close_position(C210)
    assert o.side == "buy" and o.position_intent == "buy_to_close"
    assert C210 not in fake.positions


async def test_account_reports_options_level_and_buying_power(broker, fake):
    a = await broker.account()
    assert a.options_trading_level == 3 and a.options_buying_power == pytest.approx(100_000)
