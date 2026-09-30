"""Option orders end to end through the trading service — the same reconciliation, risk book, last gate and
order manager as every stock order — against the real alpaca-py SDK, the fake Alpaca paper API and a fake
options market (never the network, never a real account)."""

import re
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from quantpulse.core.clock import FakeClock
from quantpulse.db.models import BrokerOrderRow
from quantpulse.services.trading_options import closing_intents
from quantpulse.services.trading_risk import OptionLegIntent, OptionOrderIntent
from tests.fakes.options_market import FakeOptionsMarket

from .conftest import NOW
from .test_brain_cycle import brain_client

ENABLED = {"alpaca_trading_enabled": True, "trading_dry_run": False}
OWNS = {"brain_mode": "paper_execution"}
U = "UPA"


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


async def client(tmp_path, **overrides):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **{**OWNS, **ENABLED, **overrides}):
        market = FakeOptionsMarket(clock, api.feed.live_price, broker=api.fake)
        api.container.trading.options_data = market
        api.market = market
        yield api


def call_spread(market: FakeOptionsMarket, *, qty=1, cushion=0.0) -> OptionOrderIntent:
    """Buy the at-the-money call, sell the next strike up (one strike wide: a small, defined risk)."""
    long = market.pick(U, "call", moneyness=1.0)
    short = min((c for c in market.listed(U) if c.kind == "call" and c.expiration == long.expiration
                 and c.strike > long.strike), key=lambda c: c.strike)  # fmt: skip
    ql, qs = market.quote(long), market.quote(short)
    mid = (ql.mid or 0) - (qs.mid or 0)
    legs = (
        OptionLegIntent(long.symbol, "buy", 1, "buy_to_open"),
        OptionLegIntent(short.symbol, "sell", 1, "sell_to_open"),
    )
    return OptionOrderIntent(U, "bull_call_spread", legs, qty, round(mid + cushion, 2), "entry", "test thesis", True,
                             market.spot(U))  # fmt: skip


async def run(api, *option_orders, cycle=1):
    return await api.container.trading.run_brain(
        [], brain_cycle_id=cycle, scheduled=False, option_orders=list(option_orders)
    )


def mleg_bodies(fake):
    return [b for b in fake.bodies if b.get("order_class") == "mleg"]


async def test_a_spread_goes_through_the_trading_service_once(tmp_path):
    async for api in client(tmp_path):
        o = call_spread(api.market)
        cycle = await run(api, o)
        assert cycle.mode == "paper", cycle.notes
        (t,) = cycle.trades
        assert t.asset_class == "us_option" and t.family == "bull_call_spread" and t.approved, t.risk
        assert (
            t.status == "filled" and t.client_order_id.startswith("qp-brain-") and "-o" in t.client_order_id
        )
        assert {c.name for c in t.checks} >= {
            "no_naked_short",
            "max_loss",
            "greeks",
            "limit_price",
            "total_risk",
        }
        (body,) = mleg_bodies(api.fake)
        assert body["type"] == "limit" and body["time_in_force"] == "day" and float(body["qty"]) == 1
        assert {x["position_intent"] for x in body["legs"]} == {"buy_to_open", "sell_to_open"}
        held = {p.symbol: p.qty for p in await api.container.broker.positions()}
        assert held == {o.legs[0].symbol: 1.0, o.legs[1].symbol: -1.0}
        async with api.container.db.session() as s:
            (row,) = (await s.scalars(select(BrokerOrderRow))).all()
        assert row.asset_class == "us_option" and row.order_class == "mleg" and row.symbol == "UPA:MLEG"
        assert [x["symbol"] for x in row.legs] == [x.symbol for x in o.legs] and row.status == "filled"
        events = {e["kind"] for e in (await api.get("/api/v1/trading/events")).json()}
        assert {"trade_proposed", "risk_approved", "order_submitted", "order_filled"} <= events

        # the same decision again in the same slot: never sent twice
        await run(api, o, cycle=2)
        assert len(mleg_bodies(api.fake)) == 1

        # closing: one multi-leg order, legs bought back and sold together
        positions = {p.symbol: p for p in await api.container.broker.positions()}
        quotes = await api.container.trading.option_leg_quotes(list(positions), {U: api.market.spot(U)}, NOW)
        closes = closing_intents(
            positions, quotes, {U: api.market.spot(U)}, kind="take_profit", reason="test"
        )
        assert len(closes) == 1 and not closes[0].opening
        cycle = await run(api, *closes, cycle=3)
        assert cycle.trades[0].approved and cycle.trades[0].status == "filled", cycle.trades[0].risk
        assert {x["position_intent"] for x in mleg_bodies(api.fake)[-1]["legs"]} == {
            "sell_to_close",
            "buy_to_close",
        }
        assert await api.container.broker.positions() == []


async def test_nothing_unsafe_is_ever_sent(tmp_path):
    async for api in client(tmp_path):
        m = api.market
        good = call_spread(m)
        short_call = m.pick(U, "call", moneyness=1.05)
        naked = OptionOrderIntent(U, "bull_call_spread", (OptionLegIntent(good.legs[0].symbol, "buy", 1, "buy_to_open"),
                                  OptionLegIntent(short_call.symbol, "sell", 2, "sell_to_open")), 1, 0.5, "entry", "t",
                                  True, m.spot(U))  # fmt: skip
        cycle = await run(api, naked)
        assert not cycle.trades[0].approved and "no_naked_short" in cycle.trades[0].risk
        m.age = 900  # stale quotes
        cycle = await run(api, good, cycle=2)
        assert not cycle.trades[0].approved and "option_quotes" in cycle.trades[0].risk
        m.age, m.feed = 3.0, "model"  # model-priced data is never an execution quote
        cycle = await run(api, good, cycle=3)
        assert not cycle.trades[0].approved and "never an execution quote" in cycle.trades[0].risk
        m.feed, m.down = "indicative", True  # the options feed is down: entries are not sent
        cycle = await run(api, good, cycle=4)
        assert cycle.trades == [] and any("option" in n for n in cycle.notes)
        m.down = False
        api.feed.quote_age = timedelta(minutes=15)  # the underlying's own price is stale
        cycle = await run(api, good, cycle=6)
        assert all(not t.approved for t in cycle.trades) and any("fresh price" in n for n in cycle.notes), (
            cycle.notes
        )
        api.feed.quote_age = timedelta(seconds=5)
        m.half_spread = 0.40  # a wide market on every leg
        cycle = await run(api, call_spread(m), cycle=7)
        (wide,) = cycle.trades
        assert not wide.approved and "option_liquidity" in wide.risk, wide.risk
        m.half_spread = 0.02
        bogus = OptionOrderIntent(U, "long_call", (OptionLegIntent("UPA991399C00100000", "buy", 1, "buy_to_open"),),
                                  1, 1.0, "entry", "t", True, m.spot(U))  # fmt: skip
        cycle = await run(api, bogus, cycle=8)  # a contract that cannot exist
        assert cycle.trades == [] and any("option entries are not sent" in n for n in cycle.notes), (
            cycle.notes
        )
        await api.post("/api/v1/trading/kill-switch", json={"active": True, "reason": "test"})
        cycle = await run(api, good, cycle=5)
        assert all(t.status != "filled" for t in cycle.trades)
        assert mleg_bodies(api.fake) == []
    async for api in client(tmp_path / "dry", trading_dry_run=True):
        cycle = await run(api, call_spread(api.market))
        assert cycle.mode == "dry_run" and cycle.trades[0].status == "dry_run" and mleg_bodies(api.fake) == []
    async for api in client(tmp_path / "off", options_execution=False):
        cycle = await run(api, call_spread(api.market))
        assert "options_enabled" in cycle.trades[0].risk and mleg_bodies(api.fake) == []


async def test_the_daily_loss_flatten_closes_option_positions_too(tmp_path):
    async for api in client(tmp_path, trading_daily_loss_action="flatten"):
        o = call_spread(api.market)
        await run(api, o)
        assert len(await api.container.broker.positions()) == 2
        api.fake.last_equity = api.fake.equity() * 1.10  # down 9% on the day
        cycle = await run(api, cycle=2)
        kinds = {t.kind for t in cycle.trades}
        assert kinds == {"daily_loss_flatten"} and all(t.approved for t in cycle.trades), [
            t.risk for t in cycle.trades
        ]
        assert await api.container.broker.positions() == []


def test_quantpulse_never_exercises_an_option():
    """Exercise is never requested: positions are closed before expiration; OCC settles anything left."""
    src = Path("src/quantpulse")
    hits = [
        str(p)
        for p in src.rglob("*.py")
        if re.search(r"exercise_options_position|/exercise\b", p.read_text())
    ]
    assert hits == []
