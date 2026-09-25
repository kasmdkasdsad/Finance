"""The Alpaca paper broker: the real alpaca-py SDK against a fake paper API (never the network)."""

import pytest
import requests

from quantpulse.providers import alpaca_trading as at
from quantpulse.providers.alpaca_trading import (
    AlpacaPaperBroker,
    BrokerError,
    BrokerNotConfigured,
    DuplicateClientOrderId,
    OrderRejected,
    OrderSpec,
)
from tests.fakes.alpaca_paper import FakeAlpacaPaper

KEY, SECRET = "PKFAKEKEYID", "fake-secret-value"


@pytest.fixture
def fake():
    return FakeAlpacaPaper(equity=100_000.0)


@pytest.fixture
def broker(fake):
    return AlpacaPaperBroker(KEY, SECRET, transport=fake)


def test_client_is_paper_only_and_never_shows_credentials(broker, fake):
    assert broker.base_url == "https://paper-api.alpaca.markets"
    client = broker._sdk()
    assert str(client._base_url.value) == at.PAPER_URL  # TradingClient(..., paper=True)
    assert client._sandbox is True
    assert KEY not in repr(broker) and SECRET not in repr(broker)


def test_a_client_pointing_anywhere_else_is_refused(monkeypatch):
    class LiveClient:
        _base_url = "https://api.alpaca.markets"

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(at, "TradingClient", LiveClient)
    with pytest.raises(at.NotPaperTrading):
        AlpacaPaperBroker(KEY, SECRET)._sdk()


async def test_unconfigured_broker_says_what_to_set():
    b = AlpacaPaperBroker(None, None)
    assert not b.configured()
    with pytest.raises(BrokerNotConfigured, match="QP_ALPACA_API_KEY_ID"):
        await b.account()


async def test_account_positions_and_clock(broker, fake):
    fake.hold("AAPL", 10, 190.0, price=200.0)
    fake.last_equity = 99_800.0
    a = await broker.account()
    assert a.equity == pytest.approx(100_100.0) and a.cash == pytest.approx(98_100.0)
    assert a.day_pl == pytest.approx(300.0) and a.account_number.startswith("…")
    assert not a.blocked
    [p] = await broker.positions()
    assert (p.symbol, p.qty, p.avg_entry_price, p.current_price) == ("AAPL", 10.0, 190.0, 200.0)
    assert p.unrealized_pl == pytest.approx(100.0) and p.market_value == pytest.approx(2000.0)
    clock = await broker.clock()
    assert clock.is_open and clock.next_close > clock.timestamp
    assert all(fake.auth_headers_seen)  # keys travel only as request headers


async def test_submit_lookup_cancel_and_close(broker, fake):
    fake.prices.update(MSFT=400.0, AMD=100.0)
    o = await broker.submit(OrderSpec("MSFT", "buy", 5, "limit", "qp-t-MSFT-b", 400.4))
    assert (o.status, o.filled_qty, o.filled_avg_price, o.side) == ("filled", 5.0, 400.4, "buy")
    body = fake.orders[o.id]
    assert body["client_order_id"] == "qp-t-MSFT-b" and body["time_in_force"] == "day"
    assert (await broker.order_by_client_id("qp-t-MSFT-b")).id == o.id
    assert await broker.order_by_client_id("qp-never-sent") is None
    assert (await broker.order(o.id)).client_order_id == "qp-t-MSFT-b"

    fake.fill_mode["AMD"] = "accept"
    resting = await broker.submit(OrderSpec("AMD", "buy", 3, "limit", "qp-t-AMD-b", 99.0))
    assert resting.is_open and [x.symbol for x in await broker.open_orders()] == ["AMD"]
    await broker.cancel(resting.id)
    assert (await broker.order(resting.id)).status == "canceled"

    fake.fill_mode["AMD"] = "accept"
    await broker.submit(OrderSpec("AMD", "buy", 3, "limit", "qp-t-AMD-b2", 99.0))
    assert await broker.cancel_all() == 1 and await broker.open_orders() == []

    closed = await broker.close_position("MSFT")
    assert (closed.side, closed.qty, closed.status) == ("sell", 5.0, "filled")
    fake.hold("NVDA", 4, 100.0)
    assert await broker.close_all_positions() == 1 and await broker.positions() == []
    history = await broker.orders("closed")
    assert {x.symbol for x in history} >= {"MSFT", "AMD", "NVDA"}


async def test_errors_are_translated(broker, fake):
    fake.prices["AAPL"] = 100.0
    await broker.submit(OrderSpec("AAPL", "buy", 1, "market", "qp-t-dup-b"))
    with pytest.raises(DuplicateClientOrderId):
        await broker.submit(OrderSpec("AAPL", "buy", 1, "market", "qp-t-dup-b"))
    fake.fill_mode["TSLA"] = "reject"
    with pytest.raises(OrderRejected, match="insufficient buying power") as rejected:
        await broker.submit(OrderSpec("TSLA", "buy", 1, "market", "qp-t-TSLA-b"))
    assert rejected.value.status_code == 403 and not rejected.value.ambiguous
    fake.fail_status = 500
    with pytest.raises(BrokerError) as server:
        await broker.account()
    assert server.value.status_code == 500 and server.value.ambiguous
    fake.fail_status = 401
    with pytest.raises(BrokerNotConfigured, match="paper"):
        await broker.account()


async def test_a_timeout_is_ambiguous_never_a_plain_failure(broker, fake):
    fake.fill_mode["META"] = "timeout"
    with pytest.raises(BrokerError) as exc:
        await broker.submit(OrderSpec("META", "buy", 2, "market", "qp-t-META-b"))
    assert exc.value.ambiguous and "unknown" in str(exc.value)
    # Alpaca did receive it: the client id finds it, so it must never be resent.
    assert (await broker.order_by_client_id("qp-t-META-b")).status == "filled"


def test_requests_get_a_timeout(broker):
    session = broker._sdk()._session
    assert isinstance(session, requests.Session) and session._timeout == at.DEFAULT_TIMEOUT
    assert broker._sdk()._retry_codes == [429]  # a 504 on POST is never blindly resent by the SDK


def test_order_specs_are_validated():
    with pytest.raises(ValueError):
        OrderSpec("AAPL", "buy", 0, "market", "qp-x")
    with pytest.raises(ValueError):
        OrderSpec("AAPL", "buy", 1, "limit", "qp-x")
    with pytest.raises(ValueError):
        OrderSpec("AAPL", "buy", 1, "market", "")


async def test_the_test_suite_cannot_reach_a_real_alpaca_account():
    """Guard rail: without the fake transport the SDK would use the network, which tests refuse."""
    real = AlpacaPaperBroker(KEY, SECRET)  # no fake transport
    with pytest.raises(BrokerError):
        await real.account()
