"""The order manager against a real (temporary) database and the fake Alpaca paper API."""

from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.db import repositories as repo
from quantpulse.db.models import BrokerOrderRow
from quantpulse.providers.alpaca_trading import AlpacaPaperBroker, OrderSpec
from quantpulse.services.order_manager import (
    SUBMIT_FAILED,
    SUBMIT_UNKNOWN,
    OrderManager,
    client_order_id,
)
from quantpulse.services.trading_risk import OrderIntent
from tests.fakes.alpaca_paper import FakeAlpacaPaper

NOW = datetime(2026, 9, 25, 14, 30, tzinfo=UTC)


@pytest.fixture
def clock():
    return FakeClock(NOW)


@pytest.fixture
def fake(clock):
    f = FakeAlpacaPaper(clock=clock)
    f.prices.update(AAPL=200.0, MSFT=400.0, NVDA=100.0)
    return f


@pytest.fixture
def manager(database, fake, clock):
    return OrderManager(database, AlpacaPaperBroker("PKTEST", "SECRET", transport=fake), clock)


def intent(symbol="AAPL", side="buy", qty=10.0, price=200.0) -> OrderIntent:
    return OrderIntent(symbol, side, qty, price, "entry", "strong trend", score=1.5)


async def rows(database) -> dict[str, BrokerOrderRow]:
    async with database.session() as s:
        return {r.client_order_id: r for r in await repo.broker_orders(s, 100)}


async def events(database) -> list[str]:
    async with database.session() as s:
        return [e.kind for e in reversed(await repo.trading_events(s, 200))]


def test_client_order_ids_are_deterministic():
    assert client_order_id("20260925T1030", "BRK.B", "buy") == "qp-20260925T1030-BRK_B-b"
    assert client_order_id("20260925T1030", "AAPL", "sell") == client_order_id("20260925T1030", "aapl", "s")


async def test_new_order_fills_and_is_recorded(manager, database, fake):
    sub = await manager.submit(
        intent(), cid="qp-t1-AAPL-b", order_type="marketable_limit", limit_price=200.2, cycle_id=None
    )
    assert sub.submitted and sub.status == "filled" and not sub.duplicate
    r = (await rows(database))["qp-t1-AAPL-b"]
    assert (r.status, r.filled_quantity, r.average_fill_price, r.order_type) == (
        "filled",
        10.0,
        200.2,
        "marketable_limit",
    )
    assert (
        r.alpaca_order_id and r.reason == "strong trend" and r.signal_score == 1.5 and r.filled_at is not None
    )
    assert await events(database) == ["order_submitted", "order_filled"]
    assert fake.orders[r.alpaca_order_id]["type"] == "limit"


async def test_the_same_client_id_is_never_sent_twice(manager, database, fake):
    first = await manager.submit(
        intent(), cid="qp-t1-AAPL-b", order_type="market", limit_price=None, cycle_id=None
    )
    again = await manager.submit(
        intent(), cid="qp-t1-AAPL-b", order_type="market", limit_price=None, cycle_id=None
    )
    assert first.submitted and not again.submitted and again.duplicate
    assert len(fake.orders) == 1 and fake.log.count(("POST", "/v2/orders")) == 1
    assert "duplicate_prevented" in await events(database)


async def test_restart_after_the_request_got_through(manager, database, fake, clock):
    """A crash after Alpaca accepted the order but before QuantPulse recorded the answer: the write-ahead
    row blocks a resend, and reconciliation adopts Alpaca's order."""
    fake.prices["MSFT"] = 400.0
    await AlpacaPaperBroker("PKTEST", "SECRET", transport=fake).submit(
        OrderSpec("MSFT", "buy", 3, "market", "qp-t2-MSFT-b")
    )
    async with database.session() as s:
        s.add(
            BrokerOrderRow(
                client_order_id="qp-t2-MSFT-b",
                symbol="MSFT",
                side="buy",
                quantity=3,
                order_type="market",
                status="pending_submit",
                filled_quantity=0.0,
                strategy="quantpulse",
                created_at=NOW,
                updated_at=NOW,
            )
        )
    again = await manager.submit(
        intent("MSFT", qty=3), cid="qp-t2-MSFT-b", order_type="market", limit_price=None, cycle_id=None
    )
    assert again.duplicate and len(fake.orders) == 1
    report = await manager.reconcile()
    assert report.resolved_unknown == 1
    assert (await rows(database))["qp-t2-MSFT-b"].status == "filled"


async def test_restart_before_the_request_left(manager, database, clock):
    """A crash between the write-ahead record and the request: Alpaca never saw it; after a grace period
    reconciliation closes it out as submit_failed (and it was never sent twice)."""
    async with database.session() as s:
        s.add(
            BrokerOrderRow(
                client_order_id="qp-t3-NVDA-b",
                symbol="NVDA",
                side="buy",
                quantity=5,
                order_type="market",
                status="pending_submit",
                filled_quantity=0.0,
                strategy="quantpulse",
                created_at=NOW,
                updated_at=NOW,
            )
        )
    assert await manager.unresolved_symbols() == {"NVDA"}
    await manager.reconcile()
    assert (await rows(database))["qp-t3-NVDA-b"].status == "pending_submit"  # still inside the grace period
    clock.advance(180)
    await manager.reconcile()
    assert (await rows(database))["qp-t3-NVDA-b"].status == SUBMIT_FAILED
    assert await manager.unresolved_symbols() == set()


async def test_timeouts_are_looked_up_never_resent(manager, database, fake):
    fake.fill_mode["AAPL"] = "timeout"  # Alpaca got it, the answer was lost
    sub = await manager.submit(
        intent(), cid="qp-t4-AAPL-b", order_type="market", limit_price=None, cycle_id=None
    )
    assert sub.submitted and sub.status == "filled" and len(fake.orders) == 1


async def test_dropped_connection_leaves_an_unknown_order_for_reconciliation(
    manager, database, fake, monkeypatch
):
    import requests

    original = fake._submit

    def drop(request, body):  # the request never reached Alpaca
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr(fake, "_submit", drop)
    sub = await manager.submit(
        intent(), cid="qp-t5-AAPL-b", order_type="market", limit_price=None, cycle_id=None
    )
    assert not sub.submitted and sub.status == SUBMIT_UNKNOWN and "not resent" in (sub.error or "")
    monkeypatch.setattr(fake, "_submit", original)
    again = await manager.submit(
        intent(), cid="qp-t5-AAPL-b", order_type="market", limit_price=None, cycle_id=None
    )
    assert again.duplicate and fake.orders == {}  # the same id is never tried again
    assert "order_unknown" in await events(database)


async def test_partial_fill_then_completion(manager, database, fake):
    fake.fill_mode["AAPL"] = "partial"
    sub = await manager.submit(
        intent(qty=10), cid="qp-t6-AAPL-b", order_type="market", limit_price=None, cycle_id=None
    )
    assert sub.status == "partially_filled"
    r = (await rows(database))["qp-t6-AAPL-b"]
    assert r.filled_quantity == 5.0 and r.status == "partially_filled"
    fake.complete("qp-t6-AAPL-b")
    report = await manager.reconcile()
    assert report.updated == 1
    r = (await rows(database))["qp-t6-AAPL-b"]
    assert (r.status, r.filled_quantity) == ("filled", 10.0)
    assert (await events(database))[-2:] == ["order_partially_filled", "order_filled"]


async def test_rejection_is_recorded_not_retried(manager, database, fake):
    fake.fill_mode["NVDA"] = "reject"
    sub = await manager.submit(
        intent("NVDA", qty=5, price=100.0),
        cid="qp-t7-NVDA-b",
        order_type="market",
        limit_price=None,
        cycle_id=None,
    )
    assert not sub.submitted and sub.status == "rejected" and "insufficient buying power" in (sub.error or "")
    assert (await rows(database))["qp-t7-NVDA-b"].status == "rejected"
    assert "order_rejected" in await events(database)


async def test_stale_orders_are_canceled_and_reconciled(manager, database, fake, clock):
    fake.fill_mode["MSFT"] = "accept"
    await manager.submit(
        intent("MSFT", qty=2, price=400.0),
        cid="qp-t8-MSFT-b",
        order_type="limit",
        limit_price=390.0,
        cycle_id=None,
    )
    broker = manager._broker
    assert await manager.cancel_stale(await broker.open_orders()) == []  # still fresh
    clock.advance(timedelta(minutes=25).total_seconds())
    assert await manager.cancel_stale(await broker.open_orders()) == ["qp-t8-MSFT-b"]
    await manager.reconcile()
    assert (await rows(database))["qp-t8-MSFT-b"].status == "canceled"
    assert "order_canceled" in await events(database)


async def test_reconciliation_adopts_orders_placed_elsewhere(manager, database, fake):
    await manager._broker.submit(OrderSpec("AAPL", "buy", 1, "market", "manual-from-alpaca-dashboard"))
    report = await manager.reconcile()
    assert report.added == 1
    r = (await rows(database))["manual-from-alpaca-dashboard"]
    assert r.strategy == "external" and r.status == "filled"
    assert await manager.last_trades(NOW - timedelta(hours=1)) == {}  # manual orders never set the cooldown
