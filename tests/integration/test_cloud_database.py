"""Moving to the cloud database without losing the Brain's history, on the fake Alpaca paper account.

A Brain that has traded on the PC's SQLite file is copied into an empty database — PostgreSQL when the suite
runs with ``QP_TEST_POSTGRES_URL`` (as in the cloud), otherwise another SQLite file — and the API on the new
database shows the same history, still refuses to resend an order already sent, and keeps working."""

import pytest
from sqlalchemy import create_engine, text

from quantpulse.core.clock import FakeClock
from quantpulse.db.migrate import sync_url
from quantpulse.db.transfer import TransferError, transfer
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.pg import database_url

from .conftest import NOW
from .test_brain_cycle import brain_client, run_cycle, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS, brain_orders


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def counts(url: str) -> dict[str, int]:
    engine = create_engine(sync_url(url))
    try:
        with engine.connect() as c:
            return {t: c.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar_one() for t in
                    ("brain_cycles", "brain_opinions", "brain_consensus", "brain_decisions", "brain_predictions",
                     "brain_executions", "brain_opportunity_outcomes", "broker_orders", "trading_events")}  # fmt: skip
    finally:
        engine.dispose()


async def test_the_brains_history_moves_to_the_cloud_database_intact(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    source = f"sqlite+aiosqlite:///{tmp_path / 'pc.db'}"  # the PC's database
    async for api in brain_client(tmp_path, clock, fake=fake, database_url=source, **OWNS, **ENABLED):
        await run_cycle(api)
        sent = [o["client_order_id"] for o in brain_orders(fake)]
        assert sent
    before = counts(source)
    assert before["brain_cycles"] and before["brain_executions"] and before["broker_orders"]

    target = database_url(tmp_path / "cloud.db")  # PostgreSQL with QP_TEST_POSTGRES_URL
    out = transfer(source, target, progress=lambda _m: None)
    assert out["rows"] > 0 and counts(target) == before
    with pytest.raises(TransferError, match="not empty"):  # never merged into, never overwritten
        transfer(source, target, progress=lambda _m: None)

    async for api in brain_client(tmp_path, clock, fake=fake, database_url=target, **OWNS, **ENABLED):
        cycles = (await api.get(f"{API}/cycles")).json()
        assert len(cycles) == before["brain_cycles"]
        ledger = (await api.get(f"{API}/executions")).json()["executions"]
        assert {r["client_order_id"] for r in ledger} == set(sent)
        again = await run_cycle(api)  # same slot on the new database: the orders are known, never resent
        assert [o["client_order_id"] for o in brain_orders(fake)] == sent
        assert again["summary"]["orders_sent"] == 0
        assert len((await api.get(f"{API}/cycles")).json()) == before["brain_cycles"] + 1  # ids keep counting
