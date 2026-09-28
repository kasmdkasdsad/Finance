"""Running the Brain-owned account day to day, end to end on the fakes: the pre-market check and the close
recorded per day, reconciliation during the session, the audit trail of every trade, and the market-data
report (how often data stopped the Brain, and what SIP would change)."""

from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.brain.audit import STAGES
from quantpulse.core.clock import FakeClock
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import TrendFeed

from .conftest import NOW
from .test_brain_cycle import WIDE, brain_client, run_cycle, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS, posts

PRE_MARKET = datetime(2026, 9, 28, 12, 50, tzinfo=UTC)  # Monday 08:50 New York
AFTER_HOURS = datetime(2026, 9, 25, 20, 50, tzinfo=UTC)  # Friday 16:50 New York


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


async def test_every_trade_has_an_audit_trail_from_the_idea_to_the_fill(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        await run_cycle(api)
        done = [t for t in (await api.get(f"{API}/trades")).json() if t["sent"]]
        assert done and all(t["client_order_id"].startswith("qp-brain-") for t in done)
        first = done[-1]["decision_id"]
        trail = (await api.get(f"{API}/decisions/{first}/audit")).json()
        assert [s["stage"] for s in trail["stages"]] == list(STAGES)
        st = {s["stage"]: s for s in trail["stages"]}
        for name in ("data", "agents", "opinions", "evidence", "disagreement", "consensus", "portfolio_fit",
                     "portfolio_decision", "risk_check", "order", "alpaca_response", "execution", "fill"):  # fmt: skip
            assert st[name]["status"] == "done", (name, st[name])
        assert trail["gaps"] == [] and trail["sent"]
        assert st["order"]["detail"]["client_order_id"] == done[-1]["client_order_id"]
        assert any(e["kind"] == "order_filled" for e in st["alpaca_response"]["detail"]["events"])
        assert st["risk_check"]["detail"][
            "at_execution"
        ]  # the checks at the moment of sending, not only the preview
        assert st["portfolio_decision"]["detail"]["entry"]["thesis"]
        assert st["evidence"]["detail"]["supporting"] and st["disagreement"]["detail"]["sources"]
        assert st["execution"]["detail"]["ledger"]["client_order_id"] == done[-1]["client_order_id"]
        # later stages are pending, not missing: the fill is reconciled next cycle, the call graded at its horizon
        for name in (
            "position",
            "pnl",
            "benchmark_relative",
            "prediction_grade",
            "decision_quality",
            "lesson",
        ):
            assert st[name]["status"] == "pending", (name, st[name])
        assert "open until their horizon" in st["prediction_grade"]["summary"]
        clock.advance(31 * 60)
        await run_cycle(api)  # the fill is now a position with its thesis
        trail = (await api.get(f"{API}/decisions/{first}/audit")).json()
        st = {s["stage"]: s for s in trail["stages"]}
        assert st["position"]["status"] == "done" and st["position"]["detail"]["thesis"]
        assert st["pnl"]["status"] == "done" and st["pnl"]["detail"]["kind"] == "unrealised"
        rel = st["benchmark_relative"]
        assert (
            rel["status"] == "done" and rel["detail"]["final"] is False and "so far (open)" in rel["summary"]
        )
        assert trail["gaps"] == [] and set(trail["pending"]) == {
            "prediction_grade",
            "decision_quality",
            "lesson",
        }
        traces = (await api.get(f"{API}/traces")).json()
        assert traces["trades"] >= 1 and traces["with_gaps"] == 0, [(t["action"], t["gaps"]) for t in traces["trades_detail"]]
        assert traces["in_progress"] == traces["trades"]
        assert "traceable" in traces["headline"]
        assert (await api.get(f"{API}/decisions/999999/audit")).status_code == 404


async def test_the_data_report_counts_what_stale_quotes_stopped(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    feed = TrendFeed(clock, drifts=WIDE)
    feed.quote_age = timedelta(minutes=30)  # older than the 10-minute limit
    async for api in brain_client(tmp_path, clock, feed=feed, **OWNS, **ENABLED):
        cycle = await run_cycle(api)
        assert "data_quality" in cycle["summary"]["entry_halts"]
        report = (await api.get(f"{API}/data-report")).json()
        assert (
            report["how_often"]["cycles_in_session"] == 1 and report["how_often"]["data_blocked_cycles"] == 1
        )
        assert report["how_often"]["by_day"] == [{"day": "2026-09-25", "cycles": 1, "data_blocked": 1}]
        assert "TRADING BLOCKED" in report["headline"]
        assert report["quote_age"]["statuses"].get("stale") and report["quote_age"]["limit_s"] == 600
        sip = report["sip"]
        assert sip["configured_feed"] == "iex" and "QP_ALPACA_STOCK_FEED=sip" in sip["decision"]
        assert "never buys it" in sip["cost"] and "not a forecast of returns" in sip["expected_benefit"]
        assert posts(api.fake) == []  # nothing is bought on stale data


async def test_the_pre_market_check_is_recorded_with_the_day(tmp_path):
    clock = FakeClock(PRE_MARKET)
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        fake.hold("UPA", 10, 60.0)
        assert "premarket_check" in await api.container.brain.supervisor.tick()
        day = (await api.get(f"{API}/sessions")).json()["sessions"][0]
        assert day["day"] == "2026-09-28" and day["owner"] == "brain"
        checks = {c["name"]: c for c in day["premarket"]["checks"]}
        assert (
            checks["paper_endpoint"]["ok"]
            and checks["paper_endpoint"]["detail"] == "https://paper-api.alpaca.markets"
        )
        assert checks["account"]["ok"] and checks["reconciliation"]["ok"]
        assert checks["calendar"]["trading_day"] and "overnight" in checks
        assert day["premarket"]["positions"]["UPA"]["qty"] == 10
        assert posts(fake) == []


async def test_the_close_records_the_day_and_the_session_reconciles(tmp_path):
    clock = FakeClock(AFTER_HOURS)
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        fake.hold("UPA", 10, 60.0)
        fake.last_equity = fake.equity() / 1.01  # up 1% on the day
        assert "session_close" in await api.container.brain.supervisor.tick()
        day = (await api.get(f"{API}/sessions")).json()["sessions"][0]
        assert day["day"] == "2026-09-25" and day["positions"] == 1
        assert day["day_return"] == pytest.approx(0.01, abs=1e-6) and day["equity_close"] == fake.equity()
        assert day["benchmark_return"] is not None or day["close"]["note"]
        assert day["orders_sent"] == 0 and day["close"]["positions"]["UPA"]["qty"] == 10
    clock = FakeClock(NOW)  # during the session the Brain-owned account is reconciled every few minutes
    async for api in brain_client(tmp_path / "open", clock, **OWNS, **ENABLED):
        assert "reconcile" in await api.container.brain.supervisor.tick()
