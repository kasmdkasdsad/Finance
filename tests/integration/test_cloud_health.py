"""Health monitoring, alerts and failing closed — against the fake Alpaca paper API.

Every part of the deployment is checked; while the database, Alpaca or the last reconciliation fails, new
Brain orders are refused at the last gate and resume by themselves once the check passes; an execution
anomaly turns the Brain kill switch on; alerts go out on changes (ntfy, webhook, heartbeat — all mocked).
"""

import asyncio
import json
from datetime import timedelta

import httpx
import pytest
from sqlalchemy.exc import OperationalError

from quantpulse.core.clock import FakeClock
from quantpulse.db.session import Database
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import TrendFeed

from .conftest import NOW
from .test_brain_cycle import WIDE, brain_client, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS, TRADING

ORDERS = "/v2/orders"
NTFY = "https://ntfy.example/qp-alerts-topic-secret"
HOOK = "https://hooks.example/services/secret-hook"
BEAT = "https://hc.example/ping/secret-uuid"
ALERTS = {"alert_ntfy_url": NTFY, "alert_webhook_url": HOOK, "heartbeat_url": BEAT}


@pytest.fixture(autouse=True)
def network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    mock_net.post(NTFY).respond(200)
    mock_net.post(HOOK).respond(200)
    mock_net.get(url__startswith=BEAT).respond(200)
    return mock_net


def order_posts(fake: FakeAlpacaPaper) -> int:
    return sum(1 for m, p in fake.log if m == "POST" and p == ORDERS)


def shared(clock):
    return FakeAlpacaPaper(clock=clock), TrendFeed(clock, drifts=WIDE)


def statuses(report) -> dict[str, str]:
    return {k: v["status"] for k, v in report["parts"].items()}


# --------------------------------------------------------------------------- the report
async def test_a_healthy_deployment_reports_every_part(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        assert "cycle" in await api.container.brain.supervisor.tick()
        report = (await api.get("/api/v1/system/health", params={"fresh": True})).json()
        got = statuses(report)
        assert {k: got[k] for k in ("api", "database", "supervisor", "alpaca", "reconciliation", "last_cycle",
                                    "kill_switches")} == dict.fromkeys(
            ("api", "database", "supervisor", "alpaca", "reconciliation", "last_cycle", "kill_switches"), "ok"
        )  # fmt: skip
        assert got["scheduler"] == "warn"  # background jobs are off in the tests
        assert report["status"] == "warn" and report["order_blockers"] == []
        sup = report["parts"]["supervisor"]
        assert sup["lease"]["mine"] and sup["next_cycle_at"] == (NOW + timedelta(minutes=30)).isoformat()
        assert report["parts"]["market_data"]["status"] in ("ok", "warn")
        assert (
            "postgresql" in report["parts"]["database"]["detail"]
            or "sqlite" in report["parts"]["database"]["detail"]
        )


# --------------------------------------------------------------------------- failing closed
async def test_alpaca_unreachable_fails_brain_orders_closed_until_it_answers_again(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        c = api.container
        fake.outage = True
        report = await c.health.check()
        assert report["parts"]["alpaca"]["status"] == "fail" and report["status"] == "fail"
        assert any("alpaca failing" in b for b in await c.trading.pre_submit_blockers("brain"))
        fake.outage = False  # Alpaca flaps back, but the last check said it failed: a cycle now sends nothing
        forced = (await api.post(f"{API}/run")).json()
        assert order_posts(fake) == 0
        blocked = [
            d for d in forced["decisions"] if d["quantity"] and (d.get("execution") or {}).get("reason")
        ]
        assert blocked and all("alpaca failing" in d["execution"]["reason"] for d in blocked)
        clock.advance(61)  # the next check passes: orders go again, by themselves
        assert (await c.health.check())["parts"]["alpaca"]["status"] == "ok"
        assert not any("health check" in b for b in await c.trading.pre_submit_blockers("brain"))
        await api.post(f"{API}/run")
        assert order_posts(fake) > 0


async def test_a_failed_reconciliation_holds_brain_orders_until_one_succeeds(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        t = api.container.trading
        await t.reconcile("startup")
        fake.outage = True
        with pytest.raises(Exception):  # noqa: B017  (whatever the broker raised)
            await t.reconcile("periodic")
        fake.outage = False
        assert t.reconcile_error is not None
        blockers = await t.pre_submit_blockers("brain")
        assert any("the last reconciliation with Alpaca failed" in b for b in blockers)
        assert not any("reconciliation" in b for b in await t.pre_submit_blockers("brain", flatten=True))
        report = await api.container.health.check()
        assert report["parts"]["reconciliation"]["status"] == "fail"
        events = [e["kind"] for e in (await api.get(f"{TRADING}/events")).json()]
        assert "reconciliation_failed" in events
        await t.reconcile("periodic")  # succeeds: orders may go again
        assert t.reconcile_error is None
        assert not any("reconciliation" in b for b in await t.pre_submit_blockers("brain"))


async def test_a_database_failure_is_reported_and_blocks_brain_orders(tmp_path, monkeypatch):
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED, **ALERTS):
        c = api.container
        await c.health.run()  # baseline (and the start-up note)
        real = Database.session

        def down(self):
            raise OperationalError("SELECT 1", {}, Exception("database unreachable"))

        monkeypatch.setattr(Database, "session", down)
        assert await c.health.run() == "health fail"
        report = c.health.last
        assert report["parts"]["database"]["status"] == "fail"
        assert any("database failing" in b for b in report["order_blockers"])
        assert any(a["title"] == "Database failing" for a in c.alerts.sent)  # sent even though not recorded
        monkeypatch.setattr(Database, "session", real)
        assert await c.health.run() != "health fail"
        assert any(a["title"] == "Database recovered" for a in c.alerts.sent)


async def test_an_execution_anomaly_turns_the_brain_kill_switch_on(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "reject"  # Alpaca refuses every order
    async for api in brain_client(
        tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED, **ALERTS, brain_anomaly_orders_per_hour=2
    ):
        c = api.container
        await c.brain.supervisor.tick()
        assert len(await c.health.anomalies()) == 1
        await c.health.run()  # one rejection: below the limit, nothing happens
        assert not (await api.get(f"{API}/kill-switch")).json()["active"]
        clock.advance(31 * 60)
        await c.brain.supervisor.tick()  # the next slot: rejected again
        rejected = await c.health.anomalies()
        assert len(rejected) >= 2, rejected
        await c.health.run()
        kill = (await api.get(f"{API}/kill-switch")).json()
        assert kill["active"] and kill["reason"].startswith("automatic: ")
        assert any(a["kind"] == "execution_anomaly" and a["severity"] == "critical" for a in c.alerts.sent)
        sent = order_posts(fake)
        fake.default_mode = "fill"
        clock.advance(31 * 60)
        await c.brain.supervisor.tick()
        assert order_posts(fake) == sent  # stopped until a person releases it


# --------------------------------------------------------------------------- alerts
async def test_alerts_reach_ntfy_the_webhook_and_the_heartbeat(tmp_path, monkeypatch, network):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED, **ALERTS):
        c = api.container
        await c.brain.supervisor.tick()
        await c.health.run()
        ntfy = [call.request for call in network.calls if str(call.request.url) == NTFY]
        assert ntfy and ntfy[0].headers["Title"] == "QuantPulse: QuantPulse started"
        hook = [json.loads(call.request.content) for call in network.calls if str(call.request.url) == HOOK]
        assert hook and "QuantPulse started" in hook[0]["text"] and hook[0]["content"]
        beats = [str(call.request.url) for call in network.calls if str(call.request.url).startswith(BEAT)]
        assert beats == [BEAT]
        # the Brain kill switch from the phone: an alert
        body = {"active": True, "reason": "from my phone", "cancel_open_orders": True}
        assert (await api.post(f"{API}/kill-switch", json=body)).status_code == 200
        await c.health.run()
        assert any(
            a["title"] == "Brain kill switch ON" and a["message"] == "from my phone" for a in c.alerts.sent
        )
        # unhealthy: the heartbeat reports the failure at once
        fake.outage = True
        clock.advance(61)
        await c.health.run()
        beats = [str(call.request.url) for call in network.calls if str(call.request.url).startswith(BEAT)]
        assert beats[-1] == f"{BEAT}/fail"
        assert any(a["title"] == "Alpaca paper API unreachable" for a in c.alerts.sent)
        listed = (await api.get("/api/v1/system/alerts")).json()
        assert listed["channels"] == ["ntfy", "webhook"] and listed["heartbeat"] is True
        text = json.dumps(listed)
        for secret in (NTFY, HOOK, BEAT, "secret-uuid", "topic-secret"):
            assert secret not in text


async def test_an_alert_is_not_repeated_within_the_cooldown_and_delivery_failures_never_raise(
    tmp_path, network
):
    from quantpulse.services.alerts import Alert

    network.post(NTFY).mock(side_effect=httpx.ConnectError("no route"))
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **ALERTS):
        alerts = api.container.alerts
        assert await alerts.send(Alert("x", "Something", "detail"))
        assert not await alerts.send(Alert("x", "Something", "detail"))  # within the cool-down
        assert alerts.sent[-1]["delivered"] == {"ntfy": False, "webhook": True}
        clock.advance(31 * 60)
        assert await alerts.send(Alert("x", "Something", "detail"))


async def test_an_unexpected_position_and_a_stalled_supervisor_are_alerted(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED, **ALERTS):
        c = api.container
        await c.brain.supervisor.tick()
        await c.health.run()
        fake.hold("MIDC", 7, feed.live_price("MIDC"))  # bought by hand, outside the Brain
        clock.advance(6 * 60)
        await c.brain.supervisor.tick()  # the monitor sees it
        forever = asyncio.get_running_loop().create_task(asyncio.Event().wait())
        c.poller._tasks = [forever]  # the scheduler "runs" but the supervisor stops ticking
        try:
            await c.health.run()
            assert any(a["kind"] == "unexpected_position" and "MIDC" in a["message"] for a in c.alerts.sent)
            clock.advance(6 * 60)
            await c.health.run()
            assert c.health.last["parts"]["supervisor"]["status"] == "fail"
            assert any(a["title"] == "Brain stopped" and "stalled" in a["message"] for a in c.alerts.sent)
        finally:
            forever.cancel()
            c.poller._tasks = []


# --------------------------------------------------------------------------- supervisor, execution, cycles
def kinds(c) -> list[str]:
    return [a["kind"] for a in c.alerts.sent]


async def test_a_lost_lease_and_a_takeover_after_a_crash_alert_once(tmp_path, monkeypatch):
    from .test_render_deploy import start

    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    a_gen, a = await start(tmp_path, clock, fake, feed, **ALERTS)
    b_gen, b = await start(tmp_path, clock, fake, feed, **ALERTS)
    try:
        await a.container.brain.supervisor.tick()
        a.container.lease.stop_heartbeat()  # A hangs past its lease ...
        clock.advance(a.container.lease.ttl.total_seconds() + 1)
        await b.container.brain.supervisor.tick()  # ... B takes over from a lease that was never released
        await b.container.health.run()
        await b.container.health.run()
        assert kinds(b.container).count("supervisor_takeover") == 1
        assert (await a.container.brain.supervisor.tick()).startswith("standby")  # A comes back: it lost
        await a.container.health.run()
        await a.container.health.run()
        assert kinds(a.container).count("supervisor_lost") == 1
    finally:
        await b_gen.stop()
        await a_gen.stop()


async def test_a_standby_reports_a_leader_that_holds_the_lease_but_stopped_ticking(tmp_path, monkeypatch):
    from .test_render_deploy import start

    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    a_gen, a = await start(tmp_path, clock, fake, feed)
    b_gen, b = await start(tmp_path, clock, fake, feed, **ALERTS)
    try:
        await a.container.brain.supervisor.tick()
        await b.container.brain.supervisor.tick()
        assert (await b.container.health.check())["parts"]["supervisor"]["status"] == "standby"
        clock.advance(6 * 60)
        # A still renews its lease (its heartbeat task runs) but never ticks
        await a.container.lease.acquire()
        await b.container.brain.supervisor.tick()
        part = (await b.container.health.check())["parts"]["supervisor"]
        assert part["status"] == "fail" and "has not ticked" in part["detail"]
    finally:
        await b_gen.stop()
        await a_gen.stop()


async def test_autonomous_execution_blocked_in_the_session_alerts_after_ten_minutes(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, trading_dry_run=True,
                                  alpaca_trading_enabled=True, brain_mode="paper_execution", **ALERTS):  # fmt: skip
        c = api.container
        await c.brain.supervisor.tick()
        await c.health.run()
        assert "execution_blocked" not in kinds(c)  # not yet: it may be a moment
        for _ in range(3):
            clock.advance(5 * 60)
            await c.brain.supervisor.tick()
            await c.health.run()
        blocked = [a for a in c.alerts.sent if a["kind"] == "execution_blocked"]
        assert len(blocked) == 1 and "QP_TRADING_DRY_RUN=true" in blocked[0]["message"]


async def test_nothing_alerts_about_execution_while_the_market_is_closed(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    with_stock_model(monkeypatch)
    clock = FakeClock(datetime(2026, 9, 26, 15, 0, tzinfo=UTC))  # a Saturday
    fake, feed = shared(clock)
    fake.market_open = False
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED, **ALERTS):
        c = api.container
        for _ in range(4):
            await c.brain.supervisor.tick()
            await c.health.run()
            clock.advance(5 * 60)
        assert "execution_blocked" not in kinds(c)


async def test_repeated_failed_cycles_alert(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED, **ALERTS):
        c = api.container

        async def broken(*args, **kwargs):
            raise RuntimeError("a bug in a Brain step")

        monkeypatch.setattr(c.brain.orchestrator, "_context", broken)
        for _ in range(3):
            await api.post(f"{API}/run")
            clock.advance(60)
        report = await c.health.check()
        assert report["parts"]["last_cycle"]["failed_streak"] >= 3
        await c.health.run()
        assert "cycles_failing" in kinds(c)
