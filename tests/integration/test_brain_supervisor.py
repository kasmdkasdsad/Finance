"""Events and the supervisor: what the brain does by itself while the server runs — by market session and
by event, within its rate limits — and that none of it ever reaches Alpaca's order endpoint."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from quantpulse.brain.events import Event, EventBus, EventType, TradingEventBridge
from quantpulse.core.clock import FakeClock
from quantpulse.db.models import BrainEventRow, TradingEventRow
from tests.fakes.alpaca_paper import FakeAlpacaPaper

from .conftest import NOW
from .test_brain_cycle import BRAIN, _no_network, brain_client, only_reads  # noqa: F401

SATURDAY = datetime(2026, 9, 26, 15, 30, tzinfo=UTC)  # 11:30 New York
AFTER_HOURS = datetime(2026, 9, 25, 20, 50, tzinfo=UTC)  # Friday 16:50 New York
PRE_MARKET = datetime(2026, 9, 28, 12, 50, tzinfo=UTC)  # Monday 08:50 New York


async def test_event_bus_persists_drops_repeats_and_isolates_handlers(database):
    clock = FakeClock(NOW)
    bus = EventBus(database, clock)
    seen: list[str] = []

    async def good(e: Event) -> None:
        seen.append(e.subject or "")

    async def broken(e: Event) -> None:
        raise RuntimeError("handler bug")

    bus.subscribe([EventType.PRICE_MOVE_DETECTED], broken)
    bus.subscribe([EventType.PRICE_MOVE_DETECTED], good)
    kept = await bus.publish([Event(EventType.PRICE_MOVE_DETECTED, "AAA", {"move_sigma": 4.1})])
    assert len(kept) == 1 and seen == ["AAA"]  # the broken handler did not stop the good one
    again = await bus.publish([Event(EventType.PRICE_MOVE_DETECTED, "AAA", {"move_sigma": 4.3})])
    assert again == [] and bus.dropped == 1  # a repeat inside the cooldown carries no news
    clock.advance(2 * 3600)
    assert len(await bus.publish([Event(EventType.PRICE_MOVE_DETECTED, "AAA", {})])) == 1
    history = await bus.history(type="PriceMoveDetected")
    assert [h["subject"] for h in history] == ["AAA", "AAA"]


async def test_the_trading_bridge_reads_orders_and_risk_events_without_writing(database):
    bridge = TradingEventBridge(database)
    async with database.session() as s:
        s.add(
            TradingEventRow(kind="order_filled", symbol="OLD", message="before the brain watched", details={})
        )
    assert await bridge.poll() == []  # starts from now: history is not news
    async with database.session() as s:
        s.add_all(
            [
                TradingEventRow(kind="order_submitted", symbol="AAA", message="sent", details={}),
                TradingEventRow(kind="order_filled", symbol="AAA", message="filled", details={}),
                TradingEventRow(kind="daily_loss_limit_reached", symbol=None, message="stop", details={}),
                TradingEventRow(kind="signal_generated", symbol="BBB", message="noise", details={}),
            ]
        )
    got = [(e.type, e.subject) for e in await bridge.poll()]
    assert got == [
        (EventType.ORDER_SUBMITTED, "AAA"),
        (EventType.ORDER_FILLED, "AAA"),
        (EventType.RISK_LIMIT_TRIGGERED, "@portfolio"),
    ]
    assert await bridge.poll() == []
    async with database.session() as s:
        assert len((await s.scalars(select(TradingEventRow))).all()) == 5  # read only


async def test_cycles_emit_events_and_notice_portfolio_changes(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        api.fake.hold("UPA", 40, 60.0)
        await api.post(f"{BRAIN}/run")
        events = (await api.get(f"{BRAIN}/events", params={"limit": 500})).json()
        types = {e["type"] for e in events}
        assert "AgentCompleted" in types and "OpportunityDetected" in types
        assert "PortfolioChanged" not in types  # nothing to compare with yet
        api.fake.hold("UPB", 10, 80.0)
        clock.advance(600)
        await api.post(f"{BRAIN}/run")
        changed = (await api.get(f"{BRAIN}/events", params={"type": "PositionChanged"})).json()
        assert [e["subject"] for e in changed] == ["UPB"] and changed[0]["payload"]["qty_after"] == 10
        assert (await api.get(f"{BRAIN}/events", params={"type": "PortfolioChanged"})).json()
        assert only_reads(api.fake)


async def test_market_hours_schedule_events_and_rate_limits(tmp_path):
    clock = FakeClock(NOW)  # Friday 10:00 New York, market open
    async for api in brain_client(tmp_path, clock, brain_max_event_cycles_per_hour=1):
        sup = api.container.brain.supervisor
        first = await sup.tick()
        assert "cycle" in first  # the scheduled full cycle ran
        cycles = (await api.get(f"{BRAIN}/cycles")).json()
        assert len(cycles) == 1 and cycles[0]["trigger"] == "supervisor: scheduled"
        assert (await sup.tick()).startswith("idle")  # not due again yet

        bus = api.container.brain.bus
        await bus.publish([Event(EventType.PRICE_MOVE_DETECTED, "UPB", {"move_sigma": 5.0, "held": False}),
                           Event(EventType.VOLUME_SPIKE_DETECTED, "UPC", {})])  # fmt: skip
        assert {k[1] for k in sup.queue} == {("UPB",), ("UPC",)}
        done = await sup.tick()
        assert done.startswith("event:")  # one focused cycle; the other waits for the next hour
        assert len(sup.queue) == 1
        event_cycle = (await api.get(f"{BRAIN}/cycles")).json()[0]
        assert event_cycle["kind"] == "event"
        detail = (await api.get(f"{BRAIN}/cycles/{event_cycle['id']}")).json()
        skipped = {r["agent_id"]: r["reason"] for r in detail["runs"] if r["status"] == "skipped"}
        assert skipped.get("valuation") == "not needed for a event cycle"  # routed: only the relevant agents
        assert event_cycle["focus"][0]["symbol"] in {"UPB", "UPC"}

        clock.advance(31 * 60)
        later = await sup.tick()
        assert "cycle" in later and "monitor" in later
        status = (await api.get(f"{BRAIN}/supervisor")).json()
        assert status["session"] == "market_open" and status["enabled"] and not status["paused"]
        assert status["limits"]["max_event_cycles_per_hour"] == 1 and status["recent"]
        assert only_reads(api.fake)


async def test_monitor_turns_a_big_move_into_an_event_and_a_focused_cycle(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        sup = api.container.brain.supervisor
        await sup.tick()  # the scheduled cycle gives the monitor its watch list and volatilities
        api.feed.live_move["UPC"] = 0.15  # UPC jumps 15% intraday
        clock.advance(6 * 60)
        await sup.tick()
        moves = (await api.get(f"{BRAIN}/events", params={"type": "PriceMoveDetected"})).json()
        assert any(e["subject"] == "UPC" and e["payload"]["source"] == "monitor" for e in moves)
        cycles = (await api.get(f"{BRAIN}/cycles")).json()
        assert (
            any(c["kind"] == "event" and "PriceMoveDetected" in c["trigger"] for c in cycles)
            or ("event", ("UPC",)) in sup.queue
        )
        assert only_reads(api.fake)


async def test_off_hours_learn_review_and_research_once_a_day(tmp_path):
    for moment, expected in ((AFTER_HOURS, {"learn", "review"}), (SATURDAY, {"offday_learn", "deep"}),
                             (PRE_MARKET, {"premarket_learn", "premarket"})):  # fmt: skip
        clock = FakeClock(moment)
        fake = FakeAlpacaPaper(clock=clock)
        fake.market_open = False
        async for api in brain_client(tmp_path / moment.strftime("%a%H"), clock, fake=fake):
            sup = api.container.brain.supervisor
            done = set((await sup.tick()).split(", "))
            assert done == expected, (moment, done)
            assert (await sup.tick()).startswith("idle")  # once a day
            kinds = [c["kind"] for c in (await api.get(f"{BRAIN}/cycles")).json()]
            assert kinds == [
                {"review": "portfolio", "deep": "deep", "premarket": "full"}[
                    next(t for t in expected if t in ("review", "deep", "premarket"))
                ]
            ]
            assert only_reads(api.fake)


async def test_pause_resume_and_disable(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        r = await api.post(f"{BRAIN}/supervisor", json={"paused": True})
        assert r.status_code == 200 and r.json()["paused"] is True
        assert await api.container.brain.supervisor.tick() == "paused"
        assert (await api.get(f"{BRAIN}/cycles")).json() == []
        await api.post(f"{BRAIN}/supervisor", json={"paused": False})
        assert "cycle" in await api.container.brain.supervisor.tick()
    async for api in brain_client(tmp_path / "off", clock, brain_supervisor_enabled=False):
        assert await api.container.brain.supervisor.tick() == "disabled"
    async for api in brain_client(tmp_path / "remote", clock, client_host="203.0.113.9"):
        assert (await api.post(f"{BRAIN}/supervisor", json={"paused": True})).status_code == 403


async def test_trading_events_wake_a_portfolio_review(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        sup = api.container.brain.supervisor
        await sup.tick()  # first tick: the bridge starts watching, the scheduled cycle runs
        async with api.container.db.session() as s:
            s.add(TradingEventRow(kind="order_filled", symbol="UPA", message="filled 10 UPA", details={}))
        done = await sup.tick()
        assert "portfolio:-" in done
        filled = (await api.get(f"{BRAIN}/events", params={"type": "OrderFilled"})).json()
        assert filled and filled[0]["payload"]["source"] == "trading"
        assert (await api.get(f"{BRAIN}/cycles")).json()[0]["kind"] == "portfolio"
        async with api.container.db.session() as s:
            rows = (await s.scalars(select(BrainEventRow).where(BrainEventRow.type == "OrderFilled"))).all()
        assert len(rows) == 1
        assert only_reads(api.fake)


@pytest.mark.parametrize("kind", ["portfolio", "deep", "event"])
async def test_routing_narrows_or_widens_the_cycle(tmp_path, kind):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        api.fake.hold("UPA", 40, 60.0)
        cycle = (
            await api.post(f"{BRAIN}/run", json={"kind": kind, "symbols": ["DNA"] if kind == "event" else []})
        ).json()
        focus = {f["symbol"] for f in cycle["focus"]}
        if kind == "portfolio":
            assert focus == {"UPA"}
        elif kind == "event":
            assert focus == {"UPA", "DNA"}
        else:
            assert len(focus) > 9  # twice the pre-screen and opportunity budget
        ran = {r["agent_id"] for r in cycle["runs"] if r["status"] == "ok"}
        if kind != "deep":
            assert "valuation" not in ran and "technical" in ran
