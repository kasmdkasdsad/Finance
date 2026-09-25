"""Alpaca paper trading end to end: signal → portfolio → risk → order manager → (fake) Alpaca paper API →
fills → reconciliation, through the HTTP API. The market data and the Alpaca paper account are fakes;
nothing touches the network or a real account."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.providers.alpaca_trading import AlpacaPaperBroker
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import STOCKS, TrendFeed

from .conftest import NOW, _client, make_settings

KEY, SECRET = "PKTESTKEYID0001", "test-secret-value-0001"
BASE = "/api/v1/trading"
PAPER = dict(alpaca_trading_enabled=True, trading_dry_run=False)


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    """Every outbound HTTP call in these tests must be mocked: unmocked ones fail at once."""
    return mock_net


def settings_for(tmp_path, **overrides):
    base = dict(
        enable_live_data=True,
        alpaca_api_key_id=KEY,
        alpaca_api_secret_key=SECRET,
        trading_universe=",".join(STOCKS),
        trading_etfs=["SPY", "QQQ"],
        trading_signal_weights={"momentum": 0.4, "trend": 0.3, "volume": 0.15, "volatility": 0.15},
        trading_use_implied_vol=False,
        trading_earnings_blackout_days=0,
        trading_fill_wait_seconds=0,
    )
    base.update(overrides)
    return make_settings(tmp_path, **base)


async def trading_client(
    tmp_path, clock, fake=None, client_host="127.0.0.1", sessions=320, feed=None, **overrides
):
    fake = fake or FakeAlpacaPaper(clock=clock)
    broker = AlpacaPaperBroker(KEY, SECRET, transport=fake)
    async for api in _client(
        settings_for(tmp_path, **overrides), clock, client_host=client_host, broker=broker
    ):
        feed = feed or TrendFeed(clock, sessions=sessions)
        api.container.market._providers[:] = [feed]
        for s in feed.bars:
            fake.prices[s] = feed.live_price(s)
        api.fake, api.feed = fake, feed
        yield api


def buys(cycle):
    return [t for t in cycle["trades"] if t["side"] == "buy"]


async def test_status_and_account_without_keys(tmp_path, clock):
    async for api in _client(make_settings(tmp_path), clock):
        st = (await api.get(f"{BASE}/status")).json()
        assert st["paper"] is True and st["endpoint"] == "https://paper-api.alpaca.markets"
        assert st["banner"] == "ALPACA PAPER TRADING — SIMULATED MONEY ONLY"
        assert not st["broker_configured"] and st["mode"] == "dry_run" and not st["can_submit"]
        assert st["mode_banner"] == "DRY RUN — NO ORDERS WILL BE SUBMITTED"
        assert any("QP_ALPACA_API_KEY_ID" in w for w in st["warnings"])
        r = await api.get(f"{BASE}/account")
        assert r.status_code == 503 and "not configured" in r.json()["detail"]
        assert (await api.post(f"{BASE}/run")).status_code == 503
        assert (await api.get(f"{BASE}/proposed")).json() is None
        assert await api.container.trading.run_scheduled() == "not configured (no Alpaca paper keys)"


async def test_default_settings_are_a_safe_dry_run(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock):  # defaults: trading disabled, dry run on
        st = (await api.get(f"{BASE}/status")).json()
        assert (
            st["broker_configured"]
            and not st["trading_enabled"]
            and st["dry_run"]
            and st["mode"] == "dry_run"
        )
        acct = (await api.get(f"{BASE}/account")).json()
        assert (
            acct["paper"]
            and acct["equity"] == pytest.approx(100_000.0)
            and acct["account_number"].startswith("…")
        )

        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["status"] == "completed" and cycle["mode"] == "dry_run"
        assert "DRY RUN — NO ORDERS WILL BE SUBMITTED" in cycle["notes"]
        assert cycle["regime"]["label"] == "bullish" and cycle["gross_target"] == pytest.approx(0.95)
        top = [s["symbol"] for s in cycle["signals"][:4]]
        assert set(top) <= {"UPA", "UPB", "UPC", "UPD", "UPE"}
        assert all(
            set(s["components"])
            == {"momentum", "trend", "volume", "volatility", "fundamental", "model", "regime"}
            for s in cycle["signals"]
        )
        entries = buys(cycle)
        assert entries and all(
            t["approved"] and t["status"] == "dry_run" and t["client_order_id"] is None for t in entries
        )
        assert all(t["notional"] <= 15_000 for t in entries)
        assert {t["symbol"] for t in entries} <= {t["symbol"] for t in cycle["targets"]}
        # nothing was sent to Alpaca
        assert ("POST", "/v2/orders") not in api.fake.log and api.fake.orders == {}
        assert (await api.get(f"{BASE}/proposed")).json()["id"] == cycle["id"]
        kinds = {e["kind"] for e in (await api.get(f"{BASE}/events")).json()}
        assert {
            "baseline_recorded",
            "signal_generated",
            "trade_proposed",
            "risk_approved",
            "cycle_completed",
        } <= kinds


async def test_paper_execution_submits_fills_and_reconciles(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        st = (await api.get(f"{BASE}/status")).json()
        assert st["mode"] == "paper" and st["can_submit"] and "EXECUTION ACTIVE" in st["mode_banner"]
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["mode"] == "paper" and cycle["status"] == "completed"
        sent = buys(cycle)
        assert sent and all(t["status"] == "filled" for t in sent)
        assert all(t["client_order_id"] == f"qp-m20260925T1000-{t['symbol']}-b" for t in sent)
        assert all(t["order_type"] == "marketable_limit" and t["limit_price"] > t["est_price"] for t in sent)
        assert len(api.fake.orders) == len(sent)

        positions = (await api.get(f"{BASE}/positions")).json()
        assert {p["symbol"] for p in positions} == {t["symbol"] for t in sent}
        assert all(p["target_weight"] > p["weight"] > 0 and p["signal_score"] is not None for p in positions)
        orders = (await api.get(f"{BASE}/orders")).json()
        assert all(o["source"] == "alpaca" and o["strategy"] == "quantpulse" and o["reason"] for o in orders)
        risk = (await api.get(f"{BASE}/risk")).json()
        assert risk["positions"] == len(sent) and 0 < risk["exposure_pct"] <= 0.95 and risk["can_submit"]

        # The same minute again (double click, poller retry): the cycle is not re-run, nothing is resent.
        again = (await api.post(f"{BASE}/run")).json()
        assert again["id"] == cycle["id"] and len(api.fake.orders) == len(sent)

        # Half an hour later the strategy scales further into its targets (no churn, no duplicates).
        clock.advance(1800)
        later = (await api.post(f"{BASE}/run")).json()
        adds = buys(later)
        assert later["cycle_key"] == "m20260925T1030" and adds
        assert all(t["kind"] == "add" and t["client_order_id"].startswith("qp-m20260925T1030-") for t in adds)
        assert not [t for t in later["trades"] if t["side"] == "sell"]
        acct = (await api.get(f"{BASE}/account")).json()
        assert (
            acct["exposure_pct"] > 0.8
            and acct["total_pl"] is not None
            and acct["baseline_equity"] == pytest.approx(100_000.0)
        )
        kinds = [e["kind"] for e in (await api.get(f"{BASE}/events")).json()]
        assert {"order_submitted", "order_filled", "risk_approved", "cycle_completed"} <= set(kinds)


async def test_restart_does_not_duplicate_orders(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        first = (await api.post(f"{BASE}/run")).json()
    n = len(fake.orders)
    assert n == len(buys(first))
    # a new process on the same database and the same Alpaca account
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        rec = (await api.post(f"{BASE}/reconcile")).json()
        assert rec["positions"] == n and rec["orders_added"] == 0
        again = (await api.post(f"{BASE}/run")).json()
        assert again["id"] == first["id"]  # same minute: the recorded cycle is returned
        assert len(fake.orders) == n
        cid = buys(first)[0]["client_order_id"]
        assert (
            fake.log.count(("POST", "/v2/orders")) == n
            and len({o["client_order_id"] for o in fake.orders.values()}) == n
        )
        assert cid in fake.by_client


async def test_partial_fills_rejections_and_timeouts(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.fill_mode.update(UPA="partial", UPB="reject", UPC="timeout")
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        by = {t["symbol"]: t for t in buys(cycle)}
        assert by["UPA"]["status"] == "partially_filled"
        assert by["UPB"]["status"] == "rejected" and "insufficient buying power" in by["UPB"]["error"]
        assert by["UPC"]["status"] == "filled"  # the timed-out request had arrived: found by client id
        assert sum(1 for o in fake.orders.values() if o["symbol"] == "UPC") == 1  # never resent
        kinds = {e["kind"] for e in (await api.get(f"{BASE}/events")).json()}
        assert {"order_partially_filled", "order_rejected", "order_filled"} <= kinds
        # the partial order is still working: the next cycle leaves UPA alone rather than stacking orders
        clock.advance(600)
        nxt = (await api.post(f"{BASE}/run")).json()
        assert "UPA" not in {t["symbol"] for t in nxt["trades"]}
        assert "still working" in nxt["skipped"].get("UPA", "")
        # once it is old enough, a later cycle cancels the stale remainder before re-planning
        clock.advance(1500)
        (await api.post(f"{BASE}/run")).json()
        upa = [o for o in fake.orders.values() if o["symbol"] == "UPA"]
        assert upa[0]["status"] == "canceled"


async def test_kill_switch_stops_new_orders(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        ks = (await api.post(f"{BASE}/kill-switch", json={"active": True, "reason": "testing"})).json()
        assert ks["active"] and ks["source"] == "runtime" and ks["reason"] == "testing"
        st = (await api.get(f"{BASE}/status")).json()
        assert st["mode"] == "dry_run" and not st["can_submit"]
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["mode"] == "dry_run" and cycle["trades"]
        assert all(not t["approved"] and "kill_switch" in t["risk"] for t in cycle["trades"])
        assert api.fake.orders == {}
        assert (await api.post(f"{BASE}/kill-switch", json={"active": False})).json()["active"] is False
        clock.advance(60)
        assert buys((await api.post(f"{BASE}/run")).json())[0]["status"] == "filled"
        kinds = {e["kind"] for e in (await api.get(f"{BASE}/events")).json()}
        assert {"kill_switch_activated", "kill_switch_released"} <= kinds


async def test_env_kill_switch_cannot_be_released_from_the_api(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, trading_kill_switch=True, **PAPER):
        st = (await api.get(f"{BASE}/status")).json()
        assert st["kill_switch"] == {
            "active": True,
            "source": "env",
            "reason": "QP_TRADING_KILL_SWITCH=true",
            "changed_at": None,
        }
        r = await api.post(f"{BASE}/kill-switch", json={"active": False})
        assert r.status_code == 422 and "QP_TRADING_KILL_SWITCH" in r.json()["detail"]


async def test_cancel_all_and_close_all_need_confirmation(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.hold("UPA", 10, 300.0)
    fake.hold("DNB", 20, 100.0)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        fake.fill_mode["FLAT"] = "accept"
        from quantpulse.providers.alpaca_trading import OrderSpec

        await api.container.broker.submit(OrderSpec("FLAT", "buy", 1, "limit", "manual-resting", 1.0))
        assert (await api.post(f"{BASE}/cancel-all", json={"confirm": False})).status_code == 422
        out = (await api.post(f"{BASE}/cancel-all", json={"confirm": True})).json()
        assert (
            out["canceled"] == 1 and (await api.get(f"{BASE}/orders", params={"status": "open"})).json() == []
        )

        bad = await api.post(f"{BASE}/close-all", json={"confirm": "close all please"})
        assert bad.status_code == 422 and "CLOSE ALL" in bad.json()["detail"]
        assert (await api.post(f"{BASE}/close-all", json={})).status_code == 422
        done = (await api.post(f"{BASE}/close-all", json={"confirm": "CLOSE ALL"})).json()
        assert done["mode"] == "paper" and done["submitted"] == 2
        assert all(t["order_type"] == "market" and t["kind"] == "flatten" for t in done["trades"])
        assert fake.positions == {}


async def test_close_all_is_a_preview_in_dry_run(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.hold("UPA", 10, 300.0)
    async for api in trading_client(tmp_path, clock, fake=fake):
        out = (await api.post(f"{BASE}/close-all", json={"confirm": "CLOSE ALL"})).json()
        assert out["mode"] == "dry_run" and out["submitted"] == 0 and "DRY RUN" in out["message"]
        assert [t["symbol"] for t in out["trades"]] == ["UPA"] and "UPA" in fake.positions
        r = await api.post(f"{BASE}/cancel-all", json={"confirm": True})
        assert r.status_code == 422 and "QP_ALPACA_TRADING_ENABLED=false" in r.json()["detail"]


async def test_market_closed_and_missing_live_data_block_orders(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["trades"] and all("market_open" in t["risk"] for t in cycle["trades"])
        assert fake.orders == {}
        fake.market_open = True
        api.feed.quotes_enabled = False  # the live quote feed is down
        clock.advance(60)
        dark = (await api.post(f"{BASE}/run")).json()
        assert (
            dark["trades"] == []
            and dark["signals"]
            and all(s["entry_blocks"] == ["no live quote"] for s in dark["signals"])
        )
        assert fake.orders == {}


async def test_synthetic_prices_are_never_traded(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, enable_live_data=False, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["data_status"] == "synthetic" and cycle["trades"] == []
        assert any("SYNTHETIC" in n for n in cycle["notes"]) and fake.orders == {}
        st = (await api.get(f"{BASE}/status")).json()
        assert any("QP_ENABLE_LIVE_DATA=false" in w for w in st["warnings"])


async def test_stop_loss_and_daily_loss_limit(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        price = api.feed.live_price("UPA")
        fake.hold("UPA", 20, price * 1.12, price=price)  # down ~11%
        fake.last_equity = fake.equity() / 0.955  # the account is down 4.5% today
        cycle = (await api.post(f"{BASE}/run")).json()
        stop = [t for t in cycle["trades"] if t["symbol"] == "UPA"]
        assert stop[0]["kind"] == "stop_loss" and stop[0]["status"] == "filled"
        assert all(not t["approved"] and "daily_loss" in t["risk"] for t in buys(cycle))
        assert "UPA" not in fake.positions
        kinds = [e["kind"] for e in (await api.get(f"{BASE}/events")).json()]
        assert kinds.count("daily_loss_limit_reached") == 1
        assert (await api.get(f"{BASE}/risk")).json()["daily_loss_limit_hit"] is True


async def test_daily_loss_flatten_policy(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, trading_daily_loss_action="flatten", **PAPER):
        fake.hold("UPB", 10, api.feed.live_price("UPB"))
        fake.last_equity = fake.equity() / 0.95
        cycle = (await api.post(f"{BASE}/run")).json()
        assert [t["kind"] for t in cycle["trades"]] == ["daily_loss_flatten"]
        assert fake.positions == {}


async def test_scheduler_runs_each_slot_once(tmp_path):
    """Scheduled cycles stay dry runs until paper execution was used once by hand (arming), then each
    slot runs once in paper mode."""
    clock = FakeClock(datetime(2026, 9, 25, 13, 45, tzinfo=UTC))  # 09:45 New York
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        trading = api.container.trading
        assert await trading.run_scheduled() == "no cycle due"
        assert any(e["kind"] == "reconciliation_completed" for e in (await api.get(f"{BASE}/events")).json())
        clock.advance(20 * 60)  # 10:05
        first = await trading.run_scheduled()
        assert first.startswith("cycle 20260925T1000-dry completed (dry_run")
        assert fake.orders == {} and ("POST", "/v2/orders") not in fake.log
        dry = (await api.get(f"{BASE}/proposed")).json()
        assert any("not armed" in n for n in dry["notes"])
        st = (await api.get(f"{BASE}/status")).json()
        assert st["can_submit"] and st["submit_blockers"] == []  # a manual cycle would send orders
        assert not st["scheduler_armed"] and st["scheduled_mode"] == "dry_run"
        assert await trading.run_scheduled() == "cycle 20260925T1000-dry completed"

        manual = (await api.post(f"{BASE}/run")).json()  # the user's explicit first paper cycle arms it
        assert manual["mode"] == "paper" and manual["cycle_key"] == "m20260925T1005" and fake.orders
        st = (await api.get(f"{BASE}/status")).json()
        assert st["scheduler_armed"] and st["scheduled_mode"] == "paper"
        assert "paper_armed" in {e["kind"] for e in (await api.get(f"{BASE}/events")).json()}

        assert (await trading.run_scheduled()).startswith("cycle 20260925T1000 completed (paper")
        assert await trading.run_scheduled() == "cycle 20260925T1000 completed"
        n = len(fake.orders)
        clock.advance(30 * 60)  # 10:35
        assert (await trading.run_scheduled()).startswith("cycle 20260925T1030 completed (paper")
        assert len(fake.orders) >= n
        status = (await api.get(f"{BASE}/status")).json()
        assert (
            status["next_cycle_at"].startswith("2026-09-25T11:00")
            and status["last_cycle"]["trigger"] == "schedule"
        )
        cycles = (await api.get(f"{BASE}/cycles")).json()
        assert [c["cycle_key"] for c in cycles] == [
            "20260925T1030",
            "20260925T1000",
            "m20260925T1005",
            "20260925T1000-dry",
        ]


async def test_scheduler_without_arming_sends_from_the_first_slot(tmp_path):
    clock = FakeClock(datetime(2026, 9, 25, 14, 5, tzinfo=UTC))  # 10:05 New York
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(
        tmp_path, clock, fake=fake, trading_scheduler_requires_arming=False, **PAPER
    ):
        assert (await api.container.trading.run_scheduled()).startswith(
            "cycle 20260925T1000 completed (paper"
        )
        assert fake.orders


async def test_order_endpoints_refuse_remote_callers_without_a_token(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, client_host="203.0.113.9", **PAPER):
        for path, body in (
            ("/run", None),
            ("/kill-switch", {"active": True}),
            ("/cancel-all", {"confirm": True}),
            ("/close-all", {"confirm": "CLOSE ALL"}),
        ):
            r = await api.post(f"{BASE}{path}", json=body)
            assert r.status_code == 403 and "QP_API_TOKEN" in r.json()["detail"]
        assert (await api.get(f"{BASE}/status")).status_code == 200  # read-only views stay available
        assert api.fake.orders == {}
    async for api in trading_client(tmp_path, clock, client_host="203.0.113.9", api_token="tok-123", **PAPER):
        assert (await api.post(f"{BASE}/run")).status_code == 401
        r = await api.post(f"{BASE}/run", headers={"X-API-Key": "tok-123"})
        assert r.status_code == 200 and r.json()["mode"] == "paper"


async def test_credentials_never_leave_the_server(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        await api.post(f"{BASE}/run")
        bodies = []
        for path in (
            "/status",
            "/account",
            "/positions",
            "/orders",
            "/proposed",
            "/risk",
            "/events",
            "/cycles",
            "/performance",
            "/job",
        ):
            r = await api.get(f"{BASE}{path}")
            assert r.status_code == 200, path
            bodies.append(r.text)
        bodies.append((await api.get("/api/v1/system/status")).text)
        text = "\n".join(bodies)
        assert KEY not in text and SECRET not in text
        assert json.loads(bodies[0])["paper"] is True


async def test_performance_comes_from_recorded_cycles_and_fills(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        empty = (await api.get(f"{BASE}/performance")).json()
        assert empty["days"] == 0 and empty["sharpe"] is None and empty["notes"]
        await api.post(f"{BASE}/run")
        perf = (await api.get(f"{BASE}/performance")).json()
        assert perf["days"] == 1 and perf["sharpe"] is None and perf["round_trips"] == 0
        assert perf["turnover"] and perf["turnover"] > 0 and perf["avg_exposure"] is not None


async def test_the_stock_model_feeds_the_opportunity_score(tmp_path):
    clock = FakeClock(NOW)
    weights = {"momentum": 0.3, "trend": 0.2, "model": 0.3, "fundamental": 0.2}
    async for api in trading_client(
        tmp_path,
        clock,
        sessions=800,
        trading_signal_weights=weights,
        model_universe=",".join(STOCKS),
        model_type="ridge",
        trading_model_wait_seconds=120,
    ):
        cycle = (await api.post(f"{BASE}/run", params={"wait": 300})).json()
        assert cycle["status"] == "completed"
        assert any(n.startswith("stock model (") for n in cycle["notes"])
        scored = {s["symbol"]: s for s in cycle["signals"] if s["model_z"] is not None}
        assert set(scored) <= set(STOCKS) and len(scored) >= 5  # ETFs have no model score
        assert any(abs(s["components"]["model"]) > 0.1 for s in scored.values())


async def test_a_model_without_enough_history_counts_as_neutral(tmp_path):
    clock = FakeClock(NOW)
    weights = {"momentum": 0.5, "model": 0.5}
    async for api in trading_client(
        tmp_path, clock, trading_signal_weights=weights, model_universe=",".join(STOCKS)
    ):
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["status"] == "completed"
        assert any("stock model unavailable" in n and "neutral" in n for n in cycle["notes"])
        assert all(s["components"]["model"] == 0 for s in cycle["signals"])


async def test_manual_runs_can_be_forced_to_dry_run(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        cycle = (await api.post(f"{BASE}/run", params={"dry_run": True})).json()
        assert cycle["mode"] == "dry_run" and api.fake.orders == {}
        job = (await api.get(f"{BASE}/job")).json()
        assert job["kind"] == "trading" and job["status"] == "done"


async def test_cycle_detail_and_event_filters(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        detail = (await api.get(f"{BASE}/cycles/{cycle['id']}")).json()
        assert detail["trades"] == cycle["trades"] and detail["regime"]["label"] == "bullish"
        assert (await api.get(f"{BASE}/cycles/999")).status_code == 404
        only = (await api.get(f"{BASE}/events", params={"kind": "order_filled"})).json()
        assert only and {e["kind"] for e in only} == {"order_filled"}
        assert all(e["client_order_id"].startswith("qp-") for e in only)


def test_slots_follow_the_session():
    from quantpulse.services.trading import cycle_slots

    day = datetime(2026, 9, 25).date()
    slots = cycle_slots(day, datetime.strptime("10:00", "%H:%M").time(), 30, 15)
    assert (
        slots[0].strftime("%H:%M") == "10:00" and slots[-1].strftime("%H:%M") == "15:30" and len(slots) == 12
    )
    early = cycle_slots(datetime(2026, 11, 27).date(), datetime.strptime("10:00", "%H:%M").time(), 30, 15)
    assert early[-1].strftime("%H:%M") == "12:30"  # the day after Thanksgiving closes at 13:00
    assert cycle_slots(datetime(2026, 9, 26).date(), datetime.strptime("10:00", "%H:%M").time(), 30, 15) == []
    assert timedelta(minutes=30) == slots[1] - slots[0]


async def test_kill_switch_cancels_only_quantpulse_orders(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.default_mode = "accept"  # orders rest instead of filling
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        from quantpulse.providers.alpaca_trading import OrderSpec

        await api.container.broker.submit(OrderSpec("FLAT", "buy", 1, "limit", "my-own-manual-order", 1.0))
        cycle = (await api.post(f"{BASE}/run")).json()
        ours = [t["client_order_id"] for t in buys(cycle)]
        assert ours and all(t["status"] == "accepted" for t in buys(cycle))
        await api.post(f"{BASE}/kill-switch", json={"active": True, "cancel_open_orders": True})
        status = {o["client_order_id"]: o["status"] for o in fake.orders.values()}
        assert all(status[cid] == "canceled" for cid in ours)
        assert status["my-own-manual-order"] == "accepted"  # the user's own order is left alone


async def test_cycles_cut_short_by_a_restart_are_closed_out(tmp_path):
    from quantpulse.db import repositories as repo

    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        async with api.container.db.session() as s:
            await repo.create_trading_cycle(
                s,
                cycle_key="20260925T0930",
                trigger="schedule",
                mode="paper",
                status="running",
                started_at=NOW - timedelta(minutes=40),
                regime={},
                positions=[],
                signals=[],
                targets=[],
                trades=[],
                plan={},
                notes=[],
            )
        await api.container.trading.reconcile("startup")
        [row] = (await api.get(f"{BASE}/cycles")).json()
        assert row["status"] == "failed"
        detail = (await api.get(f"{BASE}/cycles/{row['id']}")).json()
        assert "interrupted by a restart" in detail["error"]
