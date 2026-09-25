"""Regression tests for the path from a risk-approved trade to the Alpaca paper account.

Each test drives the real stack (HTTP API → trading service → risk engine → order manager → the official
alpaca-py SDK) against the fake Alpaca *paper* API, and checks what actually reached it: the POST
/v2/orders requests and their JSON bodies. Nothing touches the network or a real account.

A  dry run                           → zero submissions
B  trading disabled                  → zero submissions
C  paper + enabled + not dry run     → the orders reach Alpaca
D  kill switch                       → zero submissions
E  risk rejection                    → zero submissions
F  repeated cycle                    → no duplicate order
G  Alpaca rejects                    → the error is persisted and shown
H  success                           → the Alpaca order id is persisted
I  reconciliation                    → local records follow Alpaca
J  live endpoint                     → cannot be selected
K  missing / refused credentials     → a clear error, never a secret
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from quantpulse.config import Settings
from quantpulse.core.clock import FakeClock
from quantpulse.db import repositories as repo
from quantpulse.domain.trading_portfolio import PortfolioPlan, ProposedTrade
from quantpulse.providers.alpaca_trading import PAPER_URL, AlpacaPaperBroker
from quantpulse.services import trading as trading_mod
from quantpulse.services.order_manager import SUBMIT_FAILED
from quantpulse.services.trading_risk import OrderIntent
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import TrendFeed

from .conftest import NOW, _client, make_settings
from .test_trading import BASE, KEY, PAPER, SECRET, buys, trading_client

PHRASE = "SUBMIT ONE PAPER TEST ORDER"


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    return mock_net


def order_posts(fake: FakeAlpacaPaper) -> int:
    return sum(1 for x in fake.log if x == ("POST", "/v2/orders"))


def writes(fake: FakeAlpacaPaper) -> list[tuple[str, str]]:
    return [x for x in fake.log if x[0] in ("POST", "DELETE", "PATCH")]


# --------------------------------------------------------------------------- A, B, D, E: nothing is sent
async def test_a_dry_run_sends_nothing(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, alpaca_trading_enabled=True, trading_dry_run=True):
        st = (await api.get(f"{BASE}/status")).json()
        assert st["mode"] == "dry_run" and not st["can_submit"]
        assert any("QP_TRADING_DRY_RUN=true" in b for b in st["submit_blockers"])
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["mode"] == "dry_run" and cycle["cycle_key"].endswith("-dry")
        entries = buys(cycle)
        assert entries and all(
            t["approved"] and t["stage"] == "risk_approved" and t["alpaca_order_id"] is None for t in entries
        )
        assert any(n.startswith("Not sent to Alpaca because: QP_TRADING_DRY_RUN") for n in cycle["notes"])
        assert order_posts(api.fake) == 0 and api.fake.bodies == [] and writes(api.fake) == []
        [summary] = (await api.get(f"{BASE}/cycles")).json()
        assert summary["orders_submitted"] == 0 and summary["trades_approved"] == len(cycle["trades"])


async def test_b_trading_disabled_sends_nothing(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, alpaca_trading_enabled=False, trading_dry_run=False):
        st = (await api.get(f"{BASE}/status")).json()
        assert st["mode"] == "dry_run"
        assert st["submit_blockers"] == ["QP_ALPACA_TRADING_ENABLED=false: order submission is disabled"]
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["mode"] == "dry_run" and buys(cycle)
        assert order_posts(api.fake) == 0 and writes(api.fake) == []
        r = await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})
        assert r.status_code == 422 and "NOT sent" in r.json()["detail"]
        assert order_posts(api.fake) == 0


async def test_d_kill_switch_blocks_every_order(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, trading_kill_switch=True, **PAPER):
        st = (await api.get(f"{BASE}/status")).json()
        assert st["mode"] == "dry_run"
        assert any("kill switch ON (QP_TRADING_KILL_SWITCH=true" in b for b in st["submit_blockers"])
        await api.post(f"{BASE}/run")
        assert order_posts(api.fake) == 0
    async for api in trading_client(tmp_path / "runtime", clock, **PAPER):
        await api.post(f"{BASE}/kill-switch", json={"active": True, "reason": "test"})
        st = (await api.get(f"{BASE}/status")).json()
        assert st["mode"] == "dry_run" and any("dashboard/API: test" in b for b in st["submit_blockers"])
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["mode"] == "dry_run" and all(t["alpaca_order_id"] is None for t in cycle["trades"])
        r = await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})
        assert r.status_code == 422 and "kill switch ON" in r.json()["detail"]
        assert order_posts(api.fake) == 0
        await api.post(f"{BASE}/kill-switch", json={"active": False})
        clock.advance(60)
        assert (await api.post(f"{BASE}/run")).json()["mode"] == "paper"
        assert order_posts(api.fake) > 0


async def test_e_risk_rejection_sends_nothing(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False  # Alpaca's clock says closed: every order fails market_open
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["mode"] == "paper" and cycle["trades"]
        assert all(t["stage"] == "risk_rejected" and not t["approved"] for t in cycle["trades"])
        assert all(
            any(c["name"] == "market_open" and not c["passed"] for c in t["checks"]) for t in cycle["trades"]
        )
        assert order_posts(fake) == 0
    fake = FakeAlpacaPaper(clock=clock)
    fake.buying_power_override = 500.0  # the planner sizes by equity; only the risk engine sees this
    async for api in trading_client(tmp_path / "bp", clock, fake=fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        rejected = buys(cycle)
        assert rejected and all(not t["approved"] and "buying_power" in t["risk"] for t in rejected)
        assert order_posts(fake) == 0


# --------------------------------------------------------------------------- C, H: orders reach Alpaca
async def test_c_h_paper_mode_sends_the_orders_and_keeps_their_alpaca_ids(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        st = (await api.get(f"{BASE}/status")).json()
        assert st["can_submit"] and st["submit_blockers"] == []
        cycle = (await api.post(f"{BASE}/run")).json()
        approved = [t for t in cycle["trades"] if t["approved"]]
        assert approved and order_posts(api.fake) == len(approved)
        by_cid = {b["client_order_id"]: b for b in api.fake.bodies}
        for t in approved:
            body = by_cid[t["client_order_id"]]
            assert (body["symbol"], body["side"], body["time_in_force"]) == (t["symbol"], t["side"], "day")
            assert body["qty"] == pytest.approx(t["qty"]) and body["type"] == "limit"
            assert body["limit_price"] == pytest.approx(t["limit_price"])
            # H: the Alpaca order id is kept on the trade, on the order row and in /orders
            assert t["alpaca_order_id"] == api.fake.by_client[t["client_order_id"]]
            assert t["stage"] == "filled" and t["filled_qty"] == pytest.approx(t["qty"])
        async with api.container.db.session() as s:
            rows = await repo.broker_orders_by_client_ids(s, [t["client_order_id"] for t in approved])
        assert {r.alpaca_order_id for r in rows.values()} == {t["alpaca_order_id"] for t in approved}
        listed = {o["client_order_id"]: o for o in (await api.get(f"{BASE}/orders")).json()}
        assert all(listed[t["client_order_id"]]["alpaca_order_id"] == t["alpaca_order_id"] for t in approved)
        [summary] = (await api.get(f"{BASE}/cycles")).json()
        assert summary["orders_submitted"] == len(approved) == summary["orders_filled"]
        events = (await api.get(f"{BASE}/events", params={"kind": "order_submitted"})).json()
        assert all("Alpaca order" in e["message"] for e in events) and len(events) == len(approved)


# --------------------------------------------------------------------------- F: duplicates
async def test_f_a_repeated_cycle_never_duplicates_an_order(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        first = (await api.post(f"{BASE}/run")).json()
        sent = order_posts(api.fake)
        assert sent > 0
        again = (await api.post(f"{BASE}/run")).json()  # a double click in the same minute
        assert again["id"] == first["id"] and order_posts(api.fake) == sent
        # the order manager refuses a client order id it already used, without calling Alpaca
        t = buys(first)[0]
        intent = OrderIntent(t["symbol"], "buy", t["qty"], t["est_price"], "entry", "again")
        sub = await api.container.trading.orders.submit(
            intent, cid=t["client_order_id"], order_type="limit", limit_price=t["limit_price"], cycle_id=None
        )
        assert sub.duplicate and not sub.submitted and order_posts(api.fake) == sent


async def test_f_a_dry_run_never_blocks_the_paper_run_of_the_same_minute(tmp_path):
    """D1: the dry run and the paper run of one minute used to share a cycle key, so the paper run
    silently returned the dry run and sent nothing."""
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        dry = (await api.post(f"{BASE}/run", params={"dry_run": True})).json()
        assert dry["mode"] == "dry_run" and dry["cycle_key"] == "m20260925T1000-dry"
        assert any("dry run requested" in n for n in dry["notes"]) and order_posts(api.fake) == 0
        paper = (await api.post(f"{BASE}/run")).json()
        assert paper["mode"] == "paper" and paper["cycle_key"] == "m20260925T1000"
        assert order_posts(api.fake) == sum(1 for t in paper["trades"] if t["approved"]) > 0


# --------------------------------------------------------------------------- G: Alpaca rejects
async def test_g_an_alpaca_rejection_is_recorded_and_shown(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.default_mode = "reject"
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        sent = [t for t in cycle["trades"] if t["approved"]]
        assert sent and order_posts(fake) == len(sent)
        for t in sent:
            assert t["stage"] == "rejected" and t["status"] == "rejected" and t["alpaca_order_id"] is None
            assert "insufficient buying power" in t["error"] and "HTTP 403" in t["error"]
        async with api.container.db.session() as s:
            rows = await repo.broker_orders_by_client_ids(s, [t["client_order_id"] for t in sent])
        assert all(r.status == "rejected" and "insufficient buying power" in r.error for r in rows.values())
        listed = [o for o in (await api.get(f"{BASE}/orders")).json() if o["status"] == "rejected"]
        assert len(listed) == len(sent) and all(o["source"] == "quantpulse" and o["error"] for o in listed)
        kinds = [e["kind"] for e in (await api.get(f"{BASE}/events")).json()]
        assert kinds.count("order_rejected") == len(sent)
        [summary] = (await api.get(f"{BASE}/cycles")).json()
        assert summary["orders_submitted"] == 0 and summary["orders_failed"] == len(sent)
        assert (await api.get(f"{BASE}/proposed")).json()["trades"][0]["stage"] == "rejected"


async def test_an_invalid_order_is_marked_failed_never_left_pending(tmp_path):
    """D2: an order the SDK cannot build used to escape before the outcome was recorded, leaving a
    ``pending_submit`` row that blocked the symbol until reconciliation."""
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        manager = api.container.trading.orders
        bad = OrderIntent("UPA", "buy", 0.0, 100.0, "entry", "zero quantity")
        sub = await manager.submit(
            bad, cid="qp-t-UPA-b", order_type="market", limit_price=None, cycle_id=None
        )
        assert sub.status == SUBMIT_FAILED and not sub.submitted and "never sent" in sub.error
        async with api.container.db.session() as s:
            row = await repo.get_broker_order(s, "qp-t-UPA-b")
        assert row.status == SUBMIT_FAILED and order_posts(api.fake) == 0
        assert "UPA" not in await manager.unresolved_symbols()


# --------------------------------------------------------------------------- I: reconciliation
async def test_i_reconciliation_follows_alpaca(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.default_mode = "accept"  # orders rest on the book
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        working = buys(cycle)
        assert working and all(t["stage"] == "accepted" and t["alpaca_order_id"] for t in working)
        first, *rest = working
        fake.complete(first["client_order_id"])  # Alpaca fills one...
        fake._cancel(fake.orders[fake.by_client[rest[0]["client_order_id"]]])  # ...and cancels another
        rec = (await api.post(f"{BASE}/reconcile")).json()
        assert rec["orders_updated"] >= 2
        trades = {t["client_order_id"]: t for t in (await api.get(f"{BASE}/proposed")).json()["trades"]}
        done = trades[first["client_order_id"]]
        alpaca = fake.orders[fake.by_client[first["client_order_id"]]]
        assert done["stage"] == "filled" and done["filled_qty"] == pytest.approx(float(alpaca["filled_qty"]))
        assert done["filled_avg_price"] == pytest.approx(float(alpaca["filled_avg_price"]))
        assert trades[rest[0]["client_order_id"]]["stage"] == "canceled"
        positions = {p["symbol"]: p["qty"] for p in (await api.get(f"{BASE}/positions")).json()}
        assert positions == {s: p["qty"] for s, p in fake.positions.items()}


# --------------------------------------------------------------------------- J: never live
def test_j_the_live_endpoint_cannot_be_selected(monkeypatch):
    with pytest.raises(ValidationError, match="PAPER"):
        Settings(_env_file=None, alpaca_paper=False)
    monkeypatch.setenv("QP_ALPACA_PAPER", "false")
    with pytest.raises(ValidationError, match="PAPER"):
        Settings(_env_file=None)
    monkeypatch.delenv("QP_ALPACA_PAPER")
    # the SDK's and QuantPulse's URL variables change nothing: there is no trading URL setting at all
    for name in ("APCA_API_BASE_URL", "ALPACA_BASE_URL", "QP_ALPACA_BASE_URL", "QP_ALPACA_TRADING_URL"):
        monkeypatch.setenv(name, "https://api.alpaca.markets")
    s = Settings(_env_file=None, alpaca_api_key_id=KEY, alpaca_api_secret_key=SECRET)
    assert not [f for f in Settings.model_fields if "alpaca" in f and "url" in f and f != "alpaca_data_url"]
    assert AlpacaPaperBroker(KEY, SECRET).verify_paper_client() == PAPER_URL
    assert s.alpaca_paper is True


# --------------------------------------------------------------------------- K: credentials
async def test_k_missing_credentials_give_a_clear_error_without_secrets(tmp_path, clock):
    settings = make_settings(
        tmp_path, alpaca_api_key_id=KEY, alpaca_trading_enabled=True, trading_dry_run=False
    )
    async for api in _client(settings, clock):
        st = (await api.get(f"{BASE}/status")).json()
        assert not st["broker_configured"] and st["mode"] == "dry_run"
        assert any("QP_ALPACA_API_SECRET_KEY" in b for b in st["submit_blockers"])
        r = await api.post(f"{BASE}/run")
        assert r.status_code == 503 and "QP_ALPACA_API_SECRET_KEY" in r.json()["detail"]
        diag = (await api.get(f"{BASE}/diagnostics")).json()
        checks = {c["name"]: c for c in diag["checks"]}
        assert (
            checks["credentials"]["ok"] is False
            and "QP_ALPACA_API_SECRET_KEY" in checks["credentials"]["detail"]
        )
        assert diag["credentials"]["key_id_set"] and not diag["credentials"]["secret_set"]
        assert checks["account"]["ok"] is None and not diag["test_order"]["allowed"]
        assert KEY not in r.text and KEY not in str(diag)


async def test_k_refused_keys_are_reported_without_secrets(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        fake.fail_status = 401  # Alpaca refuses the keys (e.g. live keys used against the paper API)
        r = await api.get(f"{BASE}/diagnostics")
        diag = r.json()
        checks = {c["name"]: c for c in diag["checks"]}
        assert checks["account"]["ok"] is False and "HTTP 401" in checks["account"]["detail"]
        assert "paper" in checks["account"]["detail"] and checks["positions"]["ok"] is None
        assert diag["credentials"]["key_id_looks_like_paper"] is True
        assert KEY not in r.text and SECRET not in r.text and writes(fake) == []


# --------------------------------------------------------------------------- diagnostics and the test order
async def test_diagnostics_check_the_whole_connection_and_never_trade(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.hold("UPA", 3, 50.0)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        r = await api.get(f"{BASE}/diagnostics", params={"symbols": "UPA,UPB"})
        diag = r.json()
        assert diag["endpoint"] == PAPER_URL and diag["endpoint_verified"] and not diag["orders_sent"]
        assert [c["name"] for c in diag["checks"]] == [
            "settings",
            "credentials",
            "sdk_client",
            "account",
            "market_clock",
            "positions",
            "open_orders",
            "quotes",
        ]
        assert all(c["ok"] for c in diag["checks"])
        assert diag["account"]["paper"] and diag["account"]["equity"] > 0
        assert [p["symbol"] for p in diag["positions"]] == ["UPA"] and diag["open_orders"] == []
        assert {q["symbol"] for q in diag["quotes"]} == {"UPA", "UPB"}
        assert all(q["spread_ok"] and q["spread_bps"] < 30 for q in diag["quotes"])
        assert diag["mode"] == "paper" and diag["test_order"]["allowed"]
        assert diag["config"]["sources"] and diag["config"]["drift"] == []
        assert writes(fake) == [] and SECRET not in r.text and KEY not in r.text


async def test_the_test_order_needs_the_phrase_and_paper_execution(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock):  # defaults: disabled, dry run
        r = await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})
        assert r.status_code == 422 and "QP_TRADING_DRY_RUN=true" in r.json()["detail"]
    async for api in trading_client(tmp_path / "paper", clock, **PAPER):
        r = await api.post(f"{BASE}/test-order", json={"confirm": "yes"})
        assert r.status_code == 422 and PHRASE in r.json()["detail"]
        assert order_posts(api.fake) == 0
    async for api in trading_client(tmp_path / "remote", clock, client_host="203.0.113.9", **PAPER):
        r = await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})
        assert r.status_code == 403 and order_posts(api.fake) == 0


async def test_the_test_order_rests_then_is_canceled(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.fill_mode["SPY"] = "accept"
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        assert not (await api.get(f"{BASE}/status")).json()["scheduler_armed"]
        r = await api.post(f"{BASE}/test-order", json={"confirm": PHRASE, "symbol": "spy"})
        out = r.json()
        assert r.status_code == 200 and out["sent"] and out["canceled"], out
        assert out["alpaca_order_id"] == fake.by_client[out["client_order_id"]]
        assert out["statuses_seen"] == ["accepted", "canceled"] and out["final_status"] == "canceled"
        assert out["client_order_id"].startswith("qp-test-20260925T100000-SPY")
        [body] = fake.bodies
        bid = api.feed.live_price("SPY") * 0.9998
        assert body["type"] == "limit" and body["qty"] == 1 and body["side"] == "buy"
        assert body["limit_price"] == pytest.approx(round(bid * 0.9, 2))
        assert fake.positions == {} and order_posts(fake) == 1
        async with api.container.db.session() as s:
            row = await repo.get_broker_order(s, out["client_order_id"])
        assert row.strategy == "diagnostic" and row.alpaca_order_id == out["alpaca_order_id"]
        assert row.status == "canceled"
        assert (await api.get(f"{BASE}/status")).json()["scheduler_armed"]  # the user exercised paper once
        again = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        assert not again["sent"] and "one_test_at_a_time" in again["message"] and order_posts(fake) == 1


async def test_the_fill_test_order_spends_cash_never_margin(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        out = (
            await api.post(f"{BASE}/test-order", json={"confirm": PHRASE, "mode": "fill", "notional": 10})
        ).json()
        assert out["sent"] and out["final_status"] == "filled" and out["filled_qty"] > 0
        [body] = fake.bodies
        assert body["notional"] == 10.0 and "qty" not in body and body["type"] == "market"
        assert fake.positions["SPY"]["qty"] * fake.price("SPY") == pytest.approx(10.0)
    fake = FakeAlpacaPaper(clock=clock)
    fake.cash, fake.buying_power_override = -500.0, 5_000.0  # margin buying power but negative cash
    async for api in trading_client(tmp_path / "margin", clock, fake=fake, **PAPER):
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE, "mode": "fill"})).json()
        assert not out["sent"] and "buying_power" in out["message"] and order_posts(fake) == 0
        r = await api.post(f"{BASE}/test-order", json={"confirm": PHRASE, "mode": "fill", "notional": 30})
        assert r.status_code == 422  # $25 at most
    fake = FakeAlpacaPaper(clock=clock)
    fake.fill_mode["SPY"] = "reject"
    async for api in trading_client(tmp_path / "refused", clock, fake=fake, **PAPER):
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        assert not out["sent"] and out["final_status"] == "rejected" and out["alpaca_order_id"] is None
        assert out["message"].startswith("Alpaca refused the test order") and "HTTP 403" in out["error"]
        assert not (await api.get(f"{BASE}/status")).json()["scheduler_armed"]


# --------------------------------------------------------------------------- the user's account
# Positions and proposed trades from the reported dry run (Alpaca marks scaled so that equity ≈ $41,000.62,
# cash ≈ −$17,028.93 and long market value ≈ $58,029.55, as on the account).
HELD = {
    "ADBE": (122.424494693, 254.97),
    "XLY": (193.176808371, 119.78),
    "TSLA": (8.0, 402.80),
    "AMZN": (1.0, 271.60),
    "NVDA": (0.75, 242.38),
}
ENTRIES = {
    "AMD": (7, 630.43),
    "MSFT": (5, 516.20),
    "TMO": (13, 674.38),
    "META": (6, 753.17),
    "SMCI": (63, 43.24),
}


async def scenario(tmp_path, monkeypatch, fake, **overrides):
    clock = FakeClock(NOW)
    fake.clock = clock
    prices = {s: p for s, (_, p) in {**HELD, **ENTRIES}.items()} | {"SPY": 660.0, "QQQ": 590.0}
    feed = TrendFeed(clock, drifts=dict.fromkeys(prices, 0.0), anchors=prices)
    feed.live_move = dict.fromkeys(prices, 0.0)
    for sym, (qty, px) in HELD.items():
        fake.hold(sym, qty, px, price=px)
    fake.cash, fake.buying_power_override = -17_028.93, 91_983.71
    fake.last_equity = fake.equity()

    def plan(*args, **kwargs):
        eq = fake.equity()
        sells = [
            ProposedTrade(s, "sell", q, p, "exit", "left the target portfolio", q * p / eq, 0.0, None, True)
            for s, (q, p) in HELD.items()
        ]
        entries = [
            ProposedTrade(s, "buy", q, p, "entry", "new position", 0.0, q * p / eq, 1.5)
            for s, (q, p) in ENTRIES.items()
        ]
        return PortfolioPlan(gross_target=0.56, targets={}, trades=[*sells, *entries])

    monkeypatch.setattr(trading_mod, "build_plan", plan)
    settings = dict(trading_universe=",".join(prices), trading_etfs=["SPY", "QQQ", "XLY"], **overrides)
    async for api in trading_client(tmp_path, clock, fake=fake, feed=feed, **settings):
        yield api


async def test_the_users_account_dry_run_approves_all_ten_and_sends_nothing(tmp_path, monkeypatch):
    fake = FakeAlpacaPaper()
    async for api in scenario(tmp_path, monkeypatch, fake):
        acct = (await api.get(f"{BASE}/account")).json()
        assert acct["equity"] == pytest.approx(41_000.62, abs=1.0) and acct["cash"] == pytest.approx(
            -17_028.93
        )
        assert acct["buying_power"] == pytest.approx(91_983.71)
        cycle = (await api.post(f"{BASE}/run")).json()
        assert len(cycle["trades"]) == 10 and all(t["approved"] for t in cycle["trades"]), [
            (t["symbol"], t["risk"]) for t in cycle["trades"] if not t["approved"]
        ]
        assert any("assumes the approved sells" in n for n in cycle["notes"])
        assert order_posts(fake) == 0


async def test_the_users_account_in_paper_mode(tmp_path, monkeypatch):
    """Fractional full exits go first as market orders (never capped by the order-size limit), the account
    is re-read once they fill (negative cash becomes positive), and only then are the buys checked and sent
    — from cash, never margin, within the position limits, with no short sale."""
    fake = FakeAlpacaPaper()
    async for api in scenario(tmp_path, monkeypatch, fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        assert cycle["mode"] == "paper" and all(t["approved"] for t in cycle["trades"])
        sides = [b["side"] for b in fake.bodies]
        assert sides == ["sell"] * 5 + ["buy"] * 5  # every exit before any entry
        sent = {b["symbol"]: b for b in fake.bodies}
        assert sent["ADBE"]["qty"] == 122.424494693 and sent["ADBE"]["type"] == "market"
        assert sent["XLY"]["qty"] == 193.176808371 and sent["NVDA"]["qty"] == 0.75
        assert sent["NVDA"]["type"] == "market" and sent["TSLA"]["type"] == "limit"
        adbe = next(t for t in cycle["trades"] if t["symbol"] == "ADBE")
        assert adbe["notional"] > 15_000 and "closes the position" in next(
            c["detail"] for c in adbe["checks"] if c["name"] == "order_size"
        )
        assert set(fake.positions) == set(ENTRIES)
        assert {s: p["qty"] for s, p in fake.positions.items()} == {s: q for s, (q, _) in ENTRIES.items()}
        assert fake.cash > 0  # no margin was used
        equity = fake.equity()
        assert all(p["qty"] * fake.price(s) <= 0.30 * equity for s, p in fake.positions.items())
        assert all(t["stage"] == "filled" and t["alpaca_order_id"] for t in cycle["trades"])
        [summary] = (await api.get(f"{BASE}/cycles")).json()
        assert summary["orders_submitted"] == summary["orders_filled"] == 10


async def test_the_users_account_never_buys_on_margin_while_sells_are_unfilled(tmp_path, monkeypatch):
    fake = FakeAlpacaPaper()
    fake.fill_mode.update(ADBE="accept", XLY="accept")  # the two big exits have not filled yet
    async for api in scenario(tmp_path, monkeypatch, fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        assert [b["side"] for b in fake.bodies] == ["sell"] * 5  # no buy was sent
        entries = buys(cycle)
        assert all(t["stage"] == "risk_rejected" and "buying_power" in t["risk"] for t in entries)
        assert any("not filled" in n and "ADBE" in n for n in cycle["notes"])
        assert fake.cash < 0 and not set(ENTRIES) & set(fake.positions)


async def test_a_partial_sell_can_never_go_short(tmp_path, monkeypatch):
    fake = FakeAlpacaPaper()
    async for api in scenario(tmp_path, monkeypatch, fake, **PAPER):
        orig = HELD["NVDA"]
        monkeypatch.setitem(HELD, "NVDA", (5.0, orig[1]))  # propose selling more than the 0.75 held
        cycle = (await api.post(f"{BASE}/run")).json()
        nvda = next(t for t in cycle["trades"] if t["symbol"] == "NVDA")
        assert not nvda["approved"] and "no_short" in nvda["risk"]
        assert "NVDA" not in {b["symbol"] for b in fake.bodies}


def test_dry_and_paper_keys_differ():
    assert trading_mod.TradingService._cycle_key("20260925T1000", "paper") == "20260925T1000"
    assert trading_mod.TradingService._cycle_key("20260925T1000", "dry_run") == "20260925T1000-dry"


async def test_status_reports_where_the_switches_came_from(tmp_path):
    clock = FakeClock(NOW)
    async for api in trading_client(tmp_path, clock, **PAPER):
        cfg = (await api.get(f"{BASE}/status")).json()["config"]
        assert cfg["env_file"] is None and not cfg["restart_required"]
        sources = {x["variable"]: x for x in cfg["sources"]}
        assert sources["QP_ALPACA_API_KEY_ID"]["value"] == "set"
        assert sources["QP_TRADING_DRY_RUN"]["value"] == "false"
        assert cfg["database"].endswith("api.db")


async def test_order_ids_and_stages_survive_a_restart_of_the_view(tmp_path):
    """Cycle records are snapshots; the views overlay each order's reconciled state on them."""
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.default_mode = "partial"
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        cycle = (await api.post(f"{BASE}/run")).json()
        partial = [t for t in buys(cycle) if t["stage"] == "partially_filled"]
        assert partial
        for t in partial:
            fake.complete(t["client_order_id"])
        clock.advance(timedelta(minutes=1).total_seconds())
        await api.post(f"{BASE}/reconcile")
        view = (await api.get(f"{BASE}/cycles/{cycle['id']}")).json()
        stages = {t["client_order_id"]: t["stage"] for t in view["trades"]}
        assert all(stages[t["client_order_id"]] == "filled" for t in partial)
        assert datetime.fromisoformat(view["trades"][0]["submitted_at"]).tzinfo is not None
        assert view["started_at"].startswith(NOW.astimezone(UTC).strftime("%Y-%m-%dT%H:%M"))
