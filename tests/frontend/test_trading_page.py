"""The Alpaca Paper Trading page against a real API wired to a fake Alpaca paper account and fake market data."""

import threading
import time

import httpx
import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from quantpulse.api.app import create_app
from quantpulse.config import Settings
from quantpulse.core.clock import FakeClock
from quantpulse.providers.alpaca_trading import AlpacaPaperBroker
from quantpulse.services.container import Container
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import STOCKS, TrendFeed
from tests.frontend.conftest import _free_port
from tests.frontend.test_pages import assert_clean, page
from tests.integration.conftest import NOW

INTENDED_ALERTS = ("PAPER EXECUTION ACTIVE", "KILL SWITCH ON", "Kill switch is ON")


def assert_ok(at: AppTest) -> None:
    """No exceptions, and no red box except the page's deliberate execution / kill-switch banners."""
    assert not at.exception, [e.value for e in at.exception]
    unexpected = [e.value for e in at.error if not any(a in e.value for a in INTENDED_ALERTS)]
    assert not unexpected, unexpected


@pytest.fixture(scope="module")
def trading_server(tmp_path_factory):
    db = tmp_path_factory.mktemp("trading-ui") / "ui.db"
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{db}",
        enable_live_data=True,
        polling_enabled=False,
        log_level="WARNING",
        market_providers=["yahoo"],
        alpaca_api_key_id="PKUITEST",
        alpaca_api_secret_key="ui-secret",
        alpaca_trading_enabled=True,
        trading_dry_run=False,
        trading_universe=",".join(STOCKS),
        trading_etfs=["SPY", "QQQ"],
        trading_signal_weights={"momentum": 0.4, "trend": 0.3, "volume": 0.15, "volatility": 0.15},
        trading_use_implied_vol=False,
        trading_earnings_blackout_days=0,
        trading_fill_wait_seconds=0,
    )
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    container = Container(
        settings, clock=clock, broker=AlpacaPaperBroker("PKUITEST", "ui-secret", transport=fake)
    )
    feed = TrendFeed(clock)
    container.market._providers[:] = [feed]
    for s in feed.bars:
        fake.prices[s] = feed.live_price(s)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(container=container), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("API server did not start")
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    with httpx.Client(base_url=url, timeout=120) as client:
        client.post("/api/v1/trading/run").raise_for_status()  # one paper cycle so every view has content
    yield url, fake, clock
    server.should_exit = True
    thread.join(timeout=10)


def test_unconfigured_page_explains_setup(api_server):
    at = page("trading", api_server)
    at.run()
    assert_clean(at)
    assert any("SIMULATED MONEY ONLY" in w.value for w in at.warning)
    assert any("NO ORDERS WILL BE SUBMITTED" in i.value for i in at.info)
    assert any("QP_ALPACA_API_KEY_ID" in i.value for i in at.info)


def test_every_view_renders_with_banners_and_account(trading_server):
    url, _, _ = trading_server
    at = page("trading", url)
    at.run()
    assert_ok(at)
    assert any(w.value == "**ALPACA PAPER TRADING — SIMULATED MONEY ONLY**" for w in at.warning)
    assert any("PAPER EXECUTION ACTIVE" in e.value for e in at.error)
    labels = {m.label for m in at.metric}
    assert {"Equity", "Cash", "Buying power", "Today's P/L", "Total P/L", "Mode", "Kill switch"} <= labels
    assert at.dataframe and at.get("plotly_chart")  # positions and current-vs-target weights
    labels = {m.label for m in at.metric}
    assert {"Broker", "Dry run", "Scheduler", "Last cycle", "Next cycle"} <= labels
    for view in ("Strategy", "Orders", "Risk", "Activity", "Performance", "Controls", "Diagnostics"):
        at.segmented_control(key="trade_view").set_value(view).run()
        assert_ok(at)
    at.segmented_control(key="trade_view").set_value("Strategy").run()
    assert any(m.label == "Market regime" and m.value == "Bullish" for m in at.metric)
    assert len(at.dataframe) >= 3  # opportunities, targets, portfolio (+ trades)
    at.segmented_control(key="trade_view").set_value("Risk").run()
    assert {"Daily P/L", "Exposure", "Positions", "Largest position", "Kill switch", "Dry run"} <= {
        m.label for m in at.metric
    }


def test_controls_run_kill_switch_and_guarded_close_all(trading_server):
    url, fake, clock = trading_server
    clock.advance(60)
    at = page("trading", url, {"trade_view": "Controls"})
    at.run()
    assert_ok(at)
    at.button(key="trade_run").click().run()
    assert_ok(at)
    assert any("trade(s) proposed" in s.value for s in at.success)

    at.button(key="FormSubmitter:trade_kill-Activate kill switch").click().run()
    assert_ok(at)
    assert any("KILL SWITCH ON" in e.value for e in at.error)
    at.button(key="trade_release").click().run()
    assert_ok(at)
    assert not any("KILL SWITCH ON" in e.value for e in at.error)

    assert fake.positions  # the paper cycle bought something
    close = at.button(key="trade_close_all")
    assert close.disabled  # no one-click flattening
    at.text_input(key="trade_close_phrase").set_value("close").run()
    assert at.button(key="trade_close_all").disabled
    at.text_input(key="trade_close_phrase").set_value("CLOSE ALL").run()
    assert not at.button(key="trade_close_all").disabled
    at.button(key="trade_close_all").click().run()
    assert_ok(at)
    assert any("closing order(s)" in s.value for s in at.success)
    assert fake.positions == {}


def test_trades_show_their_stage_and_alpaca_order_id(trading_server):
    url, fake, _ = trading_server
    at = page("trading", url, {"trade_view": "Strategy"})
    at.run()
    assert_ok(at)
    trades = next(d.value for d in at.dataframe if "stage_label" in d.value.columns)
    sent = trades[trades["alpaca_order_id"].notna()]
    assert len(sent) and set(sent["stage_label"]) <= {"✓ Filled"}
    assert set(sent["alpaca_order_id"]) <= {o["id"] for o in fake.orders.values()}


def test_diagnostics_view_checks_the_connection_and_guards_the_test_order(trading_server):
    url, fake, clock = trading_server
    clock.advance(120)
    at = page("trading", url, {"trade_view": "Diagnostics"})
    at.run()
    assert_ok(at)
    at.text_input(key="trade_diag_symbols").set_value("UPA,UPB")
    at.button(key="trade_diag_run").click().run()
    assert_ok(at)
    checks = at.dataframe[0].value
    assert list(checks["step"])[:4] == ["settings", "credentials", "sdk_client", "account"]
    assert set(checks[""]) == {"✓"}
    assert any("verified paper" in c.value for c in at.caption)

    before = len(fake.orders)
    assert at.button(key="trade_test_send").disabled  # the phrase is required
    at.text_input(key="trade_test_phrase").set_value("SUBMIT ONE PAPER TEST ORDER").run()
    assert not at.button(key="trade_test_send").disabled
    fake.fill_mode["SPY"] = "accept"
    at.button(key="trade_test_send").click().run()
    assert_ok(at)
    assert any("it was canceled" in s.value for s in at.success), [s.value for s in at.success]
    assert len(fake.orders) == before + 1
