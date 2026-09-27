"""The Brain page against a real API wired to a fake Alpaca paper account and fake market data."""

import threading
import time

import httpx
import pytest
import uvicorn

from quantpulse.api.app import create_app
from quantpulse.config import Settings
from quantpulse.core.clock import FakeClock
from quantpulse.providers.alpaca_trading import AlpacaPaperBroker
from quantpulse.services.container import Container
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import TrendFeed
from tests.frontend.conftest import _free_port
from tests.frontend.test_pages import assert_clean, page
from tests.integration.conftest import NOW
from tests.integration.test_brain_cycle import STOCKS, WIDE


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [str(m.value) for m in at.metric] + [m.label for m in at.metric]
    return "\n".join(parts)


@pytest.fixture(scope="module")
def brain_server(tmp_path_factory):
    db = tmp_path_factory.mktemp("brain-ui") / "ui.db"
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{db}",
        enable_live_data=True,
        polling_enabled=False,
        log_level="WARNING",
        market_providers=["yahoo"],
        alpaca_api_key_id="PKUITEST",
        alpaca_api_secret_key="ui-secret",
        trading_universe=",".join(STOCKS),
        trading_etfs=["SPY", "QQQ"],
        trading_signal_weights={"momentum": 0.4, "trend": 0.3, "volume": 0.15, "volatility": 0.15},
        trading_use_implied_vol=False,
        trading_earnings_blackout_days=0,
        brain_use_stock_model=False,
        brain_options_analysis=False,
        brain_catalyst_analysis=False,
    )
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.hold("UPA", 40, 60.0)
    container = Container(
        settings, clock=clock, broker=AlpacaPaperBroker("PKUITEST", "ui-secret", transport=fake)
    )
    feed = TrendFeed(clock, drifts=WIDE)
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
        client.post("/api/v1/brain/run").raise_for_status()  # one cycle so every tab has content
    yield url, fake
    server.should_exit = True
    thread.join(timeout=10)


def test_empty_brain_explains_itself(api_server):
    at = page("brain", api_server).run()
    assert_clean(at)
    assert "never sends an order" in _texts(at)


def test_every_layer_is_shown_and_kept_apart(brain_server):
    url, fake = brain_server
    at = page("brain", url).run()
    assert_clean(at)
    text = _texts(at)
    for layer in ("Agent analysis", "Consensus", "risk preview", "Broker execution"):
        assert layer.lower() in text.lower(), layer
    assert "Orders sent by the Brain" in text and "Mode" in text
    frames = [d.value for d in at.dataframe]
    agents = next(f for f in frames if "track record" in f.columns)
    assert len(agents) == 16 and set(agents["this cycle"]) <= {"ok", "skipped", "failed", "timeout", "—"}
    assert (agents["track record"] == "unproven (no evaluated predictions yet)").all()
    consensus = next(f for f in frames if "disagreement" in f.columns)
    assert {"supporting", "neutral", "opposing", "confidence", "data"} <= set(consensus.columns)
    actions = next(f for f in frames if "④ execution" in f.columns)
    assert (actions["④ execution"] == "not sent (the Brain never sends orders)").all()
    assert set(actions["③ risk preview"]) <= {
        "Risk engine: would allow (recommendation only)",
        "Risk engine: rejected",
        "Blocked by a data veto",
        "No trade proposed",
    }
    opportunities = next(f for f in frames if "outcome" in f.columns)
    assert len(opportunities) and {"kind", "subject", "idea", "strength", "what"} <= set(
        opportunities.columns
    )
    assert any("Risk posture" in m.value for m in at.markdown)
    assert any("detection" in m.value for m in at.markdown)  # one idea followed through the pipeline
    assert any("Bull case" in m.value for m in at.markdown) and any(
        "Bear case" in m.value for m in at.markdown
    )
    assert any("matured predictions graded against real closing prices" in m.value for m in at.markdown)
    assert any("the Brain never sends orders" in m.value for m in at.markdown if "supervisor" in m.value)
    assert any("only a person can promote" in m.value for m in at.markdown)
    assert any("Nothing is applied automatically" in m.value for m in at.markdown)
    events = next(f for f in frames if "event" in f.columns and "source" in f.columns)
    assert "AgentCompleted" in set(events["event"])
    assert all(m == "GET" for m, _ in fake.log) and fake.orders == {}


def test_running_a_cycle_from_the_page_sends_no_order(brain_server):
    url, fake = brain_server
    at = page("brain", url).run()
    at.text_input[0].input("DNA")
    at.button[0].click().run()
    assert_clean(at)
    assert any("0 orders sent" in s.value for s in at.success)
    assert all(m == "GET" for m, _ in fake.log) and fake.orders == {}
