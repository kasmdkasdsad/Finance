"""The Brain page against a real API wired to a fake Alpaca paper account and fake market data."""

import threading
import time

import httpx
import pytest
import uvicorn

from frontend.views import brain as brain_view
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


def _html(at) -> str:
    """The page's own HTML pieces (status lines, facts)."""
    return "\n".join(e.proto.body for e in at.get("html"))


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [str(m.value) for m in at.metric] + [m.label for m in at.metric] + [_html(at)]
    return "\n".join(parts)


def _every_view(url: str) -> list:
    """The Brain page once per tab (and per Learning view): only the open tab is computed."""
    runs = []
    for tab in brain_view.TABS:
        for view in brain_view.LEARNING_VIEWS if tab == "Learning" else [None]:
            state = {"brain_tab": tab, **({"brain_learn_view": view} if view else {})}
            at = page("brain", url, state).run()
            assert_clean(at)
            runs.append(at)
    return runs


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
        brain_mode="paper_execution",  # the Brain owns the account; QP_ALPACA_TRADING_ENABLED stays false
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
    text = _texts(at)
    assert (
        "the trading service executes the Brain's decisions" in text and "The Brain has not run yet" in text
    )
    assert "orders are not sent right now" in _html(at)  # no Alpaca keys


def test_every_layer_is_shown_and_kept_apart(brain_server):
    url, fake = brain_server
    runs = _every_view(url)
    text = "\n".join(_texts(at) for at in runs)
    markdown = [m.value for at in runs for m in at.markdown]
    captions = [c.value for at in runs for c in at.caption]
    infos = [i.value for at in runs for i in at.info]
    frames = [d.value for at in runs for d in at.dataframe]
    expanders = [e.label for at in runs for e in [*at.expander, *at.status]]  # with an icon: at.status
    for layer in ("Agent analysis", "Consensus", "risk preview", "Broker execution"):
        assert layer.lower() in text.lower(), layer
    assert "Orders sent by the Brain" in text and "**Mode**" in text
    assert len(brain_view.TABS) == 6  # was 18: the same content, grouped
    agents = next(f for f in frames if "track record" in f.columns)
    assert len(agents) == 21 and set(agents["this cycle"]) <= {"ok", "skipped", "failed", "timeout", "—"}
    assert (agents["track record"] == "unproven (no evaluated predictions yet)").all()
    consensus = next(f for f in frames if "disagreement" in f.columns)
    assert {"supporting", "neutral", "opposing", "confidence", "data"} <= set(consensus.columns)
    actions = next(f for f in frames if "④ execution" in f.columns)
    assert all(e.startswith("not sent") or e == "no trade" for e in actions["④ execution"])
    assert set(actions["③ risk preview"]) <= {
        "Risk engine: would allow",
        "Risk engine: rejected",
        "Blocked by a data veto",
        "No trade proposed",
    }
    opportunities = next(f for f in frames if "outcome" in f.columns)
    assert len(opportunities) and {"kind", "subject", "idea", "strength", "what"} <= set(
        opportunities.columns
    )
    assert any("Risk posture" in m for m in markdown)
    assert any("detection" in m for m in markdown)  # one idea followed through the pipeline
    assert any("Bull case" in m for m in markdown) and any("Bear case" in m for m in markdown)
    assert any("matured predictions graded against real closing prices" in m for m in markdown)
    assert any("executed by the trading service" in m for m in markdown if "supervisor" in m)
    assert any("only a person can promote" in m for m in markdown)
    assert any("Nothing is applied automatically" in m for m in markdown)
    assert any("Every analysis is deterministic" in i for i in infos)  # no language model configured
    assert any(m.startswith("**Language models** not in use") for m in markdown)
    assert any("The Brain's paper book" in m for m in markdown)  # its own, hypothetical portfolio
    assert any("owned by the Brain" in e for e in expanders)
    assert any("**Audit trail**" in m for m in markdown)
    assert any("SIP report" in m for m in markdown) and any("Decision." in m for m in markdown)
    assert any("**Trading days**" in m for m in markdown)
    assert "Known limitations" in expanders and "How the Brain decides" in expanders
    assert any("60-session evaluation" in m for m in markdown)
    assert any("Positions and their theses" in m for m in markdown)
    assert all(any(b.label == "STOP BRAIN ORDERS" for b in at.button) for at in runs)  # on every tab
    # why it traded or not (trading is disabled here: nothing can be sent), and the execution tab
    assert any(i.startswith("no ") for i in infos), infos
    assert any("**Final execution audit**" in m for m in markdown)
    assert any("**Execution ledger**" in m for m in markdown)
    assert any("execution quality is unproven" in c for c in captions)
    # the experiment view: checkpoints, reviews, the record, ideas, behaviour, data — nothing declared
    assert any("**The paper experiment**" in m for m in markdown)
    assert any("20 / 40 / 60-session checkpoints" in m for m in markdown)
    assert any("**What the record says**" in m for m in markdown)
    assert any("**Behaviour**" in m for m in markdown)
    assert any("When data stopped trading" in m for m in markdown)
    assert any("never solved by a looser" in c for c in captions)
    assert any("Traceability:" in c for c in captions)
    events = next(f for f in frames if "event" in f.columns and "source" in f.columns)
    assert "AgentCompleted" in set(events["event"])
    assert all(m == "GET" for m, _ in fake.log) and fake.orders == {}


def test_only_the_open_tab_is_computed(brain_server):
    """A lazy page: the first view asks the API for the status, the account state and the cycle only."""
    url, _ = brain_server
    at = page("brain", url).run()
    assert_clean(at)
    assert not any("track record" in d.value.columns for d in at.dataframe)  # the Agents tab was not computed
    assert any("④ execution" in d.value.columns for d in at.dataframe)  # the Decision tab was


def test_running_a_cycle_from_the_page_sends_no_order(brain_server):
    url, fake = brain_server
    at = page("brain", url).run()
    next(t for t in at.text_input if t.label.startswith("Also study")).input("DNA")
    next(b for b in at.button if b.label == "Run a cycle now").click().run()
    assert_clean(at)
    assert any("0 orders sent" in s.value for s in at.success)
    assert all(m == "GET" for m, _ in fake.log) and fake.orders == {}


def test_the_brain_kill_switch_is_one_click_away(brain_server):
    url, fake = brain_server
    at = page("brain", url).run()
    next(b for b in at.button if b.label == "STOP BRAIN ORDERS").click().run()
    assert not at.exception
    assert [e.value for e in at.error] == ["BRAIN KILL SWITCH ON — no new Brain orders (runtime)."]
    assert httpx.get(f"{url}/api/v1/brain/kill-switch").json()["active"]
    next(b for b in at.button if b.label == "Allow Brain orders again").click().run()
    assert not httpx.get(f"{url}/api/v1/brain/kill-switch").json()["active"]
    assert all(m == "GET" for m, _ in fake.log) and fake.orders == {}
