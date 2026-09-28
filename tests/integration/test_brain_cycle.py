"""The brain end to end, through the HTTP API, on the fake market feed and the fake Alpaca paper account.

A cycle perceives (account, clock, prices, live quotes, indicators, regime), runs the five deterministic
agents, builds a consensus per subject, proposes portfolio actions, has the existing risk engine preview
each trade and persists everything. It never sends an order: every Alpaca request it makes is a read.
"""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from quantpulse.brain.agents.technical import TechnicalAgent
from quantpulse.brain.llm import ModelResponse, register_provider, unregister_provider
from quantpulse.core.clock import FakeClock
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import DRIFTS, TrendFeed

from .conftest import NOW
from .test_trading import trading_client

BRAIN = "/api/v1/brain"
AGENTS = {
    "data_quality",
    "market_regime",
    "technical",
    "momentum",
    "mean_reversion",
    "volatility",
    "statistical",
    "fundamental",
    "valuation",
    "factor",
    "options",
    "catalyst",
    "portfolio",
    "research",
    "situational_awareness",
    "strategy_lab",
    "briefing",
}
# without a stock model run, live option chains or an earnings calendar (the fakes have none) these skip
NEEDS_RESEARCH = {"fundamental", "valuation", "factor", "options", "catalyst", "strategy_lab"}
NEEDS_MODEL = {"briefing"}  # skips itself unless a language model is configured (none is, by default)
RAN = AGENTS - NEEDS_RESEARCH - NEEDS_MODEL
TRADES = {"buy", "increase", "reduce", "close", "sell"}

# a cross-section wide enough for momentum ranking: the standard trend names plus more of each kind
WIDE = {
    **DRIFTS,
    **{f"UP{c}": 0.0028 - 0.0002 * i for i, c in enumerate("FGHIJ")},
    **{f"MID{c}": 0.0004 for c in "CDE"},
    **{f"DN{c}": -0.0015 - 0.0003 * i for i, c in enumerate("DEF")},
}
STOCKS = [s for s in WIDE if s not in ("SPY", "QQQ")]


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)  # sectors are optional context
    return mock_net


async def brain_client(tmp_path, clock, fake=None, feed=None, client_host="127.0.0.1", **overrides):
    overrides.setdefault("brain_use_stock_model", False)  # never start a model training run in these tests
    overrides.setdefault("brain_options_analysis", False)  # the fakes have no option chains or SEC filings
    overrides.setdefault("brain_catalyst_analysis", False)
    feed = feed or TrendFeed(clock, drifts=WIDE)
    async for api in trading_client(
        tmp_path,
        clock,
        fake=fake,
        feed=feed,
        client_host=client_host,
        trading_universe=",".join(STOCKS),
        **overrides,
    ):
        yield api


async def seed_book(api, positions: dict[str, tuple[float, float]]) -> None:
    """Positions already in the Brain's paper book (test data): symbol -> (quantity, price paid)."""
    from quantpulse.db.models import BrainBookPositionRow, BrainStateRow

    await api.container.brain.book.load()  # opens the book at its starting capital
    now = api.container.clock.now()
    async with api.container.db.session() as s:
        row = await s.get(BrainStateRow, "book")
        cash = float(row.value["cash"])
        for sym, (qty, price) in positions.items():
            s.add(BrainBookPositionRow(symbol=sym, qty=qty, avg_cost=price, last_price=price, opened_at=now,
                                       updated_at=now, stop_price=round(price * 0.92, 4)))  # fmt: skip
            cash -= qty * price
        row.value = {**row.value, "cash": cash}


def with_stock_model(monkeypatch) -> None:
    """A completed stock-model run as test data (through the real service interface): the fundamental,
    valuation and factor agents then see the uptrend names as better — a second and third source of
    evidence beside prices, which a trade needs before the consensus is confident."""
    from datetime import date as _date

    from quantpulse.services.model import ModelService
    from tests.unit.test_brain_agents import snapshot, universe_features

    feats = universe_features()
    feats = pd.concat([feats, feats.iloc[: len(STOCKS)].set_axis(STOCKS)])
    probs = {s: 0.62 if s.startswith("UP") else 0.40 if s.startswith("DN") else 0.5 for s in STOCKS}

    async def model_snapshot(self, *, wait=None):
        return snapshot(feats, probs, as_of=_date(2026, 9, 24))

    monkeypatch.setattr(ModelService, "trading_snapshot", model_snapshot)


async def run_cycle(api, **body):
    r = await api.post(f"{BRAIN}/run", json=body or None)
    assert r.status_code == 200, r.text
    return r.json()


def by(items, key="subject"):
    return {x[key]: x for x in items}


def only_reads(fake: FakeAlpacaPaper) -> bool:
    return all(method == "GET" for method, _ in fake.log) and fake.orders == {}


async def test_a_full_cycle_perceives_thinks_proposes_and_sends_nothing(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, brain_use_stock_model=True):
        api.fake.hold("UPA", 40, 60.0)  # the strategy's Alpaca position: context only for the Brain
        await seed_book(api, {"UPA": (40, 60.0)})  # the Brain's own (hypothetical) position
        cycle = await run_cycle(api)

        # perception: session, regime, the paper account, focus
        assert cycle["status"] == "completed" and cycle["error"] is None
        assert cycle["mode"] == "paper_recommendation" and cycle["session"] == "market_open"
        assert cycle["regime"]["label"] == "bullish" and cycle["market"]["clock"] == "alpaca"
        assert cycle["portfolio"]["available"] and cycle["portfolio"]["positions"]["UPA"]["qty"] == 40
        assert "paper book" in cycle["portfolio"]["owner"]  # two portfolios, one owner each
        alpaca = cycle["portfolio"]["alpaca_account"]
        assert "trading strategy" in alpaca["owner"] and alpaca["positions"]["UPA"]["qty"] == 40
        controls = cycle["portfolio"]["trading_controls"]  # read, never acted on
        assert not controls["orders_would_reach_alpaca"]
        assert any("QP_ALPACA_TRADING_ENABLED=false" in b for b in controls["blockers"])
        focus = [f["symbol"] for f in cycle["focus"]]
        assert focus[0] == "UPA" and cycle["focus"][0]["reason"] == "held position"
        assert 1 < len(focus) <= 1 + 6 + 8 and "SPY" not in focus  # holding + opportunities + pre-screen

        # every agent ran and its run is persisted
        assert {a["agent_id"] for a in cycle["agents"]} == AGENTS
        status_of = {r["agent_id"]: r["status"] for r in cycle["runs"]}
        with_model = RAN | {"fundamental", "valuation", "factor"}
        assert status_of == {a: "ok" if a in with_model else "skipped" for a in AGENTS}
        reasons = {r["agent_id"]: r["reason"] for r in cycle["runs"] if r["status"] == "skipped"}
        assert "option chains" in reasons["options"]
        assert "QP_BRAIN_LLM_PROVIDER=none" in reasons["briefing"]

        # structured findings: each focus symbol seen by data quality, technical and momentum
        seen: dict[str, set[str]] = {}
        for o in cycle["opinions"]:
            seen.setdefault(o["subject"], set()).add(o["agent_id"])
            assert -1 <= o["score"] <= 1 and 0 <= o["confidence"] <= 1 and o["thesis"]
        for s in focus:
            assert {"data_quality", "technical", "momentum"} <= seen[s]
        assert seen["@market"] == {"data_quality", "market_regime", "volatility", "situational_awareness"}
        assert seen["@portfolio"] == {"portfolio"}
        tech = next(o for o in cycle["opinions"] if o["agent_id"] == "technical" and o["subject"] == "UPC")
        assert tech["stance"] == "bullish" and tech["evidence"] and tech["invalidation"]

        # consensus per subject, with the vote of each agent and its (unproven) reliability
        consensus = by(cycle["consensus"])
        assert set(consensus) == {"@market", *focus}
        upc = consensus["UPC"]
        assert upc["stance"] == "bullish" and not upc["unknown"] and upc["supporting"] >= 2
        votes = upc["detail"]["votes"]
        assert {"technical", "momentum"} <= {v["agent_id"] for v in votes} <= with_model
        assert upc["supporting"] + upc["neutral"] + upc["opposing"] == len(votes)
        assert all(
            v["reliability"] == {"weight": 1.0, "status": "unproven", "n": 0, "verdict": "unproven"}
            for v in votes
        )

        # proposed actions, each trade previewed by the deterministic risk engine; nothing sent
        decisions = by(cycle["decisions"])
        assert set(decisions) == set(focus)
        assert decisions["UPA"]["action"] in {"hold", "increase"}
        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        assert trades and sum(d["action"] == "buy" for d in trades) <= 2  # brain_max_new_positions_per_cycle
        for d in trades:
            assert d["risk"]["checks"] and d["status"] in {"recommended", "risk_rejected", "blocked"}
            ex = d["execution"]
            assert ex["sent"] is False and "nothing is sent to Alpaca" in ex["reason"]
            if d["risk_approved"]:
                assert d["notional"] <= 15_000  # sized within the risk engine's per-order limit
            if d["status"] == "recommended":  # simulated in the Brain's paper book, not sent anywhere
                fill = ex["book"]
                side = 1 if fill["side"] == "buy" else -1
                assert side * (fill["fill_price"] - d["est_price"]) > 0 and fill["slippage_bps"] > 0
                assert fill["cost"] > 0 and fill["qty"] <= d["quantity"]
            else:
                assert "book" not in ex  # blocked or rejected: nothing simulated either
        assert any(d["status"] == "recommended" for d in trades)
        assert cycle["summary"]["book_fills"] == sum(d["status"] == "recommended" for d in trades)
        for d in cycle["decisions"]:
            if d["action"] not in TRADES:
                assert d["status"] == "no_trade" and d["rationale"]["reasons"]
        assert cycle["summary"]["orders_sent"] == 0
        assert only_reads(api.fake) and ("POST", "/v2/orders") not in api.fake.log

        # the trading service was not involved: no trading cycle, no proposal there
        assert (await api.get("/api/v1/trading/proposed")).json() is None

        # predictions recorded for later grading (nothing evaluated: learning is not built yet)
        assert cycle["predictions_recorded"] > 0
        preds = await api.container.brain.store.predictions()
        assert len(preds) == cycle["predictions_recorded"]
        assert all(p.status == "open" and p.realized_return is None for p in preds)
        assert all(p.due_date > date(2026, 9, 25) and p.entry_price for p in preds)
        assert {p.source_type for p in preds} == {"agent", "consensus"}
        assert not {p.source_id for p in preds} & {
            "data_quality",
            "portfolio",
        }  # constraints are not forecasts
        # each claim carries what it takes to judge it later — and no invented expectation
        for p in preds:
            assert p.expected_return is None  # uncalibrated until the source has a graded record
            assert p.context["vol"] and p.context["portfolio"]["posture"] in (
                "normal",
                "cautious",
                "defensive",
            )
            assert p.context["thesis"]
            assert p.subject == "@market" or p.context["data_status"] in ("fresh", "live")
        agent_pred = next(p for p in preds if p.source_type == "agent" and p.subject == "UPA")
        assert agent_pred.context["portfolio"]["held"] and agent_pred.context["evidence"]
        assert agent_pred.context["consensus"]["stance"] in ("bullish", "bearish", "neutral", "unknown")
        team = [p for p in preds if p.source_type == "consensus"]
        assert team and all(p.context["agents"]["supporting"] and p.context["sources"] for p in team)
        assert all(p.source_version == "2" for p in team)

        # the same cycle, read back
        again = (await api.get(f"{BRAIN}/cycles/{cycle['id']}")).json()
        assert again == cycle
        listed = (await api.get(f"{BRAIN}/cycles")).json()
        assert [c["id"] for c in listed] == [cycle["id"]] and "opinions" not in listed[0]

        status = (await api.get(f"{BRAIN}/status")).json()
        assert status["paper_only"] and status["last_cycle"]["id"] == cycle["id"]
        assert status["agents"] == {"registered": 17, "enabled": 17} and not status["running"]
        assert status["open_predictions"] == len(preds) and "predictions graded" in status["learning"]

        agents = by((await api.get(f"{BRAIN}/agents")).json(), "id")
        assert set(agents) == AGENTS
        assert agents["technical"]["runs"]["runs"] == 1 and agents["technical"]["runs"]["failures"] == 0
        assert all(a["performance"] == [] for a in agents.values())  # no invented track record

        # memory: what it saw (short term), this investigation (working), what changed (long term)
        short = by((await api.get(f"{BRAIN}/memory", params={"tier": "short_term"})).json(), "kind")
        assert set(short) == {"market_state", "portfolio_state"}
        assert short["market_state"]["data"]["regime"] == "bullish"
        long = (await api.get(f"{BRAIN}/memory", params={"tier": "long_term"})).json()
        assert [m["kind"] for m in long].count("regime_change") == 1
        # memory keeps what the Brain did (its book's fills), not every proposal of every cycle
        filled = {f["symbol"] for f in cycle["portfolio"]["book"]["fills"]}
        assert filled and {m["subject"] for m in long if m["kind"] == "trade"} == filled
        assert not [m for m in long if m["kind"] == "decision"]


async def test_modes_dry_run_and_research_only(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path / "dry", clock, brain_mode="dry_run", brain_use_stock_model=True):
        cycle = await run_cycle(api)
        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        assert trades and {d["status"] for d in trades} <= {"dry_run_approved", "risk_rejected", "blocked"}
        assert all(d["mode"] == "dry_run" for d in cycle["decisions"]) and only_reads(api.fake)
    async for api in brain_client(tmp_path / "research", clock, brain_mode="research_only"):
        cycle = await run_cycle(api)
        assert cycle["mode"] == "research_only" and cycle["decisions"] == []
        assert cycle["consensus"] and cycle["predictions_recorded"] > 0  # it still thinks and remembers
        assert only_reads(api.fake)


async def test_a_repeated_view_is_recorded_once_a_day(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        first = await run_cycle(api)
        clock.advance(1800)
        second = await run_cycle(api)
        assert first["predictions_recorded"] > 0
        assert (
            second["predictions_recorded"] < first["predictions_recorded"]
        )  # the same views: not new claims
        preds = await api.container.brain.store.predictions()
        keys = [(p.source_id, p.subject, p.horizon_days, p.direction) for p in preds]
        assert len(keys) == len(set(keys))


async def test_memory_accumulates_across_cycles_without_duplicating_state(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        first = await run_cycle(api)
        clock.advance(15 * 60)
        second = await run_cycle(api, kind="portfolio")
        assert second["id"] == first["id"] + 1 and second["kind"] == "portfolio"
        cycles = (await api.get(f"{BRAIN}/cycles")).json()
        assert [c["id"] for c in cycles] == [second["id"], first["id"]]
        short = (await api.get(f"{BRAIN}/memory", params={"tier": "short_term"})).json()
        assert sorted(m["kind"] for m in short) == ["market_state", "portfolio_state"]  # updated in place
        assert all(m["cycle_id"] == second["id"] for m in short)
        working = (await api.get(f"{BRAIN}/memory", params={"tier": "working"})).json()
        assert sorted(m["key"] for m in working) == [f"cycle:{first['id']}", f"cycle:{second['id']}"]
        long = (
            await api.get(f"{BRAIN}/memory", params={"tier": "long_term", "kind": "regime_change"})
        ).json()
        assert len(long) == 1  # the regime did not change between the cycles
        agents = by((await api.get(f"{BRAIN}/agents")).json(), "id")
        assert all(a["runs"]["runs"] == 2 for a in agents.values())


async def test_a_failing_agent_is_recorded_and_the_cycle_goes_on(tmp_path, monkeypatch):
    async def broken(self, ctx, subjects):
        raise RuntimeError("indicator blew up")

    monkeypatch.setattr(TechnicalAgent, "analyze", broken)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        cycle = await run_cycle(api)
        assert cycle["status"] == "completed"
        runs = by(cycle["runs"], "agent_id")
        assert runs["technical"]["status"] == "failed"
        assert runs["technical"]["reason"] == "RuntimeError: indicator blew up"
        assert {a for a, r in runs.items() if r["status"] == "ok"} == RAN - {"technical"}
        assert cycle["summary"]["agents_failed"] == 1
        assert not any(o["agent_id"] == "technical" for o in cycle["opinions"])
        # the others still vote; the failed agent's voice is simply missing
        for c in cycle["consensus"]:
            assert "technical" not in {v["agent_id"] for v in c["detail"]["votes"]}
        agents = by((await api.get(f"{BRAIN}/agents")).json(), "id")
        assert agents["technical"]["runs"]["failures"] == 1


async def test_price_only_evidence_is_one_source_and_proposes_no_new_position(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):  # no stock model, options or calendar: prices only
        cycle = await run_cycle(api)
        stocks = [c for c in cycle["consensus"] if c["subject"] != "@market"]
        assert stocks
        for c in stocks:
            detail = c["detail"]
            assert set(detail["sources"]) <= {"prices"} and detail["independent_sources"] <= 1
            gone = {m["agent_id"]: m for m in detail["missing"]}
            assert gone["factor"]["kind"] == "skipped" and "stock model" in gone["factor"]["reason"]
            assert gone["options"]["source"] == "options"
            assert any("one source only (prices)" in u or "no source" in u for u in detail["uncertainty"])
        assert not [
            d for d in cycle["decisions"] if d["action"] == "buy"
        ]  # several agents agreeing is not enough
        assert only_reads(api.fake)


async def test_checks_fail_closed_when_their_agent_does_not_run(tmp_path, monkeypatch):
    from quantpulse.brain.agents.research import SituationalAwarenessAgent

    async def broken(self, ctx, subjects):
        raise RuntimeError("posture blew up")

    monkeypatch.setattr(SituationalAwarenessAgent, "analyze", broken)
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, brain_use_stock_model=True):
        r = await api.post(f"{BRAIN}/agents/data_quality", json={"enabled": False})
        assert r.status_code == 200
        cycle = await run_cycle(api)
        assert cycle["status"] == "completed"
        situation = cycle["market"]["situation"]
        assert situation["posture"] == "cautious" and "did not run" in situation["reasons"][0]
        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        assert trades
        for d in trades:  # nothing is executable without the data-quality check
            assert d["status"] in ("blocked", "risk_rejected")
            assert any("data-quality check did not run" in b for b in d["rationale"]["blocked_by"])
        assert only_reads(api.fake)


async def test_stale_quotes_are_vetoed(tmp_path):
    clock = FakeClock(NOW)
    feed = TrendFeed(clock, drifts=WIDE)
    feed.quote_age = timedelta(minutes=30)  # older than the 10-minute limit
    async for api in brain_client(tmp_path, clock, feed=feed):
        api.fake.hold("UPA", 40, 60.0)
        cycle = await run_cycle(api)
        assert cycle["status"] == "completed"
        states = cycle["data_quality"]["states"]
        assert states and set(states.values()) == {"stale"}
        market = cycle["data_quality"]["market"]
        assert market["veto"] and "usable live quotes" in market["veto"]
        for c in cycle["consensus"]:
            if c["subject"] != "@market":
                assert c["data_quality"] == "stale"
                assert c["vetoes"] and c["vetoes"][0]["reason"].startswith("data stale")
        assert not [d for d in cycle["decisions"] if d["status"] in {"recommended", "dry_run_approved"}]
        assert only_reads(api.fake)


async def test_every_cycle_explains_its_market_data(tmp_path):
    clock = FakeClock(NOW)
    feed = TrendFeed(clock, drifts=WIDE)
    feed.quote_age = timedelta(minutes=30)
    async for api in brain_client(tmp_path, clock, feed=feed):
        cycle = await run_cycle(api)
        dq = cycle["data_quality"]
        focus = [f["symbol"] for f in cycle["focus"]]
        assert set(dq["diagnosis"]) == set(focus)
        assert {d["status"] for d in dq["diagnosis"].values()} == {"stale"}
        assert all("limit 600s" in d["reasons"][0] for d in dq["diagnosis"].values())
        report = dq["feed"]
        assert report["clock_skew_s"] == 0.0 and report["market_open"] and not report["healthy"]
        assert report["counts"]["stale"] == report["symbols"]
        assert only_reads(api.fake)


async def test_a_wrong_system_clock_makes_nothing_executable(tmp_path):
    clock = FakeClock(NOW)
    alpaca_time = FakeClock(NOW - timedelta(seconds=90))  # this computer runs 90s ahead of Alpaca
    fake = FakeAlpacaPaper(clock=alpaca_time)
    async for api in brain_client(tmp_path, clock, fake=fake):
        cycle = await run_cycle(api)
        report = cycle["data_quality"]["feed"]
        assert report["clock_skew_s"] == 90.0 and "90.0s ahead of Alpaca's" in report["headline"]
        veto = cycle["data_quality"]["market"]["veto"]
        assert "system clock is +90s off Alpaca's" in veto
        assert not [d for d in cycle["decisions"] if d["status"] in {"recommended", "dry_run_approved"}]
        assert only_reads(api.fake)


async def test_market_closed_means_nothing_is_executable(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False  # Alpaca's clock is authoritative
    async for api in brain_client(tmp_path, clock, fake=fake):
        cycle = await run_cycle(api)
        assert cycle["status"] == "completed" and cycle["market"]["open"] is False
        assert cycle["data_quality"]["market"]["veto"] == "market closed"
        assert set(cycle["data_quality"]["states"].values()) == {"market_closed"}
        for d in cycle["decisions"]:
            if d["action"] in TRADES:
                assert d["status"] in {"blocked", "risk_rejected"}
                assert any("market" in b for b in d["rationale"]["blocked_by"])
        assert only_reads(api.fake)


class AlpacaDown(FakeAlpacaPaper):
    """The paper API answers 503 to everything."""

    def send(self, request, **kwargs):
        self.fail_status = 503
        return super().send(request, **kwargs)


async def test_broker_unavailable_degrades_to_analysis_only(tmp_path):
    clock = FakeClock(NOW)
    fake = AlpacaDown(clock=clock)
    async for api in brain_client(tmp_path, clock, fake=fake):
        cycle = await run_cycle(api)
        assert cycle["status"] == "completed"
        alpaca = cycle["portfolio"]["alpaca_account"]
        assert alpaca["available"] is False and alpaca["error"]
        assert cycle["market"]["clock"] == "calendar"  # fell back to the exchange calendar
        assert "broker unavailable" in cycle["data_quality"]["market"]["veto"]
        assert cycle["consensus"]  # the analysis still happened
        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        assert all(d["status"] in ("blocked", "risk_rejected") for d in trades)  # nothing executable
        assert cycle["summary"]["book_fills"] == 0
        assert ("POST", "/v2/orders") not in fake.log


async def test_agent_controls_and_background_runs(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        r = await api.post(f"{BRAIN}/agents/momentum", json={"enabled": False})
        assert r.status_code == 200 and r.json()["enabled"] is False
        assert (await api.get(f"{BRAIN}/agents/momentum")).json()["enabled"] is False
        assert (await api.get(f"{BRAIN}/agents/nobody")).status_code == 404
        assert (await api.post(f"{BRAIN}/agents/nobody", json={"enabled": True})).status_code == 404
        assert (await api.get(f"{BRAIN}/status")).json()["agents"] == {"registered": 17, "enabled": 16}

        cycle = await run_cycle(api, symbols=["DNA"])
        runs = by(cycle["runs"], "agent_id")
        assert runs["momentum"]["status"] == "skipped" and runs["momentum"]["reason"] == "disabled"
        assert "DNA" in [f["symbol"] for f in cycle["focus"]]  # a requested symbol is studied
        await api.post(f"{BRAIN}/agents/momentum", json={"enabled": True})

        r = await api.post(f"{BRAIN}/run", params={"wait": 0})  # answer at once, run in the background
        assert r.status_code in (200, 202)
        if r.status_code == 202:
            job = r.json()
            assert r.headers["Location"] == f"/api/v1/jobs/{job['id']}"
            await api.container.jobs.wait(api.container.jobs.latest("brain-cycle"), 60)
        cycles = (await api.get(f"{BRAIN}/cycles")).json()
        assert len(cycles) == 2 and cycles[0]["status"] == "completed"
        assert (await api.get(f"{BRAIN}/cycles/999")).status_code == 404
        assert (await api.post(f"{BRAIN}/run", json={"symbols": ["X"] * 26})).status_code == 422
        assert only_reads(api.fake)


async def test_brain_controls_refuse_remote_callers_without_a_token(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, client_host="203.0.113.9"):
        for path, body in (("/run", None), ("/agents/momentum", {"enabled": False})):
            r = await api.post(f"{BRAIN}{path}", json=body)
            assert r.status_code == 403 and "QP_API_TOKEN" in r.json()["detail"]
        assert (await api.get(f"{BRAIN}/status")).status_code == 200  # read-only views stay available
        assert (await api.get(f"{BRAIN}/cycles")).json() == []
    async for api in brain_client(tmp_path / "token", clock, client_host="203.0.113.9", api_token="tok-123"):
        assert (await api.post(f"{BRAIN}/run")).status_code == 401
        r = await api.post(f"{BRAIN}/run", headers={"X-API-Key": "tok-123"})
        assert r.status_code == 200 and r.json()["status"] == "completed"
        assert only_reads(api.fake)


async def test_every_agent_takes_part_when_its_data_exists(tmp_path, monkeypatch):
    """Stock-model run, live option chains and an earnings calendar supplied as test data through the real
    service interfaces: all thirteen agents run and vote, and an imminent release blocks new risk."""
    from datetime import UTC, datetime

    from quantpulse.core.gateway import Resolved
    from quantpulse.schemas.common import DataStatus, Provenance
    from quantpulse.schemas.reference import CompanyEvents, CompanyProfile
    from quantpulse.services.model import ModelService
    from quantpulse.services.options import OptionsService
    from quantpulse.services.reference import ReferenceService
    from tests.unit.test_brain_agents import chain, snapshot, universe_features

    now = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
    prov = Provenance(status=DataStatus.LIVE, provider="test", as_of=now, fetched_at=now)
    feats = universe_features()
    feats = pd.concat([feats, feats.iloc[: len(STOCKS)].set_axis(STOCKS)])
    probs = {s: 0.62 if s.startswith("UP") else 0.40 if s.startswith("DN") else 0.5 for s in STOCKS}

    async def model_snapshot(self, *, wait=None):
        return snapshot(feats, probs, as_of=date(2026, 9, 24))

    async def option_chain(self, symbol, expirations=None, max_expirations=8, *, force_refresh=False):
        return Resolved(chain(spot=100.0), prov), {}

    async def events(self, symbol, *, force_refresh=False):
        profile = CompanyProfile(symbol=symbol, cik="0", name=symbol, sector="12", sector_label="Other")
        release = datetime(2026, 9, 15, 11, 0, tzinfo=UTC)  # before the open: reacts the same session
        return Resolved(
            CompanyEvents(profile=profile, earnings=[release], earnings_since=date(2024, 1, 1)), prov
        )

    async def next_earnings(self, symbol):
        return (date(2026, 9, 27), "scheduled") if symbol == "UPB" else (date(2026, 11, 20), "estimated")

    monkeypatch.setattr(ModelService, "trading_snapshot", model_snapshot)
    monkeypatch.setattr(OptionsService, "chain", option_chain)
    monkeypatch.setattr(ReferenceService, "events", events)
    monkeypatch.setattr(ReferenceService, "next_earnings", next_earnings)
    clock = FakeClock(NOW)
    async for api in brain_client(
        tmp_path, clock, brain_use_stock_model=True, brain_options_analysis=True, brain_catalyst_analysis=True
    ):
        cycle = await run_cycle(api)
        assert cycle["status"] == "completed"
        statuses = {r["agent_id"]: r["status"] for r in cycle["runs"]}
        # nothing promoted, no language model
        assert statuses == {a: "skipped" if a in ("strategy_lab", "briefing") else "ok" for a in AGENTS}
        voters = {v["agent_id"] for c in cycle["consensus"] for v in c["detail"]["votes"]}
        assert {"fundamental", "valuation", "factor", "options", "technical", "momentum"} <= voters
        catalyst = [o for o in cycle["opinions"] if o["agent_id"] == "catalyst"]
        assert catalyst and all(o["meta"]["next"] for o in catalyst)
        upb = by(cycle["decisions"])["UPB"]
        assert upb["action"] in {"watch", "no_action"} and upb["status"] == "no_trade"
        if upb["action"] == "watch":
            assert "earnings in 2 day(s)" in upb["rationale"]["reasons"][0]
        factor = next(o for o in cycle["opinions"] if o["agent_id"] == "factor" and o["subject"] == "UPC")
        assert factor["stance"] == "bullish" and factor["evidence"][0]["name"] == "prob_outperform"
        assert only_reads(api.fake)


async def test_opportunities_debates_and_posture_are_recorded(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        cycle = await run_cycle(api)
        assert cycle["market"]["situation"]["posture"] == "normal"
        assert cycle["summary"]["posture"] == "normal"
        ops = cycle["opportunities"]
        assert ops and all(o["stages"][0]["stage"] == "detection" for o in ops)
        assert sum(cycle["summary"]["opportunities"].values()) == len(ops)
        focused = [o for o in ops if o["status"] not in ("not_analysed", "context", "rejected_data")]
        for o in focused:  # every analysed idea went through the whole pipeline
            stages = [s["stage"] for s in o["stages"]]
            for stage in ("data_validation", "relevant_agents", "research", "bull_case", "bear_case",
                          "devils_advocate", "consensus"):  # fmt: skip
                assert stage in stages, (o["subject"], stages)
        served = (await api.get(f"{BRAIN}/opportunities")).json()
        assert {o["id"] for o in served} == {o["id"] for o in ops}
        kind = ops[0]["kind"]
        assert all(
            o["kind"] == kind for o in (await api.get(f"{BRAIN}/opportunities", params={"kind": kind})).json()
        )

        debates = by(cycle["debates"])
        assert set(debates) == {c["subject"] for c in cycle["consensus"] if c["subject"] != "@market"}
        verdicts = {d["verdict"] for d in debates.values()}
        assert verdicts <= {"stands", "weakened", "challenged", "no view to challenge"}
        for d in debates.values():
            assert d["confidence_after"] <= d["confidence_before"] + 1e-9
            if d["verdict"] != "no view to challenge":
                assert any(o["code"] == "unproven" for o in d["objections"])
                assert any(
                    c["reasons"][-1].startswith("devil's advocate")
                    for c in cycle["consensus"]
                    if c["subject"] == d["subject"]
                )
        research = [o for o in cycle["opinions"] if o["agent_id"] == "research"]
        assert research and all(o["stance"] == "abstain" and o["meta"]["findings"] for o in research)
        assert only_reads(api.fake)


async def test_the_brain_sends_nothing_even_when_paper_orders_are_enabled(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(
        tmp_path, clock, alpaca_trading_enabled=True, trading_dry_run=False, brain_use_stock_model=True
    ):
        cycle = await run_cycle(api)
        controls = cycle["portfolio"]["trading_controls"]
        # the strategy could send (it owns the account); the Brain may not: it only recommends in this mode
        assert controls["blockers"] == [
            "QP_BRAIN_MODE=paper_recommendation: the Brain does not manage the Alpaca paper account"
        ]
        assert not controls["orders_would_reach_alpaca"]
        assert any(d["status"] == "recommended" for d in cycle["decisions"])  # it has trades it would make
        assert cycle["summary"]["orders_sent"] == 0
        assert only_reads(api.fake)  # the brain proposed them; nothing was sent


class ScriptedModel:
    """A test-only language-model provider: a fixed structured answer, every request recorded."""

    name = "scripted-test"

    def __init__(self) -> None:
        self.requests: list = []

    def unavailable(self):
        return None

    async def complete(self, request, model):
        self.requests.append((request, model))
        answer = {
            "summary": "The agents lean one way.",
            "supporting": ["trend"],
            "opposing": [],
            "watch": "x",
        }
        return ModelResponse(json.dumps(answer), model, 400, 60, "end_turn")


async def test_a_configured_model_writes_briefings_that_never_vote_or_trade(tmp_path):
    model = ScriptedModel()
    register_provider("scripted-test", lambda _s: model)
    clock = FakeClock(NOW)
    try:
        async for api in brain_client(
            tmp_path,
            clock,
            brain_llm_provider="scripted-test",
            brain_llm_fast_model="fast-test",
            brain_llm_daily_token_budget=50_000,
            brain_llm_max_briefings=2,
        ):
            cycle = await run_cycle(api)
            run = by(cycle["runs"], "agent_id")["briefing"]
            assert run["status"] == "ok" and run["model_tier"] == "fast"
            briefs = [o for o in cycle["opinions"] if o["agent_id"] == "briefing"]
            assert len(briefs) == 2 == len(model.requests)
            for o in briefs:  # context for people, never a vote or a graded forecast
                assert o["stance"] == "abstain" and o["thesis"].startswith("briefing (fast-test)")
                assert o["meta"]["briefing"]["summary"] == "The agents lean one way."
            voters = {v["agent_id"] for c in cycle["consensus"] for v in c["detail"]["votes"]}
            assert "briefing" not in voters
            preds = await api.container.brain.store.predictions()
            assert preds and all(p.source_id != "briefing" for p in preds)
            request, name = model.requests[0]
            assert name == "fast-test" and request.task.value == "summarize" and request.purpose == "briefing"

            usage = (await api.get(f"{BRAIN}/models")).json()
            assert usage["available"] and usage["models"] == {"fast": "fast-test", "strong": None}
            assert usage["usage"]["calls"] == 2 and usage["usage"]["tokens"] == 2 * 460
            assert usage["usage"]["by_purpose"] == {"briefing": 920} and len(usage["recent"]) == 2
            status = (await api.get(f"{BRAIN}/status")).json()
            assert (
                "scripted-test" in status["language_models"] and "920 of 50,000" in status["language_models"]
            )

            portfolio = await run_cycle(api, kind="portfolio")  # cheaper cycles leave the model out
            reason = by(portfolio["runs"], "agent_id")["briefing"]["reason"]
            assert reason == "not needed for a portfolio cycle" and len(model.requests) == 2
            assert only_reads(api.fake)
    finally:
        unregister_provider("scripted-test")


async def test_the_kill_switch_makes_the_brain_defensive(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        await seed_book(api, {"UPA": (40, 60.0)})
        r = await api.post(
            "/api/v1/trading/kill-switch",
            json={"active": True, "reason": "test", "cancel_open_orders": False},
        )
        assert r.status_code == 200
        cycle = await run_cycle(api)
        situation = cycle["market"]["situation"]
        assert situation["posture"] == "defensive" and "the kill switch is on" in situation["reasons"]
        actions = {d["action"] for d in cycle["decisions"]}
        assert not actions & {"buy", "increase"}  # no new risk while defensive
        for d in cycle["decisions"]:
            if d["subject"] != "UPA" and d["action"] == "watch":
                assert any("defensive posture" in r for r in d["rationale"]["reasons"])
        upa = by(cycle["decisions"])["UPA"]
        assert upa["action"] in {"hold", "de_risk", "rebalance"}
        if upa["action"] == "de_risk":
            assert upa["risk"]["checks"] and upa["quantity"] == 13  # a third of 40, whole shares
        assert only_reads(api.fake)


def extend_feed(feed: TrendFeed, until: date) -> None:
    """Let the fake market keep trading after the cycle: each symbol follows its drift to ``until``."""
    from datetime import UTC, datetime, time

    import numpy as np

    from quantpulse.core.market_calendar import NEW_YORK, next_trading_day
    from quantpulse.schemas.market import Bar

    rng = np.random.default_rng(99)
    for symbol, bars in feed.bars.items():
        price = feed.live_price(symbol)
        day = feed.clock.now().astimezone(NEW_YORK).date()
        while day <= until:
            new = price * float(np.exp(feed.drifts.get(symbol, 0.0) + rng.normal(0, 0.006)))
            stamp = datetime.combine(day, time(16, 0), NEW_YORK).astimezone(UTC)
            bars.append(Bar(timestamp=stamp, open=price, high=max(price, new) * 1.004, low=min(price, new) * 0.996,
                            close=new, volume=4_000_000))  # fmt: skip
            price, day = new, next_trading_day(day)


async def test_predictions_mature_and_the_brain_learns_from_them(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        cycle = await run_cycle(api)
        assert cycle["predictions_recorded"] > 0
        before = (await api.get(f"{BRAIN}/learning")).json()
        assert before["predictions"]["evaluated"] == 0 and before["predictions"]["open"] > 0

        # nothing is graded before its due date
        early = (await api.post(f"{BRAIN}/learn")).json()
        assert early["evaluated"] == 0 and early["reflections"] == 0

        extend_feed(api.feed, date(2026, 11, 6))
        clock.advance(40 * 86400)  # ~27 sessions later: the 5-, 10- and 21-day calls have matured
        learned = (await api.post(f"{BRAIN}/learn")).json()
        assert learned["evaluated"] > 0 and learned["voided"] == 0
        state = (await api.get(f"{BRAIN}/learning")).json()
        assert state["predictions"]["evaluated"] == learned["evaluated"]
        assert state["predictions"]["open"] == 0  # every horizon here (5 to 21 sessions) has matured
        assert state["last_run"]["evaluated"] == learned["evaluated"]
        assert state["calibration"] and all(b["n"] > 0 for b in state["calibration"])

        perf_rows = (await api.get(f"{BRAIN}/performance", params={"window": "all"})).json()
        assert {r["agent_id"] for r in perf_rows} >= {"technical", "consensus"}
        assert all(r["reliability"] is None for r in perf_rows)  # one cycle is never a track record
        assert all(r["verdict"] == "unproven" and r["n_effective"] <= r["n"] for r in perf_rows)
        assert state["measured_agents"] == []
        graded = [p for p in await api.container.brain.store.predictions() if p.status == "evaluated"]
        for p in graded:  # accuracy, risk, luck, timing and data quality are kept apart
            outcome = p.context["outcome"]
            assert outcome["data_ok"] is True and "timing" in outcome and "after_next_close" in outcome
            assert "relative_z" in outcome and isinstance(outcome["noise"], bool)
            total = outcome["timing"] + outcome["after_next_close"]
            assert total == pytest.approx(p.direction * p.realized_relative, abs=1e-5)

        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        reflections = (await api.get(f"{BRAIN}/reflections")).json()
        assert learned["reflections"] == len(reflections) >= len(trades)
        for r in reflections:
            assert r["category"] in {"earned", "unlucky", "lucky", "process_failure", "inconclusive",
                                     "block_saved_money", "block_cost_opportunity"}  # fmt: skip
            assert r["decision_quality"] in {"good", "fair", "poor"} and r["lessons"]
        again = (await api.post(f"{BRAIN}/learn")).json()
        assert again["evaluated"] == 0 and again["reflections"] == 0
        status = (await api.get(f"{BRAIN}/status")).json()
        assert f"{learned['evaluated']} predictions graded" in status["learning"]
        assert only_reads(api.fake)


async def next_session(api, clock: FakeClock) -> None:
    """Close today's session in the fake market and move to 11:00 New York on the next trading day."""
    from datetime import UTC, datetime, time

    from quantpulse.core.market_calendar import NEW_YORK, next_trading_day

    today = clock.now().astimezone(NEW_YORK).date()
    extend_feed(api.feed, today)
    nxt = datetime.combine(next_trading_day(today), time(11, 0), NEW_YORK).astimezone(UTC)
    clock.advance((nxt - clock.now()).total_seconds())
    for s in api.feed.bars:
        api.fake.prices[s] = api.feed.live_price(s)


async def test_the_brain_manages_its_own_paper_book_over_several_sessions(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, brain_use_stock_model=True):
        api.fake.hold("UPC", 25, 70.0)  # the strategy's account: never touched by the Brain's book
        first = await run_cycle(api)
        bought = {f["symbol"] for f in first["portfolio"]["book"]["fills"] if f["side"] == "buy"}
        assert bought and first["summary"]["book_fills"] >= len(bought)
        for _ in range(3):
            await next_session(api, clock)
            await run_cycle(api)
        book = (await api.get(f"{BRAIN}/book")).json()
        assert "hypothetical" in book["owner"] and book["capital"] == 100_000
        held = {p["symbol"]: p for p in book["positions"]}
        assert bought <= set(held)  # positions persist across cycles and days
        for p in held.values():  # every position says why it exists and what would end it
            assert p["qty"] > 0 and p["avg_cost"] > 0 and p["thesis"]
            assert p["stop_price"] == pytest.approx(p["avg_cost"] * 0.92, rel=0.01)
            assert p["review_after"] and p["expected_return"] is None  # uncalibrated: nothing invented
        assert len(book["equity_curve"]) == 4  # one close per session
        perf = book["performance"]
        assert perf["sessions"] == 4 and perf["too_short_to_judge"] and perf["trades"] == len(book["trades"])
        assert perf["costs"] > 0 and perf["slippage"] > 0 and "benchmark_return" in perf
        spent = sum(t["notional"] + t["cost"] for t in book["trades"] if t["side"] == "buy")
        received = sum(t["notional"] - t["cost"] for t in book["trades"] if t["side"] == "sell")
        assert book["cash"] == pytest.approx(
            100_000 - spent + received, abs=0.05
        )  # every dollar accounted for
        assert "UPC" not in held and only_reads(api.fake)  # zero orders; the Alpaca account is untouched
        assert api.fake.positions["UPC"]["qty"] == 25


async def test_a_book_position_at_its_stop_is_closed_and_its_loss_recorded(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        price = api.feed.live_price("DNA")
        await seed_book(api, {"DNA": (10, price / 0.85)})  # bought 15% higher: past the 8% stop
        cycle = await run_cycle(api)
        dna = by(cycle["decisions"])["DNA"]
        assert dna["action"] == "close" and dna["status"] == "recommended"
        fill = dna["execution"]["book"]
        assert fill["side"] == "sell" and fill["qty"] == 10 and fill["realized_pnl"] < 0
        book = (await api.get(f"{BRAIN}/book")).json()
        assert "DNA" not in {p["symbol"] for p in book["positions"]}
        assert book["performance"]["closed_trades"] == 1 and book["performance"]["closed_hit_rate"] == 0.0
        spent = sum(t["notional"] + t["cost"] for t in book["trades"] if t["side"] == "buy")
        received = sum(t["notional"] - t["cost"] for t in book["trades"] if t["side"] == "sell")
        assert book["cash"] == pytest.approx(100_000 - 10 * price / 0.85 - spent + received, abs=0.05)
        assert fill["realized_pnl"] == pytest.approx(
            (fill["fill_price"] - price / 0.85) * 10 - fill["cost"], abs=0.01
        )
        assert only_reads(api.fake)


async def test_the_book_is_reset_only_deliberately(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        await seed_book(api, {"UPA": (10, 60.0)})
        refused = await api.post(f"{BRAIN}/book/reset", json={"confirm": "yes"})
        assert refused.status_code == 422 and len((await api.get(f"{BRAIN}/book")).json()["positions"]) == 1
        done = await api.post(f"{BRAIN}/book/reset", json={"confirm": "RESET BOOK"})
        assert done.status_code == 200 and done.json()["positions"] == [] and done.json()["cash"] == 100_000
    async for api in brain_client(tmp_path / "remote", clock, client_host="203.0.113.9"):
        assert (await api.post(f"{BRAIN}/book/reset", json={"confirm": "RESET BOOK"})).status_code in (
            401,
            403,
        )
