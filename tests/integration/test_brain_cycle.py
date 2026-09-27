"""The brain end to end, through the HTTP API, on the fake market feed and the fake Alpaca paper account.

A cycle perceives (account, clock, prices, live quotes, indicators, regime), runs the five deterministic
agents, builds a consensus per subject, proposes portfolio actions, has the existing risk engine preview
each trade and persists everything. It never sends an order: every Alpaca request it makes is a read.
"""

from datetime import date, timedelta

import pytest

from quantpulse.brain.agents.technical import TechnicalAgent
from quantpulse.core.clock import FakeClock
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import DRIFTS, TrendFeed

from .conftest import NOW
from .test_trading import trading_client

BRAIN = "/api/v1/brain"
AGENTS = {"data_quality", "market_regime", "technical", "momentum", "portfolio"}
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


async def run_cycle(api, **body):
    r = await api.post(f"{BRAIN}/run", json=body or None)
    assert r.status_code == 200, r.text
    return r.json()


def by(items, key="subject"):
    return {x[key]: x for x in items}


def only_reads(fake: FakeAlpacaPaper) -> bool:
    return all(method == "GET" for method, _ in fake.log) and fake.orders == {}


async def test_a_full_cycle_perceives_thinks_proposes_and_sends_nothing(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        api.fake.hold("UPA", 40, 60.0)  # a small existing position
        cycle = await run_cycle(api)

        # perception: session, regime, the paper account, focus
        assert cycle["status"] == "completed" and cycle["error"] is None
        assert cycle["mode"] == "paper_recommendation" and cycle["session"] == "market_open"
        assert cycle["regime"]["label"] == "bullish" and cycle["market"]["clock"] == "alpaca"
        assert cycle["portfolio"]["available"] and cycle["portfolio"]["positions"]["UPA"]["qty"] == 40
        focus = [f["symbol"] for f in cycle["focus"]]
        assert focus[0] == "UPA" and cycle["focus"][0]["reason"] == "held position"
        assert 1 < len(focus) <= 9 and "SPY" not in focus

        # every agent ran and its run is persisted
        assert {a["agent_id"] for a in cycle["agents"]} == AGENTS
        assert all(a["status"] == "ok" for a in cycle["agents"])
        assert {r["agent_id"]: r["status"] for r in cycle["runs"]} == dict.fromkeys(AGENTS, "ok")

        # structured findings: each focus symbol seen by data quality, technical and momentum
        seen: dict[str, set[str]] = {}
        for o in cycle["opinions"]:
            seen.setdefault(o["subject"], set()).add(o["agent_id"])
            assert -1 <= o["score"] <= 1 and 0 <= o["confidence"] <= 1 and o["thesis"]
        for s in focus:
            assert {"data_quality", "technical", "momentum"} <= seen[s]
        assert seen["@market"] == {"data_quality", "market_regime"} and seen["@portfolio"] == {"portfolio"}
        tech = next(o for o in cycle["opinions"] if o["agent_id"] == "technical" and o["subject"] == "UPC")
        assert tech["stance"] == "bullish" and tech["evidence"] and tech["invalidation"]

        # consensus per subject, with the vote of each agent and its (unproven) reliability
        consensus = by(cycle["consensus"])
        assert set(consensus) == {"@market", *focus}
        upc = consensus["UPC"]
        assert upc["stance"] == "bullish" and not upc["unknown"] and upc["supporting"] == 2
        votes = upc["detail"]["votes"]
        assert {v["agent_id"] for v in votes} == {"technical", "momentum"}
        assert all(v["reliability"] == {"weight": 1.0, "status": "unproven", "n": 0} for v in votes)

        # proposed actions, each trade previewed by the deterministic risk engine; nothing sent
        decisions = by(cycle["decisions"])
        assert set(decisions) == set(focus)
        assert decisions["UPA"]["action"] in {"hold", "increase"}
        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        assert trades and sum(d["action"] == "buy" for d in trades) <= 2  # brain_max_new_positions_per_cycle
        for d in trades:
            assert d["risk"]["checks"] and d["status"] in {"recommended", "risk_rejected", "blocked"}
            assert d["execution"] == {"sent": False, "reason": "the brain never sends orders itself"}
            if d["risk_approved"]:
                assert d["notional"] <= 15_000  # sized within the risk engine's per-order limit
        assert any(d["status"] == "recommended" for d in trades)
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

        # the same cycle, read back
        again = (await api.get(f"{BRAIN}/cycles/{cycle['id']}")).json()
        assert again == cycle
        listed = (await api.get(f"{BRAIN}/cycles")).json()
        assert [c["id"] for c in listed] == [cycle["id"]] and "opinions" not in listed[0]

        status = (await api.get(f"{BRAIN}/status")).json()
        assert status["paper_only"] and status["last_cycle"]["id"] == cycle["id"]
        assert status["agents"] == {"registered": 5, "enabled": 5} and not status["running"]
        assert status["open_predictions"] == len(preds) and "not built yet" in status["learning"]

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
        assert {m["subject"] for m in long if m["kind"] == "decision"} == {d["subject"] for d in trades}


async def test_modes_dry_run_and_research_only(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path / "dry", clock, brain_mode="dry_run"):
        cycle = await run_cycle(api)
        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        assert trades and {d["status"] for d in trades} <= {"dry_run_approved", "risk_rejected", "blocked"}
        assert all(d["mode"] == "dry_run" for d in cycle["decisions"]) and only_reads(api.fake)
    async for api in brain_client(tmp_path / "research", clock, brain_mode="research_only"):
        cycle = await run_cycle(api)
        assert cycle["mode"] == "research_only" and cycle["decisions"] == []
        assert cycle["consensus"] and cycle["predictions_recorded"] > 0  # it still thinks and remembers
        assert only_reads(api.fake)


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
        assert {a for a, r in runs.items() if r["status"] == "ok"} == AGENTS - {"technical"}
        assert cycle["summary"]["agents_failed"] == 1
        assert not any(o["agent_id"] == "technical" for o in cycle["opinions"])
        # momentum alone is one voice: the brain says so instead of acting on it
        for c in cycle["consensus"]:
            if c["subject"] != "@market":
                assert [v["agent_id"] for v in c["detail"]["votes"]] == ["momentum"]
                assert c["confidence"] <= 0.5
        assert not [d for d in cycle["decisions"] if d["action"] in TRADES]
        agents = by((await api.get(f"{BRAIN}/agents")).json(), "id")
        assert agents["technical"]["runs"]["failures"] == 1


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
        assert cycle["portfolio"]["available"] is False and cycle["portfolio"]["error"]
        assert cycle["market"]["clock"] == "calendar"  # fell back to the exchange calendar
        assert "broker unavailable" in cycle["data_quality"]["market"]["veto"]
        assert cycle["consensus"]  # the analysis still happened
        trades = [d for d in cycle["decisions"] if d["action"] in TRADES]
        assert all(d["status"] == "not_checked" and not d["risk_approved"] for d in trades)
        assert ("POST", "/v2/orders") not in fake.log


async def test_agent_controls_and_background_runs(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock):
        r = await api.post(f"{BRAIN}/agents/momentum", json={"enabled": False})
        assert r.status_code == 200 and r.json()["enabled"] is False
        assert (await api.get(f"{BRAIN}/agents/momentum")).json()["enabled"] is False
        assert (await api.get(f"{BRAIN}/agents/nobody")).status_code == 404
        assert (await api.post(f"{BRAIN}/agents/nobody", json={"enabled": True})).status_code == 404
        assert (await api.get(f"{BRAIN}/status")).json()["agents"] == {"registered": 5, "enabled": 4}

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
