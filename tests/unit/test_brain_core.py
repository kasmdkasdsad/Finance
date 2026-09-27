"""The brain's building blocks without I/O: opinions, consensus (agreement, disagreement, "unknown",
vetoes, unproven reliability) and the agent registry (selection, skips, dependency order, isolation of a
failing or slow agent)."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd
import pytest

from quantpulse.brain.agents import default_agents
from quantpulse.brain.agents.base import Agent
from quantpulse.brain.consensus import ReliabilityBook, build_consensus
from quantpulse.brain.context import BrainContext, PortfolioState
from quantpulse.brain.registry import AgentRegistry
from quantpulse.brain.types import (
    MARKET,
    AgentFamily,
    AgentSpec,
    BrainMode,
    BrainSession,
    DataState,
    Opinion,
    Stance,
    stance_of,
    worst_state,
)
from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_risk import RiskLimits


def op(agent, score, confidence=0.7, *, subject="AAA", quality=DataState.FRESH, veto=None, version="1.0.0"):
    stance = Stance.ABSTAIN if score is None else stance_of(score)
    return Opinion(
        agent_id=agent,
        agent_version=version,
        subject=subject,
        stance=stance,
        score=score or 0.0,
        confidence=confidence,
        horizon_days=5,
        thesis=f"{agent} thinks {score}",
        data_quality=quality,
        veto=veto,
    )


def make_ctx(focus=("AAA", "BBB")) -> BrainContext:
    empty = pd.DataFrame()
    return BrainContext(
        as_of=datetime(2026, 9, 25, 14, 0, tzinfo=UTC),
        session=BrainSession.OPEN,
        market_open=True,
        clock_source="calendar",
        mode=BrainMode.DRY_RUN,
        universe=list(focus),
        close=empty,
        high=empty,
        low=empty,
        volume=empty,
        benchmark=pd.Series(dtype=float),
        benchmark_symbol="SPY",
        qqq=None,
        price_status=DataStatus.LIVE,
        quotes={},
        quality={},
        missing_quotes={},
        indicators=empty,
        market_stats={},
        regime=None,
        vix=None,
        implied_vol={},
        earnings={},
        model_z={},
        fundamentals=None,
        sectors={},
        portfolio=PortfolioState(available=True),
        data_states=dict.fromkeys(focus, DataState.FRESH),
        limits=RiskLimits(),
        kill_switch=False,
        focus=list(focus),
    )


# ---------------------------------------------------------------------------------------------- opinions
def test_opinion_scores_are_bounded_and_abstain_carries_no_view():
    o = op("x", 3.0, confidence=1.7)
    assert o.score == 1.0 and o.confidence == 1.0 and o.directional
    assert op("x", float("nan")).score == 0.0  # NaN never leaks into a score
    a = Opinion("x", "1", "AAA", Stance.ABSTAIN, 0.9, 0.9, 5, "no view")
    assert a.score == 0.0 and a.confidence == 0.0 and not a.directional
    assert worst_state([DataState.FRESH, DataState.STALE, DataState.LIVE]) is DataState.STALE
    assert worst_state([]) is DataState.UNAVAILABLE


# ---------------------------------------------------------------------------------------------- consensus
def test_agreement_gives_a_confident_view():
    c = build_consensus("AAA", [op("tech", 0.6), op("mom", 0.5)])
    assert c.stance is Stance.BULLISH and not c.unknown and c.actionable_view
    assert (c.supporting, c.opposing, c.neutral) == (2, 0, 0)
    assert c.disagreement == pytest.approx(0.0) and c.primary_disagreement is None
    assert c.confidence == pytest.approx(0.7)  # mean confidence × full coverage × fresh data
    assert all(v.reliability.status == "unproven" and v.reliability.weight == 1.0 for v in c.votes)


def test_disagreement_is_measured_kept_and_means_unknown():
    c = build_consensus("AAA", [op("tech", 0.6), op("mom", -0.55)])
    assert c.unknown and c.stance is Stance.NEUTRAL and not c.actionable_view
    assert c.disagreement > 0.5 and c.supporting + c.opposing == 2
    assert c.primary_disagreement["for"]["agent_id"] == "tech"
    assert c.primary_disagreement["against"]["agent_id"] == "mom"
    assert any("disagree" in r for r in c.reasons)


def test_one_unsure_voice_or_no_voice_is_unknown():
    lone = build_consensus("AAA", [op("tech", 0.6, confidence=0.4)])
    assert lone.unknown and any("only one view" in r for r in lone.reasons)
    single = build_consensus("AAA", [op("tech", 0.6, confidence=0.9)])
    assert single.confidence == pytest.approx(0.45)  # one view is half the coverage of two
    silent = build_consensus("AAA", [op("tech", None), op("mom", None)])
    assert silent.unknown and silent.abstaining == 2 and silent.reasons == ["no agent had a view"]


def test_stale_data_weakens_the_view_and_vetoes_stay_attached():
    fresh = build_consensus("AAA", [op("tech", 0.6), op("mom", 0.6)])
    stale = build_consensus("AAA", [op("tech", 0.6, quality=DataState.STALE), op("mom", 0.6)])
    assert stale.data_quality is DataState.STALE and stale.confidence < fresh.confidence
    vetoed = build_consensus(
        "AAA", [op("tech", 0.6), op("mom", 0.6)], [op("data_quality", 0.0, veto="quote is 20 minutes old")]
    )
    assert vetoed.vetoes == [{"agent_id": "data_quality", "reason": "quote is 20 minutes old"}]
    assert vetoed.stance is Stance.BULLISH  # the view stands; the veto blocks acting on it


def test_reliability_counts_only_once_measured():
    rows = [
        {"agent_id": "tech", "agent_version": "1.0.0", "regime": "all", "n": 12, "reliability": 0.2},
        {"agent_id": "mom", "agent_version": "1.0.0", "regime": "all", "n": 80, "reliability": 0.5},
    ]
    book = ReliabilityBook(rows, min_observations=30)
    assert book.get("tech", "1.0.0").status == "unproven" and book.get("tech", "1.0.0").weight == 1.0
    assert book.get("mom", "1.0.0").status == "measured" and book.get("mom", "1.0.0").weight == 0.5
    assert book.get("mom", "2.0.0").status == "unproven"  # a new version starts with no record
    c = build_consensus("AAA", [op("tech", 0.6), op("mom", 0.6)], reliability=book)
    weights = {v.agent_id: v.weight for v in c.votes}
    assert weights["tech"] == pytest.approx(0.7) and weights["mom"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------------------------- registry
class Probe(Agent):
    spec = AgentSpec("probe", "Probe", "test agent", AgentFamily.SPECIALIST, ("test",), ("close",))

    def __init__(self, *, fail=False, sleep=0.0, why_not=None, agent_id=None, deps=()):
        if agent_id or deps:
            self.spec = AgentSpec(
                agent_id or "probe",
                "Probe",
                "",
                AgentFamily.SPECIALIST,
                ("test",),
                ("close",),
                dependencies=deps,
            )
        self.fail, self.sleep, self.why_not = fail, sleep, why_not
        self.seen: dict[str, list[str]] = {}

    def unavailable(self, ctx):
        return self.why_not

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.fail:
            raise RuntimeError("boom")
        self.seen = {s: [o.agent_id for o in ctx.working.opinions.get(s, [])] for s in subjects}
        out = [op(self.spec.id, 0.5, subject=s) for s in subjects]
        return [*out, op(self.spec.id, 0.5, subject="NOT-ASKED")]  # out-of-scope opinions are dropped


def test_registry_rejects_duplicates_and_unknown_dependencies():
    reg = AgentRegistry([Probe()])
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Probe())
    with pytest.raises(ValueError, match="unregistered"):
        reg.register(Probe(agent_id="later", deps=("nobody",)))


def test_default_agents_are_deterministic_except_the_optional_briefing():
    reg = AgentRegistry(default_agents())
    ids = {a.spec.id: a.role for a in reg.all()}
    assert ids == {
        "data_quality": "constraint",
        "market_regime": "forecast",
        "technical": "forecast",
        "momentum": "forecast",
        "mean_reversion": "forecast",
        "volatility": "forecast",
        "statistical": "forecast",
        "fundamental": "forecast",
        "valuation": "forecast",
        "factor": "forecast",
        "options": "forecast",
        "catalyst": "forecast",
        "strategy_lab": "forecast",
        "portfolio": "constraint",
        "research": "context",
        "situational_awareness": "context",
        "briefing": "context",
    }
    assert {a.spec.id for a in reg.all() if a.spec.stage == 1} == {"research", "situational_awareness"}
    # the only model-backed agent runs last and never votes
    assert {a.spec.id for a in reg.all() if a.spec.model_tier.value != "deterministic"} == {"briefing"}
    assert reg.get("briefing").spec.stage == 2 and reg.get("briefing").role == "context"


def test_selection_explains_every_skip():
    ctx = make_ctx()
    reg = AgentRegistry([Probe(), Probe(agent_id="off"), Probe(agent_id="cannot", why_not="no options data")])
    reg.set_enabled("off", False)
    chosen, skipped = reg.select(ctx)
    assert [s.agent.spec.id for s in chosen] == ["probe"] and chosen[0].subjects == ["AAA", "BBB"]
    assert {s.agent_id: s.reason for s in skipped} == {"off": "disabled", "cannot": "no options data"}
    chosen, skipped = reg.select(make_ctx(focus=()))
    assert not chosen and {s.reason for s in skipped} >= {"nothing to analyse this cycle"}
    with pytest.raises(KeyError):
        reg.set_enabled("ghost", True)


async def test_dependencies_run_first_and_share_working_memory():
    first, second = Probe(agent_id="first"), Probe(agent_id="second", deps=("first",))
    reg = AgentRegistry([first, second])
    ctx = make_ctx()
    chosen, _ = reg.select(ctx)
    assert [[s.agent.spec.id for s in lvl] for lvl in reg.levels(chosen)] == [["first"], ["second"]]
    runs = await reg.run(ctx, chosen, timeout=5)
    assert [r.status for r in runs] == ["ok", "ok"]
    assert second.seen == {"AAA": ["first"], "BBB": ["first"]}  # it saw the first agent's findings
    assert all(len(r.opinions) == 2 for r in runs)  # the opinion on an unasked subject was dropped
    reg.set_enabled("first", False)
    _, skipped = reg.select(ctx)
    assert any(s.agent_id == "second" and "dependencies" in s.reason for s in skipped)


async def test_a_failing_or_slow_agent_never_stops_the_others():
    reg = AgentRegistry(
        [Probe(agent_id="good"), Probe(agent_id="bad", fail=True), Probe(agent_id="slow", sleep=5)]
    )
    ctx = make_ctx()
    chosen, _ = reg.select(ctx)
    runs = {r.agent_id: r for r in await reg.run(ctx, chosen, timeout=0.2)}
    assert runs["good"].status == "ok" and len(runs["good"].opinions) == 2
    assert (
        runs["bad"].status == "failed"
        and runs["bad"].error == "RuntimeError: boom"
        and not runs["bad"].opinions
    )
    assert runs["slow"].status == "timeout" and not runs["slow"].opinions and runs["slow"].cost == 0.0
    assert [o.agent_id for o in ctx.working.opinions["AAA"]] == ["good"]
    assert MARKET not in ctx.working.opinions
