"""The option agents on a synthetic chain: each answers its one question, vetoes stop a candidate, the verdict
is the weighted view of the agents that had one, and the thesis/debate/explanation say what and why."""

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from quantpulse.brain.options import agents as A
from quantpulse.brain.options.perception import UnderlyingView
from quantpulse.core.clock import FakeClock
from quantpulse.options.lab.features import DayFeatures
from quantpulse.options.lab.genome import Genome
from quantpulse.options.selection import Spec, build, evaluate
from tests.fakes.options_market import FakeOptionsMarket

NOW = datetime(2026, 9, 25, 14, 30, tzinfo=UTC)
G = Genome("bull_call_spread", "bullish", entry_signal="trend_up", dte_min=20, dte_max=45, delta_target=0.5,
           width_pct=0.05, take_profit=0.8, stop_loss=0.5, event_filter="avoid")  # fmt: skip
VERSION = {"key": "test@v1", "stage": "PAPER_ACTIVE", "expected_ror": 0.06, "version_id": 1}


@pytest.fixture
def ctx():
    clock = FakeClock(NOW)
    market = FakeOptionsMarket(clock, lambda u: 200.0)
    market.vol = 0.22
    chain = asyncio.run(market.chain("AAA"))
    feats = DayFeatures(date(2026, 9, 25), 200.0, sma50=190, sma200=170, ret5=0.01, ret20=0.05, z5=0.5, rv20=0.24,
                        rv60=0.25, iv=0.22, iv_rank=40, iv_percentile=45, iv_rv=0.92, event_days=None)  # fmt: skip
    view = UnderlyingView("AAA", NOW, spot=200.0, chain=chain, quality=chain.quality(NOW), features=feats)
    view.expiries = []
    cand = build(Spec(G.family, G.dte_min, G.dte_max, G.delta_target, G.width_pct), chain.quotes, 200.0, NOW)[
        0
    ]
    evaluate(cand, 200.0, NOW)
    return A.CandidateContext(view=view, version=VERSION, genome=G, cand=cand, now=NOW,
                              stock_view={"stance": "bullish", "confidence": 0.6, "score": 0.5},
                              risk={"approved": True, "summary": "approved"}, paper=True)  # fmt: skip


def test_a_clean_candidate_is_supported_and_explained(ctx):
    v = A.deliberate(ctx)
    assert not v["vetoes"] and v["support"] >= v["oppose"], v["opinions"]
    names = {o["agent"] for o in v["opinions"]}
    assert (
        len(names) == len(A.AGENTS) and {"OptionsRiskAgent", "LiquidityAgent", "StrategyDecayAgent"} <= names
    )
    th = A.thesis(ctx, v)
    assert th["direction"] == "bullish" and "never held into expiration" in th["exit_plan"]
    assert th["invalidation"].startswith("a close below the break-even")
    d = A.debate(ctx, v)
    assert d["devils_advocate"] and isinstance(d["bull"], list)
    text = A.explain(th, v, mode="paper", comparison={"verdict": "option +0.30 + weight 0.15 vs stock +0.20"})
    assert "PAPER:" in text and "Maximum loss" in text and "Versus shares" in text


def test_vetoes_stop_a_candidate(ctx):
    from dataclasses import replace

    stacked = replace(ctx, book={"AAA": [7]})
    assert "OptionsPortfolioAgent" in {x["agent"] for x in A.deliberate(stacked)["vetoes"]}
    risky = replace(ctx, risk={"approved": False, "summary": "max_loss: too large"})
    assert "OptionsRiskAgent" in {x["agent"] for x in A.deliberate(risky)["vetoes"]}
    decaying = replace(ctx, decay="DEGRADING")
    assert "StrategyDecayAgent" in {x["agent"] for x in A.deliberate(decaying)["vetoes"]}
    research = replace(ctx, version={**VERSION, "stage": "WALK_FORWARD"})
    assert "StrategyCriticAgent" in {x["agent"] for x in A.deliberate(research)["vetoes"]}
    ctx.view.features.event_days = 5  # earnings inside the option's life, and the strategy avoids events
    assert "EarningsEventAgent" in {x["agent"] for x in A.deliberate(ctx)["vetoes"]}
    ctx.view.features.event_days = None
    wide = replace(ctx, genome=replace(G, max_spread_pct=0.001))
    assert "LiquidityAgent" in {x["agent"] for x in A.deliberate(wide)["vetoes"]}
    ctx.view.quality = {**ctx.view.quality, "OPTIONS_DATA_QUALITY": "research"}
    assert "OptionsDataQualityAgent" in {x["agent"] for x in A.deliberate(ctx)["vetoes"]}


def test_disagreement_and_rich_volatility_weigh_against_buying(ctx):
    from dataclasses import replace

    base = A.deliberate(ctx)["score"]
    bearish = replace(ctx, stock_view={"stance": "bearish", "confidence": 0.8, "score": -0.6})
    assert A.deliberate(bearish)["score"] < base
    ctx.view.features.iv_rv = 1.8
    assert A.implied_volatility(ctx).verdict == "neutral"  # a vertical spread is close to volatility-neutral
    long_call = replace(
        ctx, genome=Genome("long_call", "bullish", entry_signal="trend_up", dte_min=20, dte_max=45)
    )
    assert A.implied_volatility(long_call).verdict == "oppose"  # buying rich volatility outright
    assert A.no_trade(["3 vetoed by LiquidityAgent"]).verdict == "abstain"


def test_every_agent_abstains_rather_than_guesses(ctx):
    ctx.view.features = None
    for fn in (A.options_regime, A.implied_volatility, A.momentum, A.mean_reversion, A.valuation,
               A.volatility_strategy, A.earnings_event):  # fmt: skip
        assert fn(ctx).verdict == "abstain", fn.__name__
    assert timedelta(0) == timedelta(0)
