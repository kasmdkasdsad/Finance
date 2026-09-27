"""Phase 3: the opportunity engine, the research and situational-awareness agents, the adversarial debate
and the deeper decision step (posture, challenged views, portfolio fit, de-risking, rebalancing)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantpulse.brain import opportunities as opp
from quantpulse.brain.agents.research import ResearchAgent, SituationalAwarenessAgent
from quantpulse.brain.consensus import build_consensus
from quantpulse.brain.debate import review
from quantpulse.brain.decisions import plan, portfolio_fit
from quantpulse.brain.types import MARKET, Action, DataState, Evidence, Opinion, Stance, stance_of
from quantpulse.providers.alpaca_trading import BrokerAccount, BrokerPosition
from tests.unit.test_brain_agents import make_ctx, path


def op(agent, subject, score, confidence=0.7, horizon=5, evidence=(), invalidation=None, meta=None):
    return Opinion(
        agent,
        "1.0.0",
        subject,
        stance_of(score),
        score,
        confidence,
        horizon,
        f"{agent} on {subject}",
        list(evidence),
        invalidation=invalidation,
        meta=meta or {},
        data_quality=DataState.FRESH,
    )


# ---------------------------------------------------------------------------------------------- detectors
def indicators(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols, index=[f"S{i}" for i in range(len(next(iter(cols.values()))))])


def test_breakout_needs_volume_and_the_trend_on_its_side():
    ind = indicators(
        breakout_20=[True, True, False, False],
        breakdown_20=[False, False, True, False],
        volume_ratio_1d=[2.0, 1.1, 2.0, 3.0],
        px_vs_sma50=[0.05, 0.05, -0.04, 0.02],
    )
    found = {o.subject: o for o in opp.breakout(ind)}
    assert set(found) == {"S0", "S2"}  # S1 lacks volume, S3 did not break out
    assert found["S0"].direction == 1 and found["S2"].direction == -1


def test_abnormal_volume_waits_for_the_first_hour_of_trading():
    ind = indicators(rel_volume=[3.0, 0.2], volume_ratio_1d=[1.0, 1.0], ret_1d=[0.03, -0.01])
    assert opp.abnormal_volume(ind, market_open=True, session_fraction=0.05) == []  # the open is front-loaded
    later = opp.abnormal_volume(ind, market_open=True, session_fraction=0.5)
    assert [o.subject for o in later] == ["S0"] and later[0].direction == 1
    closed = opp.abnormal_volume(
        indicators(volume_ratio_1d=[3.1, 1.2], ret_1d=[-0.02, 0.0]), market_open=False
    )
    assert [(o.subject, o.direction) for o in closed] == [("S0", -1)]


def test_mean_reversion_momentum_shift_and_volatility_events():
    ind = indicators(
        z20=[-2.5, 0.1, 2.4, 0.0] + [0.0] * 16,
        rsi14=[22, 50, 83, 50] + [50] * 16,
        mom_accel=[0.0, 0.3, 0.0, 0.0, *np.linspace(-0.01, 0.01, 16)],
        ret_21d=[0.0, 0.1, 0.0, 0.0] + [0.0] * 16,
        macd_cross=[0] * 20,
        move_z=[0.0, 0.0, 0.0, -4.0] + [0.0] * 16,
        vol_ratio=[1.0] * 20,
    )
    mr = {o.subject: o.direction for o in opp.mean_reversion(ind)}
    assert mr == {"S0": 1, "S2": -1}
    shift = opp.momentum_shift(ind)
    assert [o.subject for o in shift] == ["S1"] and shift[0].direction == 1
    ev = opp.volatility_event(ind, market_open=True)
    assert [o.subject for o in ev] == ["S3"] and ev[0].direction == 0


def test_relative_value_finds_a_diverged_pair_in_the_same_sector():
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(0, 0.01, 200))
    a = 50 * np.exp(common + rng.normal(0, 0.002, 200))
    b = 80 * np.exp(common + rng.normal(0, 0.002, 200))
    a[-3:] *= 0.94  # A falls away from B
    idx = pd.bdate_range(end="2026-09-24", periods=200)
    close = pd.DataFrame(
        {"AAA": a, "BBB": b, "CCC": 40 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))}, index=idx
    )
    ind = pd.DataFrame({"adv_dollar": [1e9, 1e9, 1e9]}, index=close.columns)
    pairs = opp.relative_value(close, ind, {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"})
    assert len(pairs) == 1 and pairs[0].symbols == ["AAA", "BBB"] and pairs[0].direction == 1
    assert pairs[0].evidence["correlation"] > 0.8 and pairs[0].evidence["spread_z"] < -2


def test_sector_rotation_and_regime_change():
    names = [f"{s}{i}" for s in "ABCDE" for i in range(4)]
    sectors = {n: n[0] for n in names}
    rs1 = {"A": 0.08, "B": 0.03, "C": 0.0, "D": -0.02, "E": -0.06}  # 1-month leaders
    rs3 = {"A": -0.10, "B": 0.05, "C": 0.08, "D": 0.02, "E": 0.09}  # 3-month leaders
    ind = pd.DataFrame(
        {
            "rs_1m": [rs1[n[0]] + i * 0.001 for i, n in enumerate(names)],
            "rel_strength": [rs3[n[0]] for n in names],
        },
        index=names,
    )
    rot = {o.subject: o for o in opp.sector_rotation(ind, sectors, held=["E1"])}
    assert rot["SECTOR:A"].direction == 1 and len(rot["SECTOR:A"].symbols) == 2
    assert rot["SECTOR:E"].direction == -1 and rot["SECTOR:E"].symbols == ["E1"]
    change = opp.regime_change("bearish", "bullish", -1.0)
    assert change[0].subject == MARKET and change[0].direction == -1
    assert (
        opp.regime_change("bullish", "bullish", 2.0) == [] and opp.regime_change("bullish", None, 2.0) == []
    )


def test_focus_pass_finds_earnings_catalysts_and_unusual_options():
    events = {
        "SOON": {"days_to_next": 4, "typical_move": 0.05},
        "BEAT": {"days_to_next": 80, "days_since_last": 3, "last_abnormal": 0.09, "typical_move": 0.03},
    }
    options = {"FLOW": {"volume_oi": 1.6, "pc_volume": 0.3}}
    kinds = {
        (o.kind, o.subject, o.direction) for o in opp.scan_focus(["SOON", "BEAT", "FLOW"], options, events)
    }
    assert kinds == {("earnings", "SOON", 0), ("catalyst", "BEAT", 1), ("unusual_options", "FLOW", 1)}


def test_valuation_dislocation_needs_quality():
    f = pd.DataFrame(
        {
            "earnings_yield": [0.01] * 20 + [0.15, 0.15],
            "fcf_yield": [0.01] * 20 + [0.14, 0.14],
            "book_to_market": [0.2] * 20 + [0.9, 0.9],
            "gross_profitability": [0.2] * 20 + [0.4, 0.01],
            "roe": [0.1] * 20 + [0.2, -0.3],
        },
        index=[f"S{i}" for i in range(20)] + ["GOOD", "TRAP"],
    )
    f.iloc[:20] += np.random.default_rng(1).normal(0, 0.005, (20, 5))
    found = {o.subject for o in opp.valuation_dislocation(f, list(f.index))}
    assert found == {"GOOD"}


# ---------------------------------------------------------------------------------------------- trace
async def test_trace_records_why_an_opportunity_stopped():
    ctx = make_ctx({"SPY": path(0.0003, 0.01, 1), "IN": path(0.001, 0.01, 2), "OUT": path(0.001, 0.01, 3), "BAD": path(0, 0.01, 4)},
                   focus=["IN", "BAD"])  # fmt: skip
    ctx.data_states["BAD"] = DataState.STALE
    ops = [
        opp.Opportunity("breakout", "IN", ["IN"], 1, 0.9, "IN broke out"),
        opp.Opportunity("breakout", "OUT", ["OUT"], 1, 0.8, "OUT broke out"),
        opp.Opportunity("mean_reversion", "BAD", ["BAD"], 1, 0.7, "BAD oversold"),
        opp.Opportunity("regime_change", MARKET, [], -1, 0.8, "regime changed"),
    ]
    ctx.working.add(op("technical", "IN", 0.5))
    ctx.working.add(op("momentum", "IN", 0.4))
    c = build_consensus("IN", [op("technical", "IN", 0.5), op("momentum", "IN", 0.4)])
    opp.trace(ops, focus=ctx.focus, states=ctx.data_states, market_open=True, opinions=ctx.working.opinions,
              consensus={"IN": c}, debates={}, proposals={})  # fmt: skip
    status = {o.subject: o.status for o in ops}
    assert status == {"IN": "no_action", "OUT": "not_analysed", "BAD": "rejected_data", MARKET: "context"}
    stages = [s["stage"] for s in ops[0].stages]
    assert stages[:4] == ["detection", "data_validation", "relevant_agents", "consensus"]
    heard = ops[0].stages[2]
    assert set(heard["heard"]) == {"technical", "momentum"} and heard["missing"] == ["volatility"]


# ---------------------------------------------------------------------------------------------- debate
def debate_ctx(regime="bullish", **ind):
    ctx = make_ctx({"SPY": path(0.0003, 0.01, 5), "AAA": path(0.001, 0.01, 6)})
    for k, v in ind.items():
        ctx.indicators.loc["AAA", k] = v
    ctx.working.post("regime", regime)
    return ctx


def test_devils_advocate_challenges_a_single_voice_and_weakens_one_idea():
    ctx = debate_ctx(rsi14=55.0, px_vs_sma50=0.03)
    ev = [Evidence("trend", 0.1, "price above its averages", 1, 0.8)]
    lone = [op("technical", "AAA", 0.6, 0.9, evidence=ev), op("momentum", "AAA", 0.05, 0.5)]
    for o in lone:
        ctx.working.add(o)
    c = {"AAA": build_consensus("AAA", lone)}
    before = c["AAA"].confidence
    d = review(ctx, c)["AAA"]
    assert d.verdict == "challenged" and "single_voice" in {o.code for o in d.objections}
    assert c["AAA"].confidence == pytest.approx(before * 0.7) and d.bull[0].text == "price above its averages"

    ctx = debate_ctx(rsi14=55.0, px_vs_sma50=0.03)
    trend = [op("technical", "AAA", 0.6, 0.8), op("momentum", "AAA", 0.5, 0.8)]
    for o in trend:
        ctx.working.add(o)
    prices = {"technical": "prices", "momentum": "prices"}
    c = build_consensus("AAA", trend, sources=prices)
    assert c.independent == 1 and set(c.sources) == {"prices"}
    d = review(ctx, {"AAA": c})["AAA"]
    assert d.verdict == "weakened" and {o.code for o in d.objections} >= {"one_idea", "unproven"}
    one_idea = next(o for o in d.objections if o.code == "one_idea")
    assert "prices" in one_idea.text and one_idea.haircut == 1.0  # already counted once in the consensus


def test_devils_advocate_objections_for_regime_events_extension_and_data():
    ctx = debate_ctx(regime="risk_off", rsi14=81.0, px_vs_sma50=0.2)
    ctx.working.post("event_risk", {"AAA": {"days_to_earnings": 3, "typical_move": 0.06}})
    ops = [
        op("technical", "AAA", 0.8, 0.9),
        op("factor", "AAA", 0.7, 0.9, horizon=21),
        op("valuation", "AAA", -0.6, 0.75, horizon=63),
    ]
    for o in ops:
        ctx.working.add(o)
    c = build_consensus("AAA", ops)
    c = replace(c, data_quality=DataState.STALE)
    d = review(ctx, {"AAA": c})["AAA"]
    codes = {o.code for o in d.objections}
    assert {"against_regime", "extended", "event_in_horizon", "data", "strong_opposition"} <= codes
    assert d.challenged  # stale data during the session is a high-severity objection
    assert any("earnings in 3 days" in a.text for a in d.bear)
    unknown = build_consensus("AAA", [op("technical", "AAA", 0.6), op("momentum", "AAA", -0.6)])
    assert review(ctx, {"AAA": unknown})["AAA"].verdict == "no view to challenge"


# ---------------------------------------------------------------------------------------------- agents
async def test_situational_awareness_postures():
    ctx = make_ctx({"SPY": path(0.0003, 0.01, 7), "AAA": path(0.001, 0.01, 8)})
    ctx.working.post("regime", "bullish")
    agent = SituationalAwarenessAgent()
    assert (await agent.analyze(ctx, [MARKET]))[0].meta["posture"] == "normal"
    ctx.vix = 25.0
    assert (await agent.analyze(ctx, [MARKET]))[0].meta["posture"] == "cautious"
    ctx.kill_switch = True
    o = (await agent.analyze(ctx, [MARKET]))[0]
    assert o.meta["posture"] == "defensive" and o.meta["risk_scale"] == 0.0 and o.stance is Stance.ABSTAIN
    assert ctx.working.facts["situation"]["posture"] == "defensive"


async def test_research_agent_answers_general_and_opportunity_specific_questions():
    ctx = make_ctx({"SPY": path(0.0003, 0.01, 9), "AAA": path(0.002, 0.01, 10)})
    ctx.opportunities = [opp.Opportunity("breakout", "AAA", ["AAA"], 1, 0.9, "AAA broke out")]
    ctx.events = {"AAA": {"days_to_next": 5, "next_source": "estimated"}}
    o = (await ResearchAgent().analyze(ctx, ["AAA"]))[0]
    questions = [f["question"] for f in o.meta["findings"]]
    assert "Did the breakout hold on volume?" in questions and "Is there an event ahead?" in questions
    assert o.stance is Stance.ABSTAIN and o.meta["kinds"] == ["breakout"]
    event = next(f for f in o.meta["findings"] if f["question"] == "Is there an event ahead?")
    assert event["direction"] == -1


# ---------------------------------------------------------------------------------------------- decisions
def account(equity=100_000.0, cash=None, day_pl=0.0):
    return BrokerAccount(
        account_number="…0001", status="ACTIVE", currency="USD", equity=equity, last_equity=equity / (1 + day_pl),
        cash=equity if cash is None else cash, buying_power=equity, long_market_value=0.0, short_market_value=0.0,
        portfolio_value=equity, trading_blocked=False, account_blocked=False, trade_suspended_by_user=False,
        pattern_day_trader=False, daytrade_count=0, multiplier=1.0,
    )  # fmt: skip


def position(symbol, qty, price):
    return BrokerPosition(
        symbol=symbol, qty=qty, qty_available=qty, side="long", avg_entry_price=price, current_price=price,
        market_value=qty * price, cost_basis=qty * price, unrealized_pl=0.0, unrealized_plpc=0.0,
        unrealized_intraday_pl=0.0, lastday_price=price,
    )  # fmt: skip


def planning_ctx(held=None):
    ctx = make_ctx(
        {
            "SPY": path(0.0003, 0.008, 11),
            "NEW": path(0.0008, 0.012, 12),
            "TWIN": path(0.0008, 0.012, 12),
            "HOLD": path(0.0005, 0.01, 13),
        }
    )
    ctx.portfolio.available = True
    ctx.portfolio.account = account()
    for s, (qty, px) in (held or {}).items():
        ctx.portfolio.positions[s] = position(s, qty, px)
    ctx.working.post(
        "portfolio_constraints",
        {"spendable_cash": 90_000.0, "free_slots": 5, "sector_weights": {}, "beta": 0.5},
    )
    ctx.working.post("regime", "bullish")
    return ctx  # fmt: skip


def bullish(subject):
    return build_consensus(
        subject, [op("technical", subject, 0.6, 0.8), op("factor", subject, 0.6, 0.8, horizon=21)]
    )


def test_portfolio_fit_rejects_the_same_bet_twice():
    ctx = planning_ctx(held={"TWIN": (100, float(path(0.0008, 0.012, 12)[-1]))})
    fit = portfolio_fit(ctx, "NEW", 0.05)  # NEW and TWIN follow the same generated path
    assert not fit["ok"] and fit["max_corr_with"] == "TWIN" and "same bet" in fit["notes"][0]
    proposals = {
        p.subject: p
        for p in plan(
            ctx, {"NEW": bullish("NEW")}, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15
        )
    }
    assert proposals["NEW"].action is Action.WATCH and "poor portfolio fit" in proposals["NEW"].reasons[0]


def test_postures_change_what_the_planner_proposes():
    ctx = planning_ctx(held={"HOLD": (100, float(path(0.0005, 0.01, 13)[-1]))})
    consensus = {
        "NEW": bullish("NEW"),
        "HOLD": build_consensus("HOLD", [op("technical", "HOLD", -0.05), op("factor", "HOLD", 0.05)]),
    }
    normal = {
        p.subject: p
        for p in plan(ctx, consensus, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15)
    }
    assert normal["NEW"].action is Action.BUY and normal["HOLD"].action is Action.HOLD

    ctx.working.post("situation", {"posture": "cautious", "risk_scale": 0.6, "reasons": ["VIX 25"]})
    cautious = {
        p.subject: p
        for p in plan(ctx, consensus, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15)
    }
    assert cautious["NEW"].target_weight == pytest.approx(normal["NEW"].target_weight * 0.6, rel=1e-3)
    assert any("size scaled by 60%" in r for r in cautious["NEW"].reasons)

    ctx.working.post(
        "situation", {"posture": "defensive", "risk_scale": 0.0, "reasons": ["the kill switch is on"]}
    )
    defensive = {
        p.subject: p
        for p in plan(ctx, consensus, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15)
    }
    assert defensive["NEW"].action is Action.WATCH and "defensive posture" in defensive["NEW"].reasons[0]
    assert defensive["HOLD"].action is Action.DE_RISK and defensive["HOLD"].quantity == 33  # trims a third


def test_challenged_view_blocks_new_risk_and_overweight_winners_are_rebalanced():
    ctx = planning_ctx(held={"HOLD": (400, float(path(0.0005, 0.01, 13)[-1]))})
    consensus = {"NEW": bullish("NEW"), "HOLD": bullish("HOLD")}
    for o in (op("technical", "NEW", 0.6, 0.8), op("factor", "NEW", 0.6, 0.8, horizon=21)):
        ctx.working.add(o)
    debates = review(ctx, {"NEW": consensus["NEW"]})
    debates["NEW"].verdict = "challenged"
    debates["NEW"].objections[0].severity = "high"
    proposals = {
        p.subject: p
        for p in plan(
            ctx, consensus, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15, debates=debates
        )
    }
    assert proposals["NEW"].action is Action.WATCH and "devil's advocate" in proposals["NEW"].reasons[0]
    hold = proposals["HOLD"]
    weight = ctx.portfolio.weight("HOLD")
    assert hold.action is Action.REBALANCE and hold.target_weight < weight and hold.quantity > 0  # fmt: skip
