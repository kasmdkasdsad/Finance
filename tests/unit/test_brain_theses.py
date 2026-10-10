"""Position theses and portfolio-level fit without I/O: when a thesis is broken, weakening or intact, what the
planner does about it (exit, replace), and how a new name would change the portfolio's risk."""

from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.brain.consensus import build_consensus
from quantpulse.brain.decisions import plan, portfolio_fit, portfolio_risk
from quantpulse.brain.theses import check, sessions_between, trail, trail_distance, trailing, view
from quantpulse.brain.types import Action
from quantpulse.db.models import BrainThesisRow
from tests.unit.test_brain_agents import make_ctx, path
from tests.unit.test_brain_research import bullish, op, planning_ctx

NOW = datetime(2026, 9, 25, 18, 0, tzinfo=UTC)


def thesis(**kw) -> BrainThesisRow:
    base = dict(symbol="HOLD", status="open", origin="brain", opened_at=NOW - timedelta(days=3), entry_price=100.0,
                entry_qty=10, thesis="t", stop_price=92.0, horizon_days=5, supporting=["technical", "factor"],
                opposing=[], qty=10, avg_price=100.0, last_price=103.0, return_pct=0.03, benchmark_return=0.01)  # fmt: skip
    base.update(kw)
    return BrainThesisRow(**base)


def test_sessions_held_count_trading_days_only():
    assert sessions_between(datetime(2026, 9, 25, 15, tzinfo=UTC), datetime(2026, 9, 28, 15, tzinfo=UTC)) == 1
    assert sessions_between(NOW, NOW) == 0


def test_a_thesis_is_intact_while_its_evidence_holds():
    ctx = planning_ctx()
    got = check(thesis(), ctx, bullish("HOLD"), 0.3, NOW)
    assert (
        got["status"] == "intact" and got["reasons"] == [] and got["relative_return"] == pytest.approx(0.02)
    )


def test_a_thesis_breaks_below_its_stop():
    got = check(thesis(last_price=91.0), planning_ctx(), bullish("HOLD"), 0.3, NOW)
    assert got["status"] == "broken" and "below its stop" in got["reasons"][0]


def test_a_thesis_breaks_when_its_supporters_turn_against_it():
    ctx = planning_ctx()
    ctx.working.add(op("technical", "HOLD", -0.6))
    ctx.working.add(op("factor", "HOLD", -0.5))
    got = check(thesis(), ctx, None, 0.3, NOW)
    assert (
        got["status"] == "broken" and "2 of the 2 agents that supported it now oppose it" in got["reasons"][0]
    )
    ctx = planning_ctx()
    ctx.working.add(op("technical", "HOLD", -0.6))  # one of two: weakening at most, not broken
    assert check(thesis(), ctx, bullish("HOLD"), 0.3, NOW)["status"] == "intact"


def test_a_thesis_breaks_when_the_consensus_turns_bearish():
    bearish = build_consensus("HOLD", [op("technical", "HOLD", -0.6, 0.8), op("factor", "HOLD", -0.6, 0.8)])
    got = check(thesis(), planning_ctx(), bearish, 0.3, NOW)
    assert got["status"] == "broken" and "turned bearish" in got["reasons"][0]


def test_a_thesis_that_has_not_worked_in_twice_its_horizon_is_broken():
    old = thesis(opened_at=NOW - timedelta(days=21), return_pct=0.01, benchmark_return=0.05)
    got = check(old, planning_ctx(), bullish("HOLD"), 0.3, NOW)
    assert got["status"] == "broken" and "twice its 5-session horizon" in got["reasons"][0]
    patient = thesis(
        opened_at=NOW - timedelta(days=21), horizon_days=10, return_pct=0.01, benchmark_return=0.05
    )
    got = check(patient, planning_ctx(), bullish("HOLD"), 0.3, NOW)
    assert got["status"] == "weakening"
    assert any("past its 10-session horizon" in r for r in got["reasons"])
    assert any("behind the benchmark" in r for r in got["reasons"])


def test_a_faded_consensus_weakens_a_thesis():
    neutral = build_consensus("HOLD", [op("technical", "HOLD", 0.05), op("factor", "HOLD", -0.05)])
    got = check(thesis(), planning_ctx(), neutral, 0.3, NOW)
    assert got["status"] == "weakening"


def test_a_winners_stop_trails_three_daily_moves_between_five_percent_and_the_initial_stop():
    ctx = planning_ctx()
    cap = ctx.limits.max_position_loss_pct
    for annual, expected in ((0.30, 3 * 0.30 / 252**0.5), (0.10, 0.05), (0.90, cap)):
        ctx.indicators.loc["HOLD", "risk_vol"] = annual
        assert trail_distance(ctx, "HOLD") == pytest.approx(expected, abs=1e-4)
    assert trail_distance(ctx, "NOPE") == cap  # no volatility known: the initial stop's distance


def test_the_stop_starts_trailing_at_a_ten_percent_gain_and_never_moves_down():
    row = thesis(last_price=105.0, history=[])
    trail(row, 0.06, NOW)
    assert row.peak_price == 105.0 and row.stop_price == 92.0 and not trailing(row)  # up 5%: the initial stop
    row.last_price = 112.0
    trail(row, 0.06, NOW)
    assert trailing(row) and row.stop_price == pytest.approx(112 * 0.94)  # up 12%: it can no longer lose
    assert [e["event"] for e in row.history] == ["trailing"]
    row.last_price = 120.0
    trail(row, 0.06, NOW)
    assert row.peak_price == 120.0 and row.stop_price == pytest.approx(112.8)
    for price in (116.0, 113.0):  # a pull-back: the high and the stop stay where they were
        row.last_price = price
        trail(row, 0.06, NOW)
        assert row.peak_price == 120.0 and row.stop_price == pytest.approx(112.8)
    trail(row, 0.02, NOW)  # nor does a narrower or wider distance ever lower it
    assert row.stop_price == pytest.approx(120 * 0.98)
    trail(row, 0.08, NOW)
    assert row.stop_price == pytest.approx(117.6) and [e["event"] for e in row.history] == ["trailing"]
    v = view(row)
    assert v["trailing"] and v["peak_price"] == 120.0 and v["trails_from"] == pytest.approx(110.0)


def test_falling_through_a_trailing_stop_breaks_the_thesis_and_says_what_it_keeps():
    row = thesis(last_price=120.0, history=[])
    trail(row, 0.06, NOW)
    row.last_price = 112.0
    trail(row, 0.06, NOW)
    got = check(row, planning_ctx(), bullish("HOLD"), 0.3, NOW)
    assert got["status"] == "broken"
    assert got["reasons"][0] == (
        "below its trailing stop ($112.00 ≤ $112.80, 6.0% under its high of $120.00): it keeps +12.8% on the "
        "$100.00 paid"
    )  # even with a bullish consensus: the stop is a protective exit


HELD = {"HOLD": (100, float(path(0.0005, 0.01, 13)[-1]))}
NEUTRAL = build_consensus("HOLD", [op("technical", "HOLD", 0.05), op("factor", "HOLD", -0.05)])


def planned(ctx, consensus):
    return {
        p.subject: p
        for p in plan(ctx, consensus, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15)
    }


def test_a_broken_thesis_is_exited_without_waiting_for_a_new_signal():
    ctx = planning_ctx(held=HELD)
    ctx.working.post("thesis_checks", {"HOLD": {"status": "broken", "reasons": ["its supporters turned"]}})
    p = planned(ctx, {"HOLD": NEUTRAL})["HOLD"]
    assert p.action is Action.CLOSE and p.protective and p.quantity == 100
    assert p.reasons == ["thesis broken: its supporters turned"]


def test_a_fading_holding_is_replaced_when_no_slot_is_free():
    ctx = planning_ctx(held=HELD)
    ctx.working.post(
        "portfolio_constraints", {"spendable_cash": 0.0, "free_slots": 0, "sector_weights": {}, "beta": 0.5}
    )
    consensus = {"HOLD": NEUTRAL, "NEW": bullish("NEW")}
    ctx.working.post("thesis_checks", {"HOLD": {"status": "intact", "reasons": []}})
    got = planned(ctx, consensus)
    assert got["HOLD"].action is Action.HOLD and got["NEW"].action is Action.WATCH  # an intact thesis stays

    ctx.working.post("thesis_checks", {"HOLD": {"status": "weakening", "reasons": ["past its horizon"]}})
    got = planned(ctx, consensus)
    hold, new = got["HOLD"], got["NEW"]
    assert hold.action is Action.CLOSE and not hold.protective and "replace with NEW" in hold.reasons[0]
    assert "past its horizon" in hold.reasons
    assert new.action is Action.BUY and new.quantity > 0  # paid for by the sale (the risk engine re-checks)


def test_portfolio_risk_before_and_after_a_new_name():
    ctx = make_ctx(
        {
            "SPY": path(0.0003, 0.008, 11),
            "CALM": path(0.0004, 0.008, 21),
            "STEADY": path(0.0004, 0.009, 22),
            "WILD": path(0.0010, 0.060, 23),
        }
    )
    from tests.unit.test_brain_research import account, position

    ctx.portfolio.account = account()
    for sym in ("CALM", "STEADY"):
        ctx.portfolio.positions[sym] = position(sym, 200, float(ctx.close[sym].iloc[-1]))
    ctx.working.post("portfolio_constraints", {"sector_weights": {}, "beta": 0.5})
    calm = portfolio_risk(ctx, "STEADY", 0.0)
    wild = portfolio_risk(ctx, "WILD", 0.10)
    assert 0 < wild["vol_before"] < wild["vol_after"] and 0 < wild["risk_share"] <= 1
    assert wild["hhi_after"] < wild["hhi_before"]  # a third name spreads the invested weight
    assert calm["vol_after"] == pytest.approx(calm["vol_before"])  # adding nothing changes nothing
    fit = portfolio_fit(ctx, "WILD", 0.10)
    assert not fit["ok"] and any("of the portfolio's risk" in n for n in fit["notes"])
    assert portfolio_fit(ctx, "WILD", 0.005)["ok"]  # a small enough position is fine
