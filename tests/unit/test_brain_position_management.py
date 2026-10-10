"""Position management without I/O: what the planner does with holdings beyond the consensus — take a
calibrated profit (once), halve before an overnight earnings release (once), trim the same bet held twice, a
concentrated sector and a too-volatile book — and the improvement proposals about execution and turnover.

Each trim converges (it stops once the book is back inside), only touches holdings the plan would otherwise
HOLD, and nothing here loosens a limit."""

from datetime import UTC, datetime, timedelta

from quantpulse.brain.consensus import build_consensus
from quantpulse.brain.decisions import DERISK_EVERY, PORTFOLIO_VOL_CAP, Proposal, book_volatility, plan
from quantpulse.brain.improvement import ImprovementEngine, withheld
from quantpulse.brain.store import BrainStore
from quantpulse.brain.types import Action
from quantpulse.db.models import BrainExecutionRow, BrainThesisRow
from tests.unit.test_brain_agents import make_ctx, path
from tests.unit.test_brain_research import account, bullish, op, planning_ctx, position

PRICE = float(path(0.0005, 0.01, 13)[-1])
TWIN_PRICE = float(path(0.0008, 0.012, 12)[-1])
NEUTRAL = {s: build_consensus(s, [op("technical", s, 0.05), op("factor", s, -0.05)]) for s in ("HOLD", "NEW", "TWIN", "A", "B")}  # fmt: skip


def planned(ctx, consensus=None):
    consensus = consensus or {s: NEUTRAL[s] for s in ctx.portfolio.positions}
    return {
        p.subject: p
        for p in plan(ctx, consensus, min_confidence=0.3, max_new=2, vol_budget=0.02, vol_floor=0.15)
    }


# --------------------------------------------------------------------------- take profit
def test_a_calibrated_target_takes_half_the_profit_once():
    ctx = planning_ctx(held={"HOLD": (100, PRICE)})
    ctx.working.post("theses", {"HOLD": {"target_price": round(PRICE * 0.98, 4), "entry_qty": 100,
                                         "expected_return": 0.04}})  # fmt: skip
    p = planned(ctx)["HOLD"]
    assert p.action is Action.REDUCE and p.quantity == 50
    assert p.reasons[0].startswith("take profit:") and "calibrated expected return +4.0%" in p.reasons[0]
    # already taken (70 of the 100 bought are left): held under the thesis, not cut again
    ctx = planning_ctx(held={"HOLD": (70, PRICE)})
    ctx.working.post("theses", {"HOLD": {"target_price": round(PRICE * 0.98, 4), "entry_qty": 100}})
    assert planned(ctx)["HOLD"].action is Action.HOLD


def test_no_target_is_ever_invented():
    ctx = planning_ctx(held={"HOLD": (100, PRICE)})
    ctx.working.post("theses", {"HOLD": {"target_price": None, "entry_qty": 100}})  # uncalibrated
    assert planned(ctx)["HOLD"].action is Action.HOLD
    below = planning_ctx(held={"HOLD": (100, PRICE)})
    below.working.post("theses", {"HOLD": {"target_price": round(PRICE * 1.05, 4), "entry_qty": 100}})
    assert planned(below)["HOLD"].action is Action.HOLD


# --------------------------------------------------------------------------- the no-trade band
def test_a_bullish_holding_is_topped_up_only_when_well_short_of_its_target():
    small = planning_ctx(held={"HOLD": (1, PRICE)})
    tw = planned(small, {"HOLD": bullish("HOLD")})["HOLD"].target_weight
    assert planned(small, {"HOLD": bullish("HOLD")})["HOLD"].action is Action.INCREASE
    equity = small.portfolio.equity

    def at(share: float) -> Action:
        ctx = planning_ctx(held={"HOLD": (round(share * tw * equity / PRICE), PRICE)})
        return planned(ctx, {"HOLD": bullish("HOLD")})["HOLD"].action

    assert at(0.5) is Action.INCREASE  # half its target: topped up
    # near its target (a few shares short after a price move): left alone, not a share bought every cycle
    assert at(0.85) is Action.HOLD and at(0.95) is Action.HOLD


# --------------------------------------------------------------------------- the defensive trim's pace
def defensive(held, recently=()):
    ctx = planning_ctx(held=held)
    ctx.working.post("situation", {"posture": "defensive", "risk_scale": 0.0, "reasons": ["risk-off regime"]})
    ctx.working.post("recently_derisked", list(recently))
    return ctx


def test_the_defensive_posture_trims_a_third_at_most_every_30_minutes():
    p = planned(defensive({"HOLD": (90, PRICE)}))["HOLD"]
    assert p.action is Action.DE_RISK and p.quantity == 30
    # trimmed in the last 30 minutes: held, so 5-minute cycles cut a third per half hour, not per cycle
    p = planned(defensive({"HOLD": (60, PRICE)}, recently=["HOLD"]))["HOLD"]
    assert p.action is Action.HOLD and "the next third waits" in p.reasons[0]
    assert timedelta(minutes=30) == DERISK_EVERY


async def test_recent_trims_are_read_back_from_the_decision_record(database):
    store = BrainStore(database)
    now = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
    trim = Proposal(
        subject="HOLD", action=Action.DE_RISK, confidence=1.0, reasons=["t"], quantity=30.0, est_price=PRICE
    )
    hold = Proposal(subject="KEEP", action=Action.HOLD, confidence=1.0, reasons=["t"])
    old = Proposal(
        subject="OLD", action=Action.DE_RISK, confidence=1.0, reasons=["t"], quantity=10.0, est_price=PRICE
    )
    for at, proposals in ((now - timedelta(minutes=40), [old]), (now - timedelta(minutes=10), [trim, hold])):
        cycle = await store.start_cycle(
            kind="full", trigger="t", session="market_open", mode="dry_run", now=at
        )
        await store.save_decisions(cycle, proposals, {}, "dry_run", at)
    assert await store.recent_subjects("de_risk", now - DERISK_EVERY) == ["HOLD"]


# --------------------------------------------------------------------------- the overnight
def test_the_last_half_hour_halves_a_holding_before_an_earnings_release_once():
    ctx = planning_ctx(held={"HOLD": (100, PRICE)})
    ctx.working.post("event_risk", {"HOLD": {"days_to_earnings": 0}})
    assert planned(ctx)["HOLD"].action is Action.HOLD  # earlier in the day: not yet
    ctx.working.post("near_close", {"minutes_to_close": 20.0, "derisked": []})
    p = planned(ctx)["HOLD"]
    assert p.action is Action.DE_RISK and p.quantity == 50 and p.reasons[0].startswith("overnight:")
    ctx.working.post("near_close", {"minutes_to_close": 10.0, "derisked": ["HOLD"]})
    assert planned(ctx)["HOLD"].action is Action.HOLD  # already halved today
    later = planning_ctx(held={"HOLD": (100, PRICE)})
    later.working.post("near_close", {"minutes_to_close": 20.0, "derisked": []})
    later.working.post("event_risk", {"HOLD": {"days_to_earnings": 12}})
    assert planned(later)["HOLD"].action is Action.HOLD


# --------------------------------------------------------------------------- the book as a whole
def test_the_same_bet_held_twice_is_trimmed_back_to_one_position_limit():
    qty = round(20_000 / TWIN_PRICE)  # 20% each of the same generated path: 40% in one bet (limit 30%)
    ctx = planning_ctx(held={"TWIN": (qty, TWIN_PRICE), "NEW": (qty, TWIN_PRICE)})
    got = planned(ctx)
    trimmed = [p for p in got.values() if p.action is Action.REDUCE]
    assert len(trimmed) == 1 and "the same bet as" in trimmed[0].reasons[0]
    excess = 2 * qty * TWIN_PRICE - 0.30 * 100_000
    assert abs(trimmed[0].quantity * TWIN_PRICE - excess) <= TWIN_PRICE  # the excess, in whole shares
    small = planning_ctx(held={"TWIN": (round(10_000 / TWIN_PRICE), TWIN_PRICE),
                               "NEW": (round(10_000 / TWIN_PRICE), TWIN_PRICE)})  # fmt: skip
    assert all(p.action is Action.HOLD for p in planned(small).values())  # 20% together: inside the limit


def test_a_concentrated_sector_gives_up_its_excess():
    ctx = planning_ctx(held={"HOLD": (300, PRICE)})
    ctx.sectors = {"HOLD": "Information Technology"}
    ctx.working.post("portfolio_constraints", {"spendable_cash": 0.0, "free_slots": 5, "beta": 0.5,
                                               "sector_weights": {"Information Technology": 0.55}})  # fmt: skip
    p = planned(ctx)["HOLD"]
    assert p.action is Action.REDUCE and p.reasons[0].startswith(
        "sector concentration: Information Technology"
    )
    assert abs(p.quantity * PRICE - 0.10 * 100_000) <= PRICE
    ctx.working.post("portfolio_constraints", {"spendable_cash": 0.0, "free_slots": 5, "beta": 0.5,
                                               "sector_weights": {"Information Technology": 0.40}})  # fmt: skip
    assert planned(ctx)["HOLD"].action is Action.HOLD


def test_a_too_volatile_book_trims_its_largest_risk_contributor():
    ctx = make_ctx({"SPY": path(0.0003, 0.008, 11), "A": path(0.0, 0.070, 31), "B": path(0.0, 0.055, 32)})
    ctx.portfolio.available = True
    ctx.portfolio.account = account()
    for sym in ("A", "B"):
        px = float(ctx.close[sym].iloc[-1])
        ctx.portfolio.positions[sym] = position(sym, round(28_000 / px), px)
    ctx.working.post("portfolio_constraints", {"spendable_cash": 0.0, "free_slots": 5, "sector_weights": {}})
    vol, shares = book_volatility(ctx)
    assert vol > PORTFOLIO_VOL_CAP and max(shares, key=shares.get) == "A"
    got = planned(ctx)
    assert got["A"].action is Action.DE_RISK and "portfolio volatility" in got["A"].reasons[0]
    assert got["A"].quantity == -(-ctx.portfolio.positions["A"].qty // 4) and got["B"].action is Action.HOLD


# --------------------------------------------------------------------------- improvement proposals
NOW = datetime(2026, 9, 25, 21, 0, tzinfo=UTC)


def execution(i: int, grade: str) -> BrainExecutionRow:
    return BrainExecutionRow(client_order_id=f"qp-brain-{i}-UPA-b", symbol="UPA", side="buy", action="buy",
                             reason="r", consensus={}, qty=10, order_type="market", status="filled", final=True,
                             filled_qty=10, grade=grade, cost_vs_quote_bps=25.0 if grade == "poor" else 1.0,
                             decided_at=NOW - timedelta(days=1), updated_at=NOW)  # fmt: skip


async def test_poor_execution_becomes_a_proposal_that_never_touches_a_limit(database):
    engine = ImprovementEngine(database, 30, 0.45)
    async with database.session() as s:
        s.add_all([execution(i, "poor" if i % 2 else "good") for i in range(9)])
    assert not [f for f in await engine.review(NOW) if f["kind"] == "execution"]  # 9 fills: too few
    async with database.session() as s:
        s.add_all([execution(i, "poor") for i in range(9, 14)])
    [found] = [f for f in await engine.review(NOW) if f["kind"] == "execution"]
    assert found["evidence"]["graded_fills_30d"] == 14 and found["evidence"]["poor"] == 9
    assert withheld(found) is None and "no risk limit changes" in found["proposal"]["change"]


async def test_quick_round_trips_become_a_turnover_proposal(database):
    engine = ImprovementEngine(database, 30, 0.45)
    async with database.session() as s:
        for i in range(6):
            opened = NOW - timedelta(days=3)
            s.add(BrainThesisRow(symbol=f"S{i}", status="closed", origin="brain", opened_at=opened,
                                 closed_at=opened + timedelta(hours=20), thesis="t", supporting=[], opposing=[],
                                 exit_reason="bearish consensus", qty=0, entry_price=50.0, entry_qty=10, avg_price=50.0, updated_at=NOW))  # fmt: skip
    [found] = [f for f in await engine.review(NOW) if f["target"] == "turnover"]
    assert found["evidence"]["closed_within_2_sessions"] == 6 and withheld(found) is None
