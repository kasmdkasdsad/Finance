"""The Brain's execution gates without I/O: which conditions halt new positions (exits still go), and which
decisions are handed to the trading service. The risk engine itself is not re-implemented here."""

from datetime import UTC, datetime

from quantpulse.brain.context import PortfolioState
from quantpulse.brain.decisions import Proposal
from quantpulse.brain.execution import DATA_BLOCKED, BrainExecutor, Gate, entry_halts, to_order
from quantpulse.brain.types import MARKET, Action, BrainMode, Opinion, Stance
from quantpulse.config import Settings
from quantpulse.services.trading import brain_slot
from tests.unit.test_brain_core import make_ctx
from tests.unit.test_trading_risk import account, position


def owned(ctx, acct, positions=()):
    state = PortfolioState(available=True, account=acct, positions={p.symbol: p for p in positions})
    ctx.account = ctx.portfolio = state
    ctx.mode = BrainMode.PAPER_EXECUTION
    return ctx


def codes(ctx) -> set[str]:
    return {h["code"] for h in entry_halts(ctx)}


def test_a_healthy_account_has_no_entry_halts():
    ctx = owned(make_ctx(), account(100_000, cash=75_000), [position("AAA", 250, 100.0)])
    assert entry_halts(ctx) == []


def test_each_condition_halts_new_positions():
    ctx = owned(make_ctx(), account(95_000, last_equity=100_000, cash=95_000))
    assert codes(ctx) == {"daily_loss"}  # −5% on the day, limit −4%

    ctx = owned(make_ctx(), account(100_000))
    veto = "only 20% of the universe has usable live quotes"
    dq = Opinion("data_quality", "1", MARKET, Stance.ABSTAIN, 0, 0, 1, "stale", veto=veto)
    ctx.working.opinions[MARKET].append(dq)
    halts = entry_halts(ctx)
    assert [h["code"] for h in halts] == ["data_quality"] and halts[0]["reason"].startswith(DATA_BLOCKED)

    ctx = owned(make_ctx(), account(100_000))
    ctx.working.post("system_vetoes", ["the data-quality check did not run: nothing is executable"])
    assert codes(ctx) == {"data_quality"}

    ctx = owned(make_ctx(), account(100_000))
    ctx.feed = {"clock_skew_s": 42.0}
    assert codes(ctx) == {"clock_skew"}

    ctx = owned(make_ctx(), account(100_000, blocked=True))
    assert codes(ctx) == {"inconsistent_state"}

    ctx = owned(make_ctx(), account(100_000, cash=100_000), [position("AAA", -10, 100.0)])
    assert "inconsistent_state" in codes(ctx)  # a short position although shorting is disabled

    # positions that do not add up to what the account reports
    ctx = owned(make_ctx(), account(100_000, cash=50_000, long_mv=50_000), [position("AAA", 100, 100.0)])
    assert codes(ctx) == {"inconsistent_state"}

    # exposure well above the limit, or a position far above its limit: not the Brain's doing
    ctx = owned(make_ctx(), account(100_000, cash=0, long_mv=100_000), [position("AAA", 1000, 100.0)])
    assert "unexpected_exposure" in codes(ctx)


def test_a_held_option_is_part_of_what_the_account_reports():
    """Alpaca's long market value includes long option contracts. Counting shares only, one QQQ call worth more
    than 1% of equity read as an inconsistent account and halted every new stock and option position."""
    from dataclasses import replace

    call = replace(
        position("QQQ261120C00757000", 1, 18.53, 18.53), market_value=1_853.0, asset_class="us_option"
    )
    short_leg = replace(
        position("QQQ261120C00800000", -1, 9.0, 9.0), market_value=-900.0, asset_class="us_option"
    )
    acct = account(100_000, cash=73_147, long_mv=26_853)  # 250 shares at $100, and the call
    ctx = owned(make_ctx(), acct, [position("AAA", 250, 100.0)])
    ctx.account = ctx.portfolio = replace(
        ctx.account, option_positions={c.symbol: c for c in (call, short_leg)}
    )
    assert entry_halts(ctx) == []
    # a long option the account does not report is still a mismatch
    ctx.account = ctx.portfolio = replace(ctx.account, account=account(100_000, cash=75_000))
    assert codes(ctx) == {"inconsistent_state"}


def test_the_market_closed_is_not_a_data_halt():
    ctx = owned(make_ctx(), account(100_000))
    ctx.market_open = False
    closed = Opinion("data_quality", "1", MARKET, Stance.ABSTAIN, 0, 0, 1, "closed", veto="market closed")
    ctx.working.opinions[MARKET].append(closed)
    assert entry_halts(ctx) == []  # nothing is executable anyway; that is a blocker, not a data problem


def proposal(action, *, blocked=(), approved=True, protective=False, qty=10.0):
    return Proposal("AAA", action, 0.7, ["why"], quantity=qty, est_price=100.0, blocked_by=list(blocked),
                    risk_approved=approved, protective=protective)  # fmt: skip


def test_which_decisions_are_handed_to_the_trading_service():
    ok = Gate(send=True, entries=True)
    halted = Gate(send=True, entries=False, halts=[{"code": "daily_loss", "reason": "daily loss"}])
    sendable = BrainExecutor.sendable
    assert sendable(proposal(Action.BUY), ok) is None
    assert "entries halted" in sendable(proposal(Action.BUY), halted)
    assert "entries halted" in sendable(proposal(Action.INCREASE), halted)
    assert "blocked" in sendable(proposal(Action.BUY, blocked=["market data stale"]), ok)
    assert "risk preview" in sendable(proposal(Action.BUY, approved=False), ok)
    # exits and trims go even when entries are halted
    assert sendable(proposal(Action.REDUCE), halted) is None
    assert "blocked" in sendable(proposal(Action.REDUCE, blocked=["market data stale"]), halted)
    # a protective exit is always handed on: the risk engine decides (it prices on Alpaca's mark if needed)
    assert (
        sendable(proposal(Action.CLOSE, blocked=["stale"], approved=False, protective=True), halted) is None
    )


def test_orders_carry_what_the_trading_service_needs():
    ctx = owned(make_ctx(), account(100_000), [position("AAA", 10, 100.0)])
    o = to_order(proposal(Action.REDUCE, qty=10.0), ctx)
    assert o.side == "sell" and o.closes_position and o.action == "reduce" and not o.protective
    b = to_order(proposal(Action.BUY, qty=5.0), ctx)
    assert b.side == "buy" and not b.closes_position and b.qty == 5.0


def test_the_order_slot_floors_new_york_time():
    at = datetime(2026, 9, 25, 14, 44, tzinfo=UTC)  # 10:44 New York
    assert brain_slot(at, 30) == "brain-20260925T1030"
    assert brain_slot(at, 15) == "brain-20260925T1030"
    assert brain_slot(datetime(2026, 9, 25, 14, 45, tzinfo=UTC), 15) == "brain-20260925T1045"
    assert brain_slot(at, 5) == "brain-20260925T1040"  # the default cadence: one slot per 5 minutes


def test_the_default_cadence_is_faster_but_no_protected_cap_moved():
    s = Settings(_env_file=None)
    assert (s.brain_cycle_minutes, s.brain_focus_candidates, s.brain_max_opportunities) == (5, 16, 10)
    # more cycles, same guard rails: per-cycle caps, the no-reversal cooldown and the stop-loss are unchanged
    assert s.brain_max_new_positions_per_cycle == 2 and s.trading_max_cycle_turnover_pct == 0.6
    assert s.trading_cooldown_minutes == 120.0
