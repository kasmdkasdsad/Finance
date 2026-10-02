"""The intraday agent: today's tape as a one-day view — heavy-volume moves continue, quiet ones partly
reverse, the price against VWAP and in today's range says who is in control — and silence when there is no
session, no fresh quote or only the noisy first half hour."""

from datetime import UTC, datetime

import pytest

from quantpulse.brain.agents import default_agents
from quantpulse.brain.agents.intraday import IntradayAgent
from quantpulse.brain.types import DataState, Stance
from quantpulse.services.trading_data import LiveQuote
from tests.unit.test_brain_agents import make_ctx, path, run

NOON = datetime(2026, 9, 25, 16, 0, tzinfo=UTC)  # 12:00 New York: well into the session


def quote(symbol: str, price: float, vwap: float, low: float, high: float) -> LiveQuote:
    return LiveQuote(symbol=symbol, price=price, bid=price - 0.01, ask=price + 0.01, vwap=vwap, volume=1e6,
                     day_high=high, day_low=low, day_open=vwap, timestamp=NOON, provider="test",
                     age_seconds=2.0)  # fmt: skip


def tape(**symbols: tuple[float, float | None, LiveQuote]):
    """A context whose live row says, per symbol: today's move in daily sigmas, the session-adjusted volume
    (× normal − 1) and the live quote."""
    paths = {"SPY": path(0.0004, 0.008, 1), **{s: path(0.0005, 0.012, i + 2) for i, s in enumerate(symbols)}}
    ctx = make_ctx(paths)
    ctx.as_of = NOON
    for s, (move_z, rel, q) in symbols.items():
        ctx.indicators.loc[s, "move_z"] = move_z
        ctx.indicators.loc[s, "rel_volume"] = rel
        ctx.quotes[s] = q
    return ctx


async def test_a_heavy_volume_move_above_vwap_leans_the_same_way_for_tomorrow():
    ctx = tape(UP=(2.4, 1.6, quote("UP", 105.0, 103.0, 100.0, 105.5)))  # +2.4σ on 2.6× volume, near the high
    o = (await run(IntradayAgent(), ctx))["UP"]
    assert o.stance is Stance.BULLISH and o.horizon_days == 1
    assert "tend to continue" in o.thesis and "above VWAP" in o.thesis
    assert o.invalidation and "below today's VWAP" in o.invalidation
    assert 0 < o.confidence <= 0.45  # modest by design, and unproven until its one-day calls are graded


async def test_a_quiet_selloff_leans_towards_a_partial_rebound():
    ctx = tape(DOWN=(-2.0, -0.4, quote("DOWN", 96.0, 96.5, 95.5, 99.0)))  # −2σ on 0.6× normal volume
    o = (await run(IntradayAgent(), ctx))["DOWN"]
    assert o.score > 0 and "partly reverse" in o.thesis
    heavy = tape(DOWN=(-2.0, 1.5, quote("DOWN", 95.6, 97.0, 95.5, 99.0)))  # the same move on heavy volume
    assert (await run(IntradayAgent(), heavy))["DOWN"].stance is Stance.BEARISH


async def test_it_says_nothing_without_a_session_a_fresh_quote_or_before_the_first_half_hour():
    ctx = tape(UP=(2.4, 1.6, quote("UP", 105.0, 103.0, 100.0, 105.5)))
    agent = IntradayAgent()
    assert agent.unavailable(ctx) is None
    ctx.as_of = datetime(2026, 9, 25, 13, 45, tzinfo=UTC)  # 09:45 New York
    assert "first 30 minutes" in agent.unavailable(ctx)
    ctx.as_of = datetime(2026, 11, 27, 14, 50, tzinfo=UTC)  # 09:50 on a half day (13:00 close): too early
    assert "first 30 minutes" in agent.unavailable(ctx)
    ctx.as_of = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)  # 10:00 on the half day
    assert agent.unavailable(ctx) is None
    ctx.as_of, ctx.market_open = NOON, False
    assert "market is closed" in agent.unavailable(ctx)
    ctx.market_open = True
    ctx.data_states["UP"] = DataState.STALE
    o = (await run(agent, ctx))["UP"]
    assert o.stance is Stance.ABSTAIN and "no fresh live quote" in o.thesis


@pytest.mark.parametrize("agent", [a for a in default_agents() if a.spec.id == "intraday"])
def test_it_is_a_registered_prices_vote_with_a_one_day_horizon(agent):
    # the same information family as the other price agents: it can never count as an independent second
    # source for the consensus, so the two-source rule is untouched
    assert agent.spec.source == "prices" and agent.spec.horizon_days == 1
