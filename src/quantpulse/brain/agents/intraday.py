"""Intraday agent: what today's tape says about tomorrow — a one-day view, graded at the next close.

The other price agents read daily bars, which barely change during a session; this one reads the live
quote, so each cycle in the session can see something new. From the live quote and the shared indicator
table it measures:

* today's move since the previous close, in daily standard deviations (``move_z``);
* the session-adjusted volume: today's volume against what is normal by this time of day (``rel_volume``);
* the price against today's volume-weighted average price (VWAP);
* where the price sits in today's high–low range.

The rule it tests is the documented volume–return link: a large move on heavy volume tends to continue the
next day (information is being traded, Llorente et al. 2002; the high-volume return premium, Gervais,
Kaniel and Mingelgrin 2001), while a large move on quiet volume tends to partly reverse (short-term
reversal, Jegadeesh 1990). The price against VWAP and in today's range say who is in control now.

It is one more vote from the **prices** source (the same information family as the other price agents, so
it never counts as an independent second source), with deliberately modest confidence, and — like every
agent — unproven until its own one-day calls are graded. It says nothing before the first 30 minutes of
the session (the open is noise) or without a fresh live quote.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timedelta

from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.services.trading_data import session_fraction

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, DataState, Evidence, Opinion
from .base import Agent, symbols_only
from .common import agreement, opinion, sgn, squash

SETTLE = timedelta(minutes=30)  # the first half hour after the open is noise: no view before it
HEAVY, QUIET = 0.5, -0.2  # rel_volume ("× normal − 1"): ≥ 1.5× normal is heavy, ≤ 0.8× quiet
BIG_MOVE_Z = 1.0  # a move worth reading: at least one daily standard deviation


def minutes_traded(moment: datetime) -> timedelta:
    """Time since today's 09:30 New York open (the same on half days, unlike the share of the session)."""
    local = moment.astimezone(NEW_YORK)
    return local - datetime.combine(local.date(), time(9, 30), NEW_YORK)


class IntradayAgent(Agent):
    spec = AgentSpec(
        id="intraday",
        source="prices",
        failure="no intraday vote; the daily price agents still speak for prices",
        name="Intraday tape",
        description="Today's session from the live quote — the move in daily sigmas, the volume against "
        "normal for this time of day, the price against VWAP and in today's range. A one-day view: "
        "heavy-volume moves tend to continue, quiet ones to partly reverse.",
        family=AgentFamily.SPECIALIST,
        capabilities=("intraday_momentum", "short_term_reversal", "vwap", "volume_return"),
        inputs=("indicators", "quotes"),
        subjects=("symbol",),
        priority=32,
        horizon_days=1,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        if not ctx.market_open:
            return "the market is closed: there is no session to read"
        if ctx.indicators.empty:
            return "no price history to measure today's move against"
        if session_fraction(ctx.as_of) is None or minutes_traded(ctx.as_of) < SETTLE:
            return "too early in the session: the first 30 minutes are noise"
        return None

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        frac = session_fraction(ctx.as_of) or 0.0
        return [self._one(ctx, s, frac) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str, frac: float) -> Opinion:
        q = ctx.quotes.get(s)
        state = ctx.state(s)
        if q is None or state not in (DataState.FRESH, DataState.LIVE):
            return self.abstain(s, f"no fresh live quote for today's session ({state.value})", ["live quote"])
        move_z = ctx.ind(s, "move_z")
        if move_z is None:
            return self.abstain(s, "today's move cannot be measured (no volatility history)", ["move_z"])
        rel = ctx.ind(s, "rel_volume")
        vs_vwap = (q.price / q.vwap - 1) if q.vwap and q.vwap > 0 else ctx.ind(s, "px_vs_vwap")
        span = (q.day_high - q.day_low) if q.day_high and q.day_low else None
        in_range = (q.price - q.day_low) / span if span and span > 0 and q.day_low else None

        parts: dict[str, float] = {}
        ev: list[Evidence] = []
        drift = squash(move_z, 2.5)
        if rel is not None and rel >= HEAVY:
            weight = min((rel - HEAVY) / 1.0 + 0.5, 1.0)
            parts["volume_return"] = drift * weight
            note = (
                f"a {move_z:+.1f}σ move on {rel + 1:.1f}× normal volume: heavy-volume moves tend to continue"
            )
        elif abs(move_z) >= BIG_MOVE_Z and (rel is None or rel <= QUIET):
            parts["volume_return"] = -0.6 * drift
            volume = "unknown volume" if rel is None else f"{rel + 1:.1f}× normal volume"
            note = f"a {move_z:+.1f}σ move on {volume}: quiet moves tend to partly reverse"
        else:
            parts["volume_return"] = 0.25 * drift
            volume = "volume not yet measured" if rel is None else f"{rel + 1:.1f}× normal volume"
            note = f"a {move_z:+.1f}σ move on {volume}: no clear volume signal"
        ev.append(Evidence("move_z", round(move_z, 2), note, sgn(parts["volume_return"]), 0.7, quality=state))
        if rel is not None:
            ev.append(
                Evidence("rel_volume", round(rel + 1, 2), f"{rel + 1:.1f}× normal volume by now", 0, 0.4,
                         quality=state)
            )  # fmt: skip
        if vs_vwap is not None:
            parts["vwap"] = squash(vs_vwap, 0.01)
            side = "above" if vs_vwap > 0 else "below"
            ev.append(
                Evidence("px_vs_vwap", round(vs_vwap, 4), f"{abs(vs_vwap):.2%} {side} today's VWAP",
                         sgn(vs_vwap), 0.5, quality=state)
            )  # fmt: skip
        if in_range is not None:
            parts["range"] = (in_range - 0.5) * 2 * 0.6
            ev.append(
                Evidence("day_range", round(in_range, 2), f"at {in_range:.0%} of today's range",
                         sgn(in_range - 0.5), 0.3, quality=state)
            )  # fmt: skip

        score = 0.55 * parts["volume_return"] + 0.3 * parts.get("vwap", 0.0) + 0.15 * parts.get("range", 0.0)
        size = min(abs(move_z) / 3.0, 1.0)
        confidence = (0.12 + 0.33 * size) * (0.5 + 0.5 * agreement(parts.values(), score))
        if frac < 0.3:
            confidence *= 0.7  # early in the session the tape still has a lot to say
        if ctx.regime is not None and ctx.regime.label in ("high_volatility", "risk_off"):
            confidence *= 0.8
        invalidation = None
        if q.vwap and score:
            invalidation = f"a move back {'below' if score > 0 else 'above'} today's VWAP ({q.vwap:,.2f})"
        lean = "leans up" if score > 0.1 else "leans down" if score < -0.1 else "has no lean"
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s} tape {lean} for tomorrow: {note}"
            + (
                f"; {abs(vs_vwap):.2%} {'above' if vs_vwap > 0 else 'below'} VWAP"
                if vs_vwap is not None
                else ""
            ),
            ev,
            quality=state,
            used=["live quote", "daily bars", *(["VWAP"] if q.vwap else [])],
            missing=[
                n for n, v in (("rel_volume", rel), ("vwap", vs_vwap), ("day_range", in_range)) if v is None
            ],
            invalidation=invalidation,
            meta={
                "components": {k: round(v, 3) for k, v in parts.items()},
                "session_fraction": round(frac, 2),
            },
        )
