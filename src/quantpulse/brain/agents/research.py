"""Research and situational-awareness agents: the second stage, reading what the specialists found.

**Research** works like an analyst with a checklist. For each focus symbol it answers a set of standard
questions — the general ones every idea must survive, plus the questions specific to the kind of
opportunity that brought the symbol into focus — from the data the cycle already holds, and records each
answer as evidence for or against. It casts no vote: its findings feed the bull and bear cases.

**Situational awareness** summarises the moment for the whole team — regime, market volatility, breadth,
data health, the paper account's exposure and day P/L against the daily loss limit, the kill switch — as a
risk *posture* (normal, cautious, defensive) that the decision step uses to scale or stop new risk and to
propose de-risking.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from ..context import BrainContext
from ..types import EXECUTABLE_STATES, MARKET, AgentFamily, AgentSpec, DataState, Evidence, Opinion
from .base import Agent, Role, symbols_only
from .common import opinion, sgn, symbol_quality

Finding = tuple[str, str, int]  # question, answer, +1 supports a long / −1 argues against / 0 neutral


def _sector_peers(ctx: BrainContext, s: str) -> list[str]:
    sector = ctx.sectors.get(s)
    return (
        [p for p, sec in ctx.sectors.items() if sec == sector and p != s and p in ctx.indicators.index]
        if sector
        else []
    )


def q_idiosyncratic(ctx: BrainContext, s: str) -> Finding | None:
    corr, rs = ctx.ind(s, "corr_spy_20"), ctx.ind(s, "rs_1m")
    if corr is None or rs is None:
        return None
    own = abs(rs) > 0.03 and corr < 0.6
    return (
        "Is the move the stock's own or the market's?",
        f"{'its own' if own else 'largely the market'} (1-month vs benchmark {rs:+.1%}, correlation {corr:.2f})",
        sgn(rs) if own else 0,
    )


def q_volume(ctx: BrainContext, s: str) -> Finding | None:
    ud, trend = ctx.ind(s, "updown_volume"), ctx.ind(s, "volume_trend")
    if ud is None:
        return None
    return (
        "Is volume confirming?",
        f"up-day vs down-day volume {ud:+.2f} (log ratio), 20- vs 120-day volume {trend or 0:+.0%}",
        sgn(ud) if abs(ud) > 0.1 else 0,
    )


def q_sector(ctx: BrainContext, s: str) -> Finding | None:
    peers = _sector_peers(ctx, s)
    vals = [v for v in (ctx.ind(p, "rs_1m") for p in peers) if v is not None]
    if len(vals) < 3:
        return None
    avg = sum(vals) / len(vals)
    return (
        f"Is the sector ({ctx.sectors.get(s)}) confirming?",
        f"peers' 1-month relative strength averages {avg:+.1%} across {len(vals)} names",
        sgn(avg) if abs(avg) > 0.01 else 0,
    )


def q_event(ctx: BrainContext, s: str) -> Finding | None:
    e = ctx.events.get(s) or {}
    days = e.get("days_to_next")
    if days is None:
        return None
    return (
        "Is there an event ahead?",
        f"earnings in {days} days ({e.get('next_source')})",
        -1 if 0 <= days <= 10 else 0,
    )


def q_extended(ctx: BrainContext, s: str) -> Finding | None:
    rsi, stretch = ctx.ind(s, "rsi14"), ctx.ind(s, "px_vs_sma50")
    if rsi is None or stretch is None:
        return None
    hot = rsi >= 75 or stretch >= 0.15
    cold = rsi <= 25 or stretch <= -0.15
    return (
        "Is it extended?",
        f"RSI {rsi:.0f}, {stretch:+.1%} vs its 50-day average",
        -1 if hot else 1 if cold else 0,
    )


def q_liquidity(ctx: BrainContext, s: str) -> Finding | None:
    adv = ctx.ind(s, "adv_dollar")
    qq = ctx.quality.get(s)
    if adv is None:
        return None
    ok = adv >= ctx.limits.min_dollar_volume
    spread = f", spread {qq.spread_bps:.1f} bps" if qq is not None and qq.spread_bps is not None else ""
    return ("Is it liquid enough?", f"average ${adv / 1e6:,.0f}M a day{spread}", 0 if ok else -1)


def q_breakout_confirmed(ctx: BrainContext, s: str) -> Finding | None:
    vol, close_up = ctx.ind(s, "volume_ratio_1d"), ctx.ind(s, "breakout")
    if vol is None or close_up is None:
        return None
    ok = vol >= 1.5 and close_up > 0
    return (
        "Did the breakout hold on volume?",
        f"{close_up:+.1%} beyond the prior high on {vol:.1f}× volume",
        1 if ok else -1,
    )


def q_value_quality(ctx: BrainContext, s: str) -> Finding | None:
    roe, acc = ctx.feature(s, "roe"), ctx.feature(s, "accruals")
    if roe is None:
        return None
    ok = roe > 0.08 and (acc is None or acc < 0.05)
    return (
        "Is quality good enough to avoid a value trap?",
        f"ROE {roe:.0%}" + (f", accruals {acc:+.2f}" if acc is not None else ""),
        1 if ok else -1,
    )


def q_reversion_room(ctx: BrainContext, s: str) -> Finding | None:
    trend = ctx.ind(s, "px_vs_sma200")
    if trend is None:
        return None
    return (
        "Is the long-term trend on the side of a rebound?",
        f"{trend:+.1%} vs its 200-day average",
        1 if trend > 0 else -1,
    )


GENERAL: tuple[Callable[[BrainContext, str], Finding | None], ...] = (
    q_idiosyncratic,
    q_volume,
    q_sector,
    q_event,
    q_extended,
    q_liquidity,
)
SPECIFIC: dict[str, tuple[Callable[[BrainContext, str], Finding | None], ...]] = {
    "breakout": (q_breakout_confirmed,),
    "valuation_dislocation": (q_value_quality,),
    "mean_reversion": (q_reversion_room,),
    "relative_value": (q_value_quality,),
}


class ResearchAgent(Agent):
    spec = AgentSpec(
        id="research",
        source="findings",
        failure="no research checklist; the bull and bear cases rest on the specialists' evidence alone",
        name="Research",
        description="Answers a checklist of questions per symbol (market vs own move, volume, sector, events, "
        "extension, liquidity, and opportunity-specific checks) from the cycle's data.",
        family=AgentFamily.META,
        capabilities=("research_checklist", "confirmation", "context"),
        inputs=("indicators", "events", "opportunities"),
        subjects=("symbol",),
        priority=60,
        horizon_days=21,
        stage=1,
    )
    role: ClassVar[Role] = "context"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        kinds: dict[str, set[str]] = {}
        for o in ctx.opportunities:
            for sym in o.symbols[:1]:
                kinds.setdefault(sym, set()).add(o.kind)
        return [self._one(ctx, s, kinds.get(s, set())) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str, kinds: set[str]) -> Opinion:
        questions = list(GENERAL)
        for k in sorted(kinds):
            questions.extend(SPECIFIC.get(k, ()))
        q = symbol_quality(ctx, s)
        findings = [f for f in (fn(ctx, s) for fn in questions) if f is not None]
        ev = [
            Evidence(f"research:{i}", ans, f"{question} {ans}", d, 0.5, quality=q)
            for i, (question, ans, d) in enumerate(findings)
        ]
        pro, con = sum(1 for f in findings if f[2] > 0), sum(1 for f in findings if f[2] < 0)
        ctx.working.post(
            "research", {**(ctx.working.facts.get("research") or {}), s: {"for": pro, "against": con}}
        )
        return opinion(
            self,
            s,
            0.0,
            0.0,
            f"research on {s}: {len(findings)} questions answered, {pro} support a long, {con} argue against"
            + (f" (opportunity: {', '.join(sorted(kinds))})" if kinds else ""),
            ev,
            quality=q,
            used=["indicators", "earnings events", "sector peers"],
            meta={
                "findings": [{"question": a, "answer": b, "direction": c} for a, b, c in findings],
                "kinds": sorted(kinds),
            },
            directional=False,
        )


class SituationalAwarenessAgent(Agent):
    spec = AgentSpec(
        id="situational_awareness",
        source="findings",
        failure="fails safe: if it does not run, the posture is cautious (fewer, smaller new positions)",
        name="Situational awareness",
        description="The moment in one posture — normal, cautious or defensive — from regime, volatility, "
        "breadth, data health, exposure, day P/L vs the loss limit and the kill switch.",
        family=AgentFamily.META,
        capabilities=("risk_posture", "situation_summary"),
        inputs=("regime", "portfolio", "working_memory"),
        subjects=("market",),
        priority=5,
        horizon_days=1,
        stage=1,
    )
    role: ClassVar[Role] = "context"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        facts: dict[str, Any] = ctx.working.facts
        defensive: list[str] = []
        cautious: list[str] = []
        regime = facts.get("regime")
        if ctx.kill_switch:
            defensive.append("the kill switch is on")
        if regime == "risk_off":
            defensive.append("risk-off regime")
        elif regime in ("bearish", "high_volatility"):
            cautious.append(f"{regime.replace('_', ' ')} regime")
        if ctx.vix is not None:
            if ctx.vix >= 30:
                (defensive if regime in ("bearish", "risk_off") else cautious).append(f"VIX {ctx.vix:.0f}")
            elif ctx.vix >= 22:
                cautious.append(f"VIX {ctx.vix:.0f}")
        breadth = ctx.market_stats.get("breadth_200")
        if breadth is not None and breadth < 0.4:
            cautious.append(f"only {breadth:.0%} of stocks above their 200-day average")
        pl = ctx.portfolio.account.day_pl_pct if ctx.portfolio.account else None
        limit = ctx.limits.max_daily_loss_pct
        if pl is not None and limit > 0:
            if pl <= -0.6 * limit:
                defensive.append(f"day P/L {pl:+.2%} is near the {limit:.0%} daily loss limit")
            elif pl <= -0.3 * limit:
                cautious.append(f"day P/L {pl:+.2%}")
        cons = facts.get("portfolio_constraints") or {}
        if (cons.get("exposure") or 0) >= 0.9:
            cautious.append(f"{cons['exposure']:.0%} invested")
        if cons.get("margin"):
            cautious.append("the account is on margin")
        dq = next((o for o in ctx.working.opinions.get(MARKET, []) if o.agent_id == "data_quality"), None)
        if dq is not None and dq.veto and ctx.market_open:
            cautious.append(f"data: {dq.veto}")
        if not ctx.portfolio.available:
            cautious.append("the paper account could not be read")
        posture = "defensive" if defensive else "cautious" if cautious else "normal"
        scale = {"normal": 1.0, "cautious": 0.6, "defensive": 0.0}[posture]
        reasons = defensive + cautious
        situation = {
            "posture": posture,
            "risk_scale": scale,
            "reasons": reasons,
            "session": ctx.session.value,
        }
        ctx.working.post("situation", situation)
        ev = [Evidence(f"situation:{i}", r, r, -1, 0.6) for i, r in enumerate(reasons)]
        return [
            opinion(
                self,
                MARKET,
                0.0,
                0.0,
                f"posture {posture}"
                + (f": {'; '.join(reasons)}" if reasons else ": nothing calls for caution"),
                ev,
                quality=DataState.LIVE if ctx.market_open else DataState.MARKET_CLOSED,
                used=["working memory", "paper account", "regime", "VIX"],
                meta=situation,
                directional=False,
            )
        ]


def executable(ctx: BrainContext, symbol: str) -> bool:
    return ctx.state(symbol) in EXECUTABLE_STATES
