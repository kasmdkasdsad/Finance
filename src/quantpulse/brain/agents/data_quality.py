"""Data-quality agent: can the data behind each subject be trusted right now?

It has no market view; it is a *constraint*. It reports each focus symbol's precise quote status (see
:mod:`quantpulse.brain.data_health`: fresh, live, stale, no print since the open, delayed, missing,
refused by the subscription, provider error, invalid timestamp, market closed, holiday, synthetic), what
the price and spread were measured on (IEX alone, SIP, delayed SIP), quote problems found by the existing
quote validation, thin history and provider errors — and *vetoes* any action on a subject whose data is
not executable. For the market it reports the cycle's data headline and causes, and vetoes everything
when this computer's clock is so far from Alpaca's that quote ages cannot be trusted. A veto cannot be
outvoted by bullish agents.
"""

from __future__ import annotations

from collections.abc import Sequence

from quantpulse.domain import trading_signals as ts
from quantpulse.schemas.common import DataStatus

from ..context import BrainContext
from ..data_health import CLOCK_SKEW_VETO_SECONDS
from ..types import (
    EXECUTABLE_STATES,
    MARKET,
    AgentFamily,
    AgentSpec,
    DataState,
    Evidence,
    Opinion,
    Stance,
)
from .base import Agent, symbols_only


class DataQualityAgent(Agent):
    role = "constraint"
    spec = AgentSpec(
        id="data_quality",
        source="data",
        failure="fails closed: if it does not run, the orchestrator vetoes every action (nothing is executable)",
        name="Data quality",
        description="Checks freshness, completeness and plausibility of market data and broker state; "
        "vetoes actions on data that cannot be trusted.",
        family=AgentFamily.PERCEPTION,
        capabilities=("stale_data", "bad_quotes", "spread_anomalies", "missing_data", "provider_errors"),
        inputs=("quotes", "quality", "data_states"),
        subjects=("market", "symbol"),
        priority=0,
        horizon_days=0,
    )

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        out = [self._market(ctx)] if MARKET in subjects else []
        for s in symbols_only(subjects):
            out.append(self._symbol(ctx, s))
        ctx.working.post("data_states", {s: ctx.state(s).value for s in symbols_only(subjects)})
        return out

    def _opinion(
        self, subject: str, state: DataState, thesis: str, ev: list[Evidence], veto: str | None
    ) -> Opinion:
        return Opinion(
            agent_id=self.spec.id,
            agent_version=self.spec.version,
            subject=subject,
            stance=Stance.NEUTRAL,
            score=0.0,
            confidence=1.0,
            horizon_days=0,
            thesis=thesis,
            evidence=ev,
            data_quality=state,
            veto=veto,
            meta={"gradeable": False},
        )

    def _market(self, ctx: BrainContext) -> Opinion:
        ev: list[Evidence] = []
        problems: list[str] = []
        live = sum(1 for s in ctx.universe if ctx.state(s) in EXECUTABLE_STATES)
        n = max(len(ctx.universe), 1)
        ev.append(
            Evidence("live_quotes", live, f"{live} of {len(ctx.universe)} symbols have usable live quotes")
        )
        if ctx.price_status is DataStatus.SYNTHETIC:
            problems.append("price history is synthetic")
        if not ctx.portfolio.available:
            problems.append(f"broker unavailable ({ctx.portfolio.error})")
        elif ctx.portfolio.account is not None and ctx.portfolio.account.blocked:
            problems.append("Alpaca reports the account blocked")
        for source, err in ctx.provider_errors.items():
            ev.append(
                Evidence(
                    f"provider_error:{source}",
                    err[:160],
                    f"{source} failed",
                    direction=-1,
                    quality=DataState.PROVIDER_ERROR,
                )
            )
        if ctx.market_open and live / n < 0.5:
            problems.append(f"only {live / n:.0%} of the universe has usable live quotes")
        skew = ctx.feed.get("clock_skew_s")
        if skew is not None:
            ev.append(Evidence("clock_skew_s", skew, f"this computer's clock vs Alpaca's: {skew:+.1f}s"))
            if ctx.market_open and abs(skew) > CLOCK_SKEW_VETO_SECONDS:
                problems.append(
                    f"system clock is {skew:+.0f}s off Alpaca's: quote ages cannot be trusted (synchronise it)"
                )
        for i, cause in enumerate(ctx.feed.get("causes") or []):
            ev.append(
                Evidence(
                    f"data_cause:{i}", cause[:300], cause, direction=0 if ctx.feed.get("healthy") else -1
                )
            )
        if not ctx.market_open:
            state, thesis = DataState.MARKET_CLOSED, "market closed: daily data only, nothing is executable"
        elif problems:
            state, thesis = (
                DataState.PROVIDER_ERROR if not ctx.portfolio.available else DataState.STALE,
                "; ".join(problems),
            )
        else:
            state, thesis = DataState.LIVE, f"market data usable ({live} live quotes)"
            if ctx.feed and not ctx.feed.get("healthy"):
                thesis += f"; {ctx.feed.get('headline')}"
        veto = "; ".join(problems) if problems else ("market closed" if not ctx.market_open else None)
        return self._opinion(MARKET, state, thesis, ev, veto)

    def _symbol(self, ctx: BrainContext, symbol: str) -> Opinion:
        state = ctx.state(symbol)
        ev: list[Evidence] = [Evidence("data_state", state.value, f"data state {state.value}", quality=state)]
        problems: list[str] = []
        diag = ctx.data_health.get(symbol)
        if diag is not None:
            ev.append(
                Evidence(
                    "quote_status",
                    diag.status.value,
                    f"{diag.status.value.replace('_', ' ')}: priced on {diag.coverage}"
                    + (f"; spread checked on {diag.spread_source}" if diag.spread_source else ""),
                    quality=state,
                )
            )
            if diag.reasons and state not in EXECUTABLE_STATES:
                problems.append(diag.reasons[0])
        q = ctx.quotes.get(symbol)
        if q is not None:
            ev.append(
                Evidence(
                    "trade_age_s",
                    round(q.age_seconds, 1),
                    f"last trade {q.age_seconds:.0f}s old",
                    quality=state,
                )
            )
            if q.quote_age_seconds is not None:
                ev.append(
                    Evidence(
                        "quote_age_s",
                        round(q.quote_age_seconds, 1),
                        f"bid/ask {q.quote_age_seconds:.0f}s old",
                        quality=state,
                    )
                )
        elif symbol in ctx.missing_quotes:
            ev.append(
                Evidence(
                    "quote_missing", ctx.missing_quotes[symbol], "no live quote", direction=-1, quality=state
                )
            )
        qq = ctx.quality.get(symbol)
        if qq is not None:
            ev.append(
                Evidence("spread_bps", qq.spread_bps, f"spread measured on {qq.spread_source}", quality=state)
            )
            for p in qq.problems:
                ev.append(Evidence("quote_problem", p, p, direction=-1, strength=0.6, quality=state))
            for b in qq.entry_blocks:
                problems.append(b)
                ev.append(Evidence("price_anomaly", b, b, direction=-1, strength=0.9, quality=state))
        bars = ctx.ind(symbol, "bars")
        if bars is not None and bars < ts.MIN_HISTORY:
            problems.append(f"only {bars:.0f} daily bars")
        if symbol not in ctx.indicators.index:
            problems.append("no price history")
        if state not in EXECUTABLE_STATES:
            label = diag.status.value.replace("_", " ") if diag is not None else state.value
            problems.insert(0, f"data {label}")
        veto = "; ".join(problems) if problems else None
        thesis = f"{symbol}: data {state.value}" + (f" — {veto}" if veto else ", usable")
        return self._opinion(symbol, state, thesis, ev, veto)
