"""The research jobs: what the Brain works on while the market is closed.

Each job reads the record (predictions, decisions, trades, fills, ideas, prices) and writes three things only:
its result (kept with the job), conclusions in the learning ledger (each with its evidence — UNPROVEN until the
sample suffices) and follow-up questions for the queue. Some also move a hypothesis one tested stage along the
improvement lifecycle. None of them sends an order, changes a setting, a limit, a kill switch, a weight or a
strategy's production status: what research may touch is limited to its own tables, the memory, the lab's shadow
tracking and its own preparation notes (the watchlist, upcoming events).
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from quantpulse.brain import opportunity_outcomes as ideas
from quantpulse.brain import performance as perf
from quantpulse.brain.evaluation import MarketPrices
from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import (
    BrainCycleRow,
    BrainDecisionRow,
    BrainExecutionRow,
    BrainOpinionRow,
    BrainOpportunityOutcomeRow,
    BrainReflectionRow,
    BrainSessionRow,
    BrainThesisRow,
)

from .ledger import Finding, LearningLedger, binomial_vs_half, p_two_sided, t_test_mean
from .lifecycle import Lifecycle, LifecycleError

PAPER_SHADOW_DAYS = 30  # forward data a hypothesis must be tracked on before it is evaluated
WATCHLIST_KEY = "research_watchlist"
EVENTS_KEY = "research_events"
PAPER_LIMITS = [
    "Alpaca paper fills can be kinder than real fills",
    "a short paper history; regimes it has not seen",
]


@dataclass
class JobContext:
    job: dict[str, Any]
    brain: Any  # BrainService
    settings: Settings
    clock: Clock
    ledger: LearningLedger
    lifecycle: Lifecycle
    reference: Any = None
    market: Any = None
    follow_ups: list[dict[str, Any]] = field(default_factory=list)
    learned: list[dict[str, Any]] = field(default_factory=list)

    @property
    def db(self) -> Any:
        return self.brain.db

    @property
    def min_sample(self) -> int:
        return int(self.settings.brain_min_reliability_observations)

    def now(self) -> datetime:
        return self.clock.now()

    async def learn(self, finding: Finding) -> dict[str, Any]:
        rec = await self.ledger.record(self.job["kind"], finding, self.now(), job_id=self.job.get("id"))
        self.learned.append({"id": rec["id"], "topic": rec["topic"], "status": rec["status"]})
        return rec

    def ask(self, kind: str, question: str, params: dict[str, Any] | None = None) -> None:
        self.follow_ups.append({"kind": kind, "question": question, "params": params or {}})


Handler = Callable[[JobContext], Awaitable[dict[str, Any]]]


def _d(value: datetime | None) -> date | None:
    return value.astimezone(NEW_YORK).date() if value is not None else None


def _period(times: Sequence[datetime]) -> tuple[date | None, date | None]:
    return (_d(min(times)), _d(max(times))) if times else (None, None)


def _brief(value: Any, depth: int = 0) -> Any:
    """A compact copy of a result for the job record."""
    if depth > 3:
        return "…"
    if isinstance(value, dict):
        return {str(k): _brief(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list | tuple):
        return [_brief(v, depth + 1) for v in list(value)[:25]]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def proportion_test(k: int, n: int, p0: float) -> tuple[float | None, float | None]:
    """Share and two-sided p-value of "the share is p0" (normal approximation)."""
    if n <= 0:
        return None, None
    z = (k / n - p0) / math.sqrt(p0 * (1 - p0) / n)
    return k / n, p_two_sided(z)


def two_proportions(k1: int, n1: int, k2: int, n2: int) -> tuple[float | None, float | None]:
    if n1 <= 0 or n2 <= 0:
        return None, None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    diff = k1 / n1 - k2 / n2
    return diff, (p_two_sided(diff / se) if se > 0 else None)


# ================================================================================================ GRADE
async def grade_predictions(ctx: JobContext) -> dict[str, Any]:
    """Grade every prediction whose horizon has passed (the learner; idempotent)."""
    if ctx.brain.learner is None:
        return {"skipped": "no market service: nothing can be graded"}
    summary = await ctx.brain.learn(wait=None)
    if summary.get("evaluated"):
        ctx.ask(
            "agent_calibration", "Which agents are accurate and calibrated, now that more calls are graded?"
        )
    return {"evaluated": summary.get("evaluated"), "summary": _brief(summary)}


async def grade_ideas(ctx: JobContext) -> dict[str, Any]:
    """Grade detected ideas (taken and rejected) whose horizon has passed."""
    if ctx.market is None:
        return {"skipped": "no market service"}
    counts = await ideas.evaluate_due(
        ctx.db, MarketPrices(ctx.market, ctx.clock), ctx.clock, ctx.settings.benchmark_symbol
    )
    if counts.get("evaluated"):
        ctx.ask("rejected_opportunities", "Which rejected opportunities subsequently worked?")
    return {"graded": _brief(counts)}


# ================================================================================================ ANALYZE
async def agent_calibration(ctx: JobContext) -> dict[str, Any]:
    """Each agent's graded calls against a coin flip (independent blocks), and its calibration."""
    rows = await perf.graded(ctx.db)
    by_source: dict[str, list[perf.Graded]] = defaultdict(list)
    for r in rows:
        by_source[r.source].append(r)
    out: dict[str, Any] = {}
    for source, mine in sorted(by_source.items()):
        m = perf.metrics(mine, ctx.min_sample)
        start, end = _period([r.made_at for r in mine])
        hit = m["hit_rate"]
        await ctx.learn(
            Finding(
                topic=f"agent:{source}",
                claim=f"{source}'s calls beat a coin flip",
                sample_size=m["n_effective"],
                min_sample=ctx.min_sample,
                regime="all",
                benchmark="a coin flip (50%)",
                method="block binomial test on independent call groups (Wilson CI)",
                p_value=m["p_value"],
                effect=(hit - 0.5) if hit is not None else None,
                period_start=start,
                period_end=end,
                statistics={
                    "hit_rate": hit,
                    "ci": [m["ci_low"], m["ci_high"]],
                    "brier": m["brier"],
                    "ic": m["ic"],
                    "calls": m["n"],
                },
                limitations=[
                    "calls on the same day are not independent: they are grouped first",
                    "graded against the benchmark over each call's own horizon",
                    *PAPER_LIMITS[1:],
                ],
            )
        )
        out[source] = {"n_effective": m["n_effective"], "hit_rate": hit, "verdict": m["verdict"]}
    return {"agents": out, "graded_calls": len(rows)}


async def decision_quality(ctx: JobContext) -> dict[str, Any]:
    """Decision quality separated from luck: do decisions judged sound before the outcome end well more often?"""
    await asyncio.sleep(0)
    from quantpulse.brain.reflection import reflect_on_decisions

    written = await reflect_on_decisions(ctx.db, ctx.now())
    async with ctx.db.session() as s:
        rows = list(
            (
                await s.scalars(
                    select(BrainReflectionRow).where(BrainReflectionRow.subject_type == "decision")
                )
            ).all()
        )
    rated = [
        r for r in rows if r.decision_quality in ("good", "bad") and r.outcome_quality in ("good", "bad")
    ]
    good = [r for r in rated if r.decision_quality == "good"]
    bad = [r for r in rated if r.decision_quality == "bad"]
    k1 = sum(1 for r in good if r.outcome_quality == "good")
    k2 = sum(1 for r in bad if r.outcome_quality == "good")
    diff, p = two_proportions(k1, len(good), k2, len(bad))
    start, end = _period([r.created_at for r in rated])
    mix: dict[str, int] = defaultdict(int)
    for r in rows:
        mix[r.category] += 1
    await ctx.learn(
        Finding(
            topic="decisions:quality_predicts_outcome",
            claim="decisions judged sound end well more often than unsound ones",
            sample_size=min(len(good), len(bad)),
            min_sample=ctx.min_sample,
            regime="all",
            benchmark="decisions judged unsound",
            method="two-proportion z-test",
            p_value=p,
            effect=diff,
            period_start=start,
            period_end=end,
            statistics={
                "sound": len(good),
                "sound_good_outcome": k1,
                "unsound": len(bad),
                "unsound_good_outcome": k2,
                "categories": dict(mix),
            },
            limitations=[
                "the quality rules are the Brain's own: a circular test if the rules are wrong",
                "a good decision can end badly (luck): many decisions are needed",
                *PAPER_LIMITS,
            ],
        )
    )
    return {"reflections_written": len(written), "categories": dict(mix)}


async def trade_review(ctx: JobContext) -> dict[str, Any]:
    """Winning and losing trades: win rate, average return, exit reasons, and which agents dissented on losers."""
    lessons = await ctx.brain.trade_lessons()
    async with ctx.db.session() as s:
        closed = list(
            (await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "closed"))).all()
        )
        decisions = {
            d.id: d
            for d in (
                await s.scalars(
                    select(BrainDecisionRow).where(
                        BrainDecisionRow.id.in_([t.entry_decision_id for t in closed if t.entry_decision_id])
                    )
                )
            ).all()
        }
        cycles = {d.cycle_id for d in decisions.values()}
        opinions = list(
            (await s.scalars(select(BrainOpinionRow).where(BrainOpinionRow.cycle_id.in_(cycles)))).all()
        )
    done = [t for t in closed if t.return_pct is not None]
    wins = [t for t in done if (t.return_pct or 0) > 0]
    hit, p = binomial_vs_half(len(wins), len(done))
    start, end = _period([t.closed_at for t in done if t.closed_at])
    await ctx.learn(
        Finding(
            topic="trades:win_rate",
            claim="the Brain's closed trades win more often than they lose",
            sample_size=len(done),
            min_sample=ctx.min_sample,
            regime="all",
            benchmark="a coin flip (50%)",
            method="binomial test",
            p_value=p,
            effect=(hit - 0.5) if hit is not None else None,
            period_start=start,
            period_end=end,
            statistics={"trades": len(done), "wins": len(wins), "win_rate": hit},
            limitations=["a win rate ignores the size of wins and losses", *PAPER_LIMITS],
        )
    )
    mean, t, p_mean = t_test_mean([float(t.return_pct or 0) for t in done])
    await ctx.learn(
        Finding(
            topic="trades:mean_return",
            claim="the Brain's closed trades make money on average",
            sample_size=len(done),
            min_sample=ctx.min_sample,
            regime="all",
            benchmark="zero return",
            method="one-sample t-test",
            p_value=p_mean,
            effect=mean,
            period_start=start,
            period_end=end,
            statistics={"mean_return": mean, "t": t},
            limitations=["not adjusted for the market's move over each holding period", *PAPER_LIMITS],
        )
    )
    # which agents dissented (bearish on a long) when the trade was opened — on losers vs on winners
    dissent: dict[str, dict[str, int]] = defaultdict(
        lambda: {"loss_dissent": 0, "losses": 0, "win_dissent": 0, "wins": 0}
    )
    by_cycle_subject: dict[tuple[int, str], list[BrainOpinionRow]] = defaultdict(list)
    for o in opinions:
        by_cycle_subject[(o.cycle_id, o.subject)].append(o)
    for t in done:
        d = decisions.get(t.entry_decision_id or -1)
        if d is None:
            continue
        lost = (t.return_pct or 0) <= 0
        for o in by_cycle_subject.get((d.cycle_id, t.symbol), []):
            rec = dissent[o.agent_id]
            rec["losses" if lost else "wins"] += 1
            if o.score < 0:  # it argued against the (long) position
                rec["loss_dissent" if lost else "win_dissent"] += 1
    for agent, rec in sorted(dissent.items()):
        diff, p_d = two_proportions(rec["loss_dissent"], rec["losses"], rec["win_dissent"], rec["wins"])
        await ctx.learn(
            Finding(
                topic=f"dissent:{agent}",
                claim=f"{agent} dissents more often on trades that lose than on trades that win",
                sample_size=min(rec["losses"], rec["wins"]),
                min_sample=ctx.min_sample,
                regime="all",
                benchmark="its dissent rate on winning trades",
                method="two-proportion z-test",
                p_value=p_d,
                effect=diff,
                period_start=start,
                period_end=end,
                statistics=dict(rec),
                limitations=[
                    "only trades the consensus took: the agent's dissent did not stop them",
                    *PAPER_LIMITS,
                ],
            )
        )
    for t in sorted((t for t in done if (t.return_pct or 0) < 0), key=lambda t: t.closed_at or t.opened_at)[
        -5:
    ]:
        ctx.ask("trade_post_mortem", f"Why did the {t.symbol} trade (#{t.id}) fail?", {"thesis_id": t.id})
    return {"closed_trades": len(done), "wins": len(wins), "lessons": _brief(lessons)}


async def trade_post_mortem(ctx: JobContext) -> dict[str, Any]:
    """One losing trade: what the thesis expected, what happened, and which agents disagreed at entry."""
    thesis_id = int(ctx.job["params"].get("thesis_id") or 0)
    async with ctx.db.session() as s:
        t = await s.get(BrainThesisRow, thesis_id)
        if t is None:
            return {"skipped": f"no trade #{thesis_id}"}
        d = await s.get(BrainDecisionRow, t.entry_decision_id) if t.entry_decision_id else None
        opinions: list[BrainOpinionRow] = []
        if d is not None:
            stmt = select(BrainOpinionRow).where(
                BrainOpinionRow.cycle_id == d.cycle_id, BrainOpinionRow.subject == t.symbol
            )
            opinions = list((await s.scalars(stmt)).all())
    dissenters = sorted({o.agent_id for o in opinions if o.score < 0})
    supporters = sorted({o.agent_id for o in opinions if o.score > 0})
    await ctx.learn(
        Finding(
            topic=f"postmortem:{t.symbol}:{t.id}",
            claim=f"the agents that dissented on {t.symbol} saw the loss coming",
            sample_size=1,
            min_sample=ctx.min_sample,
            regime=t.regime or "unknown",
            benchmark="the agents that supported it",
            method="a single case (no test)",
            period_start=_d(t.opened_at),
            period_end=_d(t.closed_at),
            statistics={"return": t.return_pct, "dissenters": dissenters, "supporters": supporters},
            limitations=["one trade proves nothing: aggregated in dissent:<agent> across all trades"],
        )
    )
    return {
        "symbol": t.symbol,
        "return": t.return_pct,
        "thesis": t.thesis,
        "invalidation": t.invalidation,
        "dissenters": dissenters,
        "supporters": supporters,
        "regime": t.regime,
    }


async def rejected_opportunities(ctx: JobContext) -> dict[str, Any]:
    """Were the rejections justified? Each rejection reason's ideas, after the fact; taken versus rejected."""
    rep = await ideas.report(ctx.db, ctx.min_sample)
    async with ctx.db.session() as s:
        times = list(
            (
                await s.scalars(
                    select(BrainOpportunityOutcomeRow.detected_at).where(
                        BrainOpportunityOutcomeRow.state == "evaluated"
                    )
                )
            ).all()
        )
    start, end = _period(times)
    for reason, g in rep["by_reason"].items():
        decisive = int(g["decisive"])
        share = g.get("avoided_share")
        _, p = proportion_test(round((share or 0) * decisive), decisive, 0.5)
        await ctx.learn(
            Finding(
                topic=f"rejection:{reason}",
                claim=f"rejecting ideas for '{reason}' avoided more losers than it missed winners",
                sample_size=decisive,
                min_sample=ctx.min_sample,
                regime="all",
                benchmark="a coin flip (50%)",
                method="binomial test on decisive outcomes",
                p_value=p,
                effect=(share - 0.5) if share is not None else None,
                period_start=start,
                period_end=end,
                statistics=_brief(g),
                limitations=[
                    "ideas are graded at one horizon per kind",
                    "a protected rejection (risk, data) is never "
                    "loosened on this evidence: it informs, nothing more",
                ],
            )
        )
    comp = rep["taken_vs_rejected"]
    t = comp.get("t")
    await ctx.learn(
        Finding(
            topic="ideas:taken_vs_rejected",
            claim="the ideas the Brain took did better than the ideas it rejected",
            sample_size=min(comp["taken"]["n"], comp["rejected"]["n"]),
            min_sample=ctx.min_sample,
            regime="all",
            benchmark="the rejected ideas",
            method="Welch t-test on favourable relative returns",
            p_value=p_two_sided(t) if t is not None else None,
            effect=comp.get("difference"),
            period_start=start,
            period_end=end,
            statistics=_brief(comp),
            limitations=["ideas overlap in time (not independent)", "graded against the benchmark"],
        )
    )
    return {
        "by_reason": {k: v["status"] for k, v in rep["by_reason"].items()},
        "taken_vs_rejected": comp["status"],
    }


async def missed_opportunities(ctx: JobContext) -> dict[str, Any]:
    """The last sessions' ideas the Brain did not take that worked (for the next session's preparation)."""
    since = (ctx.now() - timedelta(days=7)).astimezone(NEW_YORK).date()
    async with ctx.db.session() as s:
        rows = list(
            (
                await s.scalars(
                    select(BrainOpportunityOutcomeRow).where(
                        BrainOpportunityOutcomeRow.day >= since,
                        BrainOpportunityOutcomeRow.taken.is_(False),
                        BrainOpportunityOutcomeRow.state == "evaluated",
                    )
                )
            ).all()
        )
    missed = sorted((r for r in rows if r.verdict == "missed"), key=lambda r: -(r.favourable or 0))
    return {
        "evaluated": len(rows),
        "missed": len(missed),
        "top": [
            {
                "symbol": r.symbol,
                "kind": r.kind,
                "day": r.day.isoformat(),
                "reason": r.reason,
                "favourable": r.favourable,
            }
            for r in missed[:10]
        ],
    }


async def execution_review(ctx: JobContext) -> dict[str, Any]:
    """Execution quality (slippage against the quoted spread) and turnover (positions closed within 2 sessions)."""
    async with ctx.db.session() as s:
        fills = list(
            (await s.scalars(select(BrainExecutionRow).where(BrainExecutionRow.filled_qty > 0))).all()
        )
        closed = list(
            (await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "closed"))).all()
        )
    costs, start_end = [], []
    for e in fills:
        if not (e.filled_avg_price and e.quote_bid and e.quote_ask and e.quote_ask > e.quote_bid > 0):
            continue
        mid = (e.quote_bid + e.quote_ask) / 2
        sign = 1 if e.side == "buy" else -1
        slip_bps = sign * (e.filled_avg_price - mid) / mid * 1e4
        half = (e.quote_ask - e.quote_bid) / mid * 1e4 / 2
        costs.append(slip_bps - half)
        if e.filled_at:
            start_end.append(e.filled_at)
    mean, t, p = t_test_mean(costs)
    start, end = _period(start_end)
    await ctx.learn(
        Finding(
            topic="execution:cost_vs_half_spread",
            claim="fills cost more than half the quoted spread",
            sample_size=len(costs),
            min_sample=ctx.min_sample,
            regime="all",
            benchmark="half the quoted spread",
            method="one-sample t-test on (slippage − half spread) in bp",
            p_value=p,
            effect=mean,
            period_start=start,
            period_end=end,
            statistics={"mean_excess_bps": mean, "t": t, "fills": len(costs)},
            limitations=["the quote is the one seen at decision time (IEX on the free feed)", *PAPER_LIMITS],
        )
    )
    quick = [t_ for t_ in closed if t_.closed_at and (t_.closed_at - t_.opened_at) <= timedelta(days=2)]
    share, p_q = proportion_test(len(quick), len(closed), 0.2)
    await ctx.learn(
        Finding(
            topic="turnover:quick_closes",
            claim="more than 1 in 5 positions are closed within two days (excess turnover)",
            sample_size=len(closed),
            min_sample=ctx.min_sample,
            regime="all",
            benchmark="20% of positions",
            method="binomial test against 20%",
            p_value=p_q,
            effect=(share - 0.2) if share is not None else None,
            period_start=_period([t_.closed_at for t_ in closed if t_.closed_at])[0],
            period_end=_period([t_.closed_at for t_ in closed if t_.closed_at])[1],
            statistics={"closed": len(closed), "within_two_days": len(quick)},
            limitations=["a quick close can be right (a stop doing its job)"],
        )
    )
    return {
        "fills_measured": len(costs),
        "mean_excess_bps": mean,
        "closed": len(closed),
        "quick_closes": len(quick),
    }


async def account_vs_benchmark(ctx: JobContext) -> dict[str, Any]:
    """The paper account's daily returns against the benchmark: excess return and beta, session by session."""
    async with ctx.db.session() as s:
        days = list((await s.scalars(select(BrainSessionRow).order_by(BrainSessionRow.day))).all())
    pairs = [
        (d.day, float(d.day_return), float(d.benchmark_return))
        for d in days
        if d.day_return is not None and d.benchmark_return is not None
    ]
    excess = [a - b for _, a, b in pairs]
    mean, t, p = t_test_mean(excess)
    start, end = (pairs[0][0], pairs[-1][0]) if pairs else (None, None)
    bench = ctx.settings.benchmark_symbol
    await ctx.learn(
        Finding(
            topic="account:excess_return",
            claim=f"the paper account beats {bench} day by day",
            sample_size=len(excess),
            min_sample=ctx.min_sample,
            regime="all",
            benchmark=bench,
            method="one-sample t-test on daily excess returns",
            p_value=p,
            effect=mean,
            period_start=start,
            period_end=end,
            statistics={
                "sessions": len(excess),
                "mean_daily_excess": mean,
                "t": t,
                "annualised_excess": mean * 252 if mean is not None else None,
            },
            limitations=["daily returns are not independent of the market's regime", *PAPER_LIMITS],
        )
    )
    beta = None
    if len(pairs) >= 3:
        a = np.array([x for _, x, _ in pairs])
        b = np.array([y for _, _, y in pairs])
        if b.var() > 0:
            beta = float(np.cov(a, b)[0, 1] / b.var(ddof=1))
            resid = a - beta * b
            se = (
                math.sqrt(float(resid.var(ddof=1)) / (float(b.var(ddof=1)) * (len(b) - 1)))
                if len(b) > 2
                else None
            )
            p_beta = p_two_sided((1 - beta) / se) if se else None
            await ctx.learn(
                Finding(
                    topic="account:beta",
                    claim=f"the account moves less than {bench} (beta below 1)",
                    sample_size=len(pairs),
                    min_sample=ctx.min_sample,
                    regime="all",
                    benchmark=f"{bench} (beta 1)",
                    method="OLS beta, t-test against 1",
                    p_value=p_beta,
                    effect=1 - beta,
                    period_start=start,
                    period_end=end,
                    statistics={"beta": beta, "se": se},
                    limitations=["beta drifts with what is held", *PAPER_LIMITS[1:]],
                )
            )
    return {"sessions": len(pairs), "mean_daily_excess": mean, "beta": beta}


async def exposure_review(ctx: JobContext) -> dict[str, Any]:
    """Sector concentration and position sizes of what is held now (open theses)."""
    async with ctx.db.session() as s:
        held = list((await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "open"))).all())
    sectors: dict[str, float] = defaultdict(float)
    for t in held:
        sectors[t.sector or "unknown"] += float(t.weight or 0)
    top = max(sectors.items(), key=lambda kv: kv[1]) if sectors else None
    return {
        "positions": len(held),
        "sector_weights": {k: round(v, 4) for k, v in sorted(sectors.items())},
        "largest_sector": top[0] if top else None,
        "largest_sector_weight": round(top[1], 4) if top else None,
        "largest_position": max((float(t.weight or 0) for t in held), default=0.0),
    }


async def behavior_review(ctx: JobContext) -> dict[str, Any]:
    """The pathology monitor (round trips, concentration, herding, losing streaks...) as research input."""
    from quantpulse.brain.behavior import monitor

    rep = await monitor(ctx.db, ctx.settings, ctx.now())
    for f in rep.get("findings", [])[:5]:
        subject = str(f.get("symbol") or f.get("subject") or "")
        if f.get("check") == "repeated_thesis_losses" and subject:
            ctx.ask("trade_review", f"Why does {subject} keep losing?", {"focus": subject})
    return {"headline": rep.get("headline"), "findings": _brief(rep.get("findings", []))}


# ================================================================================================ RESEARCH / TEST
def _panel_ic(
    features: dict[str, pd.DataFrame], close: pd.DataFrame, bench: pd.Series, horizon: int = 5
) -> dict[str, dict[str, Any]]:
    """Cross-sectional rank IC of each feature against the next ``horizon`` sessions' returns, on dates
    ``horizon`` apart (non-overlapping), split in halves (in-sample / out-of-sample) and by volatility regime."""
    fwd = close.shift(-horizon) / close - 1
    bvol = bench.pct_change(fill_method=None).rolling(21).std()
    high = bvol > bvol.median()
    dates = list(close.index[252:-horizon:horizon]) if len(close.index) > 252 + horizon else []
    out: dict[str, dict[str, Any]] = {}
    for name, frame in features.items():
        rows: list[tuple[pd.Timestamp, float, bool]] = []
        for d in dates:
            x, y = frame.loc[d], fwd.loc[d]
            ok = x.notna() & y.notna()
            if int(ok.sum()) < 20:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):  # a constant day: no IC (NaN, skipped)
                ic = x[ok].rank().corr(y[ok].rank())
            if ic == ic:
                rows.append((d, float(ic), bool(high.get(d, False))))
        if len(rows) < 3:
            continue
        half = len(rows) // 2
        parts = {
            "all": rows,
            "first_half": rows[:half],
            "second_half": rows[half:],
            "high_vol": [r for r in rows if r[2]],
            "low_vol": [r for r in rows if not r[2]],
        }
        stats: dict[str, Any] = {}
        for label, part in parts.items():
            mean, t, p = t_test_mean([r[1] for r in part])
            stats[label] = {"n": len(part), "mean_ic": mean, "t": t, "p": p}
        stats["start"], stats["end"] = rows[0][0].date(), rows[-1][0].date()
        stats["series"] = [(r[0], r[1]) for r in rows]  # (date, IC): the forward record is read from here
        out[name] = stats
    return out


async def feature_research(ctx: JobContext) -> dict[str, Any]:
    """Which of the available features rank next-week returns? Tested out of sample and by volatility regime,
    with a false-discovery control across features; promising ones become feature hypotheses."""
    from quantpulse.domain.features import compute_features

    try:
        panel, meta = await ctx.brain.lab.panel()
    except Exception as exc:
        return {"skipped": f"no price panel ({type(exc).__name__}: {exc})"[:200]}
    stats = await asyncio.to_thread(lambda: _panel_ic(compute_features(panel), panel.close, panel.benchmark))
    names = sorted(stats)
    q_all = perf.benjamini_hochberg([stats[n]["all"]["p"] for n in names])
    limits = [
        "today's liquid universe (survivorship bias)",
        "next-5-session horizon only; costs ignored",
        f"{len(names)} features tested at once (false-discovery adjusted)",
    ]
    promising = []
    for name, q in zip(names, q_all, strict=True):
        st = stats[name]
        a = st["all"]
        await ctx.learn(
            Finding(
                topic=f"feature:{name}",
                claim=f"{name} ranks next-week returns (positive rank IC)",
                sample_size=a["n"],
                min_sample=ctx.min_sample,
                regime="all",
                benchmark="no predictive power (IC 0)",
                method="rank IC on non-overlapping dates, t-test, Benjamini-Hochberg across features",
                p_value=q,
                effect=a["mean_ic"],
                period_start=st["start"],
                period_end=st["end"],
                statistics={k: st[k] for k in ("all", "first_half", "second_half", "high_vol", "low_vol")},
                limitations=limits,
            )
        )
        hv, lv = st["high_vol"], st["low_vol"]
        if hv["n"] >= 3 and lv["n"] >= 3 and hv["mean_ic"] is not None and lv["mean_ic"] is not None:
            diff = hv["mean_ic"] - lv["mean_ic"]
            se = math.sqrt(sum(((x["mean_ic"] / x["t"]) ** 2 if x["t"] else 0.0) for x in (hv, lv)))
            await ctx.learn(
                Finding(
                    topic=f"feature:{name}:high_vol",
                    claim=f"{name} works better in high-volatility regimes",
                    sample_size=min(hv["n"], lv["n"]),
                    min_sample=ctx.min_sample,
                    regime="vol:high vs vol:low",
                    benchmark="the same feature in low volatility",
                    method="difference of mean ICs (z-test)",
                    p_value=p_two_sided(diff / se) if se > 0 else None,
                    effect=diff,
                    period_start=st["start"],
                    period_end=st["end"],
                    statistics={"high_vol": hv, "low_vol": lv},
                    limitations=limits,
                )
            )
        if q is not None and q < 0.05 and a["n"] >= ctx.min_sample and a["mean_ic"]:
            promising.append(name)
            await _feature_lifecycle(ctx, name, st, q)
    return {"features": len(names), "promising": promising, "data": _brief(meta)}


async def _feature_lifecycle(ctx: JobContext, name: str, st: dict[str, Any], q: float) -> None:
    sign = 1 if st["all"]["mean_ic"] > 0 else -1
    h = await ctx.lifecycle.discover(
        kind="feature",
        key=f"feature:{name}:{'+' if sign > 0 else '-'}",
        source="research",
        title=f"use {name} as a {'positive' if sign > 0 else 'negative'} ranking signal",
        detail={"feature": name, "sign": sign, "horizon_sessions": 5},
        now=ctx.now(),
        source_ref=str(ctx.job.get("id")),
    )

    def same(part: dict[str, Any], strict: bool) -> bool:
        m, p = part.get("mean_ic"), part.get("p")
        if m is None or part.get("n", 0) < 3 or m * sign <= 0:
            return False
        return (p is not None and p < 0.1) if strict else True

    checks = [
        ("HYPOTHESIS", True, {"full_sample_q": q, "mean_ic": st["all"]["mean_ic"], "n": st["all"]["n"]}),
        ("BACKTEST", same(st["first_half"], True), {"first_half": st["first_half"]}),
        ("WALK_FORWARD", same(st["second_half"], True), {"out_of_sample": st["second_half"]}),
        ("STRESS_TEST", same(st["high_vol"], False), {"high_volatility": st["high_vol"]}),
    ]
    h = await _advance_through(ctx, h, checks)
    if h["stage"] == "STRESS_TEST":  # tracked forward on new data before it can be evaluated
        last = str(st["end"])
        await ctx.lifecycle.advance(
            h["id"],
            passed=True,
            by="research",
            now=ctx.now(),
            evidence={"shadow_from": last, "days": PAPER_SHADOW_DAYS},
            wait_until=ctx.now() + timedelta(days=PAPER_SHADOW_DAYS),
        )
    elif (
        h["stage"] == "PAPER_SHADOW"
        and h["next_step_at"]
        and datetime.fromisoformat(h["next_step_at"]) <= ctx.now()
    ):
        # the evaluation reads only data that did not exist when the shadow began: a genuinely forward record
        shadow_from = pd.Timestamp(str((h["history"][-1].get("evidence") or {}).get("shadow_from")))
        forward = [ic for d, ic in st["series"] if d > shadow_from]
        if len(forward) >= 4:  # until then it keeps waiting (never rejected for lack of data)
            mean = sum(forward) / len(forward)
            await ctx.lifecycle.advance(
                h["id"],
                passed=mean * sign > 0,
                by="research",
                now=ctx.now(),
                evidence={
                    "forward_observations": len(forward),
                    "forward_mean_ic": mean,
                    "rule": "≥ 4 forward observations with the same sign",
                },
            )


async def _advance_through(
    ctx: JobContext, h: dict[str, Any], checks: list[tuple[str, bool, dict[str, Any]]]
) -> dict[str, Any]:
    """Advance one stage per passing check, in order; a failing check rejects; never past STRESS_TEST here."""
    for target, passed, evidence in checks:
        if h["stage"] in ("REJECTED", "PROTECTED_REVIEW", "RETIRED"):
            break
        if h.get("next_stage") != target:
            continue
        try:
            h = await ctx.lifecycle.advance(
                h["id"], passed=passed, evidence=evidence, by="research", now=ctx.now()
            )
        except LifecycleError:
            break
        if not passed:
            break
    return h


async def agent_combinations(ctx: JobContext) -> dict[str, Any]:
    """Would the consensus predict better without one of its agents? Leave-one-out on graded calls."""
    rows = await perf.graded(ctx.db)
    groups: dict[tuple[int, str], list[perf.Graded]] = defaultdict(list)
    for r in rows:
        if r.cycle_id is not None:
            groups[(r.cycle_id, r.subject)].append(r)
    outcomes: list[tuple[datetime, bool, dict[str, float], int, bool]] = []
    for members in groups.values():
        consensus = next((r for r in members if r.source == "consensus"), None)
        agents = [r for r in members if r.source != "consensus"]
        if consensus is None or len(agents) < 2:
            continue
        truth = 1 if consensus.relative > 0 else -1
        votes = {r.source: r.score * r.confidence for r in agents}
        high = (consensus.market_vol or 0) > 0.2
        outcomes.append((consensus.made_at, consensus.hit, votes, truth, high))
    outcomes.sort(key=lambda o: o[0])
    names = sorted({a for o in outcomes for a in o[2]})
    results: dict[str, Any] = {}
    for agent in names:
        b = c = 0  # b: full right, without wrong; c: full wrong, without right
        for _, full_hit, votes, truth, _ in outcomes:
            if agent not in votes:
                continue
            rest = sum(v for k, v in votes.items() if k != agent)
            loo_hit = (1 if rest > 0 else -1) == truth if rest else False
            b += int(full_hit and not loo_hit)
            c += int((not full_hit) and loo_hit)
        p = p_two_sided((c - b) / math.sqrt(b + c)) if b + c > 0 else None
        start, end = _period([o[0] for o in outcomes])
        await ctx.learn(
            Finding(
                topic=f"combination:without:{agent}",
                claim=f"the consensus predicts better without {agent}",
                sample_size=b + c,
                min_sample=ctx.min_sample,
                regime="all",
                benchmark="the full consensus",
                method="McNemar test on discordant calls (leave-one-out)",
                p_value=p,
                effect=float(c - b),
                period_start=start,
                period_end=end,
                statistics={"full_right_without_wrong": b, "full_wrong_without_right": c},
                limitations=[
                    "the leave-one-out vote is unweighted by measured reliability",
                    "calls on the same day are not independent",
                    "no agent is removed by this: a person decides",
                ],
            )
        )
        results[agent] = {"discordant": b + c, "net_gain_without": c - b, "p": p}
        if p is not None and p < 0.05 and c > b and b + c >= ctx.min_sample:
            h = await ctx.lifecycle.discover(
                kind="agent_combination",
                key=f"combination:without:{agent}",
                source="research",
                title=f"leave {agent} out of the consensus",
                detail={"agent": agent, "b": b, "c": c},
                now=ctx.now(),
                source_ref=str(ctx.job.get("id")),
            )
            half = len(outcomes) // 2

            def part(
                sel: Sequence[tuple[datetime, bool, dict[str, float], int, bool]], agent: str = agent
            ) -> dict[str, int]:
                bb = cc = 0
                for _, fh, votes, truth, _ in sel:
                    if agent in votes:
                        rest = sum(v for k, v in votes.items() if k != agent)
                        lh = (1 if rest > 0 else -1) == truth if rest else False
                        bb += int(fh and not lh)
                        cc += int((not fh) and lh)
                return {"b": bb, "c": cc}

            first, second = part(outcomes[:half]), part(outcomes[half:])
            stress = part([o for o in outcomes if o[4]])
            await _advance_through(
                ctx,
                h,
                [
                    ("HYPOTHESIS", True, {"b": b, "c": c, "p": p}),
                    ("BACKTEST", first["c"] > first["b"], {"first_half": first}),
                    ("WALK_FORWARD", second["c"] > second["b"], {"second_half": second}),
                    ("STRESS_TEST", stress["c"] >= stress["b"], {"high_volatility": stress}),
                ],
            )
    return {"groups": len(outcomes), "agents": results}


async def agent_redundancy(ctx: JobContext) -> dict[str, Any]:
    """Are two agents effectively measuring the same factor? Correlation of their scores on the same calls."""
    since = ctx.now() - timedelta(days=120)
    async with ctx.db.session() as s:
        rows = list(
            (
                await s.execute(
                    select(
                        BrainOpinionRow.cycle_id,
                        BrainOpinionRow.subject,
                        BrainOpinionRow.agent_id,
                        BrainOpinionRow.score,
                        BrainOpinionRow.created_at,
                    ).where(BrainOpinionRow.created_at >= since)
                )
            ).all()
        )
    table: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    times = [r[4] for r in rows]
    for cycle_id, subject, agent, score, _ in rows:
        table[agent][(cycle_id, subject)] = float(score)
    agents = sorted(table)
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            common = sorted(set(table[a]) & set(table[b]))
            if len(common) < 10:
                continue
            x = np.array([table[a][k] for k in common])
            y = np.array([table[b][k] for k in common])
            if x.std() < 1e-12 or y.std() < 1e-12:  # a constant view has no correlation
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                r = float(np.corrcoef(x, y)[0, 1])
            if math.isfinite(r):
                pairs.append({"a": a, "b": b, "n": len(common), "r": r})
    pairs.sort(key=lambda p: -abs(p["r"]))
    start, end = _period(times)
    for p in pairs[:10]:
        n, r = p["n"], max(min(abs(p["r"]), 0.9999), 0.0)
        z = (math.atanh(r) - math.atanh(0.9)) * math.sqrt(max(n - 3, 1))
        await ctx.learn(
            Finding(
                topic=f"redundancy:{p['a']}~{p['b']}",
                claim=f"{p['a']} and {p['b']} measure the same thing (|r| > 0.9)",
                sample_size=n,
                min_sample=ctx.min_sample,
                regime="all",
                benchmark="a correlation of 0.9",
                method="Fisher z-test of |r| against 0.9",
                p_value=p_two_sided(z),
                effect=abs(p["r"]) - 0.9,
                period_start=start,
                period_end=end,
                statistics=p,
                limitations=[
                    "scores on the same cycle and subject; regimes mixed",
                    "correlated is not identical",
                ],
            )
        )
    return {"agents": len(agents), "pairs": pairs[:10]}


async def strategy_research(ctx: JobContext) -> dict[str, Any]:
    """The strategy lab: propose untried templates and new generated ideas (mutations of the best out-of-sample
    strategies, combinations of the research-backed features, exploration), validate several (backtest,
    walk-forward, random portfolios, stress, the deflated Sharpe over every strategy tried), keep paper (shadow)
    tracking up to date, and move each strategy's hypothesis along its stages."""
    from quantpulse.brain.lab.generator import research_features

    lab = ctx.brain.lab
    features = research_features(await ctx.ledger.learnings(current_only=True, limit=5000))
    proposed = await lab.propose(features=features)
    try:
        validated = await lab.validate_pending(
            limit=ctx.settings.brain_lab_validations_per_run, budget=STRATEGY_BUDGET
        )
    except Exception as exc:  # no price data: reported, the rest still runs
        validated = [{"error": f"{type(exc).__name__}: {exc}"[:200]}]
    paper = (
        await lab.paper_update() if await lab.strategies("paper") or await lab.strategies("promoted") else {}
    )
    moved = await sync_strategy_hypotheses(ctx)
    for v in validated:
        if "random_percentile" in v:
            rp = v.get("random_percentile")
            await ctx.learn(
                Finding(
                    topic=f"strategy:{v['key']}:random",
                    claim=f"{v['key']} beats random portfolios of the same universe",
                    sample_size=int(((v.get("walk_forward") or {}).get("oos") or {}).get("days") or 0),
                    min_sample=ctx.min_sample,
                    regime="all",
                    benchmark="random portfolios (same size, same universe)",
                    method="percentile among random portfolios",
                    p_value=(1 - rp / 100) if rp is not None else None,
                    effect=(rp - 50) if rp is not None else None,
                    statistics=_brief(v),
                    limitations=["today's liquid universe (survivorship bias)", "modelled costs and fills"],
                )
            )
    return {
        "proposed": len(proposed),
        "generated": [p["key"] for p in proposed if p.get("source") == "generated"],
        "research_features": features,
        "validated": _brief(validated),
        "paper": _brief(paper),
        "lifecycle": moved,
    }


STRATEGY_BUDGET = timedelta(
    minutes=35
)  # no new validation starts after this (the job's timeout is 60 minutes)

BACKTEST_GATES = (
    "better than random portfolios",
    "edge survives realistic costs",
    "capacity covers the paper book",
    "robust to nearby parameters",
)
WALK_FORWARD_GATES = (
    "enough out-of-sample history",
    "positive out-of-sample Sharpe",
    "adds value out of sample (active Sharpe vs equal weight > 0)",
    "survives out of sample",
    "beats the equal-weight baseline in most folds",
    "not explained by trying many variants (deflated Sharpe)",
)
STRESS_GATES = (
    "survives doubled costs",
    "survives trading two sessions late",
    "no disproportionate losses in stress windows",
)


async def sync_strategy_hypotheses(ctx: JobContext) -> list[str]:
    """Each lab strategy is a hypothesis; its validation gates and paper record decide its stages."""
    lab = ctx.brain.lab
    moved: list[str] = []
    for st in await lab.strategies():
        h = await ctx.lifecycle.discover(
            kind="strategy",
            key=f"strategy:{st['key']}",
            title=f"strategy {st['name']}",
            source="lab",
            detail={"strategy": st["key"], "spec": st["spec"]},
            now=ctx.now(),
            source_ref=st["key"],
        )
        val = st.get("validation") or {}
        gates = {g["gate"]: g for g in val.get("gates") or []}

        def group(names: Sequence[str], gates: dict[str, Any] = gates) -> tuple[bool, dict[str, Any]]:
            present = {n: gates[n] for n in names if n in gates}
            return bool(present) and all(g["passed"] for g in present.values()), {"gates": present}

        checks: list[tuple[str, bool, dict[str, Any]]] = [("HYPOTHESIS", True, {"spec": st["spec"]})]
        if gates:
            for target, names in (
                ("BACKTEST", BACKTEST_GATES),
                ("WALK_FORWARD", WALK_FORWARD_GATES),
                ("STRESS_TEST", STRESS_GATES),
            ):
                ok, ev = group(names)
                checks.append((target, ok, {**ev, "validated_at": val.get("at")}))
        before = h["stage"]
        h = await _advance_through(ctx, h, checks)
        if h["stage"] == "STRESS_TEST" and st["status"] == "validated":
            # shadow tracking: the lab records the portfolio it would hold — no order is ever placed
            await lab.set_status(st["strategy_id"], st["version"], "paper", by="research")
            h = await ctx.lifecycle.advance(
                h["id"], passed=True, by="research", now=ctx.now(), evidence={"lab_paper_tracking": "started"}
            )
        if h["stage"] == "PAPER_SHADOW" and st["status"] in ("paper", "promoted"):
            async with ctx.db.session() as s:
                from quantpulse.db.models import BrainStrategyRow

                row = await s.scalar(
                    select(BrainStrategyRow).where(
                        BrainStrategyRow.strategy_id == st["strategy_id"],
                        BrainStrategyRow.version == st["version"],
                    )
                )
                ok, why = lab.promotable(row) if row is not None else (False, "missing")
            sessions = int((st.get("paper") or {}).get("sessions") or 0)
            if ok:
                h = await ctx.lifecycle.advance(
                    h["id"],
                    passed=True,
                    by="research",
                    now=ctx.now(),
                    evidence={"paper": st.get("paper"), "lab": why},
                )
            elif (
                sessions >= 2 * ctx.settings.brain_lab_paper_days
            ):  # long enough, still short of the benchmark
                h = await ctx.lifecycle.advance(
                    h["id"],
                    passed=False,
                    by="research",
                    now=ctx.now(),
                    evidence={"paper": st.get("paper"), "lab": why},
                )
        if h["stage"] != before:
            moved.append(f"{st['key']}: {before} → {h['stage']}")
    return moved


async def improvement_review(ctx: JobContext) -> dict[str, Any]:
    """The improvement engine's evidence-based proposals, entered into the lifecycle (protected ones parked)."""
    found = await ctx.brain.improvements.review(ctx.now())
    entered = 0
    for f in found:
        key = f"improvement:{f['kind']}:{f['target']}:{f['title']}"[:160]
        h = await ctx.lifecycle.discover(
            kind="process",
            key=key,
            title=f["title"],
            source="improvement",
            detail={"evidence": f["evidence"], "proposal": f["proposal"]},
            now=ctx.now(),
            source_ref=f["kind"],
        )
        if h["stage"] == "DISCOVERED":
            await ctx.lifecycle.advance(
                h["id"],
                passed=True,
                by="research",
                now=ctx.now(),
                evidence={
                    "evidence": _brief(f["evidence"]),
                    "next": "a person designs the test for a process change",
                },
            )
        entered += 1
    return {"proposals": len(found), "in_lifecycle": entered}


# ================================================================================================ LEARN
async def memory_maintenance(ctx: JobContext) -> dict[str, Any]:
    """Long-term memory: expire what is stale, and keep what the graded ideas taught."""
    purged = await ctx.brain.memory.purge_expired(ctx.now())
    kept = await ideas.remember(ctx.db, ctx.brain.memory, ctx.now(), ctx.min_sample)
    return {"purged": purged, "remembered": kept}


# ================================================================================================ PREPARE
async def data_quality(ctx: JobContext) -> dict[str, Any]:
    """Data availability: what each quote vendor's subscription refuses, and recent data-blocked cycles."""
    feeds = ctx.market.feed_status() if ctx.market is not None else []
    async with ctx.db.session() as s:
        days = list(
            (await s.scalars(select(BrainSessionRow).order_by(BrainSessionRow.day.desc()).limit(10))).all()
        )
    blocked = {d.day.isoformat(): {"cycles": d.cycles, "data_blocked": d.data_blocked_cycles} for d in days}
    refused = [f for f in feeds if f.get("stock_feed_error") or f.get("history_feed_refused")]
    return {"feeds": _brief(feeds), "refused": _brief(refused), "recent_sessions": blocked}


async def system_integrity(ctx: JobContext) -> dict[str, Any]:
    """The database's own consistency: the schema, cycles left running, executions without a decision."""
    from quantpulse.db import migrate

    issues: list[str] = []
    current = await asyncio.to_thread(migrate.current_revision, ctx.settings.database_url)
    head = migrate.head_revision()
    if current != head:
        issues.append(f"schema at {current}, head is {head}")
    async with ctx.db.session() as s:
        stuck = list(
            (
                await s.scalars(
                    select(BrainCycleRow.id).where(
                        BrainCycleRow.status == "running",
                        BrainCycleRow.started_at < ctx.now() - timedelta(hours=2),
                    )
                )
            ).all()
        )
        orphans = list(
            (
                await s.scalars(
                    select(BrainExecutionRow.id).where(
                        BrainExecutionRow.decision_id.is_(None), BrainExecutionRow.brain_cycle_id.is_(None)
                    )
                )
            ).all()
        )
    if stuck:
        issues.append(f"{len(stuck)} cycle(s) still marked running after 2 hours: {stuck[:5]}")
    if orphans:
        issues.append(f"{len(orphans)} execution record(s) without a decision or cycle")
    unapproved = await ctx.lifecycle.unapproved_in_production()
    if unapproved:
        issues.append(f"in PRODUCTION without a person's promotion: {unapproved}")
    return {"ok": not issues, "issues": issues, "schema": current}


async def reconcile_state(ctx: JobContext) -> dict[str, Any]:
    """Bring the order and position record in step with Alpaca (a read of the paper account; never an order)."""
    trading = ctx.brain.trading
    if not (ctx.settings.brain_owns_account and trading.broker.configured()):
        return {"skipped": "the Brain does not own the Alpaca paper account here"}
    rec = await trading.reconcile("research (market closed)")
    await ctx.brain.ledger.refresh()
    return {"positions": rec.positions, "open_orders": rec.open_orders, "updated": rec.orders_updated}


async def upcoming_events(ctx: JobContext) -> dict[str, Any]:
    """Earnings releases expected soon for what is held and watched (from the filings history available)."""
    if ctx.reference is None:
        return {"skipped": "no reference data service"}
    async with ctx.db.session() as s:
        held = [
            t.symbol
            for t in (await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "open"))).all()
        ]
    watch = (await ctx.brain.store.get_state(WATCHLIST_KEY) or {}).get("symbols", [])
    symbols = sorted({*held, *[w["symbol"] for w in watch]})[:60]
    found = await ctx.reference.events_many(symbols) if symbols else {}
    today = ctx.now().astimezone(NEW_YORK).date()
    soon: list[dict[str, Any]] = []
    for sym, res in found.items():
        dates = [d.astimezone(NEW_YORK).date() for d in res.value.earnings]
        if len(dates) >= 2:  # an estimate from the usual quarterly rhythm (no calendar provider)
            gap = (dates[-1] - dates[-2]).days
            nxt = dates[-1] + timedelta(days=gap)
            if 0 <= (nxt - today).days <= 10:
                soon.append(
                    {"symbol": sym, "estimated_next": nxt.isoformat(), "basis": "previous filing rhythm"}
                )
    await ctx.brain.store.set_state(
        EVENTS_KEY, {"at": ctx.now().isoformat(), "soon": soon, "checked": len(symbols)}, ctx.now()
    )
    return {
        "checked": len(symbols),
        "soon": soon,
        "limitation": "no earnings-calendar provider: next dates are estimated from past filings",
    }


async def watchlist_prep(ctx: JobContext) -> dict[str, Any]:
    """The next session's watchlist and research priorities: what is held, the strongest recent ideas, what
    reports soon, and the most valuable open questions."""
    async with ctx.db.session() as s:
        held = list((await s.scalars(select(BrainThesisRow).where(BrainThesisRow.status == "open"))).all())
        since = (ctx.now() - timedelta(days=5)).astimezone(NEW_YORK).date()
        recent = list(
            (
                await s.scalars(
                    select(BrainOpportunityOutcomeRow)
                    .where(BrainOpportunityOutcomeRow.day >= since)
                    .order_by(BrainOpportunityOutcomeRow.strength.desc())
                    .limit(30)
                )
            ).all()
        )
    events = {e["symbol"]: e for e in (await ctx.brain.store.get_state(EVENTS_KEY) or {}).get("soon", [])}
    picks: dict[str, list[str]] = defaultdict(list)
    for t in held:
        picks[t.symbol].append("held")
    for r in recent:
        if len(picks) >= 25 and r.symbol not in picks:
            break
        picks[r.symbol].append(f"{r.kind} idea ({'taken' if r.taken else r.reason})")
    for sym in events:
        picks[sym].append("earnings expected soon")
    symbols = [{"symbol": k, "why": sorted(set(v))} for k, v in sorted(picks.items())]
    day = ctx.now().astimezone(NEW_YORK).date().isoformat()
    await ctx.brain.store.set_state(
        WATCHLIST_KEY, {"day": day, "at": ctx.now().isoformat(), "symbols": symbols}, ctx.now()
    )
    return {"symbols": len(symbols), "watchlist": symbols[:25]}
