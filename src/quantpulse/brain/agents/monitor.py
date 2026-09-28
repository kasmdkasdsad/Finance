"""Monitoring agents — context for the decision step, never votes on direction:

* **Position monitor** — each position on the Alpaca paper account the Brain owns, against its thesis:
  distance to the stop, sessions held against the horizon, return against the benchmark since entry, and
  size against the position limit. Alerts are posted for the decision step and the dashboard; the thesis
  check itself (broken → exit) runs after the consensus.
* **Execution quality** — the Brain's own recent fills from the execution ledger: fill rate, slippage
  against the decision's price, cost against the quote as the order left, grades, quote age, spread and
  latency. With fewer than ``MIN_FILLS`` fills it says *unproven* and nothing more.
* **Learning** — the Brain's measured record (the scorecard): what is established and what is not yet —
  prediction accuracy, calibration, decision quality, agents with a verdict. It never calls anything
  reliable without the sample to show it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

from ..context import BrainContext
from ..theses import sessions_between
from ..types import MARKET, PORTFOLIO, AgentFamily, AgentSpec, DataState, Evidence, Opinion
from .base import Agent, Role, symbols_only
from .common import opinion

NEAR_STOP = 0.02  # within 2% of the stop
LAGGING = -0.05  # 5% behind the benchmark since entry
MIN_FILLS = 10  # fills before execution quality is judged at all


class PositionMonitorAgent(Agent):
    spec = AgentSpec(
        id="position_monitor",
        source="positions",
        failure="no position alerts; the thesis checks still run and the risk engine still checks every limit",
        name="Position monitor",
        description="Each position against its thesis: distance to the stop, time held against the horizon, "
        "return against the benchmark, size against the limit.",
        family=AgentFamily.PORTFOLIO,
        capabilities=("stop_distance", "horizon", "relative_performance", "position_size"),
        inputs=("portfolio", "working_memory"),
        outputs=("opinion", "position_alerts"),
        subjects=("symbol",),
        priority=35,
        horizon_days=0,
    )
    role: ClassVar[Role] = "context"

    def unavailable(self, ctx: BrainContext) -> str | None:
        if "theses" not in ctx.working.facts:
            return "position theses are kept only while the Brain owns the Alpaca paper account"
        return None

    def subjects(self, ctx: BrainContext) -> list[str]:
        return list(ctx.held)

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        theses = ctx.working.facts.get("theses") or {}
        alerts: dict[str, list[str]] = {}
        out: list[Opinion] = []
        for s in symbols_only(subjects):
            t = theses.get(s)
            if t is None:
                out.append(
                    self.abstain(s, "no thesis for this position (not opened or adopted by the Brain)")
                )
                continue
            ev: list[Evidence] = []
            mine: list[str] = []
            price, stop = t.get("last_price"), t.get("stop_price")
            if price and stop:
                gap = price / stop - 1
                ev.append(Evidence("stop_distance", round(gap, 4), f"{gap:+.1%} above its stop ${stop:,.2f}"))
                if gap <= NEAR_STOP:
                    mine.append(f"within {NEAR_STOP:.0%} of its stop")
            held = sessions_between(datetime.fromisoformat(t["opened_at"]), ctx.as_of)
            horizon = t.get("horizon_days")
            ev.append(
                Evidence(
                    "sessions_held", held, f"held {held} session(s) of a {horizon or '?'}-session horizon"
                )
            )
            if horizon and held > horizon:
                mine.append(f"past its {horizon}-session horizon")
            rel = t.get("relative_return")
            if rel is not None:
                ev.append(Evidence("relative_return", rel, f"{rel:+.1%} against the benchmark since entry"))
                if rel <= LAGGING:
                    mine.append(f"{rel:+.1%} behind the benchmark")
            w = ctx.portfolio.weight(s)
            ev.append(
                Evidence(
                    "weight", round(w, 4), f"{w:.1%} of equity (limit {ctx.limits.max_position_pct:.0%})"
                )
            )
            if w > ctx.limits.max_position_pct:
                mine.append(f"above the {ctx.limits.max_position_pct:.0%} position limit")
            if mine:
                alerts[s] = mine
            out.append(
                opinion(
                    self,
                    s,
                    0.0,
                    1.0,
                    f"{s}: " + ("; ".join(mine) if mine else "on track against its thesis"),
                    ev,
                    quality=DataState.LIVE,
                    used=["position theses", "the Alpaca paper account"],
                    meta={"alerts": mine, "thesis_origin": t.get("origin"), "gradeable": False},
                    directional=False,
                )
            )
        ctx.working.post("position_alerts", alerts)
        return out


class ExecutionQualityAgent(Agent):
    spec = AgentSpec(
        id="execution_quality",
        source="fills",
        failure="no execution context; orders still go through the risk engine and the order manager",
        name="Execution quality",
        description="The Brain's own recent fills: fill rate, slippage against the decision's price, cost "
        "against the quote as the order left, grades, quote age, spread and latency.",
        family=AgentFamily.META,
        capabilities=("fill_rate", "slippage", "spread_cost", "latency"),
        inputs=("execution_quality",),
        outputs=("opinion", "execution_quality"),
        subjects=("portfolio",),
        priority=70,
        horizon_days=0,
    )
    role: ClassVar[Role] = "context"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        q = ctx.execution_quality or {}
        filled = int(q.get("filled") or 0)
        ev = [
            Evidence("orders_sent", q.get("sent", 0), f"{q.get('sent', 0)} order(s) sent in 30 days"),
            Evidence("fill_rate", q.get("fill_rate"), f"fill rate {q.get('fill_rate')}"),
            Evidence(
                "slippage_bps", q.get("slippage_bps_mean"), f"mean slippage {q.get('slippage_bps_mean')}bp"
            ),
            Evidence(
                "cost_vs_quote_bps",
                q.get("cost_vs_quote_bps_mean"),
                f"mean cost against the quote {q.get('cost_vs_quote_bps_mean')}bp",
            ),
            Evidence("grades", str(q.get("grades") or {}), f"grades {q.get('grades') or {}}"),
        ]
        if filled < MIN_FILLS:
            verdict = f"unproven: {filled} fill(s) measured (needs {MIN_FILLS})"
        else:
            cost = q.get("cost_vs_quote_bps_mean")
            spread = q.get("spread_bps_median")
            verdict = (
                "costly: fills pay well beyond half the spread"
                if cost is not None and spread is not None and cost > spread / 2 + 10
                else "in line with the spreads paid"
            )
        ctx.working.post("execution_quality", {**q, "verdict": verdict})
        return [
            opinion(
                self,
                PORTFOLIO,
                0.0,
                1.0,
                f"execution: {verdict}",
                ev,
                quality=DataState.LIVE,
                used=["the execution ledger"],
                meta={"verdict": verdict, "gradeable": False},
                directional=False,
            )
        ]


class LearningAgent(Agent):
    spec = AgentSpec(
        id="learning",
        source="record",
        failure="no track-record context; nothing about weights or thresholds changes either way",
        name="Learning",
        description="What the Brain's measured record says — and what it does not yet: prediction accuracy, "
        "calibration, decision quality, agents with a verdict.",
        family=AgentFamily.META,
        capabilities=("track_record", "calibration", "decision_quality"),
        inputs=("track_record",),
        outputs=("opinion", "track_record"),
        subjects=("market",),
        priority=80,
        horizon_days=0,
    )
    role: ClassVar[Role] = "context"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        card = ctx.track_record or {}
        acc = card.get("prediction_accuracy") or {}
        dec = card.get("decision_quality") or {}
        agents = card.get("agent_reliability") or {}
        established = agents.get("established") or []
        ev = [
            Evidence("graded_predictions", acc.get("graded", 0), f"{acc.get('graded', 0)} graded consensus calls: {acc.get('status', 'unproven')}"),
            Evidence("hit_rate", acc.get("hit_rate"), f"hit rate {acc.get('hit_rate')} (95% interval {acc.get('ci95')})"),
            Evidence("decisions_judged", dec.get("judged", 0), f"{dec.get('judged', 0)} decisions judged: {dec.get('mix') or {}}"),
            Evidence("agents_with_a_verdict", len(established), ", ".join(established) or "none yet"),
        ]  # fmt: skip
        status = acc.get("status") or "unproven"
        thesis = (
            f"the record is {status}: {acc.get('graded', 0)} graded calls, {len(established)} agent(s) with a "
            "verdict — weights stay at their defaults until the evidence is there"
            if status != "measured"
            else f"measured record: hit rate {acc.get('hit_rate')} over {acc.get('graded')} graded calls"
        )
        ctx.working.post("track_record", {"status": status, "established_agents": established})
        return [
            opinion(
                self,
                MARKET,
                0.0,
                1.0,
                thesis,
                ev,
                quality=DataState.LIVE,
                used=["graded predictions", "reflections", "agent track records"],
                meta={"gradeable": False},
                directional=False,
            )
        ]
