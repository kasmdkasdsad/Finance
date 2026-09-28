"""Automatic reviews: the Brain looks back at its own results, daily and weekly, and writes it down.

* **Daily** (after the close): the day against the benchmark and the replaced strategy (shadow); every cycle's
  "why traded / why not"; the trades and the positions closed; the ideas considered and why they were not
  taken; what was graded (calls, decisions, ideas); execution quality; data problems; behaviour findings;
  traceability gaps — and the day's **lessons**.
* **Weekly** (after the week's last session): the week's performance, turnover and execution; the learning
  report's state and what changed since last week (agent verdicts, calibration); the rejection reasons'
  record; behaviour over the week; the daily lessons collected — and **proposals**.

Lessons are structured and deterministic — ``{topic, lesson, evidence, sample, strength}`` — and every one
says how much it rests on: one day's events are *tentative* (``one case``); only a pattern with enough
independent observations is *established*. Proposals go through the improvement engine
(:meth:`~quantpulse.brain.improvement.ImprovementEngine.submit`): one that would touch a protected control
(loss, position and order limits, kill switches, paper-only settings, data freshness and spread requirements,
account and environment checks) is recorded as ``protected_review`` for a person, never as a suggestion.
Nothing a review writes changes the Brain's behaviour: rules, thresholds and limits move only by a person's
decision, and weights only through the existing reliability rule.

Reviews are append-only (``brain_reviews``) and also kept in long-term memory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, next_trading_day
from quantpulse.db.models import (
    BrainCycleRow,
    BrainDecisionRow,
    BrainExecutionRow,
    BrainOpportunityOutcomeRow,
    BrainPredictionRow,
    BrainReflectionRow,
    BrainReviewRow,
    BrainSessionRow,
    BrainThesisRow,
)
from quantpulse.db.session import Database

from .audit import completeness
from .behavior import monitor
from .improvement import ImprovementEngine, _p
from .learning_report import build as learning_report
from .memory import LONG_TERM, MemoryStore
from .opportunity_outcomes import PROTECTED_REASONS, REASONS
from .opportunity_outcomes import report as ideas_report
from .scorecard import execution_quality
from .trade_lessons import ended_by

# rejection reasons tied to a protected control: a finding about them is recorded for a person's review
REASON_CONTROL = {
    "data_quality": "QP_TRADING_MAX_QUOTE_AGE_SECONDS",
    "data_veto": "QP_TRADING_MAX_SPREAD_BPS",
    "risk_engine": "QP_TRADING_MAX_ limits",
    "entry_halt": "QP_TRADING_DAILY_LOSS limit",
    "account": "QP_ALPACA_PAPER account checks",
}
# the Brain's own (unprotected) settings behind the other reasons, for a proposal's wording
REASON_SETTING = {
    "low_confidence": "QP_BRAIN_MIN_CONFIDENCE",
    "slot_limit": "QP_BRAIN_MAX_NEW_POSITIONS_PER_CYCLE",
    "earnings": "QP_BRAIN_EARNINGS_CAUTION_DAYS",
    "focus_budget": "QP_BRAIN_FOCUS_CANDIDATES / QP_BRAIN_MAX_OPPORTUNITIES",
}


def _lesson(
    topic: str, lesson: str, sample: int, strength: str = "tentative", **evidence: Any
) -> dict[str, Any]:
    return {"topic": topic, "lesson": lesson, "sample": sample, "strength": strength, "evidence": evidence}


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(0, 0), NEW_YORK)
    return start, start + timedelta(days=1)


def last_session_of_week(day: date) -> bool:
    return is_trading_day(day) and next_trading_day(day).isocalendar()[:2] != day.isocalendar()[:2]


class Reviewer:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        clock: Clock,
        memory: MemoryStore,
        improvements: ImprovementEngine,
    ) -> None:
        self._db = db
        self._s = settings
        self._clock = clock
        self._memory = memory
        self._improvements = improvements

    # ------------------------------------------------------------------ daily
    async def daily(self, day: date | None = None) -> dict[str, Any]:
        now = self._clock.now()
        day = day or now.astimezone(NEW_YORK).date()
        start, end = _day_bounds(day)
        async with self._db.session() as s:
            session = (await s.scalars(select(BrainSessionRow).where(BrainSessionRow.day == day))).first()
            cycles = (
                await s.scalars(
                    select(BrainCycleRow).where(
                        BrainCycleRow.started_at >= start, BrainCycleRow.started_at < end
                    )
                )
            ).all()
            decisions = (
                await s.scalars(
                    select(BrainDecisionRow).where(
                        BrainDecisionRow.created_at >= start, BrainDecisionRow.created_at < end
                    )
                )
            ).all()
            closed = (
                await s.scalars(
                    select(BrainThesisRow).where(
                        BrainThesisRow.closed_at >= start, BrainThesisRow.closed_at < end
                    )
                )
            ).all()
            ideas = (
                await s.scalars(
                    select(BrainOpportunityOutcomeRow).where(BrainOpportunityOutcomeRow.day == day)
                )
            ).all()
            ideas_graded = (
                await s.scalars(
                    select(BrainOpportunityOutcomeRow).where(
                        BrainOpportunityOutcomeRow.evaluated_at >= start,
                        BrainOpportunityOutcomeRow.evaluated_at < end,
                    )
                )
            ).all()
            graded = (
                await s.scalars(
                    select(BrainPredictionRow).where(
                        BrainPredictionRow.evaluated_at >= start, BrainPredictionRow.evaluated_at < end
                    )
                )
            ).all()
            reflections = (
                await s.scalars(
                    select(BrainReflectionRow).where(
                        BrainReflectionRow.created_at >= start, BrainReflectionRow.created_at < end
                    )
                )
            ).all()
            executions = (
                await s.scalars(
                    select(BrainExecutionRow).where(
                        BrainExecutionRow.decided_at >= start, BrainExecutionRow.decided_at < end
                    )
                )
            ).all()
        done = [c for c in cycles if c.status == "completed"]
        outcomes = Counter(((c.summary or {}).get("decision") or {}).get("outcome", "unknown") for c in done)
        why_not: Counter[str] = Counter()
        for c in done:
            dec = (c.summary or {}).get("decision") or {}
            if dec.get("outcome") == "no_trade":
                for r in dec.get("reasons") or []:
                    why_not[str(r).split(":")[0][:60]] += 1
        sent = [d for d in decisions if (d.execution or {}).get("sent")]
        behaviour = await monitor(self._db, self._s, now)
        traces = await completeness(self._db, limit=100)
        shadow = ((session.close or {}).get("strategy_shadow") or {}).get("day_return") if session else None
        body: dict[str, Any] = {
            "day": {
                "owner": session.owner if session else None,
                "return": session.day_return if session else None,
                "benchmark": session.benchmark_return if session else None,
                "excess": (session.day_return - session.benchmark_return)
                if session and session.day_return is not None and session.benchmark_return is not None
                else None,
                "shadow": shadow,
                "orders_sent": session.orders_sent if session else len(sent),
                "cycles": len(cycles),
                "failed_cycles": sum(1 for c in cycles if c.status == "failed"),
                "data_blocked_cycles": session.data_blocked_cycles if session else None,
                "halts": session.halts if session else {},
            },
            "decisions": {
                "cycles_traded": outcomes.get("traded", 0),
                "cycles_no_trade": outcomes.get("no_trade", 0),
                "why_not": dict(why_not.most_common(8)),
            },
            "trades": [
                {
                    "subject": d.subject,
                    "action": d.action,
                    "quantity": d.quantity,
                    "status": d.status,
                    "why": ((d.rationale or {}).get("reasons") or [])[:2],
                }
                for d in sent
            ],
            "closed_positions": [
                {
                    "symbol": t.symbol,
                    "realized_pnl": t.realized_pnl,
                    "ended_by": ended_by(t.exit_reason),
                    "exit_reason": t.exit_reason,
                }
                for t in closed
            ],
            "ideas": {
                "considered": len(ideas),
                "taken": sum(1 for i in ideas if i.taken),
                "why_not": dict(Counter(i.reason for i in ideas if not i.taken).most_common(8)),
                "graded_today": dict(Counter(i.verdict for i in ideas_graded)),
            },
            "graded": {
                "calls": len(graded),
                "hits": sum(1 for p in graded if p.hit),
                "consensus_calls": sum(1 for p in graded if p.source_type == "consensus"),
                "consensus_hits": sum(1 for p in graded if p.source_type == "consensus" and p.hit),
                "decisions_reflected": dict(
                    Counter(r.category for r in reflections if r.subject_type == "decision")
                ),
            },
            "execution": {
                "orders": len(executions),
                "grades": dict(Counter(e.grade or "pending" for e in executions)),
                "mean_slippage_bps": _mean(
                    [e.slippage_bps for e in executions if e.slippage_bps is not None]
                ),
                "mean_cost_vs_quote_bps": _mean(
                    [e.cost_vs_quote_bps for e in executions if e.cost_vs_quote_bps is not None]
                ),
            },
            "behaviour": [f for f in behaviour["findings"] if f["severity"] != "info"],
            "traceability": {k: traces[k] for k in ("trades", "with_gaps", "gaps_by_stage", "headline")},
        }
        lessons = self._daily_lessons(body, reflections, closed, executions)
        headline = (
            f"{day}: {len(sent)} order(s) sent in {len(done)} cycle(s); "
            + (f"day {body['day']['return']:+.2%} vs benchmark {body['day']['benchmark']:+.2%}; "
               if body["day"]["return"] is not None and body["day"]["benchmark"] is not None else "")
            + f"{body['ideas']['considered']} idea(s) considered, {body['ideas']['taken']} taken; "
            + f"{len(lessons)} lesson(s)"
        )  # fmt: skip
        return await self._store("daily", day, day, headline, body, lessons, [])

    def _daily_lessons(
        self,
        body: dict[str, Any],
        reflections: Sequence[BrainReflectionRow],
        closed: Sequence[BrainThesisRow],
        executions: Sequence[BrainExecutionRow],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in reflections:
            if r.subject_type == "decision" and r.category in (
                "process_failure", "lucky", "block_saved_money", "block_cost_opportunity"
            ) and r.lessons:  # fmt: skip
                out.append(_lesson("decision", f"{r.category.replace('_', ' ')}: {r.lessons[0]}", 1,
                                   decision_id=r.subject_id))  # fmt: skip
        for t in closed:
            how = ended_by(t.exit_reason)
            pnl = t.realized_pnl
            out.append(_lesson("position", f"{t.symbol} closed ({how.replace('_', ' ')})"
                               + (f": ${pnl:+,.2f} realised" if pnl is not None else ""), 1, thesis_id=t.id))  # fmt: skip
        poor = [e for e in executions if e.grade == "poor"]
        if poor:
            out.append(_lesson("execution", f"{len(poor)} of {len(executions)} fill(s) cost well beyond half the spread",
                               len(executions), symbols=sorted({e.symbol for e in poor})))  # fmt: skip
        d = body["day"]
        if d["data_blocked_cycles"] and d["cycles"] and d["data_blocked_cycles"] * 2 >= d["cycles"]:
            out.append(_lesson("data", f"market data blocked new positions in {d['data_blocked_cycles']} of "
                               f"{d['cycles']} cycles", d["cycles"]))  # fmt: skip
        if d["failed_cycles"]:
            out.append(_lesson("operations", f"{d['failed_cycles']} cycle(s) failed", d["cycles"]))
        for f in body["behaviour"]:
            out.append(
                _lesson("behaviour", f["finding"], f["sample"], severity=f["severity"], code=f["code"])
            )
        if body["traceability"]["with_gaps"]:
            out.append(_lesson("record", body["traceability"]["headline"], body["traceability"]["trades"]))
        if not out and body["decisions"]["cycles_no_trade"] and not body["decisions"]["cycles_traded"]:
            top = next(iter(body["decisions"]["why_not"]), "no opportunity met every requirement")
            out.append(
                _lesson(
                    "no_trade", f"no trade all day — most often: {top}", body["decisions"]["cycles_no_trade"]
                )
            )
        return out

    # ------------------------------------------------------------------ weekly
    async def weekly(self, day: date | None = None) -> dict[str, Any]:
        now = self._clock.now()
        end = day or now.astimezone(NEW_YORK).date()
        start = end - timedelta(days=end.weekday())  # Monday
        since, _ = _day_bounds(start)

        async with self._db.session() as s:
            days = (
                await s.scalars(
                    select(BrainSessionRow)
                    .where(BrainSessionRow.day >= start, BrainSessionRow.day <= end)
                    .order_by(BrainSessionRow.day)
                )
            ).all()
            dailies = (
                await s.scalars(
                    select(BrainReviewRow).where(
                        BrainReviewRow.kind == "daily",
                        BrainReviewRow.period_end >= start,
                        BrainReviewRow.period_end <= end,
                    )
                )
            ).all()
            previous = (
                await s.scalars(
                    select(BrainReviewRow)
                    .where(BrainReviewRow.kind == "weekly", BrainReviewRow.period_end < start)
                    .order_by(BrainReviewRow.period_end.desc())
                )
            ).first()
        min_n = self._s.brain_min_reliability_observations
        report = await learning_report(self._db, self._s)
        snapshot = {
            "consensus_verdict": (report["consensus"]["record"] or {}).get("verdict"),
            "consensus_ece": report["consensus"]["calibration"].get("ece"),
            "agents": {a: v["record"]["verdict"] for a, v in report["agents"].items()},
        }
        before = ((previous.body if previous else {}) or {}).get("learning_snapshot") or {}
        changes = [
            f"{a}: {before['agents'][a]} → {v}"
            for a, v in snapshot["agents"].items()
            if a in (before.get("agents") or {}) and before["agents"][a] != v
        ]
        if before.get("consensus_verdict") and before["consensus_verdict"] != snapshot["consensus_verdict"]:
            changes.append(f"consensus: {before['consensus_verdict']} → {snapshot['consensus_verdict']}")
        ideas = await ideas_report(self._db, min_n)
        week_ideas = await ideas_report(self._db, min_n, since)
        behaviour = await monitor(self._db, self._s, now, days=7)
        rets = [d.day_return for d in days if d.day_return is not None]
        bench = [d.benchmark_return for d in days if d.benchmark_return is not None]
        shadow = [float(((d.close or {}).get("strategy_shadow") or {})["day_return"]) for d in days
                  if ((d.close or {}).get("strategy_shadow") or {}).get("day_return") is not None]  # fmt: skip
        eq = [d.equity_close for d in days if d.equity_close]
        body: dict[str, Any] = {
            "performance": {
                "sessions": len(days),
                "brain": _compound(rets),
                "benchmark": _compound(bench),
                "shadow": _compound(shadow) if shadow else None,
                "worst_day": min(rets) if rets else None,
                "orders_sent": sum(d.orders_sent for d in days),
                "turnover": round(sum(d.traded_notional for d in days) / (sum(eq) / len(eq)), 4)
                if eq
                else None,
                "note": "one week says nothing about skill: reported, not judged",
            },
            "execution": await execution_quality(self._db, since),
            "learning_snapshot": snapshot,
            "learning_changes": changes,
            "calibration": report["consensus"]["calibration"],
            "agents_needing_evidence": {a: v["needs"] for a, v in report["agents"].items() if v["needs"]},
            "rejections": {
                k: {"status": v["status"], "decisive": v["decisive"], "needs": v["needs"]}
                for k, v in ideas["by_reason"].items()
            },
            "ideas_this_week": {"recorded": week_ideas["recorded"], "graded": week_ideas["graded"]},
            "behaviour": behaviour["findings"],
            "daily_lessons": [lesson for r in dailies for lesson in (r.lessons or [])],
            "traceability": (await completeness(self._db, limit=200))["headline"],
        }
        lessons = self._weekly_lessons(body, ideas, report)
        written = await self._improvements.review(now)
        written += await self._improvements.submit(self._proposals(ideas, behaviour["findings"]), now)
        proposals = [
            {"kind": p["kind"], "target": p["target"], "title": p["title"], "status": p["status"]}
            for p in written
        ]
        perf = body["performance"]
        headline = (
            f"week of {start}: {perf['sessions']} session(s)"
            + (f", Brain {perf['brain']:+.2%} vs benchmark {perf['benchmark']:+.2%}" if perf["brain"] is not None and perf["benchmark"] is not None else "")
            + (f", shadow {perf['shadow']:+.2%}" if perf["shadow"] is not None else "")
            + f"; {len(lessons)} lesson(s), {len(proposals)} proposal(s) written or refreshed"
        )  # fmt: skip
        return await self._store("weekly", start, end, headline, body, lessons, proposals)

    def _weekly_lessons(
        self, body: dict[str, Any], ideas: dict[str, Any], report: dict[str, Any]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for reason, g in ideas["by_reason"].items():
            if g["status"] in ("costing opportunities", "right more often than not"):
                out.append(_lesson("rejections", f"rejecting ideas for {g['meaning']} has been "
                                   f"{g['status']} ({g['avoided_share']:.0%} avoided over {g['decisive']} decisive ideas)",
                                   g["decisive"], "established", reason=reason))  # fmt: skip
        for agent, v in report["agents"].items():
            verdict = v["record"]["verdict"]
            if verdict in ("evidence of skill", "evidence of harm"):
                out.append(_lesson("agents", f"{agent}: {verdict} over {v['record']['n_effective']} independent calls",
                                   v["record"]["n_effective"], "established"))  # fmt: skip
            vs = v["versus_consensus"]
            if vs["status"] not in ("unproven", "no evidence either way"):
                out.append(_lesson("agents", f"{agent} against the consensus: {vs['status']}", vs["disagreements"],
                                   "established"))  # fmt: skip
        cal = report["consensus"]["calibration"]
        if cal.get("status") in ("overconfident", "underconfident"):
            out.append(_lesson("calibration", f"the consensus is {cal['status']} (calibration error {cal['ece']:.2f})",
                               cal["n_effective"], "established"))  # fmt: skip
        for change in body["learning_changes"]:
            out.append(_lesson("learning", f"changed since last week — {change}", 0, "established"))
        for f in body["behaviour"]:
            if f["severity"] in ("warning", "alert"):
                out.append(
                    _lesson("behaviour", f["finding"], f["sample"], severity=f["severity"], code=f["code"])
                )
        if not out:
            out.append(
                _lesson("evidence", "nothing established yet: the record is too short for any conclusion", 0)
            )
        return out

    def _proposals(self, ideas: dict[str, Any], findings: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Proposals from the review. A rejection reason tied to a protected control, or a finding that points at
        one, names the control and is therefore recorded for a person's review only."""
        out: list[dict[str, Any]] = []
        plan = [
            "Replay the recorded cycles with the change and count what it would have done differently.",
            "Grade the ideas it would have changed against what happened (the opportunity outcomes).",
            "If it still looks better, run it on paper beside the current rule before any adoption.",
        ]
        for reason, g in ideas["by_reason"].items():
            if g["status"] != "costing opportunities" or reason == "market_closed":
                continue
            control = REASON_CONTROL.get(reason) if reason in PROTECTED_REASONS else None
            setting = control or REASON_SETTING.get(reason) or "the rule behind it"
            out.append(
                _p("rejection", reason, f"Ideas rejected for {REASONS.get(reason, reason)} have tended to work",
                   {k: g[k] for k in ("decisive", "verdicts", "avoided_share", "ci95", "mean_favourable")},
                   f"Review {setting}: the ideas it rejected went on to work more often than not.",
                   "Fewer good ideas rejected, with no rise in the bad ideas taken.", plan)
            )  # fmt: skip
        for f in findings:
            if f["severity"] != "alert" and not (
                f["severity"] == "warning" and f["code"] in ("turnover", "agent_herding", "concentration")
            ):
                continue
            change = {
                "round_trips": "Require more before reversing a position: a minimum holding period except at a stop or a broken thesis, or a larger margin before a holding is replaced.",
                "repeated_thesis_losses": "Review the losing combination of agents: are they one idea seen twice, or wrong in this regime?",
                "turnover": "Review what drives the trading: the replacement margin, the cycle interval, the rebalance trims.",
                "agent_herding": "Check the agents' declared sources: agents moving in lockstep should count once in the consensus.",
                "concentration": "Consider a tighter QP_TRADING_MAX_POSITION_PCT or a sector limit in the risk engine.",
                "losing_streak": "Review the decisions made after the losing streak for a change of standard.",
            }.get(f["code"], "Review the finding.")
            out.append(_p("behavior", f["code"], f"Behaviour: {f['finding'][:120]}", {"finding": f},
                          change, "Behaviour back within normal bounds, with no loss of good trades.", plan))  # fmt: skip
        return out

    # ------------------------------------------------------------------ storage
    async def _store(
        self, kind: str, start: date, end: date, headline: str, body: dict[str, Any],
        lessons: list[dict[str, Any]], proposals: list[dict[str, Any]],
    ) -> dict[str, Any]:  # fmt: skip
        now = self._clock.now()
        async with self._db.session() as s:
            row = BrainReviewRow(kind=kind, period_start=start, period_end=end, headline=headline, body=body,
                                 lessons=lessons, proposals=proposals, created_at=now)  # fmt: skip
            s.add(row)
            await s.flush()
            rid = row.id
        await self._memory.remember(
            LONG_TERM,
            "review",
            "@brain",
            headline,
            now,
            key=f"review:{kind}:{end.isoformat()}",
            data={"review_id": rid, "lessons": lessons[:12], "proposals": proposals},
            tags=["review", kind],
            importance=0.6 if kind == "weekly" else 0.4,
        )
        return {"id": rid, "kind": kind, "period_start": start.isoformat(), "period_end": end.isoformat(),
                "headline": headline, "body": body, "lessons": lessons, "proposals": proposals,
                "created_at": now.isoformat()}  # fmt: skip

    async def reviews(self, kind: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            q = select(BrainReviewRow).order_by(BrainReviewRow.id.desc()).limit(limit)
            if kind:
                q = q.where(BrainReviewRow.kind == kind)
            rows = (await s.scalars(q)).all()
        return [
            {"id": r.id, "kind": r.kind, "period_start": r.period_start.isoformat(), "period_end": r.period_end.isoformat(),
             "headline": r.headline, "body": r.body, "lessons": r.lessons, "proposals": r.proposals,
             "created_at": r.created_at.isoformat()}
            for r in rows
        ]  # fmt: skip


def _mean(xs: Sequence[float]) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def _compound(rets: Sequence[float]) -> float | None:
    if not rets:
        return None
    total = 1.0
    for r in rets:
        total *= 1 + r
    return round(total - 1, 6)
