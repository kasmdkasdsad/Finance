"""Reflection: learn from decision quality, not from profit.

Every decision whose outcome is known is judged twice, independently:

* **Decision quality** — from what was known *at the time* (the stored consensus, debate and risk preview),
  never from the outcome. Hard checks: the data was executable, the risk engine allowed the trade, and the
  devil's advocate had not challenged the view. Soft checks: combined confidence, agreement among the
  agents, unresolved objections, evidence from more than one source, and whether the agents it relied on had
  a measured record. ``poor`` if a hard check failed, ``good`` if ≥ 60% of the soft checks passed, else
  ``fair``.
* **Outcome quality** — the move in the decision's favour over its horizon, relative to the benchmark:
  ``good`` beyond +0.5%, ``bad`` below −0.5%, else ``neutral``.

The four combinations are kept apart, because they teach different things:

====================  ============================================================================
earned                good decision, good outcome — the process worked; nothing to change
unlucky               good decision, bad outcome — variance; do **not** change the rules because of it
lucky                 weak decision, good outcome — do **not** repeat it because it worked
process_failure       weak decision, bad outcome — the process let a bad decision through: fix it
====================  ============================================================================

Blocked ideas (WATCH) are graded as counterfactuals: did the block (confidence, earnings, posture, fit,
challenge) save money or cost an opportunity? The reflection records the post-mortem questions and their
answers and deterministic lessons (which objections were borne out, which agents were right). Reflections
are append-only: the original reasoning is never edited.

**Failure analysis** summarises, per agent with enough graded calls, where it fails: its worst regime,
how often it is confidently wrong, whether its confidence is calibrated, and any directional bias.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select

from quantpulse.db.models import (
    BrainConsensusRow,
    BrainDebateRow,
    BrainDecisionRow,
    BrainOpinionRow,
    BrainOpportunityRow,
    BrainPredictionRow,
    BrainReflectionRow,
)
from quantpulse.db.session import Database

from .performance import Graded, metrics

LONG_SIDE = {"buy": 1, "increase": 1, "watch": 1, "hold": 1}
SHORT_SIDE = {"reduce": -1, "close": -1, "sell": -1, "de_risk": -1, "rebalance": -1}
OUTCOME_BAND = 0.005


def outcome_quality(side: int, relative: float) -> str:
    favourable = side * relative
    return "good" if favourable > OUTCOME_BAND else "bad" if favourable < -OUTCOME_BAND else "neutral"


def decision_quality(
    decision: BrainDecisionRow, consensus: BrainConsensusRow | None, debate: BrainDebateRow | None
) -> tuple[str, dict[str, Any]]:
    trade = decision.action in LONG_SIDE | SHORT_SIDE and decision.action not in ("watch", "hold")
    hard: dict[str, bool] = {}
    if trade:
        hard["data executable"] = consensus is not None and consensus.data_quality in ("fresh", "live")
        hard["risk engine allowed it"] = bool(decision.risk_approved)
    hard["not challenged by the devil's advocate"] = debate is None or debate.verdict != "challenged"
    soft: dict[str, bool] = {}
    if consensus is not None:
        soft["confidence ≥ 0.55"] = consensus.confidence >= 0.55
        soft["agents largely agreed (disagreement ≤ 0.3)"] = consensus.disagreement <= 0.3
        votes = (consensus.detail or {}).get("votes") or []
        lead = 1 if consensus.score > 0 else -1
        supporters = {v["agent_id"] for v in votes if v["score"] * lead >= 0.15}
        detail = consensus.detail or {}
        if "independent_sources" in detail:  # agents sharing a source of information count once
            soft["evidence from more than one source"] = int(detail["independent_sources"]) >= 2
        else:  # cycles recorded before sources were tracked
            soft["evidence from more than one source"] = (
                bool(supporters - {"technical", "momentum"}) and len(supporters) >= 2
            )
        soft["relied on agents with a measured record"] = any(
            (v.get("reliability") or {}).get("status") == "measured"
            for v in votes
            if v["agent_id"] in supporters
        )
    if debate is not None:
        soft["no unresolved medium/high objection"] = not any(
            o["severity"] != "low" for o in debate.objections
        )
    passed = sum(soft.values())
    score = passed / len(soft) if soft else 0.0
    if not all(hard.values()):
        quality = "poor"
    elif score >= 0.6:
        quality = "good"
    else:
        quality = "fair"
    return quality, {"hard": hard, "soft": soft, "soft_score": round(score, 3)}


def category(dq: str, oq: str, blocked: bool) -> str:
    if oq == "neutral":
        return "inconclusive"
    if blocked:
        return "block_saved_money" if oq == "bad" else "block_cost_opportunity"
    good_decision = dq == "good"
    if good_decision:
        return "earned" if oq == "good" else "unlucky"
    return "lucky" if oq == "good" else "process_failure"


LESSON = {
    "earned": "The process worked: nothing to change.",
    "unlucky": "Sound process, bad outcome: variance. Do not change the rules because of this one result.",
    "lucky": "Good outcome from a weak decision: do not repeat it because it worked.",
    "process_failure": "A weak decision that also lost: the checks that failed are the ones to tighten.",
    "block_saved_money": "The block was right: the idea would have lost.",
    "block_cost_opportunity": "The block cost an opportunity; one block proves nothing, but count how often this happens.",
    "inconclusive": "The move was too small to judge.",
}


async def reflect_on_decisions(db: Database, now: datetime) -> list[dict[str, Any]]:
    """Write a reflection for every evaluated decision that has none yet; returns them."""
    async with db.session() as s:
        done = set(
            (
                await s.scalars(
                    select(BrainReflectionRow.subject_id).where(BrainReflectionRow.subject_type == "decision")
                )
            ).all()
        )
        decisions = [
            d
            for d in (
                await s.scalars(select(BrainDecisionRow).where(BrainDecisionRow.evaluated_at.is_not(None)))
            ).all()
            if d.id not in done
            and d.action in LONG_SIDE | SHORT_SIDE
            and d.outcome.get("relative") is not None
        ]
        out: list[dict[str, Any]] = []
        for d in decisions:
            consensus = await s.get(BrainConsensusRow, d.consensus_id) if d.consensus_id else None
            debate = (
                await s.scalars(
                    select(BrainDebateRow).where(
                        BrainDebateRow.cycle_id == d.cycle_id, BrainDebateRow.subject == d.subject
                    )
                )
            ).first()
            opinions = (
                await s.scalars(
                    select(BrainOpinionRow).where(
                        BrainOpinionRow.cycle_id == d.cycle_id, BrainOpinionRow.subject == d.subject
                    )
                )
            ).all()
            kinds = sorted(
                set(
                    (
                        await s.scalars(
                            select(BrainOpportunityRow.kind).where(
                                BrainOpportunityRow.cycle_id == d.cycle_id,
                                BrainOpportunityRow.subject == d.subject,
                            )
                        )
                    ).all()
                )
            )
            side = LONG_SIDE.get(d.action) or SHORT_SIDE.get(d.action, 0)
            rel = float(d.outcome["relative"])
            blocked = d.action == "watch"
            dq, checks = decision_quality(d, consensus, debate)
            oq = outcome_quality(side, rel)
            cat = category(dq, oq, blocked)
            right = sorted(
                {
                    o.agent_id
                    for o in opinions
                    if o.stance in ("bullish", "bearish") and (o.score > 0) == (rel > 0)
                }
            )
            wrong = sorted(
                {
                    o.agent_id
                    for o in opinions
                    if o.stance in ("bullish", "bearish") and (o.score > 0) != (rel > 0)
                }
            )
            lessons = [LESSON[cat]]
            if debate is not None:
                for obj in debate.objections:
                    if obj["severity"] == "low":
                        continue
                    lessons.append(
                        f"objection '{obj['code']}' was {'borne out' if oq == 'bad' else 'not borne out'} ({obj['text']})"
                    )
            if blocked:
                blockers = list((d.rationale or {}).get("reasons", []))
                lessons.append("blocked by: " + "; ".join(blockers)[:300])
            if cat in ("process_failure", "lucky"):
                failed = [k for k, v in {**checks["hard"], **checks["soft"]}.items() if not v]
                if failed:
                    lessons.append("checks that failed at the time: " + "; ".join(failed))
            questions = {
                "What was decided?": f"{d.action.upper()} {d.subject} ({d.status})",
                "Was the data trustworthy?": checks["hard"].get("data executable", "n/a"),
                "Did the risk engine allow it?": d.risk_approved,
                "What did the devil's advocate say?": debate.verdict if debate else "no debate",
                "What happened?": f"{rel:+.2%} relative to {d.outcome.get('benchmark')} over {d.outcome.get('horizon_days')} sessions",
                "Which agents were right?": right,
                "Which agents were wrong?": wrong,
            }
            row = BrainReflectionRow(
                subject_type="decision",
                subject_id=d.id,
                category=cat,
                decision_quality=dq,
                outcome_quality=oq,
                questions=questions,
                lessons=lessons,
                evidence={
                    "checks": checks,
                    "outcome": d.outcome,
                    "cycle_id": d.cycle_id,
                    "subject": d.subject,
                    "action": d.action,
                    # structured, so recurring patterns can be counted (quantpulse.brain.patterns)
                    "objections": [
                        {"code": o["code"], "severity": o["severity"], "borne_out": oq == "bad"}
                        for o in (debate.objections if debate is not None else [])
                        if o["severity"] != "low"
                    ],
                    "kinds": kinds,
                },
                created_at=now,
            )
            s.add(row)
            out.append(
                {
                    "decision_id": d.id,
                    "subject": d.subject,
                    "category": cat,
                    "decision_quality": dq,
                    "outcome_quality": oq,
                    "lessons": lessons,
                }
            )
    return out


def failure_analysis(rows: Sequence[Graded], min_observations: int) -> list[dict[str, Any]]:
    """Per agent with at least ``min_observations`` independent graded observations: where and how it
    fails. A weakness is only named when the evidence supports it (the 95% interval of the hit rate lies
    below a coin flip); a small or noisy sample is not a finding."""
    by_agent: dict[tuple[str, str], list[Graded]] = defaultdict(list)
    for r in rows:
        by_agent[(r.source, r.version)].append(r)
    findings: list[dict[str, Any]] = []
    for (agent, version), items in sorted(by_agent.items()):
        overall = metrics(items, min_observations)
        if overall["n_effective"] < min_observations:
            continue
        notes: list[str] = []
        regimes: dict[str, list[Graded]] = defaultdict(list)
        for r in items:
            regimes[r.regime].append(r)
        for regime, sub in regimes.items():
            m = metrics(sub, min_observations)
            if m["n_effective"] >= max(10, min_observations // 3) and _below_coin(m):
                notes.append(
                    f"weak in {regime} markets: {m['hit_rate']:.0%} hit rate (95% interval "
                    f"{m['ci_low']:.0%}–{m['ci_high']:.0%}) over {m['n_effective']} independent calls"
                )
        confident = metrics([r for r in items if r.confidence >= 0.6], min_observations)
        if confident["n_effective"] >= 10 and _below_coin(confident):
            notes.append(
                f"confidently wrong: its ≥0.6-confidence calls hit {confident['hit_rate']:.0%} "
                f"(interval {confident['ci_low']:.0%}–{confident['ci_high']:.0%})"
            )
        cal = [c for c in overall["calibration"] if c.get("kind") == "bucket" and c["n"] >= 10]
        if len(cal) >= 2 and cal[-1]["hit_rate"] < cal[0]["hit_rate"] - 0.1:
            notes.append(
                "miscalibrated: its most confident calls hit less often than its least confident ones"
            )
        share = overall["calibration"][-1].get("bullish_share")
        if share is not None and (share > 0.85 or share < 0.15):
            notes.append(f"directional bias: {share:.0%} of its calls are bullish")
        if _below_coin(overall):
            notes.append(
                f"below a coin flip: {overall['hit_rate']:.0%} (interval {overall['ci_low']:.0%}–"
                f"{overall['ci_high']:.0%}) over {overall['n_effective']} independent calls"
            )
        findings.append(
            {
                "agent_id": agent,
                "version": version,
                "n": overall["n"],
                "n_effective": overall["n_effective"],
                "hit_rate": overall["hit_rate"],
                "ci": [overall["ci_low"], overall["ci_high"]],
                "verdict": overall["verdict"],
                "ic": overall["ic"],
                "brier": overall["brier"],
                "notes": notes,
            }
        )
    return findings


def _below_coin(m: dict[str, Any]) -> bool:
    return m["ci_high"] is not None and m["ci_high"] < 0.5


async def record_failure_analysis(db: Database, findings: list[dict[str, Any]], now: datetime) -> int:
    notable = [f for f in findings if f["notes"]]
    if not notable:
        return 0
    async with db.session() as s:
        s.add(
            BrainReflectionRow(
                subject_type="system",
                subject_id=None,
                category="failure_analysis",
                decision_quality=None,
                outcome_quality=None,
                questions={"Which agents fail, and where?": [f["agent_id"] for f in notable]},
                lessons=[f"{f['agent_id']}: {n}" for f in notable for n in f["notes"]],
                evidence={"agents": findings},
                created_at=now,
            )
        )
    return len(notable)


async def consensus_calibration(db: Database) -> list[dict[str, Any]]:
    """Hit rate of the consensus by its (post-debate) confidence — the evidence for or against the
    confidence threshold."""
    async with db.session() as s:
        rows = (
            await s.scalars(
                select(BrainPredictionRow).where(
                    BrainPredictionRow.status == "evaluated", BrainPredictionRow.source_type == "consensus"
                )
            )
        ).all()
    out = []
    for lo, hi in ((0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01)):
        inside = [r for r in rows if lo <= r.confidence < hi]
        if inside:
            out.append(
                {
                    "confidence": f"{lo:.2f}–{min(hi, 1.0):.2f}",
                    "n": len(inside),
                    "hit_rate": round(sum(1 for r in inside if r.hit) / len(inside), 4),
                    "mean_relative": round(
                        sum(float(r.realized_relative or 0) for r in inside) / len(inside), 5
                    ),
                }
            )
    return out
