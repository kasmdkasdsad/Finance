"""Controlled self-improvement: the brain looks at its own record and writes structured proposals.

Each proposal states the **problem**, the **evidence** (numbers from graded predictions, run history,
events, reflections and lab results — never impressions), the **proposed change**, the **expected
improvement** and a **validation plan**. Nothing is applied automatically: code, parameters and routing
only change when a person accepts a proposal, and a change is first built as a *new version* that goes
through

    PROPOSE → VERSION → TEST → BACKTEST → WALK-FORWARD → PAPER EVALUATION → COMPARE → PROMOTE ONLY IF VALIDATED

(the strategy lab implements that pipeline for strategies; agent versions are compared on graded calls,
because every version keeps its own track record). Risk controls are never a subject of proposals.

What it looks for:

* **weak agents** — a significant *evidence of harm* verdict (enough independent calls, false-discovery adjusted);
* **weak in one regime** — an agent failing in a specific market regime (a routing proposal);
* **redundant agents** — two agents whose scores on the same subjects move together (≥ 0.9 correlation);
* **missing capabilities** — agents that keep skipping for lack of data (options, earnings, the model);
* **expensive workflows** — agents or cycles that are slow for what they return;
* **stale-data problems** — symbols whose quotes keep going stale;
* **failed strategies** — lab versions rejected by validation, or promoted ones falling short in paper;
* **poor recommendations** — recurring process failures, and blocks that keep costing opportunities;
* **recurring analytical mistakes** — failure-analysis findings that repeat, objections that are (or are
  not) borne out, and a consensus threshold the calibration does not support.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import select

from quantpulse.db.models import (
    BrainAgentPerformanceRow,
    BrainAgentRunRow,
    BrainEventRow,
    BrainImprovementRow,
    BrainOpinionRow,
    BrainReflectionRow,
    BrainStrategyRow,
)
from quantpulse.db.session import Database

from .reflection import consensus_calibration

PIPELINE = [
    "PROPOSE",
    "VERSION",
    "TEST",
    "BACKTEST",
    "WALK-FORWARD",
    "PAPER EVALUATION",
    "COMPARE",
    "PROMOTE ONLY IF VALIDATED",
]
AGENT_PLAN = [
    "Implement the change as a new agent version (the current version keeps running and voting).",
    "Unit-test the new logic against recorded cycles (the same inputs must give explainable outputs).",
    "Run both versions side by side: the new version records predictions but does not vote.",
    "After enough graded calls in the same regimes, compare hit rate, Brier score and rank IC.",
    "Promote the new version only if it is better on graded calls; otherwise reject and keep the old one.",
]
LOOKBACK = timedelta(days=30)
STATUSES = ("proposed", "testing", "validated", "rejected", "applied")


def _p(
    kind: str, target: str, title: str, evidence: dict[str, Any], change: str, expected: str, plan: list[str]
) -> dict[str, Any]:
    return {
        "kind": kind,
        "target": target,
        "title": title,
        "evidence": evidence,
        "proposal": {
            "change": change,
            "expected_improvement": expected,
            "validation_plan": plan,
            "pipeline": PIPELINE,
        },
    }


class ImprovementEngine:
    def __init__(self, db: Database, min_observations: int, min_confidence: float) -> None:
        self._db = db
        self._min = min_observations
        self._min_confidence = min_confidence

    async def review(self, now: datetime) -> list[dict[str, Any]]:
        """Analyse the record and store new proposals (an open proposal about the same thing is updated
        with fresh evidence instead of duplicated). Returns the proposals written or refreshed."""
        found = [
            *await self._agents(),
            *await self._redundancy(now),
            *await self._runs(now),
            *await self._stale_data(now),
            *await self._strategies(),
            *await self._decisions(),
            *await self._calibration(),
        ]
        out: list[dict[str, Any]] = []
        async with self._db.session() as s:
            open_rows = {
                (r.kind, r.target, r.title): r
                for r in (
                    await s.scalars(
                        select(BrainImprovementRow).where(
                            BrainImprovementRow.status.in_(("proposed", "testing"))
                        )
                    )
                ).all()
            }
            for f in found:
                key = (f["kind"], f["target"], f["title"])
                row = open_rows.get(key)
                if row is None:
                    row = BrainImprovementRow(
                        status="proposed", test_result={}, created_at=now, **f, updated_at=now
                    )
                    s.add(row)
                else:
                    row.evidence, row.proposal, row.updated_at = f["evidence"], f["proposal"], now
                out.append(f)
        return out

    async def proposals(self, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        stmt = select(BrainImprovementRow).order_by(BrainImprovementRow.id.desc()).limit(limit)
        if status:
            stmt = stmt.where(BrainImprovementRow.status == status)
        async with self._db.session() as s:
            rows = (await s.scalars(stmt)).all()
        return [
            {
                "id": r.id,
                "kind": r.kind,
                "target": r.target,
                "title": r.title,
                "evidence": r.evidence,
                "proposal": r.proposal,
                "status": r.status,
                "test_result": r.test_result,
                "decided_by": r.decided_by,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]

    async def decide(
        self, improvement_id: int, status: str, by: str, note: str | None, now: datetime
    ) -> dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        async with self._db.session() as s:
            row = await s.get(BrainImprovementRow, improvement_id)
            if row is None:
                raise KeyError(improvement_id)
            row.status, row.decided_by, row.updated_at = status, by, now
            if note:
                row.test_result = {**(row.test_result or {}), "note": note, "at": now.isoformat()}
        return next(i for i in await self.proposals(limit=10_000) if i["id"] == improvement_id)

    # ------------------------------------------------------------------ analyses
    async def _agents(self) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainAgentPerformanceRow).where(BrainAgentPerformanceRow.window == "all")
                )
            ).all()
        out = []
        for r in rows:
            n = r.n_effective if r.n_effective is not None else r.n
            if r.agent_id == "consensus" or n < self._min or r.hit_rate is None:
                continue
            harm = r.verdict == "evidence of harm" if r.verdict is not None else r.hit_rate < 0.47
            evidence = {
                "graded_calls": r.n,
                "independent_calls": n,
                "hit_rate": r.hit_rate,
                "interval_95": [r.ci_low, r.ci_high],
                "q_value": r.q_value,
                "verdict": r.verdict,
                "brier": r.brier,
                "rank_ic": r.ic,
                "reliability": r.reliability,
            }
            if r.regime == "all" and harm:
                out.append(
                    _p(
                        "agent",
                        f"{r.agent_id}@{r.agent_version}",
                        f"{r.agent_id} is significantly below a coin flip on graded calls",
                        evidence,
                        f"Review {r.agent_id}'s logic: its evidence does not predict the outcome it is graded on "
                        f"({r.horizon_days}-session relative return). Until then its measured reliability already "
                        "reduces its weight.",
                        "Fewer wrong votes in the consensus and a higher consensus hit rate.",
                        AGENT_PLAN,
                    )
                )
            elif r.regime not in ("all", "unknown") and harm:
                out.append(
                    _p(
                        "routing",
                        f"{r.agent_id}@{r.agent_version}",
                        f"{r.agent_id} performs poorly in {r.regime} markets",
                        {"regime": r.regime, **evidence},
                        f"Route {r.agent_id} out of cycles in {r.regime} regimes (or give it a regime-specific "
                        "version), keeping it everywhere else.",
                        f"Remove a systematically wrong voice in {r.regime} markets.",
                        [
                            "Replay graded calls: consensus with and without the agent in that regime.",
                            *AGENT_PLAN[2:],
                        ],
                    )
                )
        return out

    async def _redundancy(self, now: datetime) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainOpinionRow).where(
                        BrainOpinionRow.created_at >= now - LOOKBACK,
                        BrainOpinionRow.stance.in_(("bullish", "bearish", "neutral")),
                    )
                )
            ).all()
        scores: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
        for o in rows:
            if not o.subject.startswith("@"):
                scores[o.agent_id][(o.cycle_id, o.subject)] = o.score
        out = []
        for a, b in itertools.combinations(sorted(scores), 2):
            shared = sorted(set(scores[a]) & set(scores[b]))
            if len(shared) < 50:
                continue
            x = np.array([scores[a][k] for k in shared])
            y = np.array([scores[b][k] for k in shared])
            if x.std() == 0 or y.std() == 0:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if corr >= 0.9:
                out.append(
                    _p(
                        "redundancy",
                        f"{a}+{b}",
                        f"{a} and {b} largely say the same thing",
                        {"shared_opinions": len(shared), "score_correlation": round(corr, 3)},
                        f"Merge {a} and {b}, or weight them as one voice, so one idea is not counted twice.",
                        "A consensus whose agreement reflects independent evidence.",
                        [
                            "Replay recorded cycles with the pair merged; compare consensus hit rates on graded calls.",
                            *AGENT_PLAN[2:],
                        ],
                    )
                )
        return out

    async def _runs(self, now: datetime) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(select(BrainAgentRunRow).where(BrainAgentRunRow.started_at >= now - LOOKBACK))
            ).all()
        by_agent: dict[str, list[BrainAgentRunRow]] = defaultdict(list)
        for r in rows:
            by_agent[r.agent_id].append(r)
        out = []
        for agent, runs in by_agent.items():
            skipped = [
                r
                for r in runs
                if r.status == "skipped" and r.reason and not r.reason.startswith(("disabled", "not needed"))
            ]
            if len(runs) >= 10 and len(skipped) / len(runs) >= 0.8:
                reason = Counter(r.reason for r in skipped).most_common(1)[0][0]
                out.append(
                    _p(
                        "capability",
                        agent,
                        f"{agent} rarely runs: {reason}",
                        {"runs": len(runs), "skipped": len(skipped), "most_common_reason": reason},
                        f"Provide the data {agent} needs ({reason}); until then it cannot contribute.",
                        f"{agent}'s evidence in the consensus.",
                        [
                            "Configure the data source (see the README's data sections).",
                            "Confirm the agent runs on real data in the next cycles (Agents tab).",
                            "Let its calls be graded before relying on it.",
                        ],
                    )
                )
            failed = [r for r in runs if r.status in ("failed", "timeout")]
            if len(runs) >= 10 and len(failed) / len(runs) >= 0.2:
                out.append(
                    _p(
                        "reliability",
                        agent,
                        f"{agent} fails often",
                        {"runs": len(runs), "failures": len(failed), "last_error": failed[-1].reason},
                        f"Fix the cause of {agent}'s failures (last: {failed[-1].reason}).",
                        "Fewer cycles with a missing voice.",
                        AGENT_PLAN,
                    )
                )
            ok = [r.duration_ms for r in runs if r.status == "ok"]
            if len(ok) >= 10 and float(np.median(ok)) > 3000:
                out.append(
                    _p(
                        "cost",
                        agent,
                        f"{agent} is slow",
                        {"median_ms": round(float(np.median(ok)), 1), "runs": len(ok)},
                        f"Cache {agent}'s intermediate results between cycles or narrow its route.",
                        "Faster cycles at the same analytical output.",
                        [
                            "Profile the agent on a recorded cycle.",
                            "Implement caching as a new version.",
                            "Check its opinions are unchanged on recorded inputs.",
                            "Deploy.",
                        ],
                    )
                )
        return out

    async def _stale_data(self, now: datetime) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainEventRow).where(
                        BrainEventRow.type == "QuoteBecameStale", BrainEventRow.created_at >= now - LOOKBACK
                    )
                )
            ).all()
        counts = Counter(r.subject for r in rows if r.subject)
        return [
            _p(
                "data",
                sym,
                f"{sym}'s quotes keep going stale",
                {"stale_events_30d": n},
                f"Check {sym}'s live quote source (feed, subscription, symbol mapping); the brain vetoes action on it "
                "whenever its data is stale.",
                "Actionable data for a symbol the brain keeps studying.",
                [
                    "Run /trading/diagnostics for the symbol.",
                    "Fix the feed or subscription.",
                    "Confirm no QuoteBecameStale events for it over the next sessions.",
                ],
            )
            for sym, n in counts.items()
            if n >= 5
        ]

    async def _strategies(self) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (await s.scalars(select(BrainStrategyRow))).all()
        out = []
        for r in rows:
            key = f"{r.strategy_id}@v{r.version}"
            v = r.validation or {}
            if r.status == "rejected":
                failed = [g["gate"] for g in v.get("gates", []) if not g["passed"]]
                out.append(
                    _p(
                        "strategy",
                        key,
                        f"{key} failed validation",
                        {"failed_gates": failed, "walk_forward": v.get("walk_forward")},
                        f"Retire {key}, or try a new version only if there is a reason (not a parameter hunt): every "
                        "extra variant raises the bar the deflated Sharpe sets.",
                        "No capital or attention on a strategy that did not survive out of sample.",
                        [
                            "Create a new version in the lab only with a stated hypothesis.",
                            "Validate it (walk-forward, deflated Sharpe, stress).",
                            "Paper-track if validated.",
                            "Compare with the previous version.",
                            "Promote only if validated.",
                        ],
                    )
                )
            paper = r.paper or {}
            if (
                r.status in ("paper", "promoted")
                and (paper.get("excess_return") or 0) < -0.03
                and (paper.get("sessions") or 0) >= 20
            ):
                out.append(
                    _p(
                        "strategy",
                        key,
                        f"{key} is falling short in paper tracking",
                        {"paper": paper},
                        f"Demote {key} (retire it or return it to paper) and re-validate on the latest data.",
                        "Stop a strategy whose live behaviour no longer matches its backtest.",
                        [
                            "Re-run validation on the latest history.",
                            "Compare paper and backtest behaviour.",
                            "Retire unless the difference is explained.",
                        ],
                    )
                )
        return out

    async def _decisions(self) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainReflectionRow).where(BrainReflectionRow.subject_type == "decision")
                )
            ).all()
        out = []
        cats = Counter(r.category for r in rows)
        judged = sum(n for c, n in cats.items() if c != "inconclusive")
        if judged >= 20 and cats["process_failure"] / judged >= 0.25:
            failed = Counter(
                check
                for r in rows
                if r.category == "process_failure"
                for check, ok in {
                    **(r.evidence.get("checks") or {}).get("hard", {}),
                    **(r.evidence.get("checks") or {}).get("soft", {}),
                }.items()
                if not ok
            )
            out.append(
                _p(
                    "decisions",
                    "planner",
                    "Weak decisions keep getting through and losing",
                    {
                        "process_failures": cats["process_failure"],
                        "judged": judged,
                        "checks_that_failed": dict(failed.most_common(5)),
                    },
                    "Tighten the checks that most often failed in these decisions (see evidence) — as a new planner "
                    "version, not an edit in place.",
                    "Fewer process failures without fewer earned outcomes.",
                    [
                        "Replay the recorded decisions with the tighter rule.",
                        "Count earned vs process failures.",
                        "Run the new rule in dry-run alongside the current one.",
                        "Adopt only if it is better on both.",
                    ],
                )
            )
        blocked = [r for r in rows if r.category in ("block_saved_money", "block_cost_opportunity")]
        if len(blocked) >= 20:
            share = sum(1 for r in blocked if r.category == "block_cost_opportunity") / len(blocked)
            if share >= 0.7:
                out.append(
                    _p(
                        "decisions",
                        "blocks",
                        "Blocked ideas mostly went on to gain",
                        {"blocked": len(blocked), "cost_opportunity_share": round(share, 3)},
                        "Review which block costs most (confidence threshold, earnings caution, posture, fit) using "
                        "the reflections' 'blocked by' lessons.",
                        "Fewer missed opportunities without letting weak decisions through.",
                        [
                            "Group blocked ideas by blocker.",
                            "Estimate each blocker's cost and saving.",
                            "Test a changed threshold in dry-run.",
                            "Adopt only with evidence on both sides.",
                        ],
                    )
                )
        return out

    async def _calibration(self) -> list[dict[str, Any]]:
        buckets = await consensus_calibration(self._db)
        big = [b for b in buckets if b["n"] >= 30]
        if len(big) < 2:
            return []
        rates = [b["hit_rate"] for b in big]
        if all(x <= y for x, y in itertools.pairwise(rates)):
            return []
        return [
            _p(
                "calibration",
                "consensus",
                "Higher consensus confidence does not mean more hits",
                {"calibration": big, "min_confidence": self._min_confidence},
                "Recalibrate the consensus confidence (weights, haircuts, coverage) so that it orders outcomes, and "
                "only then revisit QP_BRAIN_MIN_CONFIDENCE.",
                "A threshold that selects better calls.",
                [
                    "Fit the mapping on graded consensus calls from one period.",
                    "Check it on a later period.",
                    "Run the new mapping alongside the current one in recorded cycles.",
                    "Adopt only if it orders better.",
                ],
            )
        ]
