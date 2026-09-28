"""Controlled self-improvement: the brain looks at its own record and writes structured proposals.

Each proposal states the **problem**, the **evidence** (numbers from graded predictions, run history,
events, reflections and lab results — never impressions), the **proposed change**, the **expected
improvement** and a **validation plan**. Nothing is applied automatically: code, parameters and routing
only change when a person accepts a proposal, and a change is first built as a *new version* that goes
through

    PROPOSE → VERSION → TEST → BACKTEST → WALK-FORWARD → PAPER EVALUATION → COMPARE → PROMOTE ONLY IF VALIDATED

(the strategy lab implements that pipeline for strategies; agent versions are compared on graded calls,
because every version keeps its own track record). Risk controls are never a subject of proposals, and
this is enforced, not only intended: a proposal that names a protected control (:data:`PROTECTED`: loss,
position and order limits, the kill switches, the data-quality requirements, the paper-only settings, the
execution safety switches) is withheld before it is stored (:func:`withheld`).

What it looks for:

* **weak agents** — a significant *evidence of harm* verdict (enough independent calls, false-discovery adjusted);
* **weak in one regime** — an agent failing in a specific market regime (a routing proposal);
* **redundant agents** — two agents whose scores on the same subjects move together (≥ 0.9 correlation);
* **missing capabilities** — agents that keep skipping for lack of data (options, earnings, the model);
* **expensive workflows** — agents or cycles that are slow for what they return;
* **stale-data problems** — symbols whose quotes keep going stale;
* **poor execution** — the Brain's own fills (the execution ledger) costing well beyond half the spread;
* **excessive turnover** — positions the Brain closes within a session or two of opening them;
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

from quantpulse.core.market_calendar import sessions_between
from quantpulse.db.models import (
    BrainAgentPerformanceRow,
    BrainAgentRunRow,
    BrainCycleRow,
    BrainEventRow,
    BrainExecutionRow,
    BrainImprovementRow,
    BrainOpinionRow,
    BrainOpportunityRow,
    BrainReflectionRow,
    BrainStrategyRow,
    BrainThesisRow,
)
from quantpulse.db.session import Database

from .patterns import find
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
# Settings no proposal may touch. The Brain may say that data is the problem (a SIP subscription), never that
# a limit should move.
PROTECTED = (
    "QP_TRADING_MAX_",
    "QP_TRADING_MIN_",
    "QP_TRADING_REQUIRE_LIVE_DATA",
    "QP_TRADING_DAILY_LOSS",
    "QP_TRADING_CASH_BUFFER",
    "QP_TRADING_KILL_SWITCH",
    "QP_TRADING_DRY_RUN",
    "QP_TRADING_SCHEDULER_REQUIRES_ARMING",
    "QP_TRADING_ALLOW_SHORTS",
    "QP_ALPACA_PAPER",
    "QP_ALPACA_TRADING_ENABLED",
    "QP_BRAIN_KILL_SWITCH",
    "QP_BRAIN_MODE",
)
STATUSES = ("proposed", "testing", "validated", "rejected", "applied")
EXECUTION_MIN_FILLS = 10  # graded fills before execution is judged at all
POOR_SHARE = 0.3  # … and the share of poor fills that calls for a review
QUICK_SESSIONS = 2
TURNOVER_MIN = 5


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


def withheld(proposal: dict[str, Any]) -> str | None:
    """The protected control a proposal would touch (``None``: it touches none)."""
    text = repr(proposal).upper()
    return next((name for name in PROTECTED if name in text), None)


def _where(slice_: str) -> str:
    return f"{slice_[4:]}-volatility markets" if slice_.startswith("vol:") else f"{slice_} markets"


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
            *await self._execution(now),
            *await self._turnover(now),
            *await self._strategies(),
            *await self._decisions(),
            *await self._calibration(),
            *await self._horizons(),
            *await self._fit(now),
            *await self._data_sources(now),
            *await self._assumptions(),
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
                if withheld(f) is not None:  # never stored, never shown as a suggestion
                    continue
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
            elif r.regime not in ("all", "unknown") and not r.regime.startswith("at:") and harm:
                out.append(
                    _p(
                        "routing",
                        f"{r.agent_id}@{r.agent_version}",
                        f"{r.agent_id} performs poorly in {_where(r.regime)}",
                        {"regime": r.regime, **evidence},
                        f"Route {r.agent_id} out of cycles in {_where(r.regime)} (or give it a version for them), "
                        "keeping it everywhere else.",
                        f"Remove a systematically wrong voice in {_where(r.regime)}.",
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

    async def _execution(self, now: datetime) -> list[dict[str, Any]]:
        """Fills that pay well beyond half the spread, often enough to matter (graded fills only)."""
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainExecutionRow).where(
                        BrainExecutionRow.decided_at >= now - LOOKBACK, BrainExecutionRow.grade.is_not(None)
                    )
                )
            ).all()
        graded = [r for r in rows if r.grade in ("good", "fair", "poor")]
        poor = [r for r in graded if r.grade == "poor"]
        if len(graded) < EXECUTION_MIN_FILLS or len(poor) / len(graded) < POOR_SHARE:
            return []
        costs = [r.cost_vs_quote_bps for r in poor if r.cost_vs_quote_bps is not None]
        return [
            _p(
                "execution",
                "brain_orders",
                "Brain fills often cost well beyond half the spread",
                {
                    "graded_fills_30d": len(graded),
                    "poor": len(poor),
                    "poor_share": round(len(poor) / len(graded), 3),
                    "median_cost_bps_of_poor": round(float(np.median(costs)), 2) if costs else None,
                    "symbols": sorted({r.symbol for r in poor})[:10],
                    "order_types": dict(Counter(r.order_type or "unknown" for r in poor)),
                },
                "Review how Brain orders are priced and timed: marketable limits nearer the midpoint, avoiding "
                "the first and last minutes of the session, or skipping the thinnest names. Execution only — "
                "no risk limit changes.",
                "Lower cost against the quote on the same kind of orders.",
                [
                    "Compare cost against the quote by order type, time of day and spread in the ledger.",
                    "Implement the pricing change as a new version behind a setting (off by default).",
                    "Run it on paper for enough fills and compare grades with the current version.",
                    "Adopt only if cost against the quote is lower without fewer fills.",
                ],
            )
        ]

    async def _turnover(self, now: datetime) -> list[dict[str, Any]]:
        """Positions closed within ``QUICK_SESSIONS`` of opening, not at a stop: the Brain changing its mind."""
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainThesisRow).where(
                        BrainThesisRow.status == "closed",
                        BrainThesisRow.origin == "brain",
                        BrainThesisRow.closed_at >= now - LOOKBACK,
                    )
                )
            ).all()
        quick = [
            r
            for r in rows
            if r.closed_at is not None
            and sessions_between(r.opened_at, r.closed_at) <= QUICK_SESSIONS
            and "stop" not in (r.exit_reason or "").lower()
        ]
        if len(quick) < TURNOVER_MIN or len(quick) * 2 < len(rows):
            return []
        return [
            _p(
                "decision",
                "turnover",
                "The Brain closes many positions within two sessions of opening them",
                {
                    "closed_positions_30d": len(rows),
                    "closed_within_2_sessions": len(quick),
                    "exit_reasons": dict(
                        Counter((r.exit_reason or "unknown")[:60] for r in quick).most_common(5)
                    ),
                },
                "Require more before an idea is reversed: a larger margin before a holding is replaced, or a "
                "minimum holding period except at a stop or a broken thesis.",
                "Fewer round trips that pay the spread twice for no change in the evidence.",
                [
                    "Replay the recorded cycles with the stricter rule and count the trades it removes.",
                    "Check the removed trades' outcomes: were the quick exits right or wrong?",
                    "Adopt only if the removed round trips did not, on balance, avoid losses.",
                ],
            )
        ]

    async def _horizons(self) -> list[dict[str, Any]]:
        """An agent whose calls work at another horizon than the one it is graded on."""
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainAgentPerformanceRow).where(BrainAgentPerformanceRow.window == "all")
                )
            ).all()
        own = {(r.agent_id, r.agent_version): r for r in rows if r.regime == "all"}
        out = []
        for r in rows:
            if (
                not r.regime.startswith("at:")
                or r.verdict != "evidence of skill"
                or r.agent_id == "consensus"
            ):
                continue
            base = own.get((r.agent_id, r.agent_version))
            if base is None or base.verdict == "evidence of skill" or r.horizon_days == base.horizon_days:
                continue
            out.append(
                _p(
                    "horizon",
                    f"{r.agent_id}@{r.agent_version}",
                    f"{r.agent_id} is right at {r.horizon_days} sessions, not at its own {base.horizon_days}",
                    {
                        "own_horizon": base.horizon_days,
                        "own_verdict": base.verdict,
                        "better_horizon": r.horizon_days,
                        "hit_rate_there": r.hit_rate,
                        "interval_95": [r.ci_low, r.ci_high],
                        "q_value": r.q_value,
                        "independent_calls": r.n_effective,
                    },
                    f"Grade (and weigh) {r.agent_id}'s calls at {r.horizon_days} sessions in a new version.",
                    "Its votes count at the horizon where they carry information.",
                    AGENT_PLAN,
                )
            )
        return out

    async def _fit(self, now: datetime) -> list[dict[str, Any]]:
        """Kinds of opportunity that keep failing the portfolio-fit check (the book cannot use them)."""
        async with self._db.session() as s:
            rows = (
                await s.scalars(
                    select(BrainOpportunityRow).where(BrainOpportunityRow.created_at >= now - LOOKBACK)
                )
            ).all()
        tried: Counter[str] = Counter()
        failed: Counter[str] = Counter()
        for r in rows:
            fit = next((st for st in r.stages or [] if st.get("stage") == "portfolio_fit"), None)
            if fit is None:
                continue
            tried[r.kind] += 1
            failed[r.kind] += fit.get("result") == "poor fit"
        return [
            _p(
                "opportunity",
                kind,
                f"{kind.replace('_', ' ')} ideas keep failing the portfolio-fit check",
                {"checked_30d": tried[kind], "failed_fit": failed[kind]},
                f"Spend less of the focus budget on {kind.replace('_', ' ')} ideas while the book cannot absorb them "
                "(too correlated with holdings or too much in one sector), or diversify the book first.",
                "Focus spent on ideas the book can actually take.",
                [
                    "Replay recent cycles with the change: how many fit-passing ideas replace them?",
                    "Paper-track the change and compare the book's diversification and the hit rate of its trades.",
                ],
            )
            for kind in tried
            if tried[kind] >= 10 and failed[kind] / tried[kind] >= 0.6
        ]

    async def _data_sources(self, now: datetime) -> list[dict[str, Any]]:
        """Data problems that recur across cycles (from each cycle's data report)."""
        async with self._db.session() as s:
            rows = (
                await s.scalars(select(BrainCycleRow).where(BrainCycleRow.started_at >= now - LOOKBACK))
            ).all()
        reports = [((r.data_quality or {}).get("feed") or {}) for r in rows]
        open_ = [f for f in reports if f.get("market_open")]
        out = []
        iex = sum(1 for f in open_ if any("IEX-priced symbols" in c for c in f.get("causes") or []))
        if len(open_) >= 10 and iex / len(open_) >= 0.5:
            out.append(
                _p(
                    "data",
                    "stock_feed",
                    "IEX-only prices are too often stale for the Brain to act",
                    {"open_market_cycles_30d": len(open_), "cycles_with_stale_iex_prices": iex},
                    "Consider a market-data subscription with real-time SIP (QP_ALPACA_STOCK_FEED=sip). This costs "
                    "money and is a person's decision; the quote-age limit is not the fix.",
                    "More symbols with executable data; fewer vetoes for stale prices.",
                    ["Compare the share of fresh/live quotes and the vetoes before and after the change."],
                )
            )
        skewed = [
            f["clock_skew_s"]
            for f in reports
            if f.get("clock_skew_s") is not None and abs(f["clock_skew_s"]) > 2
        ]
        if len(skewed) >= 3:
            out.append(
                _p(
                    "data",
                    "system_clock",
                    "This computer's clock keeps drifting from Alpaca's",
                    {"cycles_with_skew_over_2s": len(skewed), "largest_skew_s": max(skewed, key=abs)},
                    "Synchronise the system clock (Windows: Settings → Time → Sync now; or enable automatic time).",
                    "Quote ages that are right, so fresh data is not called stale (or stale data fresh).",
                    ["Confirm the data report shows a skew under 2 s for the next sessions."],
                )
            )
        return out

    async def _assumptions(self) -> list[dict[str, Any]]:
        """Objections and hypotheses whose record is established (see quantpulse.brain.patterns)."""
        out = []
        for p in await find(self._db, self._min):
            if p["status"] != "established":
                continue
            if p["kind"] == "objection":
                strong = p["rate"] > 0.5
                change = (
                    f"Make the '{p['name']}' objection weigh more (higher severity or a larger confidence cut)."
                    if strong
                    else f"The '{p['name']}' objection rarely predicts a bad outcome: soften it."
                )
                out.append(
                    _p(
                        "assumption",
                        f"objection:{p['name']}",
                        f"the '{p['name']}' objection is {'usually right' if strong else 'usually wrong'}",
                        {k: p[k] for k in ("n", "k", "rate", "ci")},
                        change,
                        "A devil's advocate whose objections carry the weight their record supports.",
                        [
                            "Replay graded debates with the changed weight; compare decision quality and outcomes."
                        ],
                    )
                )
            elif p["kind"] == "hypothesis" and p["rate"] < 0.5:
                out.append(
                    _p(
                        "assumption",
                        f"hypothesis:{p['name']}",
                        f"trades on {p['name'].replace('_', ' ')} ideas keep losing",
                        {k: p[k] for k in ("n", "k", "rate", "ci")},
                        f"Require more independent evidence before acting on {p['name'].replace('_', ' ')} ideas, "
                        "or stop acting on them.",
                        "Fewer trades on a hypothesis the record does not support.",
                        [
                            "Paper-track the change in the book; compare the hit rate of the trades it would skip."
                        ],
                    )
                )
        return out

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
