"""The research catalogue: every kind of job, what it is worth, what it costs and how often it is asked again.

The closed-market loop works through them as

    GRADE → ANALYZE → RESEARCH → TEST → LEARN → PREPARE

(the phase of each job). Standing questions come back when their answer is older than ``refresh``; one-off
questions (a post-mortem, a person's question) are asked once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from . import handlers as h

PHASES = ("GRADE", "ANALYZE", "RESEARCH", "TEST", "LEARN", "PREPARE")


@dataclass(frozen=True)
class JobSpec:
    kind: str
    question: str
    phase: str
    cost: str  # light | medium | heavy
    value: float  # what an answer is worth, 1–10
    refresh: timedelta | None  # a standing question is asked again when its answer is this old
    handler: h.Handler
    topic: str | None = None  # the ledger topics whose conclusions tell how settled the question is
    timeout: timedelta | None = None  # None: QP_RESEARCH_JOB_TIMEOUT_MINUTES
    owner_only: bool = False  # needs the Brain to own the Alpaca paper account


H, D = timedelta(hours=1), timedelta(days=1)

CATALOG: dict[str, JobSpec] = {
    s.kind: s
    for s in (
        JobSpec("grade_predictions", "Which predictions have matured, and were they right?", "GRADE", "light", 9, 4 * H,
                h.grade_predictions),
        JobSpec("grade_ideas", "Which detected ideas have matured, and did they work?", "GRADE", "light", 8, 6 * H,
                h.grade_ideas),
        JobSpec("agent_calibration", "Which agents are accurate and calibrated?", "ANALYZE", "medium", 7, D,
                h.agent_calibration, topic="agent:"),
        JobSpec("decision_quality", "Do sound decisions end well more often than unsound ones (skill, not luck)?",
                "ANALYZE", "medium", 7, D, h.decision_quality, topic="decisions:"),
        JobSpec("trade_review", "What separates the winning trades from the losing ones?", "ANALYZE", "medium", 7, D,
                h.trade_review, topic="trades:"),
        JobSpec("trade_post_mortem", "Why did this trade fail?", "ANALYZE", "light", 6, None, h.trade_post_mortem),
        JobSpec("rejected_opportunities", "Which rejected opportunities subsequently worked — were rejections right?",
                "ANALYZE", "medium", 7, D, h.rejected_opportunities, topic="rejection:"),
        JobSpec("missed_opportunities", "What did the last sessions miss?", "ANALYZE", "light", 5, D,
                h.missed_opportunities),
        JobSpec("execution_review", "Are fills and turnover costing too much?", "ANALYZE", "light", 6, D,
                h.execution_review, topic="execution:"),
        JobSpec("account_vs_benchmark", "Does the paper account beat the benchmark, and how much market risk does it "
                "carry?", "ANALYZE", "light", 7, D, h.account_vs_benchmark, topic="account:"),
        JobSpec("exposure_review", "What sector and size concentration does the portfolio carry?", "ANALYZE", "light",
                5, D, h.exposure_review),
        JobSpec("behavior_review", "Is the Brain behaving pathologically (churn, herding, streaks)?", "ANALYZE",
                "medium", 5, D, h.behavior_review),
        JobSpec("agent_redundancy", "Are two agents effectively measuring the same factor?", "RESEARCH", "medium", 4,
                7 * D, h.agent_redundancy, topic="redundancy:"),
        JobSpec("feature_research", "Which features consistently rank future returns, out of sample and across "
                "volatility regimes?", "RESEARCH", "heavy", 5, 2 * D, h.feature_research, topic="feature:",
                timeout=timedelta(minutes=45)),
        JobSpec("agent_combinations", "Would the consensus predict better without one of its agents?", "TEST",
                "medium", 5, 7 * D, h.agent_combinations, topic="combination:"),
        JobSpec("strategy_research", "Do new strategies (templates, then generated ideas) survive backtests, "
                "walk-forward, random portfolios and stress — and how do the paper-tracked ones hold up?", "TEST",
                "heavy", 7, 3 * H,
                h.strategy_research, topic="strategy:", timeout=timedelta(minutes=60)),
        JobSpec("options_research", "Which option strategies survive model-priced backtests, walk-forward, Monte "
                "Carlo, stress and the false-discovery control — and what should the next generation try?", "TEST",
                "heavy", 8, H, h.options_research, topic="options:", timeout=timedelta(minutes=40)),
        JobSpec("improvement_review", "What weaknesses does the record show, and what evidence-based changes follow?",
                "LEARN", "medium", 5, D, h.improvement_review),
        JobSpec("memory_maintenance", "Is long-term memory current?", "LEARN", "light", 3, D, h.memory_maintenance),
        JobSpec("data_quality", "Is the data the next session needs available?", "PREPARE", "light", 6, 6 * H,
                h.data_quality),
        JobSpec("system_integrity", "Is the database consistent and the schema current?", "PREPARE", "light", 6,
                6 * H, h.system_integrity),
        JobSpec("reconcile_state", "Is the record in step with the Alpaca paper account?", "PREPARE", "light", 6,
                12 * H, h.reconcile_state, owner_only=True),
        JobSpec("upcoming_events", "What reports soon for what is held and watched?", "PREPARE", "light", 5, 12 * H,
                h.upcoming_events),
        JobSpec("watchlist_prep", "What should the next session watch, and what should research look at next?",
                "PREPARE", "light", 8, 12 * H, h.watchlist_prep),
    )
}  # fmt: skip
