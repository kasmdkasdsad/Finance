"""Strategy stages and the evidence each step requires. No stage can be skipped; no source, investor or
backtest is trusted on its own; PROVEN needs a meaningful sample of paper trades.

::

    RESEARCH → EXTRACTED → BACKTESTING → VALIDATION → WALK_FORWARD → PAPER_SHADOW → PAPER_ACTIVE → PROVEN
                                                                         (any) → RETIRED

====================  ================================================================================
EXTRACTED             the genome is explicit and valid (every parameter stated)
BACKTESTING           a backtest has run (any data grade; labelled)
VALIDATION            enough trades; positive per dollar at risk under REALISTIC *and* PESSIMISTIC fills
WALK_FORWARD          positive in the held-out validation period; overfit risk below 0.5
PAPER_SHADOW          walk-forward passed; Monte Carlo and tail stress passed; beats the baselines;
                      survives the false-discovery control across the population; the critic's attacks
PAPER_ACTIVE          enough shadow trades on live quotes over enough sessions, positive net of modelled
                      costs; the structure family is executable (default list, or approved by a person)
PROVEN                enough real paper trades, positive with a t-statistic of 2, not decaying
====================  ================================================================================

Human approval is required (and cannot be given by the system) for families outside the default
executable list and for anything undefined-risk — which is never executable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from quantpulse.options.structures import FAMILIES


class Stage(StrEnum):
    RESEARCH = "RESEARCH"
    EXTRACTED = "EXTRACTED"
    BACKTESTING = "BACKTESTING"
    VALIDATION = "VALIDATION"
    WALK_FORWARD = "WALK_FORWARD"
    PAPER_SHADOW = "PAPER_SHADOW"
    PAPER_ACTIVE = "PAPER_ACTIVE"
    PROVEN = "PROVEN"
    RETIRED = "RETIRED"


ORDER = [Stage.RESEARCH, Stage.EXTRACTED, Stage.BACKTESTING, Stage.VALIDATION, Stage.WALK_FORWARD,
         Stage.PAPER_SHADOW, Stage.PAPER_ACTIVE, Stage.PROVEN]  # fmt: skip
EXECUTABLE_STAGES = frozenset({Stage.PAPER_ACTIVE, Stage.PROVEN})
SHADOW_STAGES = frozenset({Stage.PAPER_SHADOW, Stage.PAPER_ACTIVE, Stage.PROVEN})


@dataclass(frozen=True, slots=True)
class Policy:
    min_backtest_trades: int = 30
    max_overfit_risk: float = 0.5
    max_risk_of_ruin: float = 0.01
    min_shadow_trades: int = 20
    min_shadow_sessions: int = 20
    min_paper_trades: int = 50
    min_paper_t: float = 2.0


@dataclass
class Evidence:
    genome_problems: list[str] = field(default_factory=list)
    backtests: int = 0
    ror_by_model: dict[str, float | None] = field(default_factory=dict)  # full-period, by execution model
    backtest_trades: int = 0
    validation_ror: float | None = None
    overfit_risk: float | None = None
    walkforward_passed: bool | None = None
    montecarlo_ruin: float | None = None
    tail_passed: bool | None = None
    beats_baselines: bool | None = None
    fdr_discovery: bool | None = None
    critic_survived: bool | None = None
    shadow_trades: int = 0
    shadow_sessions: int = 0
    shadow_ror: float | None = None
    paper_trades: int = 0
    paper_ror: float | None = None
    paper_t: float | None = None
    decay_status: str | None = None
    family: str = ""
    human_approved_family: bool = False


def gate(target: Stage, ev: Evidence, policy: Policy | None = None) -> list[str]:
    """Every reason ``target`` is not yet earned (empty: it is)."""
    p = policy or Policy()
    r: list[str] = []
    if target == Stage.EXTRACTED:
        r += [f"genome: {x}" for x in ev.genome_problems]
    elif target == Stage.BACKTESTING:
        if ev.backtests < 1:
            r.append("no backtest has run")
    elif target == Stage.VALIDATION:
        if ev.backtest_trades < p.min_backtest_trades:
            r.append(f"{ev.backtest_trades} backtest trades (minimum {p.min_backtest_trades})")
        for m in ("REALISTIC", "PESSIMISTIC"):
            v = ev.ror_by_model.get(m)
            if v is None or v <= 0:
                r.append(f"not positive per dollar at risk under {m} fills ({v})")
    elif target == Stage.WALK_FORWARD:
        if ev.validation_ror is None or ev.validation_ror <= 0:
            r.append(f"the held-out validation period is not positive ({ev.validation_ror})")
        if ev.overfit_risk is None or ev.overfit_risk >= p.max_overfit_risk:
            r.append(f"overfit risk {ev.overfit_risk} (must be below {p.max_overfit_risk})")
    elif target == Stage.PAPER_SHADOW:
        if not ev.walkforward_passed:
            r.append("walk-forward not passed")
        if ev.montecarlo_ruin is None or ev.montecarlo_ruin > p.max_risk_of_ruin:
            r.append(f"Monte Carlo risk of ruin {ev.montecarlo_ruin} (limit {p.max_risk_of_ruin})")
        if not ev.tail_passed:
            r.append("tail-risk stress not passed")
        if not ev.beats_baselines:
            r.append("does not beat the baselines")
        if ev.fdr_discovery is False:
            r.append("not a discovery after false-discovery-rate control across the population")
        if ev.critic_survived is False:
            r.append("did not survive the critic")
    elif target == Stage.PAPER_ACTIVE:
        if ev.shadow_trades < p.min_shadow_trades:
            r.append(f"{ev.shadow_trades} shadow trades (minimum {p.min_shadow_trades})")
        if ev.shadow_sessions < p.min_shadow_sessions:
            r.append(f"{ev.shadow_sessions} shadow sessions (minimum {p.min_shadow_sessions})")
        if ev.shadow_ror is None or ev.shadow_ror <= 0:
            r.append(f"shadow result not positive net of costs ({ev.shadow_ror})")
        fam = FAMILIES.get(ev.family)
        if fam is None or not fam.defined_risk:
            r.append(f"{ev.family}: not a defined-risk family — never executable")
        elif not fam.default_executable and not ev.human_approved_family:
            r.append(f"{ev.family} needs a person's approval before paper execution")
    elif target == Stage.PROVEN:
        if ev.paper_trades < p.min_paper_trades:
            r.append(f"{ev.paper_trades} paper trades (minimum {p.min_paper_trades})")
        if ev.paper_ror is None or ev.paper_ror <= 0 or (ev.paper_t or 0) < p.min_paper_t:
            r.append(f"paper result not established (per $ at risk {ev.paper_ror}, t {ev.paper_t})")
        if ev.decay_status not in (None, "HEALTHY"):
            r.append(f"decay status {ev.decay_status}")
    return r


def next_stage(current: Stage) -> Stage | None:
    if current in (Stage.PROVEN, Stage.RETIRED):
        return None
    return ORDER[ORDER.index(current) + 1]


def advance(current: Stage, ev: Evidence, policy: Policy | None = None) -> tuple[Stage, list[str]]:
    """Move at most ONE stage forward if its gate passes; the reasons it did not, otherwise."""
    nxt = next_stage(current)
    if nxt is None:
        return current, ["no further stage"]
    missing = gate(nxt, ev, policy)
    return (nxt if not missing else current), missing


def demote_for(current: Stage, ev: Evidence) -> tuple[Stage, str] | None:
    """Evidence that removes a strategy from paper eligibility (automatic, allowed): decay BROKEN retires it;
    DEGRADING drops an active strategy back to shadow."""
    if ev.decay_status == "BROKEN" and current != Stage.RETIRED:
        return Stage.RETIRED, "decay BROKEN: retired from paper eligibility (history kept)"
    if ev.decay_status == "DEGRADING" and current in EXECUTABLE_STAGES:
        return Stage.PAPER_SHADOW, "decay DEGRADING: back to shadow until the evidence recovers"
    return None


def stage_record(
    stage: Stage, at: str, reason: str, evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {"stage": stage.value, "at": at, "reason": reason, "evidence": evidence or {}}
