"""ExperimentGenerator and the experiment queue.

When a strategy fails in a recognisable way, the generator proposes competing, testable fixes — each an
explicit change to the genome (a child version), never an edit of the parent. Example: a strategy that loses
in low implied volatility yields "require IV rank > 40", "switch to a defined-risk premium seller", "longer
expiration", "avoid low-IV regimes", "require volatility expansion".

Every experiment carries its hypothesis, parent, changes, dataset, periods, metrics, status (QUEUED →
RUNNING → PASSED | FAILED | INCONCLUSIVE → PROMOTED | REJECTED) and decision.

**Priority is expected information, not expected profit**: an experiment on a strategy whose edge is
uncertain (few trades, wide interval) and whose evidence is still promising ranks above one on a strategy
with ten thousand observations and a tiny, well-measured edge. Concretely: the expected reduction in the
posterior standard deviation of the parent's mean if the experiment adds its trades, times the probability
the parent's mean is positive, plus a bonus for untested hypothesis types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from quantpulse.options.lab.genome import Genome

STATUSES = ("QUEUED", "RUNNING", "PASSED", "FAILED", "INCONCLUSIVE", "PROMOTED", "REJECTED")


@dataclass
class Proposal:
    hypothesis: str
    parent_hash: str
    child: Genome
    changes: dict[str, Any]
    source: str  # failure | decay | lesson | meta | research | evolution
    kind: str = "parameter"
    rationale: dict[str, Any] = field(default_factory=dict)


def _child(parent: Genome, **changes: Any) -> Genome | None:
    try:
        g = replace(parent, **changes)
    except TypeError:
        return None
    return g if g.valid and g.hash != parent.hash else None


def from_failure(parent: Genome, finding: str, evidence: Mapping[str, Any] | None = None) -> list[Proposal]:
    """Competing fixes for one observed failure mode (``finding``: low_iv | high_iv | iv_crush | slow_move |
    execution | event_losses | trend_reversal | regime:<NAME>)."""
    ev = dict(evidence or {})
    ideas: list[tuple[str, dict[str, Any], str]] = []
    if finding == "low_iv":
        ideas += [("H1: require IV rank > 40", {"iv_rank_min": 40.0}, "filter"),
                  ("H2: a defined-risk premium seller instead of buying premium",
                   {"family": "bull_put_spread", "delta_target": 0.25, "width_pct": 0.05, "take_profit": 0.5, "stop_loss": 2.0}
                   if parent.direction == "bullish" else
                   {"family": "bear_call_spread", "delta_target": 0.25, "width_pct": 0.05, "take_profit": 0.5, "stop_loss": 2.0},
                   "structure"),
                  ("H3: a longer expiration", {"dte_min": parent.dte_min + 15, "dte_max": parent.dte_max + 30}, "expiration"),
                  ("H4: avoid low-IV regimes", {"regime_filter": ("NORMAL_IV", "HIGH_IV")}, "regime"),
                  ("H5: require volatility expansion first", {"entry_signal": "iv_high"}, "signal")]  # fmt: skip
    elif finding in ("high_iv", "iv_crush"):
        ideas += [("require IV rank < 60 when buying premium", {"iv_rank_max": 60.0}, "filter"),
                  ("a spread to cut vega", {"family": "bull_call_spread" if parent.direction == "bullish" else "bear_put_spread",
                                            "width_pct": 0.05}, "structure"),
                  ("avoid earnings inside the option's life", {"event_filter": "avoid"}, "event")]  # fmt: skip
    elif finding == "slow_move":
        ideas += [("a longer expiration", {"dte_min": parent.dte_min + 15, "dte_max": parent.dte_max + 30}, "expiration"),
                  ("a higher delta (less time value)", {"delta_target": min(0.8, parent.delta_target + 0.15)}, "strike"),
                  ("a debit spread", {"family": "bull_call_spread" if parent.direction == "bullish" else "bear_put_spread",
                                      "width_pct": 0.05}, "structure")]  # fmt: skip
    elif finding == "execution":
        ideas += [("tighter liquidity filter", {"max_spread_pct": max(0.02, parent.max_spread_pct / 2)}, "liquidity"),
                  ("higher open-interest minimum", {"min_open_interest": parent.min_open_interest * 3 + 100}, "liquidity")]  # fmt: skip
    elif finding == "event_losses":
        ideas += [("avoid earnings", {"event_filter": "avoid"}, "event"),
                  ("shorter expirations that end before events", {"dte_min": 7, "dte_max": 21, "exit_dte": 3}, "expiration")]  # fmt: skip
    elif finding == "trend_reversal":
        ideas += [("require the 50/200 trend to agree", {"entry_signal": "trend_up" if parent.direction == "bullish" else "trend_down"}, "signal"),
                  ("a tighter stop", {"stop_loss": max(0.2, (parent.stop_loss or 1.0) * 0.6)}, "exit"),
                  ("a shorter holding period", {"max_hold_days": max(3, parent.max_hold_days // 2)}, "exit")]  # fmt: skip
    elif finding.startswith("regime:"):
        bad = finding.split(":", 1)[1]
        from quantpulse.options.lab.genome import REGIMES

        allowed = tuple(r for r in REGIMES if r != bad)
        ideas += [(f"avoid {bad}", {"regime_filter": allowed}, "regime")]
    out = []
    for text, changes, kind in ideas:
        child = _child(parent, **changes)
        if child is not None:
            out.append(
                Proposal(text, parent.hash, child, changes, "failure", kind, {"finding": finding, **ev})
            )
    return out


def information_value(*, n: float, sd: float, mean: float, added: float, tested_kinds: Sequence[str] = (),
                      kind: str = "parameter") -> float:  # fmt: skip
    """Expected reduction in the posterior sd of the mean from ``added`` more trades, weighted by how likely the
    parent is worth improving (P(mean > 0)), plus a bonus for a kind of hypothesis not yet tested."""
    from scipy.stats import norm

    sd = max(sd, 1e-6)
    before = sd / math.sqrt(max(n, 1.0))
    after = sd / math.sqrt(max(n, 1.0) + max(added, 0.0))
    promise = float(norm.cdf(mean / before)) if before > 0 else 0.5
    novelty = 0.25 if kind not in tested_kinds else 0.0
    return round((before - after) / sd * (0.25 + promise) + novelty, 6)


def prioritize(queue: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Queued experiments, highest expected information first (ties: oldest first)."""
    return sorted((q for q in queue if q.get("status") == "QUEUED"),
                  key=lambda q: (-(q.get("priority") or 0.0), q.get("created_at") or ""))  # fmt: skip


def decide(parent: Mapping[str, Any], child: Mapping[str, Any], *, min_trades: int = 20) -> tuple[str, str]:
    """PASSED when the child beats the parent out of sample by a margin with enough trades; FAILED when it is
    worse; INCONCLUSIVE otherwise. Passing does not promote: the child then enters the pipeline as a version."""
    n = child.get("trades") or 0
    c, p = child.get("expectancy_on_risk"), parent.get("expectancy_on_risk")
    if n < min_trades or c is None or p is None:
        return "INCONCLUSIVE", f"{n} out-of-sample trades (need {min_trades})"
    if c > p + 0.02 and c > 0:
        return "PASSED", f"child {c:+.3f} vs parent {p:+.3f} per $ at risk out of sample"
    if c < p - 0.02:
        return "FAILED", f"child {c:+.3f} worse than parent {p:+.3f}"
    return "INCONCLUSIVE", f"child {c:+.3f} vs parent {p:+.3f}: no clear difference"
