"""Self-critique and lessons: after each meaningful trade, what did we believe, what happened, and what should
change (and what should not)?

:func:`critique` turns a closed trade — its thesis, its attribution, its counterfactuals, its execution —
into structured answers: which assumptions held and which failed, what information was missing, and whether
the fault lay with the strategy, the implementation (strike, expiration, structure), the execution, or simply
an unfavourable market. :func:`lessons_from` turns critiques into candidate lessons (context, observation,
hypothesis, evidence, confidence, sample size, date range, applicability, an expiry). One trade never changes
a strategy: a lesson becomes an experiment only when replicated (``min_replications``) and is then tested out
of sample like everything else.

:func:`classify_missed` grades a rejected candidate once its outcome is known: GOOD_REJECTION,
BAD_REJECTION, MISSED_WINNER, CORRECTLY_AVOIDED_LOSER or INSUFFICIENT_DATA — so learning sees the trades
not taken as well as the ones taken (no survivorship bias).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any

MEMORIES = ("StrategyMemory", "MarketRegimeMemory", "OptionsLessonMemory", "ExecutionMemory",
            "ContractSelectionMemory", "VolatilityMemory", "FailureMemory", "CounterfactualMemory")  # fmt: skip


def critique(trade: Mapping[str, Any], *, thesis: Mapping[str, Any] | None = None,
             counterfactual: Mapping[str, Any] | None = None, execution: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    a = trade.get("attribution") or {}
    direction = trade.get("direction")
    move = trade.get("underlying_return")
    held, failed, missing = [], [], []
    dir_ok = (
        None
        if move is None or direction not in ("bullish", "bearish")
        else (move > 0) == (direction == "bullish")
    )
    if dir_ok is True:
        held.append("direction")
    elif dir_ok is False:
        failed.append("direction")
    vega = float(a.get("vega") or 0.0)
    if abs(vega) > 0.2 * abs(float(trade.get("pnl") or 0.0)) and abs(vega) > 1:
        (held if vega > 0 else failed).append("volatility")
    theta = float(a.get("theta") or 0.0)
    if theta < 0 and abs(theta) > 0.3 * abs(float(trade.get("pnl") or 1.0)):
        failed.append("time decay (the move came too slowly for the premium paid)")
    exec_cost = -float(a.get("execution") or 0.0) - float(a.get("fees") or 0.0)
    if exec_cost > 0.25 * abs(float(trade.get("pnl") or 1.0)) and exec_cost > 5:
        failed.append("execution (spread and fees ate a large share)")
    if not trade.get("features", {}).get("event_days"):
        missing.append("event calendar (no earnings date known)")
    if trade.get("features", {}).get("iv_rank") is None:
        missing.append("IV history (no IV rank)")
    structure_ok = (counterfactual or {}).get("structure_correct")
    if dir_ok and trade.get("pnl", 0) < 0:
        fault = (
            "implementation"
            if structure_ok is False
            else "execution"
            if "execution" in " ".join(failed)
            else "market"
        )
    elif dir_ok is False:
        fault = "strategy" if trade.get("pnl", 0) < 0 else "luck"
    else:
        fault = "none" if trade.get("pnl", 0) >= 0 else "market"
    change, keep = [], []
    if fault == "implementation" and (counterfactual or {}).get("best_alternative"):
        change.append(f"test {counterfactual['best_alternative']} for this setup")  # type: ignore[index]
    if "volatility" in failed:
        change.append("test an IV filter or a lower-vega structure")
    if "time decay (the move came too slowly for the premium paid)" in failed:
        change.append("test a longer expiration or a spread")
    if dir_ok:
        keep.append("the directional signal")
    return {
        "believed": (thesis or {}).get("thesis")
        or f"{direction} on {trade.get('underlying')} via {trade.get('family')}",
        "happened": f"underlying {move:+.1%}, P&L {trade.get('pnl')}"
        if move is not None
        else f"P&L {trade.get('pnl')}",
        "assumptions_held": held,
        "assumptions_failed": failed,
        "missing_information": missing,
        "fault": fault,  # strategy | implementation | execution | market | luck | none
        "should_change": change,
        "should_not_change": keep,
        "dominant_driver": max(("delta", "gamma", "theta", "vega"), key=lambda k: abs(float(a.get(k) or 0.0)))
        if a
        else None,
        "execution": dict(execution or {}),
    }


def lessons_from(critiques: Iterable[Mapping[str, Any]], *, min_replications: int = 3, today: date,
                 expiry_days: int = 365) -> list[dict[str, Any]]:  # fmt: skip
    """Group critiques by (fault, failed assumption, context); a pattern seen ``min_replications`` times
    becomes a candidate lesson with its evidence. One trade alone never does."""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for c in critiques:
        ctx = c.get("context") or {}
        for f in c.get("assumptions_failed") or ["-"]:
            groups[(c["fault"], f, str(ctx.get("family")), str(ctx.get("iv_regime")))].append(c)
    out = []
    for (fault, failed, fam, ivr), items in groups.items():
        if len(items) < min_replications or fault in ("none", "luck"):
            continue
        dates = sorted(str((i.get("context") or {}).get("exit_date") or today.isoformat()) for i in items)
        memory = {"implementation": "ContractSelectionMemory", "execution": "ExecutionMemory",
                  "strategy": "StrategyMemory", "market": "MarketRegimeMemory"}.get(fault, "OptionsLessonMemory")  # fmt: skip
        if "volatility" in failed:
            memory = "VolatilityMemory"
        out.append({
            "memory": memory,
            "kind": fault,
            "context": {"family": fam, "iv_regime": ivr},
            "observation": f"{len(items)} {fam} trades in {ivr}: {failed} failed ({fault})",
            "hypothesis": f"{fam} in {ivr} is hurt by {failed}; a variant addressing it should do better out of sample",
            "evidence": {"trades": len(items), "faults": fault},
            "confidence": round(min(0.9, len(items) / (len(items) + 10)), 3),
            "sample_size": len(items),
            "date_from": dates[0][:10],
            "date_to": dates[-1][:10],
            "applicability": {"family": fam, "iv_regime": ivr},
            "expires_on": (today + timedelta(days=expiry_days)).isoformat(),
            "status": "candidate",
        })  # fmt: skip
    return out


def classify_missed(outcome_pnl: float | None, *, rejected_for: str, would_have_passed_risk: bool | None,
                    threshold: float = 0.0) -> str:  # fmt: skip
    """How a rejection looks once the market has spoken."""
    if outcome_pnl is None:
        return "INSUFFICIENT_DATA"
    won = outcome_pnl > threshold
    safety = any(
        w in rejected_for for w in ("stale", "spread", "liquidity", "kill switch", "risk", "data", "closed")
    )
    if won:
        return "GOOD_REJECTION" if safety else "MISSED_WINNER" if would_have_passed_risk else "BAD_REJECTION"
    return "CORRECTLY_AVOIDED_LOSER"


def missed_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    by_gate: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cls = str(r.get("classification") or "UNGRADED")
        counts[cls] += 1
        by_gate[str(r.get("gate") or "?")][cls] += 1
    return {"counts": dict(counts), "by_gate": {k: dict(v) for k, v in by_gate.items()}}
