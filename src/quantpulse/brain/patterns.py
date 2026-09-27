"""Recurring patterns: what the Brain's graded history keeps saying, consolidated into memory.

Individual lessons are noisy (one bad outcome proves nothing). After each learning pass, :func:`consolidate`
counts across every graded decision and prediction and writes one long-term memory per pattern — updated
in place, so memory holds the pattern, not a pile of repetitions:

* **objections** — how often each of the devil's advocate's objections was borne out (the decision it was
  raised against went badly): an assumption that keeps being right deserves more weight, one that is
  rarely right is noise;
* **hypotheses** — how decisions on each kind of opportunity (breakouts, value dislocations, …) turned out:
  successful and failed hypotheses;
* **process** — the mix of earned / unlucky / lucky / process-failure outcomes;
* **regimes and volatility environments** — where the consensus has (or lacks) evidence of skill.

Every pattern carries its sample size and a 95% interval, and is *tentative* until the sample is large
enough and the interval excludes a coin flip; only then is it *established*. Nothing here changes a
weight or a rule: patterns are recalled into decisions as context (:func:`recall`) and feed improvement
proposals, which a person decides on.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select

from quantpulse.db.models import BrainAgentPerformanceRow, BrainReflectionRow
from quantpulse.db.session import Database

from .memory import LONG_TERM, MemoryStore
from .performance import wilson

MIN_PATTERN = 10  # observations before a pattern is even reported
TRADE_ACTIONS = frozenset({"buy", "increase", "reduce", "close", "sell", "de_risk", "rebalance"})


def _rate(k: int, n: int, min_established: int) -> dict[str, Any]:
    ci = wilson(k, n)
    established = bool(n >= min_established and ci is not None and (ci[0] > 0.5 or ci[1] < 0.5))
    return {
        "n": n,
        "k": k,
        "rate": round(k / n, 4) if n else None,
        "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "status": "established" if established else "tentative",
    }


def _interval(p: dict[str, Any]) -> str:
    return f"{p['ci'][0]:.0%}–{p['ci'][1]:.0%}" if p.get("ci") else "—"


async def find(db: Database, min_observations: int) -> list[dict[str, Any]]:
    """Every pattern with at least ``MIN_PATTERN`` observations."""
    async with db.session() as s:
        reflections = (
            await s.scalars(select(BrainReflectionRow).where(BrainReflectionRow.subject_type == "decision"))
        ).all()
        consensus = (
            await s.scalars(
                select(BrainAgentPerformanceRow).where(
                    BrainAgentPerformanceRow.agent_id == "consensus", BrainAgentPerformanceRow.window == "all"
                )
            )
        ).all()
    out: list[dict[str, Any]] = []

    objections: dict[str, list[bool]] = defaultdict(list)
    kinds: dict[str, list[bool]] = defaultdict(list)
    categories: dict[str, int] = defaultdict(int)
    for r in reflections:
        ev = r.evidence or {}
        for o in ev.get("objections") or []:
            objections[o["code"]].append(bool(o.get("borne_out")))
        if ev.get("action") in TRADE_ACTIONS and r.outcome_quality in ("good", "bad"):
            for k in ev.get("kinds") or []:
                kinds[k].append(r.outcome_quality == "good")
        categories[r.category] += 1

    for code, flags in sorted(objections.items()):
        if len(flags) < MIN_PATTERN:
            continue
        stat = _rate(sum(flags), len(flags), min_observations)
        out.append(
            {
                "kind": "objection",
                "name": code,
                "summary": f"objection '{code}' was borne out in {stat['k']} of {stat['n']} graded decisions "
                f"({stat['rate']:.0%}, 95% interval {_interval(stat)}): {stat['status']}",
                **stat,
            }
        )
    for kind, wins in sorted(kinds.items()):
        if len(wins) < MIN_PATTERN:
            continue
        stat = _rate(sum(wins), len(wins), min_observations)
        verdict = (
            "a successful hypothesis"
            if stat["status"] == "established" and stat["rate"] > 0.5
            else "a failed hypothesis"
            if stat["status"] == "established"
            else "not yet decided"
        )
        out.append(
            {
                "kind": "hypothesis",
                "name": kind,
                "summary": f"trades on {kind.replace('_', ' ')} ideas went well {stat['k']} of {stat['n']} times "
                f"({stat['rate']:.0%}, interval {_interval(stat)}): {verdict}",
                **stat,
            }
        )
    total = sum(categories.values())
    if total >= MIN_PATTERN:
        mix = {k: round(v / total, 3) for k, v in sorted(categories.items())}
        out.append(
            {
                "kind": "process",
                "name": "outcomes",
                "summary": f"{total} graded decisions: "
                + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in mix.items()),
                "n": total,
                "mix": mix,
                "status": "established" if total >= min_observations else "tentative",
            }
        )
    for row in consensus:
        if row.regime == "all" or row.regime.startswith("at:") or (row.n_effective or 0) < MIN_PATTERN:
            continue
        label = row.regime.replace("vol:", "") + (
            " volatility" if row.regime.startswith("vol:") else " regime"
        )
        out.append(
            {
                "kind": "environment",
                "name": row.regime,
                "summary": f"consensus in a {label}: {row.verdict or 'unproven'} "
                f"(hit rate {row.hit_rate or 0:.0%} over {row.n_effective} independent calls)",
                "n": row.n_effective,
                "rate": row.hit_rate,
                "ci": [row.ci_low, row.ci_high] if row.ci_low is not None else None,
                "status": "established"
                if row.verdict in ("evidence of skill", "evidence of harm")
                else "tentative",
            }
        )
    return out


async def consolidate(db: Database, memory: MemoryStore, now: datetime, min_observations: int) -> int:
    """Write every pattern to long-term memory (one entry each, updated in place); returns the count."""
    found = await find(db, min_observations)
    for p in found:
        await memory.remember(
            LONG_TERM,
            "pattern",
            p["name"][:24],
            p["summary"],
            now,
            key=f"pattern:{p['kind']}:{p['name']}",
            data=p,
            tags=["pattern", p["kind"], p["name"], p["status"]],
            importance=0.8 if p["status"] == "established" else 0.4,
        )
    return len(found)


def recall(
    subject: str,
    *,
    patterns: Sequence[dict[str, Any]],
    lessons: Sequence[dict[str, Any]],
    objections: Sequence[str],
    kinds: Sequence[str],
    regime: str | None,
    limit: int = 5,
) -> list[str]:
    """What memory says about this decision: past lessons on the subject, and the patterns for the
    objections raised, the kinds of opportunity involved and the current regime (established first)."""
    wanted = {f"pattern:objection:{c}" for c in objections} | {f"pattern:hypothesis:{k}" for k in kinds}
    if regime:
        wanted.add(f"pattern:environment:{regime}")
    hits = [p for p in patterns if p.get("key") in wanted]
    hits.sort(key=lambda p: (p["data"].get("status") != "established", -(p["data"].get("n") or 0)))
    out = [f"pattern: {p['summary']}" for p in hits]
    out += [
        f"lesson ({m['created_at']:%Y-%m-%d}): {m['summary']}" for m in lessons if m["subject"] == subject
    ][:2]
    return out[:limit]
