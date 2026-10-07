"""The strategy population and its generations — a controlled search, never a blind one.

=============  ==========================================================================================
Generation 0   documented strategies (the research library, extracted), baselines, simple deterministic ones
Generation 1   bounded variations of strategies that passed at least VALIDATION
Generation 2   combinations of complementary survivors (crossover: one's entry and filters, the other's
               structure and exits)
Generation 3   regime-specific variants: a survivor restricted to the regimes where it earned its keep
Generation 4   portfolio-aware variants: smaller risk for strategies correlated with what is already active
Generation 5+  evolutionary: tournament selection on the robust score, with a novelty bonus
Every one      random immigrants: brand-new strategies (the most novel of several random draws), at least
               two and any budget the generation's own rule leaves unused, so the search keeps trying what
               nothing in the population resembles
Exploration    while no strategy has passed VALIDATION yet, the search does not wait: two-gene mutations of
               the best-scoring candidates and random immigrants (the generations above start once one passes)
=============  ==========================================================================================

Every child records its parents, its generation, its origin and the reason it exists; a child identical to an
existing genome is dropped. Parents are never modified. Each generation has a budget (the number of new
candidates) so the search stays inside what the service can compute without disturbing trading.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from quantpulse.options.lab.extraction import extract
from quantpulse.options.lab.genome import Genome, random_genome
from quantpulse.options.lab.research import SEED, Source


@dataclass(frozen=True, slots=True)
class Member:
    """What the population knows about one strategy version."""

    key: str
    version_id: int | None
    genome: Genome
    stage: str
    score: float | None = None  # robust objective (e.g. out-of-sample expectancy per $ at risk, shrunk)
    regimes: dict[str, float] | None = None  # expectancy per $ at risk by regime
    correlation: float | None = None  # with the active book
    strengths: tuple[str, ...] = ()  # e.g. ("iv_timing",), ("trend_filter",)


@dataclass(frozen=True, slots=True)
class Child:
    genome: Genome
    generation: int
    origin: str  # seed | extraction | baseline | mutation | crossover | regime | portfolio | evolution
    parents: tuple[str, ...]
    reason: str
    source_key: str | None = None
    assumptions: dict[str, Any] | None = None


PASSED = ("VALIDATION", "WALK_FORWARD", "PAPER_SHADOW", "PAPER_ACTIVE", "PROVEN")

SIMPLE: tuple[Genome, ...] = (
    Genome("long_call", "bullish", entry_signal="trend_up", dte_min=30, dte_max=60, delta_target=0.5),
    Genome("long_put", "bearish", entry_signal="trend_down", dte_min=30, dte_max=60, delta_target=0.5),
    Genome("bull_call_spread", "bullish", entry_signal="breakout_up", dte_min=30, dte_max=60, delta_target=0.5,
           width_pct=0.05, take_profit=0.8, stop_loss=0.5),
    Genome("bear_put_spread", "bearish", entry_signal="breakout_down", dte_min=30, dte_max=60, delta_target=0.5,
           width_pct=0.05, take_profit=0.8, stop_loss=0.5),
    Genome("bull_put_spread", "bullish", entry_signal="trend_up", iv_rank_min=50, dte_min=30, dte_max=45,
           delta_target=0.25, width_pct=0.05, take_profit=0.5, stop_loss=2.0),
    Genome("bear_call_spread", "bearish", entry_signal="trend_down", iv_rank_min=50, dte_min=30, dte_max=45,
           delta_target=0.25, width_pct=0.05, take_profit=0.5, stop_loss=2.0),
    Genome("long_call", "bullish", entry_signal="reversion_up", dte_min=20, dte_max=40, delta_target=0.4,
           take_profit=0.5, stop_loss=0.5, max_hold_days=10),
)  # fmt: skip


def generation0(sources: Sequence[Source] = SEED) -> list[Child]:
    """Documented strategies (each extracted with its assumptions), then simple deterministic ones."""
    out: list[Child] = []
    for s in sources:
        e = extract(s.rules_text)
        if e.genome is not None:
            out.append(Child(e.genome, 0, "extraction", (), f"extracted from {s.title} ({s.author})", s.key,
                             {"stated": e.stated, "assumed": e.assumed, "substitution": e.substitution}))  # fmt: skip
    for g in SIMPLE:
        out.append(
            Child(g, 0, "seed", (), "a simple deterministic strategy (a reference point for the search)")
        )
    return _unique(out, set())


def _unique(children: list[Child], existing: set[str]) -> list[Child]:
    seen = set(existing)
    out = []
    for c in children:
        if c.genome.valid and c.genome.hash not in seen:
            seen.add(c.genome.hash)
            out.append(c)
    return out


def next_generation(
    population: Sequence[Member], generation: int, *, budget: int = 12, seed: int = 0
) -> list[Child]:
    """The children of ``generation`` (≥ 1) from the current population, at most ``budget`` of them."""
    rng = random.Random(seed * 1000 + generation)
    existing = {m.genome.hash for m in population}
    alive = [m for m in population if m.stage in PASSED]
    ranked = sorted(alive, key=lambda m: m.score if m.score is not None else float("-inf"), reverse=True)
    out: list[Child] = []
    if generation == 1:
        for m in ranked[: max(1, budget // 3)]:
            for _ in range(3):
                c = m.genome.mutate(rng, changes=1)
                out.append(Child(c, 1, "mutation", (m.key,), f"a bounded variation of {m.key}"))
    elif generation == 2:
        pairs = [(a, b) for i, a in enumerate(ranked) for b in ranked[i + 1 :]
                 if set(a.strengths) != set(b.strengths) and a.genome.direction == b.genome.direction]  # fmt: skip
        for a, b in pairs[:budget]:
            c = a.genome.crossover(b.genome, rng)
            out.append(Child(c, 2, "crossover", (a.key, b.key),
                             f"{a.key} ({', '.join(a.strengths) or 'entry'}) × {b.key} ({', '.join(b.strengths) or 'structure'}): "
                             "tested from scratch, not assumed better"))  # fmt: skip
    elif generation == 3:
        for m in ranked:
            good = tuple(sorted(k for k, v in (m.regimes or {}).items() if v > 0 and k in _REGIME_LABELS))
            if good and set(good) != set(m.genome.regime_filter):
                c = replace(m.genome, regime_filter=good)
                out.append(
                    Child(
                        c,
                        3,
                        "regime",
                        (m.key,),
                        f"{m.key} restricted to the regimes where it was positive: {good}",
                    )
                )
    elif generation == 4:
        for m in ranked:
            if m.correlation is not None and m.correlation > 0.6:
                c = replace(m.genome, risk_per_trade=max(0.001, round(m.genome.risk_per_trade / 2, 4)))
                out.append(Child(c, 4, "portfolio", (m.key,), f"{m.key} at half the risk: {m.correlation:.2f} correlated "
                                 "with the active book"))  # fmt: skip
    else:
        pool = ranked[: max(4, budget)]
        for _ in range(budget):
            if len(pool) < 2:
                break
            a = max(rng.sample(pool, k=min(3, len(pool))), key=lambda m: m.score or float("-inf"))
            b = max(rng.sample(pool, k=min(3, len(pool))), key=lambda m: m.score or float("-inf"))
            child = a.genome.crossover(b.genome, rng) if a.key != b.key else a.genome
            child = child.mutate(rng, changes=1)
            out.append(Child(child, generation, "evolution", tuple(sorted({a.key, b.key})),
                             "tournament selection on the robust score, crossed and mutated"))  # fmt: skip
    picked = _unique(out, existing)[: max(0, budget - IMMIGRANTS)]
    return picked + immigrants(population, picked, rng, generation, budget - len(picked))


IMMIGRANTS = 2  # brand-new random strategies in every generation
DRAWS = 8  # random draws per immigrant; the most novel one is kept


def immigrants(population: Sequence[Member], also: Sequence[Child], rng: random.Random, generation: int,
               n: int) -> list[Child]:  # fmt: skip
    """``n`` brand-new random strategies, each the most novel of ``DRAWS`` draws against everything known."""
    known = [m.genome for m in population] + [c.genome for c in also]
    seen = {g.hash for g in known}
    out: list[Child] = []
    for _ in range(max(0, n)):
        draws = [g for g in (random_genome(rng) for _ in range(DRAWS)) if g.hash not in seen]
        if not draws:
            continue
        best = max(draws, key=lambda g: novelty(g, known))
        known.append(best)
        seen.add(best.hash)
        out.append(Child(best, generation, "immigrant", (), f"a brand-new random strategy (novelty "
                         f"{novelty(best, known[:-1]):.2f} against everything tried): {best.family}, "
                         f"entry {best.entry_signal}"))  # fmt: skip
    return out


def explore(population: Sequence[Member], *, budget: int, seed: int) -> list[Child]:
    """While nothing has passed VALIDATION: two-gene mutations of the best-scoring candidates so far, and
    random immigrants. Seeded by the population's size, so each round tries something new."""
    rng = random.Random(f"{seed}:{len(population)}")
    existing = {m.genome.hash for m in population}
    scored = sorted((m for m in population if m.score is not None and m.stage != "RETIRED"),
                    key=lambda m: m.score or 0.0, reverse=True)  # fmt: skip
    out: list[Child] = []
    for m in scored[: max(1, budget // 4)]:
        for _ in range(2):
            out.append(Child(m.genome.mutate(rng, changes=2), 0, "exploration", (m.key,),
                             f"nothing has passed VALIDATION yet: a wider variation of the best-scoring {m.key}"))  # fmt: skip
    picked = _unique(out, existing)[: budget // 2]
    return picked + immigrants(population, picked, rng, 0, budget - len(picked))


_REGIME_LABELS = frozenset({"TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "CALM", "PANIC", "HIGH_IV", "LOW_IV",
                            "NORMAL_IV"})  # fmt: skip


def novelty(g: Genome, others: Sequence[Genome]) -> float:
    """How different a genome is from the rest (0 = a duplicate, 1 = shares nothing)."""
    if not others:
        return 1.0
    mine = g.canonical()
    sims = []
    for o in others:
        theirs = o.canonical()
        same = sum(1 for k in mine if mine[k] == theirs.get(k))
        sims.append(same / len(mine))
    return round(1 - max(sims), 4)
