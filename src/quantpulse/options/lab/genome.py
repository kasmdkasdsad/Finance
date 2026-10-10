"""The strategy genome: a machine-readable options strategy in which every parameter is explicit.

A genome says what to trade (the structure family and its direction), when (an entry signal and filters on
implied volatility, its rank and percentile, IV against realized volatility, the term structure, the skew,
the market regime and scheduled events), how (days to expiration, the target delta of the main leg, the
spread width), how much (the share of equity put at risk) and when to leave (take profit, stop loss, an exit
before expiration, a maximum holding time). Nothing is left vague: a strategy that cannot be written this way
is not a strategy the lab will test.

Genomes are immutable and hashed; mutation and crossover produce new genomes (and new strategy versions),
never edit old ones. The numbers in :data:`DEFAULTS` are starting points for search, not claims that they
are good.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from quantpulse.options.structures import FAMILIES

SIGNALS = (
    "always",  # enter whenever the filters pass (a pure volatility/structure strategy)
    "trend_up",  # the underlying above its 50-day average and the 50-day above the 200-day
    "trend_down",
    "momentum_up",  # 20-day return in the top of its own history
    "momentum_down",
    "breakout_up",  # a close above the 20-day high
    "breakout_down",
    "reversion_up",  # an oversold stretch (z-score of the 5-day return below −2)
    "reversion_down",
    "iv_high",  # implied volatility rich (rank filter decides how rich)
    "iv_low",
    "pre_event",  # an earnings date inside the option's life
    "random",  # a control: entries at random (seeded) — for baselines only
)
REGIMES = (
    "TRENDING_UP",
    "TRENDING_DOWN",
    "MEAN_REVERTING",
    "CALM",
    "PANIC",
    "HIGH_IV",
    "LOW_IV",
    "NORMAL_IV",
)
TERM = ("any", "contango", "backwardation")
SKEW = ("any", "put_rich", "call_rich")
EVENTS = ("ignore", "avoid", "require")
UNDERLYINGS = ("liquid_large_cap", "index_etf", "any_liquid")
INTEGER_GENES = frozenset({"dte_min", "dte_max", "max_hold_days", "exit_dte"})

# (min, max, step) for the searchable numeric genes; None-able genes may also be switched off
BOUNDS: dict[str, tuple[float, float, float]] = {
    "iv_rank_min": (0, 95, 5),
    "iv_rank_max": (5, 100, 5),
    "iv_percentile_min": (0, 95, 5),
    "iv_percentile_max": (5, 100, 5),
    "iv_rv_min": (0.5, 2.5, 0.05),
    "iv_rv_max": (0.5, 3.0, 0.05),
    "dte_min": (1, 180, 1),
    "dte_max": (2, 365, 1),
    "delta_target": (0.05, 0.9, 0.01),
    "delta_min": (0.02, 0.9, 0.01),
    "delta_max": (0.05, 0.95, 0.01),
    "width_pct": (0.01, 0.30, 0.005),
    "wing_pct": (0.01, 0.30, 0.005),
    "take_profit": (0.1, 3.0, 0.05),
    "stop_loss": (0.2, 4.0, 0.05),
    "max_hold_days": (1, 120, 1),
    "exit_dte": (0, 60, 1),
    "max_spread_pct": (0.01, 0.30, 0.01),
    "min_open_interest": (0, 5000, 50),
    "risk_per_trade": (0.001, 0.05, 0.001),
}


@dataclass(frozen=True, slots=True)
class Genome:
    family: str
    direction: str
    entry_signal: str = "always"
    underlying_filter: str = "liquid_large_cap"
    iv_rank_min: float | None = None
    iv_rank_max: float | None = None
    iv_percentile_min: float | None = None
    iv_percentile_max: float | None = None
    iv_rv_min: float | None = None
    iv_rv_max: float | None = None
    term_filter: str = "any"
    skew_filter: str = "any"
    regime_filter: tuple[str, ...] = ()
    event_filter: str = "avoid"
    dte_min: int = 30
    dte_max: int = 60
    delta_target: float = 0.30  # |delta| of the main (or short) leg
    delta_min: float | None = None
    delta_max: float | None = None
    width_pct: float | None = None  # spread width as a share of the spot (verticals, condor wings)
    wing_pct: float | None = None  # distance of a strangle's legs / a condor's short strikes from the spot
    take_profit: float | None = 0.5  # debit: +50% of the debit; credit: 50% of the credit kept
    stop_loss: float | None = 1.0  # debit: lose this share of the debit; credit: lose this multiple of it
    max_hold_days: int = 45
    exit_dte: int = 7  # close when this many days are left (never carried into expiration week by default)
    max_spread_pct: float = 0.10
    min_open_interest: float = 100
    risk_per_trade: float = 0.01  # share of equity at risk (the structure's maximum loss)
    hedge_rule: str = "none"
    exit_rule: str = "standard"
    roll_rule: str = "none"
    zero_dte: bool = False
    seed: int | None = None  # for the random-entry control only
    notes: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    # ------------------------------------------------------------------ identity
    def canonical(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("notes")
        d["regime_filter"] = sorted(d["regime_filter"])
        return d

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True).encode()).hexdigest()[:32]

    @property
    def parameter_count(self) -> int:
        """Degrees of freedom actually used: every gene switched on (a filter set, a regime named)."""
        n = 4  # family, signal, DTE window, delta — always chosen
        for name in ("iv_rank_min", "iv_rank_max", "iv_percentile_min", "iv_percentile_max", "iv_rv_min",
                     "iv_rv_max", "delta_min", "delta_max", "width_pct", "wing_pct", "take_profit", "stop_loss"):  # fmt: skip
            n += getattr(self, name) is not None
        n += self.term_filter != "any"
        n += self.skew_filter != "any"
        n += len(self.regime_filter)
        n += self.event_filter != "ignore"
        return n

    # ------------------------------------------------------------------ validity
    def problems(self) -> list[str]:
        """Every reason this genome is not a testable, explicit strategy (empty: it is)."""
        out: list[str] = []
        fam = FAMILIES.get(self.family)
        if fam is None or self.family == "stock":
            out.append(f"unknown structure family {self.family!r}")
        elif fam.direction != self.direction and self.direction not in ("neutral", "volatility", "income"):
            out.append(f"{self.family} is {fam.direction}, not {self.direction}")
        if self.entry_signal not in SIGNALS:
            out.append(f"unknown entry signal {self.entry_signal!r}")
        if self.term_filter not in TERM or self.skew_filter not in SKEW or self.event_filter not in EVENTS:
            out.append("unknown term, skew or event filter")
        if self.underlying_filter not in UNDERLYINGS:
            out.append(f"unknown underlying filter {self.underlying_filter!r}")
        bad = [r for r in self.regime_filter if r not in REGIMES]
        if bad:
            out.append(f"unknown regimes {bad}")
        if self.dte_min < 1 and not self.zero_dte:
            out.append("DTE below 1 is the separate 0DTE family")
        if self.dte_min > self.dte_max:
            out.append("dte_min above dte_max")
        if self.exit_dte >= self.dte_min and not self.zero_dte:
            out.append("exits (exit_dte) before it could ever enter")
        for lo, hi in (("iv_rank_min", "iv_rank_max"), ("iv_percentile_min", "iv_percentile_max"), ("iv_rv_min", "iv_rv_max"),
                       ("delta_min", "delta_max")):  # fmt: skip
            a, b = getattr(self, lo), getattr(self, hi)
            if a is not None and b is not None and a > b:
                out.append(f"{lo} above {hi}")
        for name, (low, high, _) in BOUNDS.items():
            v = getattr(self, name)
            if self.zero_dte and name in ("dte_min", "dte_max"):
                low = 0  # the 0DTE family has its own (research-only) range
            if v is not None and not (low <= v <= high):
                out.append(f"{name}={v} outside [{low}, {high}]")
        spreads = ("bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread", "iron_condor",
                   "call_butterfly", "put_butterfly", "broken_wing_butterfly", "reverse_iron_condor")  # fmt: skip
        if self.family in spreads and self.width_pct is None:
            out.append(f"{self.family} needs an explicit width_pct")
        if self.family in ("long_strangle", "iron_condor", "iron_butterfly") and self.wing_pct is None:
            out.append(f"{self.family} needs an explicit wing_pct")
        if self.entry_signal == "random" and self.seed is None:
            out.append("a random control needs a seed")
        return out

    @property
    def valid(self) -> bool:
        return not self.problems()

    # ------------------------------------------------------------------ search operators
    def mutate(self, rng: random.Random, changes: int = 1) -> Genome:
        """A variation: ``changes`` genes moved a few steps inside their bounds; an unset IV or delta filter
        may be switched on, and now and then a categorical gene (the term, skew or event filter) changes.
        The child is always valid — or the parent comes back unchanged."""
        optional = ("iv_rank_min", "iv_rank_max", "iv_percentile_min", "iv_percentile_max", "iv_rv_min", "iv_rv_max",
                    "delta_min", "delta_max")  # fmt: skip
        genes = [n for n in BOUNDS if getattr(self, n) is not None or n in optional]
        for _ in range(25):
            g = self
            for name in rng.sample(genes, k=min(changes, len(genes))):
                lo, hi, step = BOUNDS[name]
                cur = getattr(g, name)
                raw = rng.choice([lo + (hi - lo) * q for q in (0.25, 0.5, 0.75)]) if cur is None else (
                    cur + step * rng.choice([-3, -2, -1, 1, 2, 3]))  # fmt: skip
                value: float = min(hi, max(lo, round(round(raw / step) * step, 6)))
                g = _with(g, **{name: int(value) if name in INTEGER_GENES else value})
            if rng.random() < 0.2:
                gene, choices = rng.choice(
                    [("term_filter", TERM), ("skew_filter", SKEW), ("event_filter", EVENTS)]
                )
                g = _with(g, **{gene: rng.choice(choices)})
            if g.valid and g.hash != self.hash:
                return g
        return self

    def crossover(self, other: Genome, rng: random.Random) -> Genome:
        """Entry and filters from one parent, structure and exits from the other (when they express the
        same direction). Returns ``self`` unchanged when no valid child exists."""
        entry = ("entry_signal", "iv_rank_min", "iv_rank_max", "iv_percentile_min", "iv_percentile_max", "iv_rv_min",
                 "iv_rv_max", "term_filter", "skew_filter", "regime_filter", "event_filter")  # fmt: skip
        a, b = (self, other) if rng.random() < 0.5 else (other, self)
        child = _with(b, **{f: getattr(a, f) for f in entry})
        if child.valid and child.hash not in (self.hash, other.hash):
            return child
        return self

    # ------------------------------------------------------------------ words
    def describe(self) -> str:
        fam = FAMILIES.get(self.family)
        parts = [f"{self.family.replace('_', ' ')} ({fam.description if fam else '?'})",
                 f"entry: {self.entry_signal.replace('_', ' ')}"]  # fmt: skip
        f = []
        if self.iv_rank_min is not None or self.iv_rank_max is not None:
            f.append(
                f"IV rank {self.iv_rank_min or 0:g}–{self.iv_rank_max if self.iv_rank_max is not None else 100:g}"
            )
        if self.iv_rv_min is not None or self.iv_rv_max is not None:
            f.append(f"IV/RV {self.iv_rv_min or 0:g}–{self.iv_rv_max or math.inf:g}")
        if self.term_filter != "any":
            f.append(self.term_filter)
        if self.skew_filter != "any":
            f.append(self.skew_filter.replace("_", " "))
        if self.regime_filter:
            f.append("regimes " + "/".join(self.regime_filter))
        if self.event_filter != "ignore":
            f.append(f"{self.event_filter} earnings")
        if f:
            parts.append("filters: " + ", ".join(f))
        parts.append(f"{self.dte_min}–{self.dte_max} DTE, |delta| {self.delta_target:g}"
                     + (f", width {self.width_pct:.1%}" if self.width_pct else "")
                     + (f", wings {self.wing_pct:.1%}" if self.wing_pct else ""))  # fmt: skip
        parts.append(f"exit: take profit {self.take_profit}, stop {self.stop_loss}, at {self.exit_dte} DTE or after "
                     f"{self.max_hold_days} days; risk {self.risk_per_trade:.1%} of equity")  # fmt: skip
        return "; ".join(parts)


# Random new strategies are drawn only from the families executable by default, so one that earns its stages
# can trade (one-contract exploration, then PAPER_ACTIVE) without a person having to enable its family first.
RANDOM_FAMILIES = ("long_call", "long_put", "bull_call_spread", "bear_put_spread", "bull_put_spread",
                   "bear_call_spread")  # fmt: skip
RANDOM_ENTRIES = {
    "bullish": ("always", "trend_up", "momentum_up", "breakout_up", "reversion_up", "iv_low", "iv_high", "pre_event"),
    "bearish": ("always", "trend_down", "momentum_down", "breakout_down", "reversion_down", "iv_low", "iv_high",
                "pre_event"),
}  # fmt: skip
SPREADS = ("bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread")


def random_genome(rng: random.Random) -> Genome:
    """A brand-new strategy drawn from the whole searchable space of the default executable families: entry
    signal, IV filter, expiry window, delta, width, exits and size, all at random within their bounds. Always
    valid and defined-risk (never 0DTE, never naked)."""
    for _ in range(100):
        family = rng.choice(RANDOM_FAMILIES)
        fam = FAMILIES[family]
        credit = fam.vol == "short_vol"
        signal = rng.choice(RANDOM_ENTRIES[fam.direction])
        dte_min = rng.randrange(14, 61)
        iv: dict[str, Any] = {}
        if rng.random() < 0.5:  # credit spreads sell rich volatility; debits buy it cheap
            iv = (
                {"iv_rank_min": float(rng.randrange(30, 75, 5))}
                if credit
                else {"iv_rank_max": float(rng.randrange(30, 75, 5))}
            )
        g = Genome(
            family, fam.direction, entry_signal=signal,
            event_filter="require" if signal == "pre_event" else rng.choice(("avoid", "ignore")),
            dte_min=dte_min, dte_max=dte_min + rng.randrange(10, 46),
            delta_target=round(rng.uniform(0.15, 0.35) if credit else rng.uniform(0.25, 0.65), 2),
            width_pct=round(rng.choice((0.02, 0.03, 0.04, 0.05, 0.06, 0.08)), 3) if family in SPREADS else None,
            take_profit=rng.choice((0.25, 0.5, 0.75, 1.0)), stop_loss=rng.choice((0.5, 1.0, 1.5, 2.0)),
            max_hold_days=rng.randrange(5, 46), exit_dte=rng.randrange(1, min(dte_min, 15)),
            risk_per_trade=rng.choice((0.005, 0.01, 0.015, 0.02)), **iv,
        )  # fmt: skip
        if g.valid:
            return g
    return Genome("long_call", "bullish", entry_signal="trend_up", dte_min=30, dte_max=60, delta_target=0.5)


# The other defined-risk families, researched and shadow-traded on live quotes like any strategy — but never
# traded on the paper account until a person enables the family (QP_OPTIONS_ALLOWED_STRUCTURES and the
# promotion gate's family approval): their evidence builds while they wait. Calendars are left out: the
# backtester reads one DTE window of a chain and a calendar needs a later expiration outside it.
RESEARCH_FAMILIES = ("long_straddle", "long_strangle", "iron_condor", "call_butterfly", "put_butterfly",
                     "iron_butterfly", "broken_wing_butterfly", "reverse_iron_condor")  # fmt: skip
RESEARCH_ENTRIES = {"short_vol": ("always", "iv_high"), "long_vol": ("always", "iv_low", "pre_event")}


def research_genome(rng: random.Random) -> Genome:
    """A brand-new strategy of one of the research families (see :data:`RESEARCH_FAMILIES`): volatility sellers
    enter when volatility is rich, buyers when it is cheap or before an event. Always valid and defined-risk."""
    for _ in range(100):
        family = rng.choice(RESEARCH_FAMILIES)
        fam = FAMILIES[family]
        short = fam.vol == "short_vol"
        signal = rng.choice(RESEARCH_ENTRIES["short_vol" if short else "long_vol"])
        dte_min = rng.randrange(14, 46)
        iv: dict[str, Any] = {}
        if rng.random() < 0.6:
            iv = (
                {"iv_rank_min": float(rng.randrange(40, 80, 5))}
                if short
                else {"iv_rank_max": float(rng.randrange(25, 60, 5))}
            )
        width = round(rng.choice((0.02, 0.03, 0.04, 0.05, 0.06)), 3)
        wing = round(rng.choice((0.03, 0.04, 0.05, 0.07, 0.10)), 3)
        g = Genome(
            family, fam.direction, entry_signal=signal,
            event_filter="require" if signal == "pre_event" else "avoid" if short else rng.choice(("avoid", "ignore")),
            dte_min=dte_min, dte_max=dte_min + rng.randrange(10, 31),
            delta_target=round(rng.uniform(0.15, 0.30) if family in ("iron_condor", "broken_wing_butterfly")
                               else rng.uniform(0.25, 0.50), 2),
            width_pct=width if family not in ("long_straddle", "long_strangle", "iron_butterfly") else None,
            wing_pct=wing if family in ("long_strangle", "iron_condor", "iron_butterfly") else None,
            take_profit=rng.choice((0.25, 0.5, 0.75)) if short else rng.choice((0.5, 1.0, 1.5)),
            stop_loss=rng.choice((1.0, 1.5, 2.0)) if short else rng.choice((0.5, 0.75, 1.0)),
            max_hold_days=rng.randrange(5, 31), exit_dte=rng.randrange(1, min(dte_min, 10)),
            risk_per_trade=rng.choice((0.005, 0.01)), **iv,
        )  # fmt: skip
        if g.valid:
            return g
    return Genome("iron_condor", "neutral", entry_signal="iv_high", dte_min=30, dte_max=45, delta_target=0.2,
                  width_pct=0.04, wing_pct=0.05)  # fmt: skip


def _with(g: Genome, **changes: Any) -> Genome:
    return replace(g, **changes)


def from_dict(d: dict[str, Any]) -> Genome:
    names = {f.name for f in fields(Genome)}
    clean = {k: v for k, v in d.items() if k in names}
    if "regime_filter" in clean:
        clean["regime_filter"] = tuple(clean["regime_filter"] or ())
    return Genome(**clean)
