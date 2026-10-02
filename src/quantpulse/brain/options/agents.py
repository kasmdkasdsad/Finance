"""The option agents: deterministic specialists, each with one question to answer about a candidate trade.

Every agent returns an :class:`Opinion` — support, oppose, neutral, abstain (it lacks what it needs, and says
what) or **veto** (a hard reason never to trade it). No agent sends an order or does arithmetic a model could
get wrong: every number comes from :mod:`quantpulse.options` (payoffs, Greeks, IV) or the research lab. The
research-side agents (research, extraction, critic, experiments, meta-learning, decay) run in the lab
(:mod:`quantpulse.services.options_lab`); their conclusions reach a candidate through the strategy version's
recorded evidence, read here by :func:`strategy_critic`, :func:`meta_learning` and :func:`strategy_decay`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quantpulse.options.lab.features import iv_trend
from quantpulse.options.lab.genome import Genome
from quantpulse.options.selection import Candidate
from quantpulse.options.structures import FAMILIES

from .perception import UnderlyingView

VERDICTS = ("support", "oppose", "neutral", "abstain", "veto")


@dataclass
class Opinion:
    agent: str
    verdict: str
    score: float  # −1 … +1 (support positive)
    reasons: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "verdict": self.verdict, "score": round(self.score, 3), "reasons": self.reasons,
                "data": self.data}  # fmt: skip


@dataclass
class CandidateContext:
    view: UnderlyingView
    version: Mapping[str, Any]
    genome: Genome
    cand: Candidate
    now: datetime
    stock_view: Mapping[str, Any] | None = None  # the stock Brain's consensus on the underlying
    book: Mapping[str, Any] = field(default_factory=dict)  # open option positions, per underlying
    weight: Mapping[str, Any] | None = None  # the learned (shrunk) weight of this strategy in this context
    decay: str | None = None
    risk: Mapping[str, Any] | None = None  # the risk book's preview
    paper: bool = False  # would this be a paper order (not only shadow)?


def _op(agent: str, score: float, reasons: list[str], *, veto: bool = False, abstain: bool = False,
        **data: Any) -> Opinion:  # fmt: skip
    if veto:
        return Opinion(agent, "veto", -1.0, reasons, data)
    if abstain:
        return Opinion(agent, "abstain", 0.0, reasons, data)
    s = max(-1.0, min(1.0, score))
    verdict = "support" if s > 0.1 else "oppose" if s < -0.1 else "neutral"
    return Opinion(agent, verdict, s, reasons, data)


# ----------------------------------------------------------------------------- market and data
def data_quality(c: CandidateContext) -> Opinion:
    q = c.view.quality
    grade = q.get("OPTIONS_DATA_QUALITY")
    need = "execution" if c.paper else "research"
    reasons = [f"chain {grade}: {q.get('usable_for_execution')} of {q.get('contracts')} contracts usable for execution; "
               f"feed {q.get('OPTIONS_DATA_FEED')}"]  # fmt: skip
    if grade in ("unusable", "empty") or (need == "execution" and grade != "execution"):
        return _op(
            "OptionsDataQualityAgent", 0, [*reasons, f"not good enough for {need}"], veto=True, grade=grade
        )
    stale = [q_.symbol for q_ in c.cand.quotes if q_.age(c.now) is None or (q_.age(c.now) or 0) > 300]
    if stale:
        return _op("OptionsDataQualityAgent", 0, [*reasons, f"stale legs: {', '.join(stale)}"], veto=True)
    if q.get("OPTIONS_DATA_FEED") == "indicative":
        reasons.append("indicative feed: not firm quotes (paper limit orders only)")
    return _op("OptionsDataQualityAgent", 0.2, reasons, grade=grade)


def options_regime(c: CandidateContext) -> Opinion:
    f = c.view.features
    if f is None:
        return _op("OptionsRegimeAgent", 0, ["no features"], abstain=True)
    regime, vol = c.view.regime, c.view.vol_regime
    allowed = c.genome.regime_filter
    reasons = [f"market {regime}, volatility {vol}, {iv_trend(f).lower().replace('_', ' ')}"]
    if allowed and not ({regime, vol} & set(allowed)):
        return _op(
            "OptionsRegimeAgent", -0.6, [*reasons, f"the strategy is validated only in {', '.join(allowed)}"]
        )
    return _op("OptionsRegimeAgent", 0.1, reasons, regime=regime, vol_regime=vol)


def implied_volatility(c: CandidateContext) -> Opinion:
    f = c.view.features
    fam = FAMILIES[c.genome.family]
    if f is None or f.iv_rv is None:
        return _op("ImpliedVolatilityAgent", 0, ["IV relative to realized volatility unknown"], abstain=True)
    rich = f.iv_rv - 1.0
    # buying premium wants cheap volatility; selling premium wants rich volatility
    sign = 1 if fam.vol == "short_vol" else -1 if fam.vol == "long_vol" else 0
    score = sign * max(-1.0, min(1.0, rich / 0.3))
    rank = (
        f"IV rank {f.iv_rank:.0f}"
        if f.iv_rank is not None
        else "IV rank unknown (under 60 days of IV history)"
    )
    return _op("ImpliedVolatilityAgent", score, [f"IV/RV {f.iv_rv:.2f} ({'rich' if rich > 0 else 'cheap'}); {rank}; "
                                                  f"the structure is {fam.vol.replace('_', ' ')}"], iv_rv=f.iv_rv)  # fmt: skip


def earnings_event(c: CandidateContext) -> Opinion:
    f = c.view.features
    days = f.event_days if f else None
    life = c.cand.dte
    if days is None:
        return _op("EarningsEventAgent", 0, ["no known earnings date (index ETFs have none)"], abstain=True)
    inside = days <= life
    fam = FAMILIES[c.genome.family]
    if inside and c.genome.event_filter == "avoid":
        return _op("EarningsEventAgent", 0, [f"earnings in {days} days, inside the option's {life}-day life; the "
                                             "strategy avoids events"], veto=True)  # fmt: skip
    if inside and fam.vol == "short_vol" and c.genome.event_filter != "require":
        return _op("EarningsEventAgent", -0.6, [f"earnings in {days} days: a short-volatility position through the "
                                                "announcement's jump"])  # fmt: skip
    return _op("EarningsEventAgent", 0.1 if not inside else 0.0, [f"earnings in {days} days ({'inside' if inside else 'after'} "
                                                                  "the option's life)"])  # fmt: skip


# ----------------------------------------------------------------------------- direction
def momentum(c: CandidateContext) -> Opinion:
    f = c.view.features
    if f is None or f.ret20 is None:
        return _op("OptionsMomentumAgent", 0, ["no 20-day return"], abstain=True)
    direction = FAMILIES[c.genome.family].direction
    trend = 1 if (f.sma50 and f.sma200 and f.spot > f.sma50 > f.sma200) else -1 if (
        f.sma50 and f.sma200 and f.spot < f.sma50 < f.sma200) else 0  # fmt: skip
    s = max(-1.0, min(1.0, f.ret20 / 0.08)) * 0.6 + trend * 0.4
    if direction == "bearish":
        s = -s
    elif direction not in ("bullish",):
        s = -abs(s) * 0.5  # neutral structures prefer no strong trend
    return _op("OptionsMomentumAgent", s, [f"20-day return {f.ret20:+.1%}, trend {['down', 'none', 'up'][trend + 1]}, "
                                           f"structure {direction}"])  # fmt: skip


def mean_reversion(c: CandidateContext) -> Opinion:
    f = c.view.features
    if f is None or f.z5 is None:
        return _op("OptionsMeanReversionAgent", 0, ["no 5-day z-score"], abstain=True)
    direction = FAMILIES[c.genome.family].direction
    s = max(-1.0, min(1.0, -f.z5 / 3))  # stretched down: expect a bounce (bullish)
    if direction == "bearish":
        s = -s
    elif direction != "bullish":
        s = 0.0
    return _op("OptionsMeanReversionAgent", s * 0.5, [f"5-day move {f.z5:+.1f} standard deviations"])


def stock_view(c: CandidateContext) -> Opinion:
    """The stock Brain's consensus on the same underlying: agreement strengthens a directional option,
    disagreement weakens it (a disagreement is information, not a veto)."""
    v = c.stock_view
    direction = FAMILIES[c.genome.family].direction
    if not v or v.get("stance") not in ("bullish", "bearish") or direction not in ("bullish", "bearish"):
        return _op("OptionsStructureAgent", 0, ["no directional stock view to compare with"], abstain=True)
    agree = v["stance"] == direction
    conf = float(v.get("confidence") or 0)
    return _op("OptionsStructureAgent", (1 if agree else -1) * conf, [
        f"the stock agents are {v['stance']} (confidence {conf:.2f}): they {'agree' if agree else 'disagree'} with a "
        f"{direction} {c.genome.family.replace('_', ' ')}"])  # fmt: skip


# ----------------------------------------------------------------------------- the structure itself
def greeks(c: CandidateContext) -> Opinion:
    g = c.cand.metrics.get("greeks") or {}
    if any(g.get(k) is None for k in ("delta", "theta", "vega")):
        return _op("GreeksAgent", 0, ["Greeks unknown for a leg"], veto=c.paper, abstain=not c.paper)
    loss = c.cand.metrics.get("max_loss") or 0
    theta_share = abs(g["theta"]) * 7 / loss if loss else 0.0
    reasons = [
        f"per unit: delta {g['delta']:+.1f} shares, theta {g['theta']:+.2f} $/day, vega {g['vega']:+.2f} $/vol pt"
    ]
    if g["theta"] < 0 and theta_share > 0.15:
        return _op(
            "GreeksAgent", -0.4, [*reasons, f"a week of decay costs {theta_share:.0%} of the maximum loss"]
        )
    return _op("GreeksAgent", 0.1, reasons, greeks=g)


def contract_selection(c: CandidateContext) -> Opinion:
    g, cand = c.genome, c.cand
    reasons = [f"{cand.dte} DTE (window {g.dte_min}–{g.dte_max}), expiration {cand.expiration.isoformat()}"]
    if not g.dte_min <= cand.dte <= g.dte_max:
        return _op("ContractSelectionAgent", 0, [*reasons, "outside the strategy's window"], veto=True)
    be = cand.metrics.get("breakevens") or []
    spot = c.view.spot or 0
    if be and spot:
        dist = min(abs(b / spot - 1) for b in be)
        move = next((e.implied_move_pct for e in c.view.expiries if e.implied_move_pct), None)
        reasons.append(f"nearest break-even {dist:.1%} away; implied move {move:.1%}" if move else
                       f"nearest break-even {dist:.1%} away")  # fmt: skip
    return _op("ContractSelectionAgent", 0.1, reasons)


def liquidity(c: CandidateContext) -> Opinion:
    bad, notes = [], []
    for q in c.cand.quotes:
        sp = q.spread_pct
        notes.append(f"{q.symbol}: spread {'?' if sp is None else f'{sp:.0%}'}, OI {q.open_interest or '?'}")
        if sp is None or sp > c.genome.max_spread_pct:
            bad.append(
                f"{q.symbol} spread {'unknown' if sp is None else f'{sp:.0%}'} > {c.genome.max_spread_pct:.0%}"
            )
        if q.open_interest is not None and q.open_interest < c.genome.min_open_interest:
            bad.append(f"{q.symbol} open interest {q.open_interest:.0f} < {c.genome.min_open_interest:.0f}")
    if bad:
        return _op("LiquidityAgent", 0, bad, veto=True)
    return _op("LiquidityAgent", 0.2 * (c.cand.metrics.get("liquidity") or 0.5), notes[:4])


def statistical(c: CandidateContext) -> Opinion:
    m = c.cand.metrics
    eor, pop = m.get("expected_on_risk"), m.get("pop")
    if eor is None:
        return _op(
            "OptionsStatisticalAgent", 0, ["no distribution to price the payoff against"], abstain=True
        )
    emp = m.get("empirical") or {}
    reasons = [
        f"under the market-implied distribution: {eor:+.1%} per $ at risk, P(profit) {pop:.0%} (costs included)"
    ]
    score = eor * 5
    if emp.get("expected_on_risk") is not None:
        reasons.append(
            f"under the underlying's own recent returns: {emp['expected_on_risk']:+.1%} per $ at risk"
        )
        score = 0.5 * score + 0.5 * emp["expected_on_risk"] * 5
    return _op("OptionsStatisticalAgent", score, reasons, expected_on_risk=eor, pop=pop)


def valuation(c: CandidateContext) -> Opinion:
    f = c.view.features
    if f is None or f.rv20 is None or c.view.spot is None:
        return _op("OptionsValuationAgent", 0, ["no realized volatility to value against"], abstain=True)
    from quantpulse.options.pricing import greeks as bsm

    diff = 0.0
    for leg, q in zip(c.cand.structure.option_legs, c.cand.quotes, strict=True):
        con = leg.contract
        assert con is not None
        fair = bsm(con.kind, c.view.spot, con.strike, max(con.years(c.now), 1e-6), f.rv20).price
        diff += leg.sign * leg.ratio * ((q.mid or 0) - fair) * 100
    # paying more than realized-volatility value (diff > 0 for a buyer) is expensive
    per_risk = diff / max(c.cand.metrics.get("max_loss") or 1, 1)
    return _op("OptionsValuationAgent", -per_risk * 2, [f"${diff:+,.0f} per unit versus pricing at realized volatility "
                                                         f"({f.rv20:.0%})"], rich_vs_rv=round(diff, 2))  # fmt: skip


def volatility_strategy(c: CandidateContext) -> Opinion:
    f = c.view.features
    fam = FAMILIES[c.genome.family]
    if f is None or f.iv is None or f.rv20 is None:
        return _op("OptionsVolatilityStrategyAgent", 0, ["no IV or realized volatility"], abstain=True)
    vrp = f.iv - f.rv20
    if fam.vol == "short_vol":
        s = vrp / 0.05
    elif fam.vol == "long_vol":
        s = -vrp / 0.05
    else:
        s = 0.0
    return _op("OptionsVolatilityStrategyAgent", s * 0.5, [f"volatility risk premium {vrp:+.1%} (IV {f.iv:.0%} vs RV "
                                                             f"{f.rv20:.0%})"])  # fmt: skip


# ----------------------------------------------------------------------------- the book, risk and the strategy
def portfolio(c: CandidateContext) -> Opinion:
    held = c.book.get(c.view.underlying) or []
    if held:
        return _op("OptionsPortfolioAgent", 0, [f"already {len(held)} option position(s) on {c.view.underlying}: no "
                                                "stacking"], veto=True)  # fmt: skip
    n = sum(len(v) for v in c.book.values())
    return _op("OptionsPortfolioAgent", 0.1 if n < 3 else -0.2, [f"{n} open option position(s)"])


def risk(c: CandidateContext) -> Opinion:
    r = c.risk
    if r is None:
        return _op("OptionsRiskAgent", 0, ["no risk preview (shadow only)"], abstain=True)
    if not r.get("approved"):
        return _op(
            "OptionsRiskAgent", 0, [f"risk preview: {r.get('summary')}"], veto=c.paper, abstain=not c.paper
        )
    return _op(
        "OptionsRiskAgent", 0.2, ["risk preview approved (the risk engine decides again at the order)"]
    )


def strategy_critic(c: CandidateContext) -> Opinion:
    stage = c.version.get("stage")
    exp = c.version.get("expected_ror")
    reasons = [f"{c.version.get('key')} at {stage}"
               + (f", validated {exp:+.1%} per $ at risk out of sample (model-priced)" if exp is not None else "")]  # fmt: skip
    if stage not in ("PAPER_SHADOW", "PAPER_ACTIVE", "PROVEN"):
        return _op("StrategyCriticAgent", 0, [*reasons, "not validated for live use"], veto=True)
    return _op("StrategyCriticAgent", 0.3 if stage != "PAPER_SHADOW" else 0.1, reasons)


def meta_learning(c: CandidateContext) -> Opinion:
    w = c.weight
    if not w or not w.get("n"):
        return _op(
            "MetaLearningAgent", 0, ["no live trades of this strategy in this context yet"], abstain=True
        )
    p = float(w.get("weight") or 0.5)
    return _op("MetaLearningAgent", (p - 0.5) * 2, [f"shrunk P(edge > 0) {p:.2f} from {w['n']:.0f} trades in this "
                                                    "context (recency-weighted, never below the floor)"])  # fmt: skip


def strategy_decay(c: CandidateContext) -> Opinion:
    d = c.decay
    if d in ("DEGRADING", "BROKEN"):
        return _op("StrategyDecayAgent", 0, [f"decay status {d}"], veto=c.paper, abstain=not c.paper)
    return _op("StrategyDecayAgent", -0.2 if d == "WATCH" else 0.0, [f"decay status {d or 'no live record'}"])


AGENTS: tuple[tuple[str, Callable[[CandidateContext], Opinion], float, str], ...] = (
    ("OptionsDataQualityAgent", data_quality, 1.0, "is the chain fresh, two-sided and execution grade?"),
    ("OptionsRegimeAgent", options_regime, 0.5, "which market and volatility regime is this, and is the strategy valid in it?"),
    ("ImpliedVolatilityAgent", implied_volatility, 1.0, "is volatility cheap or rich for this structure?"),
    ("GreeksAgent", greeks, 0.5, "are the Greeks known and consistent with the thesis?"),
    ("OptionsStructureAgent", stock_view, 0.8, "does the stock Brain's view agree with the structure's direction?"),
    ("ContractSelectionAgent", contract_selection, 0.5, "is the expiration and strike right for the move?"),
    ("LiquidityAgent", liquidity, 1.0, "can every leg be traded at a sane price?"),
    ("EarningsEventAgent", earnings_event, 0.8, "is an earnings announcement inside the option's life?"),
    ("OptionsVolatilityStrategyAgent", volatility_strategy, 0.6, "is the volatility risk premium on our side?"),
    ("OptionsMomentumAgent", momentum, 0.6, "does the trend support the direction?"),
    ("OptionsMeanReversionAgent", mean_reversion, 0.4, "is the move stretched?"),
    ("OptionsStatisticalAgent", statistical, 1.5, "is the expected value per dollar at risk positive after costs?"),
    ("OptionsValuationAgent", valuation, 0.6, "is the premium rich or cheap against realized volatility?"),
    ("OptionsPortfolioAgent", portfolio, 1.0, "does it fit the option book (no stacking, concentration)?"),
    ("OptionsRiskAgent", risk, 1.0, "does the risk engine's preview approve it?"),
    ("StrategyCriticAgent", strategy_critic, 1.0, "is the strategy validated for live use?"),
    ("MetaLearningAgent", meta_learning, 0.8, "how has this strategy done in this context?"),
    ("StrategyDecayAgent", strategy_decay, 1.0, "is the strategy decaying?"),
)  # fmt: skip

# the lab-side agents, for the record of who does what
LAB_AGENTS: tuple[tuple[str, str], ...] = (
    (
        "OptionsResearchAgent",
        "keeps the research library: sources, their claims (hypotheses, never facts) and tests",
    ),
    (
        "StrategyExtractionAgent",
        "turns a documented strategy into an explicit genome, recording every assumption",
    ),
    (
        "StrategyCriticAgent",
        "attacks every strategy: costs, spreads, look-ahead, one ticker, one period, sample size",
    ),
    ("ExperimentGeneratorAgent", "proposes competing fixes for failures, ranked by expected information"),
    ("MetaLearningAgent", "learns which strategies work in which contexts (shrunk, recency-weighted)"),
    ("StrategyDecayAgent", "watches live results against validation; demotes decaying strategies"),
    ("NoTradeAgent", "decides when the best action is no trade, and says why"),
)


def deliberate(c: CandidateContext) -> dict[str, Any]:
    """Every agent's opinion and the weighted verdict: a veto stops the candidate; otherwise the score is the
    weighted mean of the agents that had a view."""
    ops = [(fn(c), w) for _, fn, w, _ in AGENTS]
    vetoes = [o for o, _ in ops if o.verdict == "veto"]
    voting = [(o, w) for o, w in ops if o.verdict in ("support", "oppose", "neutral")]
    total = sum(w for _, w in voting)
    score = sum(o.score * w for o, w in voting) / total if total else 0.0
    support = sum(1 for o, _ in voting if o.verdict == "support")
    oppose = sum(1 for o, _ in voting if o.verdict == "oppose")
    return {
        "opinions": [o.as_dict() for o, _ in ops],
        "vetoes": [{"agent": o.agent, "reasons": o.reasons} for o in vetoes],
        "score": round(score, 4),
        "support": support,
        "oppose": oppose,
        "confidence": round(abs(score) * (1 - (min(support, oppose) / max(support + oppose, 1))), 4),
    }


def no_trade(reasons: Sequence[str]) -> Opinion:
    """NoTradeAgent: the best action is sometimes nothing — recorded with its reasons."""
    return _op("NoTradeAgent", 0, list(reasons) or ["nothing cleared every check"], abstain=True)


def debate(c: CandidateContext, verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Bull and bear cases from the agents' own reasons, and the devil's advocate's strongest objection."""
    ops = verdict["opinions"]
    bull = [r for o in ops if o["verdict"] == "support" for r in o["reasons"][:1]]
    bear = [r for o in ops if o["verdict"] in ("oppose", "veto") for r in o["reasons"][:1]]
    worst = min(
        (o for o in ops if o["verdict"] in ("oppose", "veto")), key=lambda o: o["score"], default=None
    )
    loss = c.cand.metrics.get("max_loss")
    devil = (worst["reasons"][0] if worst else
             f"the whole premium at risk: ${loss:,.0f} per unit if it expires worthless" if loss else "none")  # fmt: skip
    return {"bull": bull[:5], "bear": bear[:5], "devils_advocate": devil}


def thesis(c: CandidateContext, verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The trade's thesis: why this structure, why now, what would prove it wrong."""
    fam = FAMILIES[c.genome.family]
    m = c.cand.metrics
    f = c.view.features
    be = ", ".join(f"${b:,.2f}" for b in (m.get("breakevens") or []))
    wrong = {
        "bullish": "a close below the break-even, or the trend breaking",
        "bearish": "a close above the break-even, or the down-trend ending",
        "neutral": "a move beyond a short strike",
        "volatility": "realized volatility staying below what was paid",
        "income": "a sharp fall in the shares",
    }[fam.direction]
    supporting = [r for o in verdict["opinions"] if o["verdict"] == "support" for r in o["reasons"][:1]]
    return {
        "underlying": c.view.underlying,
        "direction": fam.direction,
        "structure": c.genome.family,
        "strategy": c.version.get("key"),
        "why_now": [c.genome.describe()[:300], *supporting[:4]],
        "market_regime": c.view.regime,
        "iv_regime": c.view.vol_regime,
        "iv_rank": f.iv_rank if f else None,
        "expected_on_risk": m.get("expected_on_risk"),
        "pop": m.get("pop"),
        "max_loss": m.get("max_loss"),
        "max_profit": m.get("max_profit"),
        "breakevens": m.get("breakevens"),
        "invalidation": f"{wrong} (break-even {be})" if be else wrong,
        "exit_plan": f"take profit {c.genome.take_profit}, stop {c.genome.stop_loss}, close at {c.genome.exit_dte} DTE "
        f"or after {c.genome.max_hold_days} days — never held into expiration",
        "confidence": verdict["confidence"],
    }


def explain(
    th: Mapping[str, Any], verdict: Mapping[str, Any], *, mode: str, comparison: Mapping[str, Any] | None
) -> str:
    """Plain words: what, why, the risk and the alternative considered."""
    parts = [
        f"{mode.upper()}: {th['structure'].replace('_', ' ')} on {th['underlying']} ({th['direction']})."
    ]
    if th.get("expected_on_risk") is not None:
        parts.append(
            f"Expected {th['expected_on_risk']:+.1%} per dollar at risk, P(profit) {th.get('pop') or 0:.0%}."
        )
    if th.get("max_loss") is not None and math.isfinite(th["max_loss"]):
        parts.append(f"Maximum loss ${th['max_loss']:,.0f} per unit, known in advance.")
    parts.append(f"Wrong if: {th['invalidation']}.")
    if comparison:
        parts.append(f"Versus shares: {comparison.get('verdict')}.")
    if verdict.get("vetoes"):
        parts.append("Vetoed by " + ", ".join(v["agent"] for v in verdict["vetoes"]) + ".")
    return " ".join(parts)
