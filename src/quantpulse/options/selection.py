"""Contract selection: turn a chain into concrete candidate structures, then judge each on its whole
payoff and risk — never "the highest delta" or "the cheapest option".

:func:`build` makes, for every expiration inside the DTE window, the structure a family calls for with
strikes chosen by delta (vendor Greeks, or computed from the quote's own IV) and widths as a share of the
spot. :func:`evaluate` then prices it (mid and a realistic fill), and measures it against a distribution of
prices at expiration: the expected P&L, the probability of profit, the average of the worst 5% of outcomes,
the maximum loss and profit, the break-evens, the capital, the net Greeks, the round-trip cost of the spread
and the liquidity of its worst leg. The distribution is either the *market's* (lognormal at the ATM IV:
its expected value is roughly the cost of trading, by construction) or an *empirical* one (the underlying's
own history of moves over the same horizon, optionally shifted by a directional view). An edge is a
difference between the two that survives the costs — and it is only a hypothesis until tested.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np

from quantpulse.options import structures as s
from quantpulse.options.analytics import quote_iv
from quantpulse.options.fills import ExecutionModel, quote_fill
from quantpulse.options.liquidity import LiquidityRules
from quantpulse.options.liquidity import assess as assess_liquidity
from quantpulse.options.pricing import greeks as model_greeks
from quantpulse.options.quotes import OptionQuote


@dataclass(frozen=True, slots=True)
class Spec:
    """What to build (a genome's structural genes)."""

    family: str
    dte_min: int
    dte_max: int
    delta_target: float
    width_pct: float | None = None
    wing_pct: float | None = None
    max_spread_pct: float = 0.15
    min_open_interest: float = 0
    stock_price: float | None = None  # for covered calls, protective puts and collars (shares held)


@dataclass
class Candidate:
    structure: s.Structure
    quotes: list[OptionQuote]  # one per option leg, in leg order
    expiration: date
    dte: int
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    rejected: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.structure.key()


def _delta(q: OptionQuote, now: datetime) -> float | None:
    if q.greeks.delta is not None:
        return q.greeks.delta
    iv = quote_iv(q, now)
    if iv is None or not q.underlying_price:
        return None
    c = q.contract
    return model_greeks(c.kind, q.underlying_price, c.strike, c.years(now), iv).delta


def _by_delta(qs: Sequence[OptionQuote], kind: str, target: float, now: datetime) -> OptionQuote | None:
    """The contract of ``kind`` whose |delta| is nearest ``target`` (two-sided markets only)."""
    best, gap = None, math.inf
    for q in qs:
        if q.contract.kind != kind or not q.two_sided:
            continue
        d = _delta(q, now)
        if d is None:
            continue
        if abs(abs(d) - target) < gap:
            best, gap = q, abs(abs(d) - target)
    return best


def _by_strike(
    qs: Sequence[OptionQuote], kind: str, strike: float, *, above: bool | None = None
) -> OptionQuote | None:
    pool = [q for q in qs if q.contract.kind == kind and q.two_sided]
    if above is True:
        pool = [q for q in pool if q.contract.strike > strike - 1e-9]
    elif above is False:
        pool = [q for q in pool if q.contract.strike < strike + 1e-9]
    return min(pool, key=lambda q: abs(q.contract.strike - strike), default=None)


def _mid(q: OptionQuote) -> float:
    return q.mid or 0.0


def build(spec: Spec, quotes: Sequence[OptionQuote], spot: float, now: datetime) -> list[Candidate]:
    """One candidate per eligible expiration (the structure the family calls for, if the chain has it)."""
    by_exp: dict[date, list[OptionQuote]] = {}
    for q in quotes:
        dte = q.contract.dte(now)
        if spec.dte_min <= dte <= spec.dte_max and not q.contract.expired(now):
            by_exp.setdefault(q.contract.expiration, []).append(q)
    out: list[Candidate] = []
    for exp, qs in sorted(by_exp.items()):
        try:
            made = _one(spec, qs, spot, now)
        except s.StructureError:
            made = None
        if made is not None:
            structure, legs = made
            out.append(Candidate(structure, legs, exp, legs[0].contract.dte(now) if legs else 0))
    return out


def _one(
    spec: Spec, qs: list[OptionQuote], spot: float, now: datetime
) -> tuple[s.Structure, list[OptionQuote]] | None:
    f, t = spec.family, spec.delta_target
    w = spec.width_pct or 0.05
    wing = spec.wing_pct or 0.05
    if f == "long_call":
        c = _by_delta(qs, "call", t, now)
        return (s.long_call(c.contract, _mid(c)), [c]) if c else None
    if f == "long_put":
        p = _by_delta(qs, "put", t, now)
        return (s.long_put(p.contract, _mid(p)), [p]) if p else None
    if f == "bull_call_spread":
        lo = _by_delta(qs, "call", t, now)
        hi = _by_strike(qs, "call", lo.contract.strike * (1 + w), above=True) if lo else None
        if lo and hi and hi.contract.strike > lo.contract.strike:
            return s.bull_call_spread(lo.contract, _mid(lo), hi.contract, _mid(hi)), [lo, hi]
        return None
    if f == "bear_put_spread":
        hi = _by_delta(qs, "put", t, now)
        lo = _by_strike(qs, "put", hi.contract.strike * (1 - w), above=False) if hi else None
        if hi and lo and lo.contract.strike < hi.contract.strike:
            return s.bear_put_spread(hi.contract, _mid(hi), lo.contract, _mid(lo)), [hi, lo]
        return None
    if f == "bull_put_spread":
        short = _by_delta(qs, "put", t, now)
        long = _by_strike(qs, "put", short.contract.strike * (1 - w), above=False) if short else None
        if short and long and long.contract.strike < short.contract.strike:
            return s.bull_put_spread(short.contract, _mid(short), long.contract, _mid(long)), [short, long]
        return None
    if f == "bear_call_spread":
        short = _by_delta(qs, "call", t, now)
        long = _by_strike(qs, "call", short.contract.strike * (1 + w), above=True) if short else None
        if short and long and long.contract.strike > short.contract.strike:
            return s.bear_call_spread(short.contract, _mid(short), long.contract, _mid(long)), [short, long]
        return None
    if f == "long_straddle":
        c = _by_strike(qs, "call", spot)
        p = next(
            (
                q
                for q in qs
                if q.contract.kind == "put" and c and q.contract.strike == c.contract.strike and q.two_sided
            ),
            None,
        )
        return (s.long_straddle(c.contract, _mid(c), p.contract, _mid(p)), [c, p]) if c and p else None
    if f == "long_strangle":
        c = _by_strike(qs, "call", spot * (1 + wing), above=True)
        p = _by_strike(qs, "put", spot * (1 - wing), above=False)
        return (s.long_strangle(c.contract, _mid(c), p.contract, _mid(p)), [c, p]) if c and p else None
    if f == "iron_condor":
        sp = _by_delta(qs, "put", t, now)
        sc = _by_delta(qs, "call", t, now)
        if not (sp and sc):
            return None
        lp = _by_strike(qs, "put", sp.contract.strike * (1 - w), above=False)
        lc = _by_strike(qs, "call", sc.contract.strike * (1 + w), above=True)
        if (
            not (lp and lc)
            or lp.contract.strike >= sp.contract.strike
            or lc.contract.strike <= sc.contract.strike
        ):
            return None
        legs = [lp, sp, sc, lc]
        return s.iron_condor(
            lp.contract, _mid(lp), sp.contract, _mid(sp), sc.contract, _mid(sc), lc.contract, _mid(lc)
        ), legs
    if f == "call_butterfly":
        mid = _by_strike(qs, "call", spot)
        if not mid:
            return None
        lo = _by_strike(qs, "call", mid.contract.strike * (1 - w), above=False)
        hi = _by_strike(qs, "call", mid.contract.strike * (1 + w), above=True)
        if not (lo and hi) or not lo.contract.strike < mid.contract.strike < hi.contract.strike:
            return None
        return s.call_butterfly(lo.contract, _mid(lo), mid.contract, _mid(mid), hi.contract, _mid(hi)), [
            lo,
            mid,
            hi,
        ]
    if f == "covered_call":
        c = _by_delta(qs, "call", t, now)
        stock = spec.stock_price or spot
        return (s.covered_call(stock, c.contract, _mid(c)), [c]) if c else None
    if f == "cash_secured_put":
        p = _by_delta(qs, "put", t, now)
        return (s.cash_secured_put(p.contract, _mid(p)), [p]) if p else None
    if f == "protective_put":
        p = _by_delta(qs, "put", t, now)
        return (s.protective_put(spec.stock_price or spot, p.contract, _mid(p)), [p]) if p else None
    if f == "collar":
        p = _by_delta(qs, "put", t, now)
        c = _by_delta(qs, "call", t, now)
        if p and c and p.contract.strike < c.contract.strike:
            return s.collar(spec.stock_price or spot, p.contract, _mid(p), c.contract, _mid(c)), [p, c]
        return None
    return None


def market_distribution(spot: float, iv: float, years: float, n: int = 4000, seed: int = 7) -> np.ndarray:
    """Prices at expiration under the market's own (risk-neutral, zero-drift) lognormal at the ATM IV."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return spot * np.exp(-0.5 * iv * iv * years + iv * math.sqrt(max(years, 1e-9)) * z)


def empirical_distribution(spot: float, closes: Sequence[float], days: int, drift: float = 0.0, n: int = 4000,
                           seed: int = 7) -> np.ndarray | None:  # fmt: skip
    """Prices at expiration from the underlying's own history of ``days``-day moves (bootstrapped from the
    overlapping windows), shifted by ``drift`` (a view on the move, e.g. from the stock model)."""
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    days = max(days, 1)
    if len(c) < days + 60:
        return None
    moves = c[days:] / c[:-days]
    rng = np.random.default_rng(seed)
    return spot * rng.choice(moves, size=n) * math.exp(drift)


def evaluate(
    cand: Candidate,
    spot: float,
    now: datetime,
    *,
    terminal: np.ndarray | None = None,
    distribution: str = "market",
    fee_per_contract: float = 0.05,
    liquidity_rules: LiquidityRules | None = None,
) -> Candidate:
    """Fill in :attr:`Candidate.metrics` (and :attr:`rejected` for unusable legs)."""
    st = cand.structure
    signs = [leg.sign for leg in st.legs if leg.contract is not None]
    ratios = [leg.ratio for leg in st.legs if leg.contract is not None]
    realistic = [
        quote_fill(sg, q, ExecutionModel.REALISTIC) for sg, q in zip(signs, cand.quotes, strict=True)
    ]
    mid_debit = st.debit()
    fill_debit = mid_debit
    if all(p is not None for p in realistic):
        fill_debit = mid_debit + sum(sg * r * 100 * ((p or 0) - (q.mid or 0))
                                     for sg, r, p, q in zip(signs, ratios, realistic, cand.quotes, strict=True))  # fmt: skip
    contracts = sum(ratios)
    fee = contracts * fee_per_contract * 2  # in and out
    per_leg = []
    for q in cand.quotes:
        iv = quote_iv(q, now)
        d = _delta(q, now)
        g = q.greeks
        if g.gamma is None and iv is not None:
            v = model_greeks(q.contract.kind, spot, q.contract.strike, q.contract.years(now), iv)
            per_leg.append(
                {"delta": d, "gamma": v.gamma, "theta": v.theta, "vega": v.vega, "rho": v.rho, "iv": iv}
            )
        else:
            per_leg.append(
                {"delta": d, "gamma": g.gamma, "theta": g.theta, "vega": g.vega, "rho": g.rho, "iv": iv}
            )
    leg_greeks: list[dict[str, float | None] | None] = []
    it = iter(per_leg)
    for leg in st.legs:
        leg_greeks.append(None if leg.contract is None else next(it))
    net = st.greeks(leg_greeks)
    liq = [assess_liquidity(q, liquidity_rules) for q in cand.quotes]
    worst_liq = min((x.score for x in liq), default=0.0)
    for x in liq:
        cand.rejected.extend(f"{x.symbol}: {r}" for r in x.reasons)
    ivs = [p["iv"] for p in per_leg if p["iv"] is not None]
    atm_iv = float(np.mean(ivs)) if ivs else None
    years = cand.quotes[0].contract.years(now) if cand.quotes else 0.0
    if terminal is None and atm_iv is not None and st.single_expiry:
        terminal = market_distribution(spot, atm_iv, years)
        distribution = "market"
    max_loss, max_profit = st.max_loss(), st.max_profit()
    m: dict[str, Any] = {
        "family": st.family,
        "dte": cand.dte,
        "expiration": cand.expiration.isoformat(),
        "legs": [leg.label() for leg in st.legs],
        "mid_debit": round(mid_debit, 2),
        "fill_debit": round(fill_debit, 2),
        "spread_cost": round(fill_debit - mid_debit, 2),
        "fees": round(fee, 2),
        "max_loss": None if math.isinf(max_loss) else round(max_loss + (fill_debit - mid_debit), 2),
        "max_profit": None if math.isinf(max_profit) else round(max_profit - (fill_debit - mid_debit), 2),
        "breakevens": st.breakevens(),
        "capital": None if math.isinf(st.capital_required()) else round(st.capital_required(), 2),
        "greeks": {k: None if v is None else round(v, 4) for k, v in net.items()},
        "iv": atm_iv,
        "liquidity": worst_liq,
        "defined_risk": st.defined_risk,
    }
    if terminal is not None and st.single_expiry:
        pnl = np.asarray(st.pnl_at_expiry(terminal), dtype=float) - (fill_debit - mid_debit) - fee
        worst = np.sort(pnl)[: max(1, len(pnl) // 20)]
        m.update(
            distribution=distribution,
            expected_pnl=round(float(pnl.mean()), 2),
            pop=round(float((pnl > 0).mean()), 4),
            cvar_5=round(float(worst.mean()), 2),
            expected_on_risk=round(float(pnl.mean()) / max_loss, 4)
            if math.isfinite(max_loss) and max_loss > 0
            else None,
        )
    cand.metrics = m
    return cand


def score(cand: Candidate, *, prefer_defined: bool = True) -> float | None:
    """A ranking for candidates that passed every hard check: expected P&L per dollar at risk, discounted for
    tail loss and poor liquidity. ``None`` when the expected P&L is unknown (no distribution): unranked."""
    m = cand.metrics
    eor = m.get("expected_on_risk")
    if eor is None:
        return None
    tail = abs(m.get("cvar_5") or 0.0) / max(m.get("max_loss") or 1.0, 1.0)
    value = eor - 0.1 * tail + 0.05 * (m.get("liquidity") or 0.0)
    if prefer_defined and not m.get("defined_risk"):
        value -= 10.0
    cand.score = round(value, 6)
    return cand.score


def rank(cands: list[Candidate]) -> list[Candidate]:
    usable = [c for c in cands if not c.rejected and score(c) is not None]
    return sorted(usable, key=lambda c: c.score or -math.inf, reverse=True)
