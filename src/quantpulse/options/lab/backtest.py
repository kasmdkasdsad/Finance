"""The options backtester: one genome, a set of underlyings, a period, a chain source, an execution model.

Each day, in order: open positions are marked at the mid and checked for their exits (expiration, days left,
holding time, take profit, stop loss); then, for each underlying without a position, the entry signal and
filters are read from that day's features (the past only) and, if they pass, the day's chain is built, the
structure selected (the expiration nearest the middle of the DTE window — selection is not optimized on the
outcome), filled under the execution model, and sized so its maximum loss is ``risk_per_trade`` of equity.
Exits are filled under the same model; a position still held at expiration is settled at intrinsic value
(the exercise or assignment is recorded). Every trade keeps its entry features, regime, Greeks, IVs, the
underlying's move and a Greek-by-Greek attribution of its P&L.

The result is labelled with its data grade (``recorded`` or ``model``) and execution model, always.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from quantpulse.options.attribution import Mark, attribute_path
from quantpulse.options.fills import ASSUMPTIONS, ExecutionModel, leg_fill
from quantpulse.options.lab.chains import ChainSource, close_time
from quantpulse.options.lab.features import DayFeatures, iv_regime, iv_trend, signal, trend_regime
from quantpulse.options.lab.genome import Genome
from quantpulse.options.lab.metrics import summarize
from quantpulse.options.quotes import OptionQuote
from quantpulse.options.selection import Candidate, Spec, build
from quantpulse.options.structures import Structure

LABELS = {
    "recorded": "RECORDED chains: quotes the market actually showed",
    "model": "MODEL-PRICED chains: tests the logic and the underlying's path, not real option prices",
}


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    start: date
    end: date
    underlyings: tuple[str, ...]
    equity: float = 100_000.0
    model: ExecutionModel = ExecutionModel.REALISTIC
    fee_per_contract: float = 0.05
    seed: int = 11
    max_contracts: int = 50


@dataclass
class _Pos:
    underlying: str
    structure: Structure
    qty: int
    opened: date
    entry_fill: float  # per unit, dollars (debit > 0)
    entry_mid: float
    fees_in: float
    first_exp: date
    features: dict[str, Any]
    regime: str
    iv_regime: str
    marks: list[Mark] = field(default_factory=list)
    last_leg_mid: dict[str, float] = field(default_factory=dict)
    pending_exit: str | None = None
    entry_iv: float | None = None
    entry_greeks: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    genome: Genome
    config: BacktestConfig
    grade: str
    label: str
    trades: list[dict[str, Any]]
    days: list[date]
    equity: list[float]
    metrics: dict[str, Any]
    skipped: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "genome": self.genome.hash,
            "family": self.genome.family,
            "grade": self.grade,
            "execution_model": self.config.model.value,
            "label": f"{self.label}; fills: {ASSUMPTIONS[self.config.model].label}",
            "period": [self.config.start.isoformat(), self.config.end.isoformat()],
            "underlyings": list(self.config.underlyings),
            "metrics": self.metrics,
            "skipped": self.skipped,
        }


def _rng(genome: Genome, cfg: BacktestConfig) -> random.Random:
    h = hashlib.sha256(f"{genome.hash}:{cfg.seed}:{cfg.model}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _passes(g: Genome, f: DayFeatures, rng: random.Random) -> str | None:
    """The reason the entry is not taken today (``None``: take it)."""
    fired = signal(g.entry_signal, f, seed_value=rng.random() if g.entry_signal == "random" else None)
    if fired is None:
        return "insufficient data for the signal"
    if not fired:
        return "no signal"
    for name, value, lo, hi in (("IV rank", f.iv_rank, g.iv_rank_min, g.iv_rank_max),
                                ("IV percentile", f.iv_percentile, g.iv_percentile_min, g.iv_percentile_max),
                                ("IV/RV", f.iv_rv, g.iv_rv_min, g.iv_rv_max)):  # fmt: skip
        if lo is None and hi is None:
            continue
        if value is None:
            return f"{name} unknown"
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            return f"{name} outside the filter"
    if g.regime_filter:
        regimes = {trend_regime(f), iv_regime(f)}
        if not regimes & set(g.regime_filter):
            return "regime filter"
    if g.event_filter != "ignore":
        near = f.event_days is not None and f.event_days <= g.dte_max
        if g.event_filter == "avoid" and near:
            return "event inside the option's life"
        if g.event_filter == "require" and not near:
            return "no event inside the option's life"
    return None


def _chain_filters(g: Genome, quotes: Sequence[OptionQuote], spot: float, now: datetime) -> str | None:
    if g.term_filter == "any" and g.skew_filter == "any":
        return None
    from quantpulse.options.analytics import by_expiry, term_structure

    exps = by_expiry(quotes, spot, now)
    if g.term_filter != "any" and term_structure(exps)["shape"] != g.term_filter:
        return "term structure filter"
    if g.skew_filter != "any":
        skews = [e.skew for e in exps if e.skew is not None]
        if not skews:
            return "skew unknown"
        rich = "put_rich" if skews[0] > 0.02 else "call_rich" if skews[0] < -0.02 else "flat"
        if rich != g.skew_filter:
            return "skew filter"
    return None


def _leg_quotes(pos: _Pos, source: ChainSource, day: date) -> list[OptionQuote | None]:
    return [source.quote(leg.contract.symbol, pos.underlying, day) if leg.contract is not None else None
            for leg in pos.structure.legs]  # fmt: skip


def _mark(
    pos: _Pos, source: ChainSource, day: date, spot: float
) -> tuple[float, list[OptionQuote | None], bool]:
    """Liquidation value per unit at the mid (stale legs keep their last mid) and whether any leg was stale."""
    quotes = _leg_quotes(pos, source, day)
    value, stale = 0.0, False
    for leg, q in zip(pos.structure.legs, quotes, strict=True):
        if leg.contract is None:
            value += leg.sign * leg.units * spot
            continue
        if day >= leg.contract.expiration:
            px = leg.contract.intrinsic(spot)
        elif q is not None and q.mid is not None:
            px = q.mid
            pos.last_leg_mid[leg.contract.symbol] = px
        elif q is not None and q.ask is not None:
            px = one_sided(leg.sign, q)  # no bid: a long leg is worth nothing, a short one costs the ask
        else:
            px = pos.last_leg_mid.get(leg.contract.symbol, leg.price)
            stale = True  # no quote at all: nothing to trade against today
        value += leg.sign * leg.units * px
    return value, quotes, stale


def one_sided(sign: int, q: OptionQuote) -> float:
    """A market with an ask but no bid: selling a long leg brings nothing; buying back a short leg costs the ask."""
    return 0.0 if sign > 0 else float(q.ask or 0.0)


def _greeks_now(pos: _Pos, quotes: list[OptionQuote | None]) -> tuple[dict[str, float], float | None]:
    net: dict[str, float] = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    ivs = []
    for leg, q in zip(pos.structure.legs, quotes, strict=True):
        if leg.contract is None:
            net["delta"] += leg.sign * leg.units
            continue
        if q is None:
            continue
        for k in net:
            v = getattr(q.greeks, k)
            if v is not None:
                net[k] += leg.sign * leg.units * v
        if q.iv is not None:
            ivs.append(q.iv)
    return net, (sum(ivs) / len(ivs) if ivs else None)


def run(
    genome: Genome,
    source: ChainSource,
    features: Mapping[str, Mapping[date, DayFeatures]],
    cfg: BacktestConfig,
) -> BacktestResult:
    if not genome.valid:
        raise ValueError("invalid genome: " + "; ".join(genome.problems()))
    rng = _rng(genome, cfg)
    assume = ASSUMPTIONS[cfg.model]
    days = sorted({d for u in cfg.underlyings for d in features.get(u, {}) if cfg.start <= d <= cfg.end})
    cash = cfg.equity
    open_pos: list[_Pos] = []
    trades: list[dict[str, Any]] = []
    curve: list[float] = []
    skipped: Counter[str] = Counter()
    spec = Spec(genome.family, genome.dte_min, genome.dte_max, genome.delta_target, genome.width_pct, genome.wing_pct,
                genome.max_spread_pct, genome.min_open_interest)  # fmt: skip
    for d in days:
        now = close_time(d)
        # 1. manage what is open
        for pos in list(open_pos):
            spot = source.spot(pos.underlying, d)
            if spot is None:
                continue
            value, quotes, stale = _mark(pos, source, d, spot)
            greeks, iv = _greeks_now(pos, quotes)
            pos.marks.append(Mark(spot, iv or (pos.entry_iv or 0.0), value * pos.qty, (d - pos.opened).days,
                                  greeks["delta"] * pos.qty, greeks["gamma"] * pos.qty, greeks["theta"] * pos.qty,
                                  greeks["vega"] * pos.qty))  # fmt: skip
            reason = pos.pending_exit
            dte = (pos.first_exp - d).days
            if d >= pos.first_exp:
                reason = "expiration"
            elif reason is None:
                reason = _exit_reason(genome, pos, value, dte, (d - pos.opened).days)
                if reason and assume.exit_delay_days and reason not in ("expiration",):
                    pos.pending_exit = reason  # a day late (stress)
                    continue
            if reason is None or (stale and reason != "expiration"):
                continue  # no market to exit into today (a stale leg): try tomorrow
            proceeds, exit_mid = _close(pos, quotes, spot, d, cfg.model, reason == "expiration")
            fees_out = (
                0.0
                if reason == "expiration"
                else cfg.fee_per_contract * _contracts(pos) * assume.fee_multiplier
            )
            cash += proceeds * pos.qty - fees_out
            open_pos.remove(pos)
            trades.append(_record(genome, pos, d, reason, proceeds, exit_mid, fees_out, spot, iv))
        # 2. entries
        equity_now = cash + sum(
            _mark(p, source, d, source.spot(p.underlying, d) or 0.0)[0] * p.qty for p in open_pos
        )
        for u in cfg.underlyings:
            f = features.get(u, {}).get(d)
            if f is None or any(p.underlying == u for p in open_pos):
                continue
            why = _passes(genome, f, rng)
            if why:
                skipped[why] += 1
                continue
            chain = source.chain(u, d, (genome.dte_min, genome.dte_max))
            if chain is None or not chain.quotes:
                skipped["no chain"] += 1
                continue
            why = _chain_filters(genome, chain.quotes, chain.underlying_price, now)
            if why:
                skipped[why] += 1
                continue
            cands = build(spec, chain.quotes, chain.underlying_price, now)
            pick = _pick(cands, genome)
            if pick is None:
                skipped["no liquid structure"] += 1
                continue
            if rng.random() < assume.miss_rate:
                skipped["missed fill"] += 1
                continue
            fill = _open_fill(pick, cfg.model)
            loss = pick.structure.max_loss() + max(fill - pick.structure.debit(), 0.0)
            if not math.isfinite(loss) or loss <= 0:
                skipped["undefined risk"] += 1
                continue
            qty = min(cfg.max_contracts, int(equity_now * genome.risk_per_trade // loss))
            if qty < 1:
                skipped["too small to size"] += 1
                continue
            fees_in = (
                cfg.fee_per_contract
                * sum(leg.ratio for leg in pick.structure.option_legs)
                * qty
                * assume.fee_multiplier
            )
            cash -= fill * qty + fees_in
            g0, iv0 = _greeks_now(
                _Pos(u, pick.structure, 1, d, 0, 0, 0, pick.expiration, {}, "", ""), list(pick.quotes)
            )
            pos = _Pos(u, pick.structure, qty, d, fill, pick.structure.debit(), fees_in, pick.expiration, f.as_dict(),
                       trend_regime(f), iv_regime(f), entry_iv=iv0, entry_greeks=g0)  # fmt: skip
            for leg, q in zip(pick.structure.option_legs, pick.quotes, strict=True):
                pos.last_leg_mid[leg.contract.symbol] = q.mid or leg.price  # type: ignore[union-attr]
            pos.marks.append(Mark(chain.underlying_price, iv0 or 0.0, pick.structure.debit() * qty, 0,
                                  g0["delta"] * qty, g0["gamma"] * qty, g0["theta"] * qty, g0["vega"] * qty))  # fmt: skip
            pos.features["iv_trend"] = iv_trend(f)
            open_pos.append(pos)
        curve.append(
            cash + sum(_mark(p, source, d, source.spot(p.underlying, d) or 0.0)[0] * p.qty for p in open_pos)
        )
    # anything still open at the end is valued at the mid, not closed: it is reported as open
    metrics = summarize(trades, curve)
    metrics["open_at_end"] = len(open_pos)
    return BacktestResult(genome, cfg, source.grade, LABELS.get(source.grade, source.grade), trades, days, curve,
                          metrics, dict(skipped))  # fmt: skip


def _contracts(pos: _Pos) -> int:
    return sum(leg.ratio for leg in pos.structure.option_legs) * pos.qty


def _exit_reason(g: Genome, pos: _Pos, value: float, dte: int, held: int) -> str | None:
    if dte <= g.exit_dte:
        return "exit_dte"
    if held >= g.max_hold_days:
        return "max_hold"
    entry = pos.entry_mid
    if entry > 0:  # a debit: judged on the premium paid
        gain = value - entry
        if g.take_profit is not None and gain >= g.take_profit * entry:
            return "take_profit"
        if g.stop_loss is not None and -gain >= g.stop_loss * entry:
            return "stop_loss"
    elif entry < 0:  # a credit: judged on the premium received
        credit = -entry
        profit = credit + value  # value is the (negative) cost to close
        if g.take_profit is not None and profit >= g.take_profit * credit:
            return "take_profit"
        if g.stop_loss is not None and -profit >= g.stop_loss * credit:
            return "stop_loss"
    return None


def _pick(cands: list[Candidate], g: Genome) -> Candidate | None:
    """The expiration nearest the middle of the DTE window whose every leg is liquid enough."""
    mid = (g.dte_min + g.dte_max) / 2
    ok = []
    for c in cands:
        bad = False
        for q in c.quotes:
            spct = q.spread_pct
            if (
                spct is None
                or spct > g.max_spread_pct
                or (q.open_interest is not None and q.open_interest < g.min_open_interest)
            ):
                bad = True
                break
        if not bad:
            ok.append(c)
    return min(ok, key=lambda c: abs(c.dte - mid), default=None)


def _open_fill(c: Candidate, model: ExecutionModel) -> float:
    total = 0.0
    it = iter(c.quotes)
    for leg in c.structure.legs:
        if leg.contract is None:
            total += leg.sign * leg.units * leg.price
            continue
        q = next(it)
        total += leg.sign * leg.units * leg_fill(leg.sign, q.bid or 0.0, q.ask or 0.0, model)
    return total


def _close(pos: _Pos, quotes: list[OptionQuote | None], spot: float, day: date, model: ExecutionModel,
           settle: bool) -> tuple[float, float]:  # fmt: skip
    """Proceeds per unit of closing every leg (and the same at the mid)."""
    proceeds = mid = 0.0
    for leg, q in zip(pos.structure.legs, quotes, strict=True):
        if leg.contract is None:
            proceeds += leg.sign * leg.units * spot
            mid += leg.sign * leg.units * spot
            continue
        if settle or day >= leg.contract.expiration:
            px = leg.contract.intrinsic(spot)
            proceeds += leg.sign * leg.units * px
            mid += leg.sign * leg.units * px
            continue
        if q is not None and q.two_sided:
            m = q.mid or 0.0
            fill = leg_fill(-leg.sign, q.bid or m, q.ask or m, model)
        elif q is not None and q.ask is not None:
            m = fill = one_sided(leg.sign, q)
        else:
            m = fill = pos.last_leg_mid.get(leg.contract.symbol, leg.price)
        proceeds += leg.sign * leg.units * fill
        mid += leg.sign * leg.units * m
    return proceeds, mid


def _record(g: Genome, pos: _Pos, day: date, reason: str, proceeds: float, exit_mid: float, fees_out: float,
            spot: float, iv: float | None) -> dict[str, Any]:  # fmt: skip
    pnl = (proceeds - pos.entry_fill) * pos.qty - pos.fees_in - fees_out
    spread_cost = ((pos.entry_fill - pos.entry_mid) + (exit_mid - proceeds)) * pos.qty
    attribution = None
    if len(pos.marks) >= 2:
        a = attribute_path(pos.marks, execution=-spread_cost, fees=-(pos.fees_in + fees_out))
        attribution = a.as_dict()
    entry_spot = pos.marks[0].spot if pos.marks else spot
    settle = []
    if reason == "expiration":
        from quantpulse.options.expiration import settlement

        settle = [
            settlement(leg.contract, leg.side, leg.ratio * pos.qty, spot)
            for leg in pos.structure.option_legs
            if leg.contract is not None
        ]
    return _plain(
        {
            "underlying": pos.underlying,
            "family": pos.structure.family,
            "direction": g.direction,
            "legs": [leg.label() for leg in pos.structure.legs],
            "structure_key": pos.structure.key(),
            "qty": pos.qty,
            "entry_date": pos.opened.isoformat(),
            "exit_date": day.isoformat(),
            "days_held": (day - pos.opened).days,
            "dte_entry": (pos.first_exp - pos.opened).days,
            "dte_exit": max((pos.first_exp - day).days, 0),
            "entry_fill": round(pos.entry_fill, 2),
            "entry_mid": round(pos.entry_mid, 2),
            "exit_value": round(proceeds, 2),
            "pnl": round(pnl, 2),
            "max_loss": round(
                (pos.structure.max_loss() + max(pos.entry_fill - pos.entry_mid, 0)) * pos.qty, 2
            ),
            "fees": round(pos.fees_in + fees_out, 2),
            "spread_cost": round(spread_cost, 2),
            "exit_reason": reason,
            "entry_iv": pos.entry_iv,
            "exit_iv": iv,
            "underlying_return": round(spot / entry_spot - 1, 5) if entry_spot else None,
            "regime": pos.regime,
            "iv_regime": pos.iv_regime,
            "features": pos.features,
            "entry_greeks": {k: round(v, 4) if v is not None else None for k, v in pos.entry_greeks.items()},
            "attribution": attribution,
            "settlement": settle,
        }
    )


def _plain(x: Any) -> Any:
    """numpy scalars → Python numbers (the record is stored as JSON)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, str):
        return x.item()
    return x
