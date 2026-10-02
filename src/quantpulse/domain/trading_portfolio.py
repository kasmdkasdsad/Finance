"""Target portfolio and proposed trades for the paper-trading strategy (pure: no I/O).

1. **Risk exits first.** A holding down ``max_position_loss_pct`` is closed (stop-loss). One up
   ``take_profit_pct`` has ``take_profit_fraction`` sold once per entry price (take-profit). In a risk-off
   market, holdings below the entry threshold are closed.
2. **Strategy exits.** A holding is sold when its trend reverses (below its 20- and 50-day averages with a
   negative 10-day return), its opportunity score falls below ``exit_threshold``, the stock model turns
   from clearly positive to clearly negative since entry, or a stronger opportunity displaces it.
3. **Selection.** Holdings that pass get a head start of ``incumbent_bonus`` (limits churn); new names need
   ``entry_threshold`` plus the regime's entry penalty, an intact trend, liquidity, a live quote and no
   earnings release within the blackout. The best ``max_positions`` are kept.
4. **Sizing.** Weight ∝ conviction ÷ volatility, where conviction grows with the score and volatility is
   the largest of realised, ATR-based and implied volatility (floored at ``vol_floor``). Weights are scaled
   to the regime's share of the maximum exposure and capped per position by ``max_position_pct``, by a
   volatility budget (``position_vol_budget`` ÷ volatility) and by liquidity (``max_adv_pct`` of dollar
   volume); excess is redistributed to uncapped names. Weights under ``min_position_pct`` are dropped.
5. **Trades.** Only changes of at least ``min_weight_change`` are traded; buys are capped at
   ``max_order_notional`` per cycle (the rest follows next cycle); discretionary trades may not reverse the
   direction of a symbol's last trade within ``cooldown_minutes`` (no buy right after a sell or sell right
   after a buy) and share a per-cycle turnover budget. Sells are listed before buys.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

RISK_KINDS = frozenset({"stop_loss", "take_profit", "risk_off_exit", "flatten", "daily_loss_flatten"})


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    max_positions: int = 8
    max_position_pct: float = 0.30
    max_total_exposure_pct: float = 0.95
    cash_buffer_pct: float = 0.02
    min_position_pct: float = 0.03
    entry_threshold: float = 0.75
    exit_threshold: float = 0.0
    incumbent_bonus: float = 0.35
    min_weight_change: float = 0.02
    max_order_notional: float = 15_000.0
    min_order_notional: float = 100.0
    max_position_loss_pct: float = 0.08
    take_profit_pct: float = 0.25
    take_profit_fraction: float = 0.5
    position_vol_budget: float = 0.12
    vol_floor: float = 0.12
    max_adv_pct: float = 0.01
    max_cycle_turnover_pct: float = 0.6
    cooldown_minutes: float = 120.0
    model_flip: float = 0.5  # |model z| that counts as "clearly" positive or negative

    @classmethod
    def from_settings(cls, s: Any) -> StrategyConfig:
        return cls(
            max_positions=s.trading_max_positions,
            max_position_pct=s.trading_max_position_pct,
            max_total_exposure_pct=s.trading_max_total_exposure_pct,
            cash_buffer_pct=s.trading_cash_buffer_pct,
            min_position_pct=s.trading_min_position_pct,
            entry_threshold=s.trading_entry_threshold,
            exit_threshold=s.trading_exit_threshold,
            incumbent_bonus=s.trading_incumbent_bonus,
            min_weight_change=s.trading_min_weight_change,
            max_order_notional=s.trading_max_order_notional,
            min_order_notional=s.trading_min_order_notional,
            max_position_loss_pct=s.trading_max_position_loss_pct,
            take_profit_pct=s.trading_take_profit_pct,
            take_profit_fraction=s.trading_take_profit_fraction,
            position_vol_budget=s.trading_position_vol_budget,
            vol_floor=s.trading_vol_floor,
            max_adv_pct=s.trading_max_adv_pct,
            max_cycle_turnover_pct=s.trading_max_cycle_turnover_pct,
            cooldown_minutes=s.trading_cooldown_minutes,
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One scored, priced symbol. ``entry_blocks`` lists why it may not be *bought* (it can still be held)."""

    symbol: str
    score: float
    price: float
    risk_vol: float
    adv_dollar: float | None = None
    trend_ok: bool = True
    trend_broken: bool = False
    model_z: float | None = None
    entry_blocks: tuple[str, ...] = ()
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Holding:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_plpc: float


@dataclass(frozen=True, slots=True)
class PositionMemory:
    """What the strategy remembers about a holding since it entered."""

    entry_score: float | None = None
    entry_model_z: float | None = None
    profit_taken_basis: float | None = None  # avg entry price at which profit was last taken


@dataclass(frozen=True, slots=True)
class TargetPosition:
    symbol: str
    weight: float
    score: float
    conviction: float
    risk_vol: float
    capped_by: str | None = None
    incumbent: bool = False


@dataclass(frozen=True, slots=True)
class ProposedTrade:
    symbol: str
    side: str  # buy | sell
    qty: float
    est_price: float
    kind: str  # entry | add | trim | exit | stop_loss | take_profit | risk_off_exit | flatten | daily_loss_flatten
    reason: str
    current_weight: float
    target_weight: float
    score: float | None
    closes_position: bool = False

    @property
    def notional(self) -> float:
        return self.qty * self.est_price

    @property
    def risk_reducing(self) -> bool:
        return self.side == "sell" and (self.kind in RISK_KINDS or self.closes_position)


@dataclass
class PortfolioPlan:
    gross_target: float
    targets: dict[str, TargetPosition]
    trades: list[ProposedTrade]
    exits: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- sizing
def conviction(score: float, config: StrategyConfig) -> float:
    """Grows with the score above the exit threshold; bounded so one name cannot swamp the book."""
    return min(max(score - config.exit_threshold + 0.5, 0.25), 3.0)


def waterfill(raw: Mapping[str, float], caps: Mapping[str, float], total: float) -> dict[str, float]:
    """Distribute ``total`` in proportion to ``raw`` without exceeding ``caps``; leftovers go to names
    below their cap. The result may sum to less than ``total`` when every name is capped."""
    weights = dict.fromkeys(raw, 0.0)
    free = {s for s, r in raw.items() if r > 0 and caps.get(s, 0.0) > 0}
    remaining = total
    for _ in range(len(raw) + 1):
        if remaining <= 1e-12 or not free:
            break
        norm = sum(raw[s] for s in free)
        capped: set[str] = set()
        for s in free:
            share = remaining * raw[s] / norm
            room = caps[s] - weights[s]
            if share >= room - 1e-12:
                weights[s] = caps[s]
                capped.add(s)
        if not capped:
            for s in free:
                weights[s] += remaining * raw[s] / norm
            remaining = 0.0
            break
        free -= capped
        remaining = total - sum(weights.values())
    return weights


def size_positions(
    selected: Mapping[str, Candidate],
    incumbents: set[str],
    gross: float,
    equity: float,
    config: StrategyConfig,
) -> dict[str, TargetPosition]:
    """Conviction × inverse-volatility weights, capped and water-filled; tiny weights dropped."""
    names = dict(selected)
    targets: dict[str, TargetPosition] = {}
    for _ in range(len(selected) + 1):
        raw: dict[str, float] = {}
        caps: dict[str, float] = {}
        why: dict[str, str] = {}
        for s, c in names.items():
            vol = max(c.risk_vol if math.isfinite(c.risk_vol) else config.vol_floor, config.vol_floor)
            raw[s] = conviction(c.score, config) / vol
            options = {
                "position cap": config.max_position_pct,
                "volatility cap": config.position_vol_budget / vol,
            }
            if c.adv_dollar and c.adv_dollar > 0 and equity > 0:
                options["liquidity cap"] = config.max_adv_pct * c.adv_dollar / equity
            reason = min(options, key=lambda k: options[k])
            caps[s], why[s] = options[reason], reason
        weights = waterfill(raw, caps, gross)
        small = [s for s, w in weights.items() if w < config.min_position_pct]
        if small and len(small) < len(names):
            for s in small:
                names.pop(s)
            continue
        if small:  # everything is tiny: hold nothing rather than dust
            weights = {}
        targets = {
            s: TargetPosition(
                symbol=s,
                weight=w,
                score=names[s].score,
                conviction=conviction(names[s].score, config),
                risk_vol=max(names[s].risk_vol, config.vol_floor),
                capped_by=why[s] if w >= caps[s] - 1e-9 else None,
                incumbent=s in incumbents,
            )
            for s, w in weights.items()
            if w > 0
        }
        break
    return targets


# ----------------------------------------------------------------------------- plan
def _floor_qty(value: float) -> float:
    return float(math.floor(value + 1e-9))


def build_plan(
    candidates: Mapping[str, Candidate],
    holdings: Mapping[str, Holding],
    equity: float,
    config: StrategyConfig,
    *,
    regime_exposure: float = 1.0,
    entry_penalty: float = 0.0,
    risk_off: bool = False,
    memory: Mapping[str, PositionMemory] | None = None,
    last_traded: Mapping[str, tuple[datetime, str]] | None = None,
    working: set[str] | None = None,
    now: datetime | None = None,
) -> PortfolioPlan:
    memory = memory or {}
    last_traded = last_traded or {}
    working = working or set()
    if equity <= 0:
        return PortfolioPlan(0.0, {}, [], notes=["account equity is not positive: nothing to plan"])
    gross = min(config.max_total_exposure_pct, 1.0 - config.cash_buffer_pct) * max(regime_exposure, 0.0)
    notes: list[str] = []
    exits: dict[str, tuple[str, str]] = {}  # symbol -> (kind, reason)
    take_profit: dict[str, str] = {}
    skipped: dict[str, str] = {}

    def in_cooldown(symbol: str, side: str) -> bool:
        """Churn is reversing direction: no buy soon after a sell (or sell soon after a buy). Scaling
        further into the same direction is allowed at the next cycle."""
        last = last_traded.get(symbol)
        if last is None or now is None:
            return False
        at, last_side = last
        return last_side != side and now - at < timedelta(minutes=config.cooldown_minutes)

    # 1-2. exits for current holdings
    unscored: dict[str, Holding] = {}
    for sym, pos in holdings.items():
        c = candidates.get(sym)
        mem = memory.get(sym, PositionMemory())
        if pos.unrealized_plpc <= -config.max_position_loss_pct:
            exits[sym] = (
                "stop_loss",
                f"stop-loss: {pos.unrealized_plpc:+.1%} is past the −{config.max_position_loss_pct:.0%} limit",
            )
            continue
        if c is None:
            unscored[sym] = pos
            continue
        if risk_off and c.score < config.entry_threshold:
            exits[sym] = ("risk_off_exit", f"risk-off market: score {c.score:+.2f} is below the entry bar")
        elif c.trend_broken:
            exits[sym] = ("exit", "trend reversed: below its 20- and 50-day averages, falling over 10 days")
        elif c.score < config.exit_threshold:
            exits[sym] = (
                "exit",
                f"signal deteriorated: score {c.score:+.2f} < exit threshold {config.exit_threshold:+.2f}",
            )
        elif (
            mem.entry_model_z is not None
            and c.model_z is not None
            and mem.entry_model_z >= config.model_flip
            and c.model_z <= -config.model_flip
        ):
            exits[sym] = (
                "exit",
                f"stock model turned negative (z {mem.entry_model_z:+.2f} at entry → {c.model_z:+.2f})",
            )
        elif pos.unrealized_plpc >= config.take_profit_pct and not (
            mem.profit_taken_basis is not None and math.isclose(mem.profit_taken_basis, pos.avg_entry_price)
        ):
            take_profit[sym] = (
                f"take-profit: {pos.unrealized_plpc:+.1%} ≥ +{config.take_profit_pct:.0%}; selling "
                f"{config.take_profit_fraction:.0%} of the position"
            )
    for sym in unscored:
        skipped[sym] = "held but not scored this cycle (no live data): position left unchanged"

    # 3. selection
    ranked: list[tuple[float, str, bool]] = []
    for sym in holdings:
        if sym in exits or sym in unscored:
            continue
        c = candidates[sym]
        ranked.append((c.score + config.incumbent_bonus, sym, True))
    bar = config.entry_threshold + entry_penalty
    for sym, c in candidates.items():
        if sym in holdings:
            continue
        if c.entry_blocks:
            skipped[sym] = "; ".join(c.entry_blocks)
            continue
        if c.score < bar:
            continue
        if not c.trend_ok:
            skipped[sym] = (
                "score qualifies but the trend is not intact (price below 50-day average or 3m return ≤ 0)"
            )
            continue
        if in_cooldown(sym, "buy"):
            skipped[sym] = f"cooldown: sold within the last {config.cooldown_minutes:.0f} minutes"
            continue
        ranked.append((c.score, sym, False))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    slots = max(config.max_positions - len(unscored), 0)
    chosen = ranked[:slots]
    for rank, (_, sym, incumbent) in enumerate(ranked[slots:], start=slots + 1):
        if incumbent:
            exits[sym] = ("exit", f"displaced by stronger opportunities (ranked #{rank} of {len(ranked)})")
    if not chosen and not holdings:
        notes.append(f"no candidate reached the entry score of {bar:+.2f} with an intact trend")

    # 4. sizing
    held_unscored_w = sum(h.market_value for h in unscored.values()) / equity
    selected = {sym: candidates[sym] for _, sym, _ in chosen}
    incumbents = {sym for _, sym, inc in chosen if inc}
    targets = size_positions(selected, incumbents, max(gross - held_unscored_w, 0.0), equity, config)
    for sym in selected:
        if sym not in targets and sym in holdings:
            exits.setdefault(sym, ("exit", "target weight fell below the minimum position size"))
    if regime_exposure < 1.0:
        notes.append(f"market regime allows {regime_exposure:.0%} of the maximum exposure")

    # 5. trades
    trades: list[ProposedTrade] = []
    for sym in sorted(set(holdings) | set(targets)):
        h = holdings.get(sym)
        c = candidates.get(sym)
        price = c.price if c is not None else (h.current_price if h else 0.0)
        cur_w = (h.market_value / equity) if h else 0.0
        tgt = targets.get(sym)
        tgt_w = tgt.weight if tgt else (cur_w if sym in unscored else 0.0)
        score = c.score if c is not None else None
        if sym in unscored:
            continue
        if sym in working:
            skipped[sym] = "an order for this symbol is still working at Alpaca"
            continue
        if price <= 0:
            skipped[sym] = "no usable price"
            continue
        if h is not None and sym in exits:
            kind, reason = exits[sym]
            trades.append(
                ProposedTrade(
                    sym, "sell", h.qty, price, kind, reason, cur_w, 0.0, score, closes_position=True
                )
            )
            continue
        if h is not None and sym in take_profit:
            qty = _floor_qty(h.qty * config.take_profit_fraction)
            if qty > 0:
                trades.append(
                    ProposedTrade(
                        sym,
                        "sell",
                        qty,
                        price,
                        "take_profit",
                        take_profit[sym],
                        cur_w,
                        max(cur_w - qty * price / equity, 0.0),
                        score,
                    )
                )
                continue
        delta = tgt_w - cur_w
        if abs(delta) < config.min_weight_change:
            if h is None or abs(delta) > 1e-9:
                skipped[sym] = f"within the rebalance band (Δ weight {delta:+.1%})"
            continue
        if in_cooldown(sym, "buy" if delta > 0 else "sell"):
            skipped[sym] = (
                f"cooldown: {'sold' if delta > 0 else 'bought'} within the last {config.cooldown_minutes:.0f} "
                "minutes (no direction reversals)"
            )
            continue
        wanted = abs(delta) * equity
        notional = min(wanted, config.max_order_notional)
        if delta > 0:
            qty = _floor_qty(notional / price)
            kind = "add" if h is not None else "entry"
            reason = (
                f"{'new position' if h is None else 'add to position'}: score {score:+.2f}, target "
                f"{tgt_w:.1%} of equity"
                if score is not None
                else f"target {tgt_w:.1%} of equity"
            )
            side = "buy"
        else:
            qty = min(_floor_qty(notional / price), h.qty if h else 0.0)
            kind, side = "trim", "sell"
            reason = f"over target: {cur_w:.1%} held vs {tgt_w:.1%} target"
        if notional < wanted - 1e-6:
            reason += f" (scaling in: ${notional:,.0f} of ${wanted:,.0f} this cycle)"
        if qty <= 0 or qty * price < config.min_order_notional:
            skipped[sym] = f"order below the ${config.min_order_notional:,.0f} minimum"
            continue
        trades.append(ProposedTrade(sym, side, qty, price, kind, reason, cur_w, tgt_w, score))

    # turnover budget for discretionary trades (risk exits and full exits always go through)
    budget = config.max_cycle_turnover_pct * equity
    sells = [t for t in trades if t.side == "sell"]
    buys = sorted((t for t in trades if t.side == "buy"), key=lambda t: -(t.score or 0.0))
    kept: list[ProposedTrade] = []
    for t in [*sells, *buys]:
        if t.risk_reducing:
            kept.append(t)
            continue
        if t.notional > budget + 1e-6:
            skipped[t.symbol] = "deferred: this cycle's turnover budget is used up"
            continue
        budget -= t.notional
        kept.append(t)
    return PortfolioPlan(
        gross_target=gross,
        targets=targets,
        trades=kept,
        exits={s: r for s, (_, r) in exits.items()},
        skipped=skipped,
        notes=notes,
    )
