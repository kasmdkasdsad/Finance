"""Paper-trading ledger: simulated cash, positions and fills. No real orders are ever sent anywhere.

Execution model
    * Market orders fill at the reference price (latest quote / close) adjusted by ``slippage_bps`` against
      the trader: buys fill higher, sells lower.
    * Commission = ``commission_per_trade`` + ``commission_bps`` of notional, paid from cash.
    * Long-only, no margin: a buy may not take cash below zero and a sell may not exceed the position.
    * Fractional shares are allowed (rounded to 6 decimals), as offered by most modern brokers.

Cost basis is the volume-weighted average fill price (commission excluded); realised P&L on a sale is
``(fill − avg_cost) · qty − commission``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from quantpulse.core.errors import DomainError

Side = Literal["buy", "sell"]
QTY_DECIMALS = 6
EPS = 1e-9


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    commission_bps: float = 0.0

    def fill_price(self, side: Side, reference: float) -> float:
        if reference <= 0:
            raise DomainError("reference price must be positive")
        adj = self.slippage_bps / 10_000.0
        return reference * (1.0 + adj) if side == "buy" else reference * (1.0 - adj)

    def commission(self, notional: float) -> float:
        return self.commission_per_trade + abs(notional) * self.commission_bps / 10_000.0


@dataclass(slots=True)
class Holding:
    quantity: float
    avg_cost: float


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    reference_price: float
    commission: float
    realized_pnl: float

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class PaperBook:
    cash: float
    positions: dict[str, Holding] = field(default_factory=dict)

    def quantity(self, symbol: str) -> float:
        h = self.positions.get(symbol)
        return h.quantity if h else 0.0

    def market_value(self, prices: Mapping[str, float]) -> float:
        total = 0.0
        for symbol, h in self.positions.items():
            if symbol not in prices:
                raise DomainError(f"no price for held symbol {symbol}")
            total += h.quantity * prices[symbol]
        return total

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def buy(self, symbol: str, quantity: float, reference: float, model: ExecutionModel) -> Fill:
        qty = round(quantity, QTY_DECIMALS)
        if qty <= 0:
            raise DomainError("quantity must be positive")
        price = model.fill_price("buy", reference)
        fee = model.commission(qty * price)
        cost = qty * price + fee
        if cost > self.cash + EPS:
            raise DomainError(f"insufficient cash: need {cost:,.2f}, have {self.cash:,.2f}")
        h = self.positions.get(symbol)
        if h is None:
            self.positions[symbol] = Holding(qty, price)
        else:
            total = h.quantity + qty
            h.avg_cost = (h.quantity * h.avg_cost + qty * price) / total
            h.quantity = round(total, QTY_DECIMALS)
        self.cash -= cost
        if -EPS < self.cash < 0:
            self.cash = 0.0  # absorb floating-point dust when spending the last cent
        return Fill(symbol, "buy", qty, price, reference, fee, 0.0)

    def sell(self, symbol: str, quantity: float, reference: float, model: ExecutionModel) -> Fill:
        qty = round(quantity, QTY_DECIMALS)
        if qty <= 0:
            raise DomainError("quantity must be positive")
        h = self.positions.get(symbol)
        held = h.quantity if h else 0.0
        if h is None or qty > held + EPS:
            raise DomainError(f"cannot sell {qty:g} {symbol}: holding {held:g} (short selling is disabled)")
        qty = min(qty, held)
        price = model.fill_price("sell", reference)
        fee = model.commission(qty * price)
        realized = (price - h.avg_cost) * qty - fee
        self.cash += qty * price - fee
        remaining = round(held - qty, QTY_DECIMALS)
        if remaining <= 0:
            del self.positions[symbol]
        else:
            h.quantity = remaining
        return Fill(symbol, "sell", qty, price, reference, fee, realized)

    def max_affordable(self, reference: float, model: ExecutionModel, cash: float | None = None) -> float:
        """Largest quantity whose cost (with slippage and commission) fits in ``cash``."""
        budget = self.cash if cash is None else cash
        price = model.fill_price("buy", reference)
        budget -= model.commission_per_trade
        if budget <= 0:
            return 0.0
        qty = budget / (price * (1.0 + model.commission_bps / 10_000.0))
        return max(0.0, floor_quantity(qty))

    def rebalance(
        self,
        prices: Mapping[str, float],
        targets: Mapping[str, float],
        model: ExecutionModel,
        *,
        cash_buffer: float = 0.0,
        min_trade_value: float = 0.0,
    ) -> list[Fill]:
        """Trade towards ``targets`` (symbol → weight of equity). Sells run first, then buys; buys are
        scaled down proportionally if cash (after slippage and fees) cannot cover all of them."""
        if any(w < 0 for w in targets.values()):
            raise DomainError("target weights must be non-negative (long-only)")
        if sum(targets.values()) > 1.0 + 1e-9:
            raise DomainError("target weights sum to more than 100%")
        missing = [s for s in set(targets) | set(self.positions) if s not in prices]
        if missing:
            raise DomainError(f"missing prices for {sorted(missing)}")
        investable = self.equity(prices) * (1.0 - cash_buffer)
        fills: list[Fill] = []
        deltas = {
            s: targets.get(s, 0.0) * investable - self.quantity(s) * prices[s]
            for s in sorted(set(targets) | set(self.positions))
        }
        for s, delta in deltas.items():
            if delta < 0 and (-delta >= min_trade_value or targets.get(s, 0.0) == 0.0):
                qty = min(self.quantity(s), -delta / prices[s])
                if targets.get(s, 0.0) == 0.0:
                    qty = self.quantity(s)  # fully exit names that left the target list
                if qty > 0:
                    fills.append(self.sell(s, qty, prices[s], model))
        buys = {s: d for s, d in deltas.items() if d > 0 and d >= min_trade_value}
        needed = sum(
            model.fill_price("buy", prices[s]) * (d / prices[s]) + model.commission(d)
            for s, d in buys.items()
        )
        scale = min(1.0, self.cash / needed) if needed > 0 else 0.0
        for s, d in buys.items():
            qty = floor_quantity(d / prices[s] * scale)
            qty = min(qty, self.max_affordable(prices[s], model))
            if qty * prices[s] >= max(min_trade_value, 0.01):
                fills.append(self.buy(s, qty, prices[s], model))
        return fills


def floor_quantity(qty: float) -> float:
    """Round a share quantity down to the ledger's precision (never buys more than was paid for)."""
    factor = 10**QTY_DECIMALS
    return int(qty * factor) / factor
