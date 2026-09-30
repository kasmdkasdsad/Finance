"""A deterministic options market for tests: listed contracts on weekly Friday expirations, priced by
Black-Scholes at a set volatility, with a bid/ask spread, vendor Greeks and open interest.

It implements :class:`~quantpulse.options.data.OptionsMarketDataProvider` directly (no network) and keeps a
:class:`~tests.fakes.alpaca_paper.FakeAlpacaPaper` in step: every contract it quotes is priced at its mid
on the fake broker, so fills and position values agree with the quotes the risk engine saw.

Knobs: ``vol`` (and ``vol_by`` per underlying), ``half_spread`` (share of the mid on each side), ``age``
(seconds behind the clock), ``feed`` ("indicative", "opra" — or "model" to see it refused),
``open_interest``, ``down`` (every call raises, as an outage would).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Any

from quantpulse.core.errors import ProviderNoData
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.options.contracts import OptionContract, parse_occ
from quantpulse.options.data import ChainSnapshot, ContractInfo, OptionBar, OptionTrade
from quantpulse.options.pricing import greeks
from quantpulse.options.quotes import Greeks, OptionQuote


def fridays(start: date, weeks: int) -> list[date]:
    d = start + timedelta(days=(4 - start.weekday()) % 7)
    return [d + timedelta(weeks=i) for i in range(weeks)]


def strike_step(spot: float) -> float:
    return 1.0 if spot < 50 else 2.5 if spot < 150 else 5.0


class FakeOptionsMarket:
    name = "fake_options"

    def __init__(
        self,
        clock: Any,
        spot: Callable[[str], float],
        *,
        broker: Any = None,
        vol: float = 0.30,
        weeks: int = 10,
        strikes_each_side: int = 8,
    ) -> None:
        self.clock = clock
        self.spot = spot
        self.broker = broker
        self.vol = vol
        self.vol_by: dict[str, float] = {}
        self.weeks = weeks
        self.strikes_each_side = strikes_each_side
        self.half_spread = 0.02
        self.age = 3.0
        self._feed = "indicative"
        self.open_interest = 2500.0
        self.down = False
        self.calls: list[str] = []

    # ------------------------------------------------------------------ protocol
    def configured(self) -> bool:
        return True

    @property
    def feed(self) -> str:
        return self._feed

    @feed.setter
    def feed(self, value: str) -> None:
        self._feed = value

    def _check(self, what: str) -> None:
        self.calls.append(what)
        if self.down:
            raise ProviderNoData(self.name, "simulated options data outage")

    def listed(self, underlying: str) -> list[OptionContract]:
        spot = self.spot(underlying)
        step = strike_step(spot)
        atm = round(spot / step) * step
        strikes = [
            atm + step * k
            for k in range(-self.strikes_each_side, self.strikes_each_side + 1)
            if atm + step * k > 0
        ]
        today = self.clock.now().astimezone(NEW_YORK).date()
        out = []
        for exp in fridays(today, self.weeks):
            for k in strikes:
                for kind in ("call", "put"):
                    out.append(OptionContract(underlying, exp, kind, float(k)))  # type: ignore[arg-type]
        return out

    def quote(self, c: OptionContract) -> OptionQuote:
        now = self.clock.now()
        spot = self.spot(c.underlying)
        vol = self.vol_by.get(c.underlying, self.vol)
        v = greeks(c.kind, spot, c.strike, max(c.years(now), 1e-6), vol)
        mid = max(v.price, 0.01)
        hs = max(mid * self.half_spread, 0.01)
        bid = round(max(mid - hs, 0.0), 2)
        ask = round(mid + hs, 2)
        if self.broker is not None:
            self.broker.prices[c.symbol] = round((bid + ask) / 2, 4) if bid > 0 else ask
        stamp = now - timedelta(seconds=self.age)
        return OptionQuote(
            contract=c,
            bid=bid if bid > 0 else None,
            ask=ask,
            quote_at=stamp,
            feed=self._feed,  # type: ignore[arg-type]
            source=self.name,
            bid_size=10,
            ask_size=10,
            volume=500,
            open_interest=self.open_interest,
            iv=vol,
            greeks=Greeks(v.delta, v.gamma, v.theta, v.vega, v.rho, "vendor"),
            underlying_price=spot,
            underlying_at=now - timedelta(seconds=1),
        )

    async def contracts(
        self,
        underlying: str,
        *,
        expiration_from: date | None = None,
        expiration_to: date | None = None,
        strike_from: float | None = None,
        strike_to: float | None = None,
    ) -> list[ContractInfo]:
        self._check("contracts")
        out = []
        for c in self.listed(underlying):
            if (expiration_from and c.expiration < expiration_from) or (
                expiration_to and c.expiration > expiration_to
            ):
                continue
            if (strike_from is not None and c.strike < strike_from) or (
                strike_to is not None and c.strike > strike_to
            ):
                continue
            out.append(ContractInfo(c, True, "active", self.open_interest))
        return out

    async def chain(
        self, underlying: str, *, expiration_from: date | None = None, expiration_to: date | None = None
    ) -> ChainSnapshot:
        self._check("chain")
        quotes = [
            self.quote(c)
            for c in self.listed(underlying)
            if not (expiration_from and c.expiration < expiration_from)
            and not (expiration_to and c.expiration > expiration_to)
        ]
        now = self.clock.now()
        return ChainSnapshot(underlying, self.spot(underlying), now - timedelta(seconds=1), now, self._feed,
                             self.name, quotes, [])  # fmt: skip

    async def latest_quotes(self, symbols: Sequence[str]) -> dict[str, OptionQuote]:
        self._check("latest_quotes")
        return {s: self.quote(parse_occ(s)) for s in symbols}

    async def latest_trades(self, symbols: Sequence[str]) -> dict[str, OptionTrade]:
        self._check("latest_trades")
        return {}

    async def bars(self, symbols: Sequence[str], start: date, end: date) -> dict[str, list[OptionBar]]:
        self._check("bars")
        return {}

    def stream(self, symbols: Sequence[str]) -> AsyncIterator[OptionQuote | OptionTrade]:
        raise NotImplementedError("the fake options market does not stream")

    # ------------------------------------------------------------------ helpers for tests
    def pick(self, underlying: str, kind: str, *, moneyness: float, dte_min: int = 14) -> OptionContract:
        """The listed contract nearest ``spot × moneyness`` on the first expiration at least ``dte_min`` away."""
        now: datetime = self.clock.now()
        spot = self.spot(underlying)
        listed = [c for c in self.listed(underlying) if c.kind == kind and c.dte(now) >= dte_min]
        exp = min(c.expiration for c in listed)
        return min((c for c in listed if c.expiration == exp), key=lambda c: abs(c.strike - spot * moneyness))
