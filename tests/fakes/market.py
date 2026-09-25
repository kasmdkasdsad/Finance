"""A deterministic market-data vendor for trading tests: trending, flat and falling stocks with live quotes.

Implements the provider protocols the market service uses (``history``, ``histories`` for bulk panels,
``quote`` and multi-symbol ``quotes`` with VWAP and bid/ask), so the trading cycle runs on "live" data
without the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta

import numpy as np

from quantpulse.core.errors import ProviderNoData
from quantpulse.core.market_calendar import NEW_YORK, previous_trading_day
from quantpulse.schemas.market import Bar, PriceHistory, Quote

# daily log drift per symbol: steady uptrends, a flat name, downtrends and the index ETFs
DRIFTS: dict[str, float] = {
    "UPA": 0.0040,
    "UPB": 0.0034,
    "UPC": 0.0030,
    "UPD": 0.0026,
    "UPE": 0.0022,
    "MIDA": 0.0010,
    "MIDB": 0.0006,
    "FLAT": 0.0,
    "DNA": -0.0025,
    "DNB": -0.0030,
    "DNC": -0.0035,
    "SPY": 0.0007,
    "QQQ": 0.0009,
}
STOCKS = [s for s in DRIFTS if s not in ("SPY", "QQQ")]


def sessions_before(day: date, n: int) -> list[date]:
    out: list[date] = []
    d = day
    while len(out) < n:
        d = previous_trading_day(d)
        out.append(d)
    return sorted(out)


class TrendFeed:
    name = "trendfeed"

    def __init__(
        self, clock, drifts: dict[str, float] | None = None, sessions: int = 320, seed: int = 11
    ) -> None:
        self.clock = clock
        self.drifts = dict(drifts or DRIFTS)
        self.live_move: dict[str, float] = {}  # today's move vs the last close, per symbol
        self.quotes_enabled = True
        self.quote_age = timedelta(seconds=5)
        today = clock.now().astimezone(NEW_YORK).date()
        self.days = sessions_before(today, sessions)
        rng = np.random.default_rng(seed)
        self.bars: dict[str, list[Bar]] = {}
        for i, (symbol, drift) in enumerate(sorted(self.drifts.items())):
            noise = rng.normal(0.0, 0.006, len(self.days))
            logp = np.log(50.0 + 10 * i) + np.cumsum(drift + noise)
            closes = np.exp(logp)
            bars = []
            for d, c, prev in zip(self.days, closes, np.r_[closes[0], closes[:-1]], strict=True):
                high, low = max(c, prev) * 1.004, min(c, prev) * 0.996
                stamp = datetime.combine(d, time(16, 0), NEW_YORK).astimezone(UTC)
                bars.append(Bar(timestamp=stamp, open=prev, high=high, low=low, close=c, volume=4_000_000))
            self.bars[symbol] = bars

    def configured(self) -> bool:
        return True

    def last_close(self, symbol: str) -> float:
        return self.bars[symbol][-1].close

    def live_price(self, symbol: str) -> float:
        move = self.live_move.get(symbol, self.drifts.get(symbol, 0.0))
        return self.last_close(symbol) * float(np.exp(move))

    async def histories(self, symbols: Sequence[str], interval: str, start: datetime, end: datetime):
        out = {}
        for s in symbols:
            if s in self.bars:
                bars = [b for b in self.bars[s] if start <= b.timestamp <= min(end, self.clock.now())]
                if bars:
                    out[s] = PriceHistory(symbol=s, interval=interval, bars=bars)
        return out

    async def history(self, symbol: str, interval: str, start: datetime, end: datetime) -> PriceHistory:
        got = await self.histories([symbol], interval, start, end)
        if symbol not in got:
            raise ProviderNoData(self.name, f"no bars for {symbol}")
        return got[symbol]

    def _quote(self, symbol: str) -> Quote:
        price = self.live_price(symbol)
        return Quote(
            symbol=symbol,
            price=price,
            previous_close=self.last_close(symbol),
            bid=price * 0.9998,
            ask=price * 1.0002,
            volume=1_500_000,
            vwap=price * 0.999,
            day_open=self.last_close(symbol),
            day_high=price * 1.002,
            day_low=min(price, self.last_close(symbol)) * 0.998,
            timestamp=self.clock.now() - self.quote_age,
        )

    async def quote(self, symbol: str) -> Quote:
        if symbol not in self.bars or not self.quotes_enabled:
            raise ProviderNoData(self.name, f"no quote for {symbol}")
        return self._quote(symbol)

    async def quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        if not self.quotes_enabled:
            raise ProviderNoData(self.name, "quotes unavailable")
        return {s: self._quote(s) for s in symbols if s in self.bars}
