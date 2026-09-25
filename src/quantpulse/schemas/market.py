"""Market data schemas: quotes and OHLCV bars."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from quantpulse.schemas.common import StrictModel, Symbol

Interval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
INTRADAY_INTERVALS: frozenset[str] = frozenset({"1m", "5m", "15m", "30m", "1h"})


class Quote(StrictModel):
    symbol: str
    price: float = Field(gt=0)
    previous_close: float | None = Field(default=None, gt=0)
    change: float | None = None
    change_percent: float | None = None
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    bid_size: float | None = Field(default=None, ge=0)
    ask_size: float | None = Field(default=None, ge=0)
    day_open: float | None = Field(default=None, gt=0)
    day_high: float | None = Field(default=None, gt=0)
    day_low: float | None = Field(default=None, gt=0)
    volume: float | None = Field(default=None, ge=0)
    vwap: float | None = Field(default=None, gt=0, description="Today's volume-weighted average price")
    currency: str = "USD"
    exchange: str | None = None
    name: str | None = None
    market_cap: float | None = Field(default=None, ge=0)
    shares_outstanding: float | None = Field(default=None, gt=0)
    dividend_yield: float | None = Field(default=None, ge=0, le=1)
    timestamp: AwareDatetime
    quote_timestamp: AwareDatetime | None = Field(
        default=None, description="When the bid/ask was quoted (the price's timestamp is the last trade's)"
    )
    feed: str | None = Field(
        default=None, description="Vendor feed, e.g. 'iex' (one exchange) or 'sip' (all US exchanges)"
    )

    @model_validator(mode="after")
    def _derive_change(self) -> Quote:
        if self.previous_close and self.change is None:
            self.change = self.price - self.previous_close
        if self.previous_close and self.change_percent is None:
            self.change_percent = (self.price / self.previous_close - 1.0) * 100.0
        return self

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask and self.ask >= self.bid > 0:
            return 0.5 * (self.bid + self.ask)
        return None


class Bar(StrictModel):
    timestamp: AwareDatetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _consistent_range(self) -> Bar:
        if self.high + 1e-9 < max(self.open, self.close, self.low) or self.low - 1e-9 > min(
            self.open, self.close, self.high
        ):
            raise ValueError("bar high/low inconsistent with open/close")
        return self

    @classmethod
    def sanitized(
        cls,
        timestamp: AwareDatetime,
        open_: float | None,
        high: float | None,
        low: float | None,
        close: float | None,
        volume: float | None,
    ) -> Bar | None:
        """Build a bar from possibly-dirty vendor values; returns ``None`` if unusable.

        Vendors occasionally publish bars whose high/low do not bracket open/close by a few ticks;
        those are widened rather than discarded. Missing OHLC fields fall back to the close.
        """
        if close is None or not math.isfinite(close) or close <= 0:
            return None
        o = open_ if open_ is not None and math.isfinite(open_) and open_ > 0 else close
        h = high if high is not None and math.isfinite(high) and high > 0 else max(o, close)
        lo = low if low is not None and math.isfinite(low) and low > 0 else min(o, close)
        h = max(h, o, close, lo)
        lo = min(lo, o, close, h)
        vol = volume if volume is not None and math.isfinite(volume) and volume >= 0 else 0.0
        return cls(timestamp=timestamp, open=o, high=h, low=lo, close=close, volume=vol)


class PriceHistory(StrictModel):
    symbol: str
    interval: Interval
    currency: str = "USD"
    bars: list[Bar]

    @field_validator("bars")
    @classmethod
    def _sorted_unique(cls, bars: list[Bar]) -> list[Bar]:
        dedup: dict[object, Bar] = {}
        for bar in bars:
            dedup[bar.timestamp] = bar
        return [dedup[k] for k in sorted(dedup)]  # type: ignore[type-var]

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]


class QuoteBatch(StrictModel):
    quotes: dict[str, Quote]


class HistoryQuery(StrictModel):
    symbol: Symbol
    interval: Interval = "1d"
    lookback_days: int = Field(default=365, ge=1, le=3650)
