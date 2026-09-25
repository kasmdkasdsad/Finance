"""Universe price panels: warehouse-first loading, incremental tails, re-adjustments and delistings."""

import functools
from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.core.errors import ProviderNoData
from quantpulse.providers import synthetic
from quantpulse.schemas.common import DataStatus
from quantpulse.schemas.market import PriceHistory

from .conftest import _client, make_settings

ANCHOR = datetime(2027, 3, 31, 21, 0, tzinfo=UTC)
FRIDAY = datetime(2026, 9, 25, 20, 30, tzinfo=UTC)  # 16:30 New York, after the close
DEAD_UNTIL = datetime(2024, 6, 28, 21, 0, tzinfo=UTC)


@functools.lru_cache(maxsize=16)
def _full(symbol):
    return synthetic.synthetic_history(symbol, "1d", ANCHOR - timedelta(days=2400), ANCHOR, ANCHOR)


class BulkFeed:
    name = "bulk"

    def __init__(self, clock, known):
        self.clock, self.known, self.calls, self.factor = clock, set(known), [], {}

    def configured(self):
        return True

    async def histories(self, symbols, interval, start, end):
        self.calls.append((tuple(symbols), start))
        now = self.clock.now()
        out = {}
        for s in symbols:
            if s not in self.known:
                continue
            f = self.factor.get(s, 1.0)
            bars = [
                b.model_copy(
                    update={"open": b.open * f, "high": b.high * f, "low": b.low * f, "close": b.close * f}
                )
                for b in _full(s).bars
                if start <= b.timestamp <= min(end, now)
            ]
            out[s] = PriceHistory(symbol=s, interval=interval, bars=bars)
        return out

    async def history(self, symbol, interval, start, end):  # pragma: no cover - the batch path is used
        raise AssertionError("per-symbol calls must not go to the bulk feed")

    async def quote(self, symbol):  # pragma: no cover
        raise ProviderNoData(self.name, "no quotes")


class SingleFeed:
    """Knows only a delisted company, whose bars stop in mid-2024."""

    name = "single"

    def __init__(self):
        self.calls = []

    def configured(self):
        return True

    async def history(self, symbol, interval, start, end):
        self.calls.append(symbol)
        if symbol != "DEAD":
            raise ProviderNoData(self.name, f"no bars for {symbol}")
        bars = [b for b in _full(symbol).bars if start <= b.timestamp <= min(end, DEAD_UNTIL)]
        return PriceHistory(symbol=symbol, interval=interval, bars=bars)

    async def quote(self, symbol):  # pragma: no cover
        raise ProviderNoData(self.name, "no quotes")


async def test_panel_is_incremental_and_honest(tmp_path, mock_net):
    clock = FakeClock(FRIDAY)
    universe = ["AAA", "BBB", "DEAD", "GHOST", "SPY"]
    async for api in _client(make_settings(tmp_path, enable_live_data=True), clock):
        market = api.container.market
        bulk, single = BulkFeed(clock, {"AAA", "BBB", "SPY"}), SingleFeed()
        market._providers[:] = [bulk, single]
        stages: list[tuple[float, str]] = []

        def record(fraction: float, stage: str, out: list = stages) -> None:
            out.append((fraction, stage))

        panel = await market.daily_panel(universe, 1825, progress=record)
        assert set(panel.frames) == {"AAA", "BBB", "DEAD", "SPY"}
        assert "GHOST" in panel.missing and "no bars" in panel.missing["GHOST"]
        assert {panel.status(s) for s in panel.frames} == {DataStatus.LIVE}
        assert panel.provenance["DEAD"].provider == "single"
        assert panel.frames["AAA"].index[-1] == datetime(2026, 9, 25)
        assert panel.frames["DEAD"].index[-1] == datetime(2024, 6, 28)
        assert stages[-1] == (1.0, "prices ready") and all(0 <= f <= 1 for f, _ in stages)
        first_start = bulk.calls[0][1]
        assert first_start == FRIDAY - timedelta(days=1825)
        assert sorted(single.calls) == ["DEAD", "GHOST"]

        # Same evening: everything comes from the warehouse, nothing is downloaded again.
        n_bulk, n_single = len(bulk.calls), len(single.calls)
        again = await market.daily_panel(universe, 1825)
        assert (len(bulk.calls), len(single.calls)) == (n_bulk, n_single)
        assert {again.status(s) for s in again.frames} == {DataStatus.CACHED}
        assert "delisted" in again.provenance["DEAD"].message
        assert again.frames["BBB"].equals(panel.frames["BBB"])

        # Next session: only the tail of the live listings is requested, in one bulk call.
        clock.advance((datetime(2026, 9, 28, 21, 0, tzinfo=UTC) - FRIDAY).total_seconds())
        monday = await market.daily_panel(universe, 1825)
        tail_calls = bulk.calls[n_bulk:]
        assert len(tail_calls) == 1 and set(tail_calls[0][0]) == {"AAA", "BBB", "SPY"}
        assert tail_calls[0][1] >= FRIDAY - timedelta(days=12)
        assert len(single.calls) == n_single  # the finished listing and the unknown ticker are left alone
        assert monday.frames["AAA"].index[-1] == datetime(2026, 9, 28)
        assert monday.status("AAA") is DataStatus.LIVE and monday.status("DEAD") is DataStatus.CACHED

        # A 2-for-1 split re-bases the vendor's whole history: the tail no longer matches, so AAA is
        # downloaded again in full and old and new bars never mix.
        bulk.factor["AAA"] = 0.5
        clock.advance(24 * 3600)
        split = await market.daily_panel(universe, 1825)
        old_day = datetime(2023, 3, 1)
        assert split.frames["AAA"].loc[old_day, "close"] == pytest.approx(
            monday.frames["AAA"].loc[old_day, "close"] * 0.5
        )
        assert split.frames["BBB"].loc[old_day, "close"] == pytest.approx(
            monday.frames["BBB"].loc[old_day, "close"]
        )
        assert any(
            call[0] == ("AAA",) and call[1] == clock.now() - timedelta(days=1825) for call in bulk.calls
        )


async def test_panel_offline_is_synthetic(tmp_path, clock):
    async for api in _client(make_settings(tmp_path), clock):
        panel = await api.container.market.daily_panel(["AAA", "SPY"], 900)
        assert set(panel.frames) == {"AAA", "SPY"} and panel.status("AAA") is DataStatus.SYNTHETIC
        assert list(panel.summary()) == ["prices:synthetic:synthetic"]
