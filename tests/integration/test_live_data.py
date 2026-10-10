"""The stale-market-data fix, end to end through the API, on the fake Alpaca paper account and fake feeds.

The symptom: a paper test order refused with ``live_data: quote is 5886s old`` while the market was open.
The age was the last IEX *trade* — one exchange's print — although IEX's own bid/ask was live. The fix
measures a price's age on its freshest reliable observation and takes trading prices from the broker's own
feed first; the quote-age limit (600s) and the live-data requirement are unchanged.
"""

from datetime import timedelta

from quantpulse.core.clock import FakeClock
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import TrendFeed

from .conftest import NOW
from .test_brain_cycle import WIDE, brain_client, run_cycle
from .test_paper_execution import PHRASE, order_posts
from .test_trading import BASE, PAPER, trading_client


def iex_feed(clock, trade_age: float, bidask_age: float | None, name: str | None = None) -> TrendFeed:
    feed = TrendFeed(clock)
    feed.feed = "iex"
    feed.quote_age = timedelta(seconds=trade_age)
    feed.bidask_age = None if bidask_age is None else timedelta(seconds=bidask_age)
    if name:
        feed.name = name
    return feed


def live_check(out):
    return next(c for c in out["checks"] if c["name"] == "live_data")


async def test_an_hour_old_iex_print_with_a_live_iex_book_is_live_data(tmp_path):
    clock = FakeClock(NOW)  # Friday 10:00 New York, the market is open
    fake = FakeAlpacaPaper(clock=clock)
    fake.fill_mode["SPY"] = "accept"
    feed = iex_feed(clock, trade_age=5886, bidask_age=2)  # the symptom: no IEX print for 98 minutes
    async for api in trading_client(tmp_path, clock, fake=fake, feed=feed, **PAPER):
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        check = live_check(out)
        assert out["sent"] and check["passed"], out
        assert "IEX bid/ask midpoint (trendfeed), 2s old" in check["detail"]
        diag = (await api.get(f"{BASE}/diagnostics", params={"symbols": "SPY"})).json()
        [q] = diag["quotes"]
        assert q["price_source"] == "IEX bid/ask midpoint (trendfeed)"
        assert q["price_age_seconds"] == 2 and q["trade_age_seconds"] == 5886 and q["spread_ok"]


async def test_a_quiet_book_is_still_stale_and_the_refusal_says_what_was_measured(tmp_path):
    clock = FakeClock(NOW)
    feed = iex_feed(clock, trade_age=5886, bidask_age=5000)  # IEX's book has not moved for 83 minutes either
    async for api in trading_client(tmp_path, clock, feed=feed, **PAPER):
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        check = live_check(out)
        assert not out["sent"] and not check["passed"]
        assert check["detail"].startswith(
            "quote is 5000s old (limit 600s): the IEX bid/ask midpoint (trendfeed)"
        )
        assert "New York" in check["detail"] and "market is closed" not in check["detail"]
        assert order_posts(api.fake) == 0
    wide = iex_feed(clock, trade_age=5886, bidask_age=2)
    wide.half_spread = 0.004  # an 80bp IEX book is not a price: the old print stays the price, and stale
    async for api in trading_client(tmp_path / "wide", clock, feed=wide, **PAPER):
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        assert not out["sent"] and "the last IEX trade (trendfeed)" in live_check(out)["detail"]
        assert order_posts(api.fake) == 0


async def test_outside_the_session_the_refusal_says_the_market_is_closed(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False  # Alpaca's clock: closed
    feed = iex_feed(clock, trade_age=5886, bidask_age=5886)
    async for api in trading_client(tmp_path, clock, fake=fake, feed=feed, **PAPER):
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        detail = live_check(out)["detail"]
        assert not out["sent"] and "the market is closed, so no live price exists" in detail
        assert order_posts(fake) == 0


async def test_trading_prices_come_from_the_brokers_feed_before_a_delayed_vendor(tmp_path):
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.fill_mode["SPY"] = "accept"
    async for api in trading_client(tmp_path, clock, fake=fake, **PAPER):
        delayed = iex_feed(clock, trade_age=900, bidask_age=None, name="polygon")  # a 15-minute-delayed plan
        broker = iex_feed(clock, trade_age=3, bidask_age=1, name="alpaca")
        api.container.market._providers[:] = [delayed, broker]  # QP_MARKET_PROVIDERS starts with polygon
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        assert out["sent"] and "(alpaca)" in live_check(out)["detail"], out
        api.container.market._providers[:] = [delayed]  # only the delayed vendor: correctly refused
        clock.advance(120)
        out = (await api.post(f"{BASE}/test-order", json={"confirm": PHRASE})).json()
        assert not out["sent"] and "(polygon)" in live_check(out)["detail"]


async def test_the_brain_sees_live_data_when_only_the_iex_print_is_old(tmp_path):
    clock = FakeClock(NOW)
    feed = TrendFeed(clock, drifts=WIDE)
    feed.feed, feed.quote_age, feed.bidask_age = "iex", timedelta(seconds=5886), timedelta(seconds=2)
    async for api in brain_client(tmp_path, clock, feed=feed):
        cycle = await run_cycle(api)
        states = cycle["data_quality"]["states"]
        assert states and set(states.values()) <= {"fresh", "live"}
        assert not (cycle["data_quality"]["market"] or {}).get("veto")
        diag = next(iter(cycle["data_quality"]["diagnosis"].values()))
        assert diag["price_source"].startswith("IEX bid/ask midpoint") and diag["price_age_s"] == 2
        assert diag["trade_age_s"] == 5886 and any("no IEX print for 5,886s" in r for r in diag["reasons"])
