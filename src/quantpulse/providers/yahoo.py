"""Yahoo Finance (keyless) provider: quotes, OHLCV bars, dividends, option chains and analyst estimates.

Chart data (``/v8/finance/chart``) needs no authentication. Quote batches, option chains and
``quoteSummary`` require Yahoo's cookie + "crumb" handshake, which is performed lazily and cached.
Yahoo aggressively rate-limits datacenter IPs; failures are reported as provider errors so the gateway
can fail over (Polygon/Alpaca) or fall back to warehouse / synthetic data.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from quantpulse.core.errors import ProviderError, ProviderHTTPError, ProviderNoData
from quantpulse.core.http import HttpClient
from quantpulse.providers.base import (
    WireModel,
    epoch_to_date,
    epoch_to_datetime,
    finite_or_none,
    parse_wire,
    positive_or_none,
    require,
)
from quantpulse.schemas.fundamentals import AnalystEstimates, AnalystPeriodEstimate
from quantpulse.schemas.market import INTRADAY_INTERVALS, Bar, Interval, PriceHistory, Quote
from quantpulse.schemas.options import OptionChain, OptionContract

NAME = "yahoo"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
COOKIE_URL = "https://fc.yahoo.com"
CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": BROWSER_UA, "Accept": "application/json,text/plain,*/*"}
INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}
# Yahoo only serves recent intraday history.
INTRADAY_MAX_DAYS: dict[str, int] = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729}
CRUMB_TTL_SECONDS = 3600.0


# ----------------------------------------------------------------------------- wire schemas
class _Raw(WireModel):
    raw: float | None = None


class _ChartMeta(WireModel):
    symbol: str
    currency: str | None = None
    exchangeName: str | None = None
    fullExchangeName: str | None = None
    regularMarketPrice: float | None = None
    regularMarketTime: int | None = None
    chartPreviousClose: float | None = None
    previousClose: float | None = None
    regularMarketDayHigh: float | None = None
    regularMarketDayLow: float | None = None
    regularMarketVolume: float | None = None
    longName: str | None = None
    shortName: str | None = None


class _QuoteArrays(WireModel):
    open: list[float | None] = []
    high: list[float | None] = []
    low: list[float | None] = []
    close: list[float | None] = []
    volume: list[float | None] = []


class _AdjClose(WireModel):
    adjclose: list[float | None] = []


class _Indicators(WireModel):
    quote: list[_QuoteArrays] = []
    adjclose: list[_AdjClose] | None = None


class _Dividend(WireModel):
    amount: float
    date: int


class _Events(WireModel):
    dividends: dict[str, _Dividend] | None = None


class _ChartResult(WireModel):
    meta: _ChartMeta
    timestamp: list[int] | None = None
    indicators: _Indicators | None = None
    events: _Events | None = None


class _Error(WireModel):
    code: str | None = None
    description: str | None = None


class _Chart(WireModel):
    result: list[_ChartResult] | None = None
    error: _Error | None = None


class _ChartEnvelope(WireModel):
    chart: _Chart


class _QuoteItem(WireModel):
    symbol: str
    regularMarketPrice: float | None = None
    regularMarketPreviousClose: float | None = None
    regularMarketOpen: float | None = None
    regularMarketDayHigh: float | None = None
    regularMarketDayLow: float | None = None
    regularMarketVolume: float | None = None
    regularMarketTime: int | None = None
    bid: float | None = None
    ask: float | None = None
    bidSize: float | None = None
    askSize: float | None = None
    currency: str | None = None
    fullExchangeName: str | None = None
    longName: str | None = None
    shortName: str | None = None
    marketCap: float | None = None
    sharesOutstanding: float | None = None
    trailingAnnualDividendYield: float | None = None


class _QuoteResponse(WireModel):
    result: list[_QuoteItem] = []
    error: _Error | None = None


class _QuoteEnvelope(WireModel):
    quoteResponse: _QuoteResponse


class _OptContract(WireModel):
    contractSymbol: str
    strike: float
    bid: float | None = None
    ask: float | None = None
    lastPrice: float | None = None
    volume: float | None = None
    openInterest: float | None = None
    impliedVolatility: float | None = None
    inTheMoney: bool | None = None
    lastTradeDate: int | None = None
    expiration: int | None = None


class _OptSet(WireModel):
    expirationDate: int
    calls: list[_OptContract] = []
    puts: list[_OptContract] = []


class _OptQuote(WireModel):
    regularMarketPrice: float | None = None
    regularMarketTime: int | None = None


class _OptResult(WireModel):
    underlyingSymbol: str
    expirationDates: list[int] = []
    quote: _OptQuote | None = None
    options: list[_OptSet] = []


class _OptChain(WireModel):
    result: list[_OptResult] | None = None
    error: _Error | None = None


class _OptEnvelope(WireModel):
    optionChain: _OptChain


class _Estimate(WireModel):
    avg: _Raw | None = None
    low: _Raw | None = None
    high: _Raw | None = None
    numberOfAnalysts: _Raw | None = None
    growth: _Raw | None = None


class _Trend(WireModel):
    period: str
    endDate: str | None = None
    growth: _Raw | None = None
    earningsEstimate: _Estimate | None = None
    revenueEstimate: _Estimate | None = None


class _EarningsTrend(WireModel):
    trend: list[_Trend] = []


class _FinancialData(WireModel):
    targetMeanPrice: _Raw | None = None
    targetHighPrice: _Raw | None = None
    targetLowPrice: _Raw | None = None
    recommendationMean: _Raw | None = None
    recommendationKey: str | None = None
    numberOfAnalystOpinions: _Raw | None = None


class _KeyStats(WireModel):
    beta: _Raw | None = None


class _SummaryResult(WireModel):
    financialData: _FinancialData | None = None
    defaultKeyStatistics: _KeyStats | None = None
    earningsTrend: _EarningsTrend | None = None


class _Summary(WireModel):
    result: list[_SummaryResult] | None = None
    error: _Error | None = None


class _SummaryEnvelope(WireModel):
    quoteSummary: _Summary


def _raw(v: _Raw | None) -> float | None:
    return finite_or_none(v.raw) if v is not None else None


# ----------------------------------------------------------------------------- provider
class YahooFinance:
    name = NAME

    def __init__(self, http: HttpClient) -> None:
        self._http = http
        self._crumb: str | None = None
        self._crumb_expires = 0.0
        self._crumb_lock = asyncio.Lock()

    def configured(self) -> bool:
        return True

    # ---------------------------------------------------------------- auth handshake
    async def _get_crumb(self, force: bool = False) -> str:
        async with self._crumb_lock:
            if not force and self._crumb and time.monotonic() < self._crumb_expires:
                return self._crumb
            # The consent cookie lands in the shared client's jar (domain-scoped to .yahoo.com).
            await self._http.request(
                NAME, "GET", COOKIE_URL, headers=HEADERS, expected=(200, 301, 302, 403, 404)
            )
            resp = await self._http.request(NAME, "GET", CRUMB_URL, headers=HEADERS)
            crumb = resp.text.strip()
            if not crumb or len(crumb) > 64 or "<" in crumb or " " in crumb:
                raise ProviderError(NAME, "could not obtain Yahoo crumb")
            self._crumb = crumb
            self._crumb_expires = time.monotonic() + CRUMB_TTL_SECONDS
            return crumb

    async def _authed_json(self, url: str, params: dict[str, object]) -> object:
        crumb = await self._get_crumb()
        try:
            return await self._http.get_json(NAME, url, params={**params, "crumb": crumb}, headers=HEADERS)
        except ProviderHTTPError as exc:
            if exc.status_code not in (401, 403):
                raise
        crumb = await self._get_crumb(force=True)  # crumb expired/invalidated: one retry
        return await self._http.get_json(NAME, url, params={**params, "crumb": crumb}, headers=HEADERS)

    # ---------------------------------------------------------------- quotes
    async def quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        try:
            return await self._quotes_batch(symbols)
        except ProviderError:
            # Crumb-less fallback: one chart request per symbol.
            results = await asyncio.gather(
                *(self._quote_from_chart(s) for s in symbols), return_exceptions=True
            )
            quotes = {s: r for s, r in zip(symbols, results, strict=True) if isinstance(r, Quote)}
            if not quotes:
                first = next((r for r in results if isinstance(r, BaseException)), None)
                if isinstance(first, ProviderError):
                    raise first from None
                raise ProviderNoData(NAME, "no quotes returned") from None
            return quotes

    async def quote(self, symbol: str) -> Quote:
        quotes = await self.quotes([symbol])
        require(symbol in quotes, NAME, f"no quote for {symbol}")
        return quotes[symbol]

    async def _quotes_batch(self, symbols: Sequence[str]) -> dict[str, Quote]:
        payload = await self._authed_json(QUOTE_URL, {"symbols": ",".join(symbols)})
        env = parse_wire(NAME, _QuoteEnvelope, payload)
        out: dict[str, Quote] = {}
        for item in env.quoteResponse.result:
            price = positive_or_none(item.regularMarketPrice)
            if price is None or item.regularMarketTime is None:
                continue
            out[item.symbol.upper()] = Quote(
                symbol=item.symbol.upper(),
                price=price,
                previous_close=positive_or_none(item.regularMarketPreviousClose),
                bid=positive_or_none(item.bid),
                ask=positive_or_none(item.ask),
                bid_size=positive_or_none(item.bidSize),
                ask_size=positive_or_none(item.askSize),
                day_open=positive_or_none(item.regularMarketOpen),
                day_high=positive_or_none(item.regularMarketDayHigh),
                day_low=positive_or_none(item.regularMarketDayLow),
                volume=finite_or_none(item.regularMarketVolume),
                currency=item.currency or "USD",
                exchange=item.fullExchangeName,
                name=item.longName or item.shortName,
                market_cap=positive_or_none(item.marketCap),
                shares_outstanding=positive_or_none(item.sharesOutstanding),
                dividend_yield=_clamp_yield(item.trailingAnnualDividendYield),
                timestamp=epoch_to_datetime(item.regularMarketTime),
            )
        require(bool(out), NAME, "empty quote response")
        return out

    async def _chart(self, symbol: str, params: dict[str, object]) -> _ChartResult:
        payload = await self._http.get_json(
            NAME, CHART_URL.format(symbol=symbol), params=params, headers=HEADERS
        )
        env = parse_wire(NAME, _ChartEnvelope, payload)
        if env.chart.error is not None and env.chart.error.code:
            raise ProviderNoData(NAME, f"{env.chart.error.code}: {env.chart.error.description}")
        require(bool(env.chart.result), NAME, f"no chart data for {symbol}")
        return env.chart.result[0]  # type: ignore[index]

    async def _quote_from_chart(self, symbol: str) -> Quote:
        result = await self._chart(symbol, {"range": "1d", "interval": "1d"})
        meta = result.meta
        price = positive_or_none(meta.regularMarketPrice)
        require(price is not None and meta.regularMarketTime is not None, NAME, f"no price for {symbol}")
        return Quote(
            symbol=symbol,
            price=price,
            previous_close=positive_or_none(meta.previousClose or meta.chartPreviousClose),
            day_high=positive_or_none(meta.regularMarketDayHigh),
            day_low=positive_or_none(meta.regularMarketDayLow),
            volume=finite_or_none(meta.regularMarketVolume),
            currency=meta.currency or "USD",
            exchange=meta.fullExchangeName or meta.exchangeName,
            name=meta.longName or meta.shortName,
            timestamp=epoch_to_datetime(meta.regularMarketTime),  # type: ignore[arg-type]
        )

    # ---------------------------------------------------------------- history / dividends
    async def history(self, symbol: str, interval: Interval, start: datetime, end: datetime) -> PriceHistory:
        if interval in INTRADAY_MAX_DAYS:
            start = max(start, end - timedelta(days=INTRADAY_MAX_DAYS[interval]))
        result = await self._chart(
            symbol,
            {
                "period1": int(start.timestamp()),
                "period2": int(end.timestamp()),
                "interval": INTERVAL_MAP[interval],
                "includePrePost": "false",
                "events": "div,splits",
            },
        )
        return _history_from_chart(symbol, interval, result)

    async def dividend_yield(self, symbol: str) -> float | None:
        """Trailing-twelve-month dividends divided by the current price (keyless)."""
        result = await self._chart(symbol, {"range": "1y", "interval": "1d", "events": "div"})
        price = positive_or_none(result.meta.regularMarketPrice)
        if price is None:
            return None
        cutoff = datetime.now(UTC) - timedelta(days=365)
        divs = (result.events.dividends or {}) if result.events else {}
        total = sum(d.amount for d in divs.values() if epoch_to_datetime(d.date) >= cutoff)
        return _clamp_yield(total / price)

    # ---------------------------------------------------------------- options
    async def option_chain(
        self, symbol: str, expirations: Sequence[date] | None, max_expirations: int = 8
    ) -> OptionChain:
        first = await self._options_page(symbol, None)
        all_exp = sorted({epoch_to_date(e) for e in first.expirationDates})
        require(bool(all_exp), NAME, f"no listed options for {symbol}")
        wanted = [e for e in (expirations or all_exp[:max_expirations]) if e in all_exp]
        require(bool(wanted), NAME, "requested expirations are not listed")
        pages: dict[date, _OptResult] = {}
        for opt_set in first.options:
            pages[epoch_to_date(opt_set.expirationDate)] = first
        missing = [e for e in wanted if e not in pages]
        results = await asyncio.gather(*(self._options_page(symbol, e) for e in missing))
        for e, res in zip(missing, results, strict=True):
            pages[e] = res

        spot = positive_or_none(first.quote.regularMarketPrice) if first.quote else None
        require(spot is not None, NAME, f"no underlying price for {symbol}")
        as_of = (
            epoch_to_datetime(first.quote.regularMarketTime)
            if first.quote and first.quote.regularMarketTime
            else datetime.now(UTC)
        )
        contracts: list[OptionContract] = []
        for exp in wanted:
            for opt_set in pages[exp].options:
                if epoch_to_date(opt_set.expirationDate) != exp:
                    continue
                for kind, items in (("call", opt_set.calls), ("put", opt_set.puts)):
                    for c in items:
                        contracts.append(
                            OptionContract(
                                contract_symbol=c.contractSymbol,
                                kind=kind,
                                strike=c.strike,
                                expiration=exp,
                                bid=positive_or_none(c.bid),
                                ask=positive_or_none(c.ask),
                                last=positive_or_none(c.lastPrice),
                                volume=finite_or_none(c.volume),
                                open_interest=finite_or_none(c.openInterest),
                                implied_volatility=positive_or_none(c.impliedVolatility),
                                in_the_money=c.inTheMoney,
                                last_trade=epoch_to_datetime(c.lastTradeDate) if c.lastTradeDate else None,
                            )
                        )
        require(bool(contracts), NAME, f"empty option chain for {symbol}")
        return OptionChain(
            underlying=symbol,
            underlying_price=spot,
            as_of=as_of,
            expirations=all_exp,
            contracts=contracts,
        )

    async def _options_page(self, symbol: str, expiration: date | None) -> _OptResult:
        params: dict[str, object] = {}
        if expiration is not None:
            params["date"] = int(
                datetime(expiration.year, expiration.month, expiration.day, tzinfo=UTC).timestamp()
            )
        payload = await self._authed_json(OPTIONS_URL.format(symbol=symbol), params)
        env = parse_wire(NAME, _OptEnvelope, payload)
        require(bool(env.optionChain.result), NAME, f"no option chain for {symbol}")
        return env.optionChain.result[0]  # type: ignore[index]

    # ---------------------------------------------------------------- analyst estimates
    async def estimates(self, symbol: str) -> AnalystEstimates:
        payload = await self._authed_json(
            SUMMARY_URL.format(symbol=symbol),
            {"modules": "financialData,defaultKeyStatistics,earningsTrend"},
        )
        env = parse_wire(NAME, _SummaryEnvelope, payload)
        require(bool(env.quoteSummary.result), NAME, f"no quoteSummary for {symbol}")
        res = env.quoteSummary.result[0]  # type: ignore[index]
        fd = res.financialData or _FinancialData()
        periods: list[AnalystPeriodEstimate] = []
        ltg: float | None = None
        for t in res.earningsTrend.trend if res.earningsTrend else []:
            if t.period == "+5y":
                ltg = _raw(t.growth)
                continue
            if t.period not in ("0q", "+1q", "0y", "+1y"):
                continue
            rev = t.revenueEstimate or _Estimate()
            eps = t.earningsEstimate or _Estimate()
            periods.append(
                AnalystPeriodEstimate(
                    period=t.period,
                    end_date=date.fromisoformat(t.endDate) if t.endDate else None,
                    revenue_avg=_raw(rev.avg),
                    revenue_low=_raw(rev.low),
                    revenue_high=_raw(rev.high),
                    revenue_growth=_raw(rev.growth),
                    eps_avg=_raw(eps.avg),
                    eps_growth=_raw(eps.growth),
                    analysts=int(n)
                    if (n := _raw(rev.numberOfAnalysts) or _raw(eps.numberOfAnalysts))
                    else None,
                )
            )
        count = _raw(fd.numberOfAnalystOpinions)
        return AnalystEstimates(
            symbol=symbol,
            target_mean_price=_raw(fd.targetMeanPrice),
            target_high_price=_raw(fd.targetHighPrice),
            target_low_price=_raw(fd.targetLowPrice),
            recommendation_mean=_raw(fd.recommendationMean),
            recommendation_key=fd.recommendationKey,
            analyst_count=int(count) if count else None,
            long_term_growth=ltg,
            beta=_raw(res.defaultKeyStatistics.beta) if res.defaultKeyStatistics else None,
            periods=periods,
        )


def _clamp_yield(value: float | None) -> float | None:
    v = finite_or_none(value)
    if v is None or v < 0 or v > 1:
        return None
    return v


def _history_from_chart(symbol: str, interval: Interval, result: _ChartResult) -> PriceHistory:
    stamps = result.timestamp or []
    arrays = result.indicators.quote[0] if result.indicators and result.indicators.quote else _QuoteArrays()
    adj: list[float | None] = []
    if interval not in INTRADAY_INTERVALS and result.indicators and result.indicators.adjclose:
        adj = result.indicators.adjclose[0].adjclose
    bars: list[Bar] = []
    for i, ts in enumerate(stamps):

        def at(seq: list[float | None], i: int = i) -> float | None:
            return finite_or_none(seq[i]) if i < len(seq) else None

        o, h, lo, c, v = at(arrays.open), at(arrays.high), at(arrays.low), at(arrays.close), at(arrays.volume)
        a = at(adj) if adj else None
        if c is not None and c > 0 and a is not None and a > 0:
            factor = a / c  # split + dividend adjustment applied to the whole candle
            o = o * factor if o is not None else None
            h = h * factor if h is not None else None
            lo = lo * factor if lo is not None else None
            c = a
        bar = Bar.sanitized(epoch_to_datetime(ts), o, h, lo, c, v)
        if bar is not None:
            bars.append(bar)
    require(bool(bars), NAME, f"no bars for {symbol}")
    return PriceHistory(symbol=symbol, interval=interval, currency=result.meta.currency or "USD", bars=bars)
