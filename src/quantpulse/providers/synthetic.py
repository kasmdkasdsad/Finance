"""Deterministic synthetic data used when every live source (and the warehouse) is unavailable.

Design goals: *clearly labelled* (always served with ``DataStatus.SYNTHETIC``), deterministic (the same
request returns the same numbers, so charts do not flicker), and internally consistent (quotes agree
with bars; asset returns share a market factor so portfolio correlations and betas are realistic).
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import math
from datetime import UTC, date, datetime, time, timedelta

import numpy as np

from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, previous_trading_day
from quantpulse.domain.earnings import reaction_day
from quantpulse.domain.fundamental_factors import CompanyFacts, Fact
from quantpulse.domain.sectors import FF12_NAMES
from quantpulse.quant.black_scholes import bsm_price
from quantpulse.quant.vol_surface import years_to_expiry
from quantpulse.schemas.fundamentals import (
    AnalystEstimates,
    AnalystPeriodEstimate,
    CompanyFundamentals,
    FinancialStatement,
)
from quantpulse.schemas.market import Bar, Interval, PriceHistory, Quote
from quantpulse.schemas.options import OptionChain, OptionContract, YieldCurve, YieldPoint
from quantpulse.schemas.reference import CompanyEvents, CompanyProfile
from quantpulse.schemas.sports import Game, League, MarketLine, TeamRef, TeamScore
from quantpulse.schemas.vehicle import FuelPrice, FuelPriceSeries

MARKET_VOL = 0.16
MARKET_DRIFT = 0.07
CACHED_SPAN_DAYS = 1900  # windows starting within ~5 years of the anchor are sliced from a memoised series


def _digest(*parts: object) -> bytes:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()


def _seed(*parts: object) -> int:
    return int.from_bytes(_digest(*parts)[:8], "big")


def _unit(*parts: object) -> float:
    """Deterministic uniform number in [0, 1)."""
    return (_seed(*parts) % 10_000_000) / 10_000_000


def _normal(*parts: object) -> float:
    """Deterministic standard-normal draw (Box-Muller on two 64-bit uniforms taken from one hash)."""
    d = _digest(*parts)
    u1 = (int.from_bytes(d[:8], "big") + 1) / (2**64 + 1)  # in (0, 1): log is finite
    u2 = int.from_bytes(d[8:16], "big") / 2**64
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def symbol_profile(symbol: str) -> tuple[float, float, float]:
    """(anchor price, beta, idiosyncratic annual vol) for a symbol."""
    if symbol in ("SPY", "^GSPC"):
        return 560.0, 1.0, 0.0
    if symbol == "QQQ":
        return 480.0, 1.2, 0.05
    price = 20.0 + 480.0 * _unit("price", symbol)
    beta = 0.6 + 0.8 * _unit("beta", symbol)
    idio = 0.12 + 0.25 * _unit("idio", symbol)
    return round(price, 2), beta, idio


def _trading_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if is_trading_day(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _anchor_day(now: datetime) -> date:
    today = now.astimezone(NEW_YORK).date()
    return today if is_trading_day(today) else previous_trading_day(today)


@functools.lru_cache(maxsize=16384)
def _market_shock(day: date) -> float:
    return _normal("mkt", day)


def _log_return(symbol: str, day: date) -> float:
    _, beta, idio = symbol_profile(symbol)
    market = _market_shock(day) * MARKET_VOL / math.sqrt(252) + MARKET_DRIFT / 252
    own = _normal("ret", symbol, day) * idio / math.sqrt(252)
    if day in _earnings_reaction_days(symbol):
        own += _normal("earn-jump", symbol, day) * 0.03 * (0.6 + _unit("earn-size", symbol))
    return beta * market + own


# ----------------------------------------------------------------------------- company events
EARNINGS_EPOCH = date(2000, 1, 3)
EARNINGS_CYCLE_DAYS = 91


def _synthetic_earnings(symbol: str) -> list[datetime]:
    """Quarterly releases (±3 days of jitter) from 2000 to 2035, before the open or after the close."""
    if symbol in ("SPY", "QQQ", "^GSPC"):
        return []
    before_open = _unit("earn-time", symbol) < 0.4
    first = EARNINGS_EPOCH + timedelta(days=int(_unit("earn-offset", symbol) * EARNINGS_CYCLE_DAYS))
    out: list[datetime] = []
    k = 0
    while True:
        d = first + timedelta(days=k * EARNINGS_CYCLE_DAYS + round(_normal("earn-jitter", symbol, k) * 2))
        if d.year > 2035:
            return out
        while d.weekday() >= 5:
            d += timedelta(days=1)
        at = time(7, 0) if before_open else time(16, 30)
        out.append(datetime.combine(d, at, NEW_YORK).astimezone(UTC))
        k += 1


@functools.lru_cache(maxsize=4096)
def _earnings_reaction_days(symbol: str) -> frozenset[date]:
    return frozenset(reaction_day(e) for e in _synthetic_earnings(symbol))


def synthetic_company_events(symbol: str, since: date, now: datetime) -> CompanyEvents:
    codes = [c for c in FF12_NAMES if c != "Other"]
    sector = codes[int(_unit("sector", symbol) * len(codes))]
    events = [e for e in _synthetic_earnings(symbol) if e.date() >= since and e <= now]
    return CompanyEvents(
        profile=CompanyProfile(
            symbol=symbol,
            cik=f"{_seed('cik', symbol) % 10**9:010d}",
            name=f"{symbol} (synthetic)",
            sic=None,
            sic_description=None,
            sector=sector,
            sector_label=FF12_NAMES[sector],
        ),
        earnings=events,
        earnings_since=since,
    )


def _closes(symbol: str, days: list[date]) -> np.ndarray:
    rets = np.array([_log_return(symbol, d) for d in days])
    # price(d_i) = anchor_price * exp(-sum of returns after d_i)
    suffix = np.concatenate([np.cumsum(rets[::-1])[::-1][1:], [0.0]])
    base, _, _ = symbol_profile(symbol)
    return base * np.exp(-suffix)


@functools.lru_cache(maxsize=512)
def _cached_series(symbol: str, anchor: date) -> tuple[tuple[date, ...], np.ndarray]:
    span = _trading_days(anchor - timedelta(days=CACHED_SPAN_DAYS), anchor)
    return tuple(span), _closes(symbol, span)


def daily_closes(symbol: str, start: date, now: datetime) -> tuple[list[date], np.ndarray]:
    """Closing prices for every trading day in [start, anchor], anchored so the last close is the
    symbol's anchor price. Windows are mutually consistent regardless of ``start``."""
    anchor = _anchor_day(now)
    if start > anchor:
        return [], np.array([])
    if start >= anchor - timedelta(days=CACHED_SPAN_DAYS):
        cached_days, cached = _cached_series(symbol, anchor)
        i = bisect.bisect_left(cached_days, start)
        return list(cached_days[i:]), cached[i:].copy()
    days = _trading_days(start, anchor)
    if not days:
        return [], np.array([])
    return days, _closes(symbol, days)


def synthetic_history(
    symbol: str, interval: Interval, start: datetime, end: datetime, now: datetime
) -> PriceHistory:
    first_day = start.astimezone(NEW_YORK).date() - timedelta(days=7)
    days, closes = daily_closes(symbol, first_day, now)
    end_day = end.astimezone(NEW_YORK).date()
    start_day = start.astimezone(NEW_YORK).date()
    bars: list[Bar] = []
    if interval in ("1d", "1wk", "1mo"):
        daily: list[Bar] = []
        for i, d in enumerate(days):
            if i == 0:
                continue
            prev, close = closes[i - 1], closes[i]
            gap = _normal("gap", symbol, d) * 0.003
            open_ = prev * math.exp(gap)
            spread = abs(_normal("rng", symbol, d)) * 0.008 + 0.002
            high = max(open_, close) * (1 + spread / 2)
            low = min(open_, close) * (1 - spread / 2)
            vol = 1e6 * (5 + 20 * _unit("vol", symbol)) * (0.7 + 0.6 * _unit("v", symbol, d))
            if start_day <= d <= end_day:
                stamp = datetime.combine(d, time(16, 0), NEW_YORK).astimezone(UTC)
                daily.append(
                    Bar(timestamp=stamp, open=open_, high=high, low=low, close=close, volume=round(vol))
                )
        bars = daily if interval == "1d" else _aggregate(daily, interval)
    else:
        minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[interval]
        steps = int(390 / minutes)
        for i, d in enumerate(days):
            if i == 0 or not (start_day <= d <= end_day):
                continue
            prev, close = float(closes[i - 1]), float(closes[i])
            rng = np.random.default_rng(_seed("intraday", symbol, d, interval))
            noise = rng.standard_normal(steps) * 0.0015 * math.sqrt(minutes)
            walk = np.cumsum(noise)
            bridge = walk - np.linspace(0, 1, steps) * walk[-1]  # pin the path to the daily close
            path = prev * np.exp(np.linspace(0, math.log(close / prev), steps) + bridge)
            open_session = datetime.combine(d, time(9, 30), NEW_YORK)
            last = prev
            for k in range(steps):
                stamp = (open_session + timedelta(minutes=k * minutes)).astimezone(UTC)
                if stamp > now:
                    break
                c = float(path[k])
                hi, lo = max(last, c) * 1.0005, min(last, c) * 0.9995
                bars.append(
                    Bar(
                        timestamp=stamp,
                        open=last,
                        high=hi,
                        low=lo,
                        close=c,
                        volume=round(2e5 * (1 + _unit("iv", symbol, d, k))),
                    )
                )
                last = c
    return PriceHistory(symbol=symbol, interval=interval, bars=bars)


def _aggregate(daily: list[Bar], interval: Interval) -> list[Bar]:
    groups: dict[tuple[int, int], list[Bar]] = {}
    for bar in daily:
        local = bar.timestamp.astimezone(NEW_YORK).date()
        key = local.isocalendar()[:2] if interval == "1wk" else (local.year, local.month)
        groups.setdefault(key, []).append(bar)
    out = []
    for key in sorted(groups):
        g = groups[key]
        out.append(
            Bar(
                timestamp=g[-1].timestamp,
                open=g[0].open,
                high=max(b.high for b in g),
                low=min(b.low for b in g),
                close=g[-1].close,
                volume=sum(b.volume for b in g),
            )
        )
    return out


def synthetic_quote(symbol: str, now: datetime) -> Quote:
    days, closes = daily_closes(symbol, _anchor_day(now) - timedelta(days=10), now)
    close, prev = float(closes[-1]), float(closes[-2])
    minute_bucket = int(now.timestamp() // 60)
    wiggle = math.exp(_normal("tick", symbol, minute_bucket) * 0.0008)
    price = round(close * wiggle, 2)
    spread = max(0.01, round(price * 0.0002, 2))
    return Quote(
        symbol=symbol,
        price=price,
        previous_close=round(prev, 2),
        bid=round(price - spread / 2, 2),
        ask=round(price + spread / 2, 2),
        day_open=round(prev * math.exp(_normal("gap", symbol, days[-1]) * 0.003), 2),
        day_high=round(max(price, close, prev) * 1.004, 2),
        day_low=round(min(price, close, prev) * 0.996, 2),
        volume=float(round(1e6 * (5 + 20 * _unit("vol", symbol)))),
        name=f"{symbol} (synthetic)",
        dividend_yield=round(0.03 * _unit("div", symbol), 4),
        timestamp=now,
    )


# ----------------------------------------------------------------------------- rates & options
SYNTHETIC_CURVE_PCT = [
    ("1 Mo", 1 / 12, 4.00),
    ("2 Mo", 2 / 12, 4.02),
    ("3 Mo", 0.25, 4.05),
    ("4 Mo", 4 / 12, 4.06),
    ("6 Mo", 0.5, 4.08),
    ("1 Yr", 1.0, 4.05),
    ("2 Yr", 2.0, 3.95),
    ("3 Yr", 3.0, 3.92),
    ("5 Yr", 5.0, 3.98),
    ("7 Yr", 7.0, 4.08),
    ("10 Yr", 10.0, 4.20),
    ("20 Yr", 20.0, 4.55),
    ("30 Yr", 30.0, 4.50),
]


def synthetic_curve(today: date) -> YieldCurve:
    return YieldCurve(
        as_of=today, points=[YieldPoint(tenor=t, years=y, rate=r / 100.0) for t, y, r in SYNTHETIC_CURVE_PCT]
    )


def _strike_step(spot: float) -> float:
    for limit, step in ((25, 0.5), (100, 1.0), (250, 2.5), (1000, 5.0)):
        if spot < limit:
            return step
    return 10.0


def synthetic_expirations(now: datetime, count: int = 10) -> list[date]:
    today = now.astimezone(NEW_YORK).date()
    fridays: list[date] = []
    cursor = today + timedelta(days=(4 - today.weekday()) % 7)
    while len(fridays) < 6:
        if (cursor - today).days >= 2:
            fridays.append(cursor)
        cursor += timedelta(days=7)
    monthly: list[date] = []
    year, month = today.year, today.month
    while len(monthly) < 6:
        first = date(year, month, 1)
        third_friday = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
        if third_friday > fridays[-1]:
            monthly.append(third_friday)
        month += 1
        if month > 12:
            year, month = year + 1, 1
    out = []
    for d in sorted(set(fridays + monthly)):
        exp = d if is_trading_day(d) else previous_trading_day(d)
        out.append(exp)
    return sorted(set(out))[:count]


def synthetic_option_chain(
    symbol: str,
    spot: float,
    now: datetime,
    rate_at,
    dividend_yield: float,
    expirations: list[date] | None = None,
    max_expirations: int = 8,
) -> OptionChain:
    all_exp = synthetic_expirations(now)
    default = all_exp[:max_expirations]
    wanted = [e for e in (expirations or default) if e in all_exp] or default
    atm = 0.18 + 0.25 * _unit("atm", symbol)
    step = _strike_step(spot)
    lo = math.floor(spot * 0.65 / step) * step
    hi = math.ceil(spot * 1.35 / step) * step
    strikes = np.arange(lo, hi + step / 2, step)
    contracts: list[OptionContract] = []
    for exp in wanted:
        t = years_to_expiry(exp, now)
        if t <= 0:
            continue
        r = rate_at(t)
        fwd = spot * math.exp((r - dividend_yield) * t)
        for k in strikes:
            if k <= 0:
                continue
            m = math.log(k / fwd)
            iv = max(0.05, atm * (1 - 0.35 * m + 0.9 * m * m) * (1 + 0.04 * math.sqrt(t)))
            for kind in ("call", "put"):
                px = bsm_price(spot, float(k), t, r, iv, dividend_yield, kind)
                if px < 0.01:
                    continue
                half = max(0.01, 0.015 * px) / 2
                bid = max(0.01, round(px - half, 2))
                ask = round(px + half, 2)
                occ = f"{symbol}{exp:%y%m%d}{'C' if kind == 'call' else 'P'}{round(float(k) * 1000):08d}"
                contracts.append(
                    OptionContract(
                        contract_symbol=occ,
                        kind=kind,
                        strike=float(k),
                        expiration=exp,
                        bid=bid,
                        ask=ask,
                        last=round(px, 2),
                        volume=float(int(500 * _unit("ov", occ))),
                        open_interest=float(int(5000 * _unit("oi", occ) * math.exp(-8 * m * m))),
                        implied_volatility=round(iv, 4),
                        in_the_money=(k < spot) if kind == "call" else (k > spot),
                    )
                )
    return OptionChain(
        underlying=symbol, underlying_price=spot, as_of=now, expirations=all_exp, contracts=contracts
    )


# ----------------------------------------------------------------------------- fundamentals
def synthetic_fundamentals(symbol: str, today: date) -> CompanyFundamentals:
    base_rev = 5e9 + 395e9 * _unit("rev", symbol)
    growth = 0.03 + 0.12 * _unit("growth", symbol)
    margin = 0.12 + 0.18 * _unit("margin", symbol)
    price, _, _ = symbol_profile(symbol)
    ps_ratio = 3 + 5 * _unit("ps", symbol)
    shares = base_rev * ps_ratio / price
    statements = []
    last_year = today.year - 1
    for i, fy in enumerate(range(last_year - 4, last_year + 1)):
        rev = base_rev / (1 + growth) ** (4 - i)
        ebit = rev * margin
        pretax = ebit * 0.98
        tax = pretax * 0.18
        statements.append(
            FinancialStatement(
                fiscal_year=fy,
                period_end=date(fy, 12, 31),
                form="10-K",
                revenue=rev,
                gross_profit=rev * (margin + 0.25),
                operating_income=ebit,
                pretax_income=pretax,
                income_tax=tax,
                net_income=pretax - tax,
                interest_expense=rev * 0.005,
                depreciation_amortization=rev * 0.04,
                total_assets=rev * 1.3,
                total_liabilities=rev * 0.7,
                stockholders_equity=rev * 0.6,
                cash=rev * 0.1,
                total_debt=rev * 0.25,
                current_assets=rev * 0.45,
                current_liabilities=rev * 0.35,
                operating_cash_flow=ebit * 0.95,
                capital_expenditure=rev * 0.05,
                diluted_shares=shares,
                diluted_eps=(pretax - tax) / shares,
            )
        )
    return CompanyFundamentals(
        symbol=symbol,
        cik=None,
        name=f"{symbol} (synthetic)",
        shares_outstanding=shares,
        shares_as_of=today,
        statements=statements,
    )


def synthetic_company_facts(symbol: str, now: datetime) -> CompanyFacts:
    """Six years of annual facts with year-to-year noise, and a public float marked to synthetic prices."""
    base_rev = 5e9 + 395e9 * _unit("rev", symbol)
    growth = 0.03 + 0.12 * _unit("growth", symbol)
    margin = 0.12 + 0.18 * _unit("margin", symbol)
    anchor_price, _, _ = symbol_profile(symbol)
    shares = base_rev * (3 + 5 * _unit("ps", symbol)) / anchor_price
    last = now.astimezone(NEW_YORK).year - 1
    first_price_day = _anchor_day(now) - timedelta(days=CACHED_SPAN_DAYS)
    days, closes = daily_closes(symbol, first_price_day, now)
    facts = CompanyFacts()
    for y in range(last - 5, last + 1):
        noise = [_normal(tag, symbol, y) for tag in ("rev", "margin", "cash", "assets")]
        rev = base_rev / (1 + growth) ** (last - y) * math.exp(0.05 * noise[0])
        m = max(0.02, margin + 0.03 * noise[1])
        ni = rev * m * 0.8
        start, end = date(y, 1, 1), date(y, 12, 31)
        facts.net_income.append(Fact(end, ni, start))
        facts.operating_cash_flow.append(Fact(end, ni * (1.1 + 0.2 * noise[2]), start))
        facts.capex.append(Fact(end, rev * 0.05, start))
        facts.gross_profit.append(Fact(end, rev * (m + 0.25), start))
        facts.assets.append(Fact(end, rev * 1.3 * math.exp(0.05 * noise[3])))
        facts.equity.append(Fact(end, rev * 0.6))
        float_day = date(y, 6, 30)
        if days and days[0] <= float_day:
            i = bisect.bisect_right(days, float_day) - 1
            facts.public_float.append(Fact(float_day, shares * float(closes[i]) * 0.9))
    return facts


def synthetic_estimates(symbol: str, today: date) -> AnalystEstimates:
    growth = 0.03 + 0.12 * _unit("growth", symbol)
    base_rev = 5e9 + 395e9 * _unit("rev", symbol)
    price, _, _ = symbol_profile(symbol)
    return AnalystEstimates(
        symbol=symbol,
        target_mean_price=round(price * (1.05 + 0.1 * _unit("tgt", symbol)), 2),
        recommendation_mean=round(1.8 + 1.2 * _unit("rec", symbol), 2),
        analyst_count=int(10 + 30 * _unit("n", symbol)),
        beta=round(symbol_profile(symbol)[1], 2),
        periods=[
            AnalystPeriodEstimate(
                period=f"{today.year + k}-12-31",
                end_date=date(today.year + k, 12, 31),
                revenue_avg=base_rev * (1 + growth) ** (k + 1),
                revenue_growth=growth,
            )
            for k in range(0, 2)
        ],
    )


# ----------------------------------------------------------------------------- fuel
GRADE_OFFSET = {"regular": 0.0, "midgrade": 0.45, "premium": 0.85, "diesel": 0.60}


def synthetic_fuel_series(
    region: str, region_name: str, grade: str, today: date, weeks: int = 104
) -> FuelPriceSeries:
    offset = 1.2 if region in ("SCA", "Y05LA", "Y05SF") else -0.35 + 0.8 * _unit("region", region)
    monday = today - timedelta(days=today.weekday())
    history: list[FuelPrice] = []
    level = 0.0
    for k in range(weeks - 1, -1, -1):
        period = monday - timedelta(weeks=k)
        level = 0.85 * level + _normal("fuel", region, period) * 0.04
        seasonal = 0.15 * math.sin(2 * math.pi * (period.timetuple().tm_yday - 60) / 365.25)
        price = round(max(1.5, 3.15 + offset + GRADE_OFFSET[grade] + seasonal + level), 3)
        history.append(
            FuelPrice(region=region, region_name=region_name, grade=grade, price=price, period=period)
        )
    return FuelPriceSeries(
        region=region, region_name=region_name, grade=grade, latest=history[-1], history=history
    )


# ----------------------------------------------------------------------------- sports
NFL_TEAMS = [
    ("ARI", "Arizona Cardinals"),
    ("ATL", "Atlanta Falcons"),
    ("BAL", "Baltimore Ravens"),
    ("BUF", "Buffalo Bills"),
    ("CAR", "Carolina Panthers"),
    ("CHI", "Chicago Bears"),
    ("CIN", "Cincinnati Bengals"),
    ("CLE", "Cleveland Browns"),
    ("DAL", "Dallas Cowboys"),
    ("DEN", "Denver Broncos"),
    ("DET", "Detroit Lions"),
    ("GB", "Green Bay Packers"),
    ("HOU", "Houston Texans"),
    ("IND", "Indianapolis Colts"),
    ("JAX", "Jacksonville Jaguars"),
    ("KC", "Kansas City Chiefs"),
    ("LV", "Las Vegas Raiders"),
    ("LAC", "Los Angeles Chargers"),
    ("LAR", "Los Angeles Rams"),
    ("MIA", "Miami Dolphins"),
    ("MIN", "Minnesota Vikings"),
    ("NE", "New England Patriots"),
    ("NO", "New Orleans Saints"),
    ("NYG", "New York Giants"),
    ("NYJ", "New York Jets"),
    ("PHI", "Philadelphia Eagles"),
    ("PIT", "Pittsburgh Steelers"),
    ("SF", "San Francisco 49ers"),
    ("SEA", "Seattle Seahawks"),
    ("TB", "Tampa Bay Buccaneers"),
    ("TEN", "Tennessee Titans"),
    ("WSH", "Washington Commanders"),
]
CFB_TEAMS = [
    ("ALA", "Alabama Crimson Tide"),
    ("UGA", "Georgia Bulldogs"),
    ("OSU", "Ohio State Buckeyes"),
    ("MICH", "Michigan Wolverines"),
    ("TEX", "Texas Longhorns"),
    ("ORE", "Oregon Ducks"),
    ("PSU", "Penn State Nittany Lions"),
    ("ND", "Notre Dame Fighting Irish"),
    ("LSU", "LSU Tigers"),
    ("CLEM", "Clemson Tigers"),
    ("FSU", "Florida State Seminoles"),
    ("FLA", "Florida Gators"),
    ("UCF", "UCF Knights"),
    ("MIA", "Miami Hurricanes"),
    ("USC", "USC Trojans"),
    ("OU", "Oklahoma Sooners"),
    ("TENN", "Tennessee Volunteers"),
    ("MISS", "Ole Miss Rebels"),
    ("UTAH", "Utah Utes"),
    ("WASH", "Washington Huskies"),
    ("KSU", "Kansas State Wildcats"),
    ("ISU", "Iowa State Cyclones"),
    ("BYU", "BYU Cougars"),
    ("ARIZ", "Arizona Wildcats"),
]


def _teams(league: League) -> list[TeamRef]:
    roster = NFL_TEAMS if league == "nfl" else CFB_TEAMS
    return [TeamRef(id=f"syn-{abbr}", abbreviation=abbr, name=name) for abbr, name in roster]


def season_start(league: League, season: int) -> datetime:
    """Kickoff of week 1 (NFL: Thursday after Labor Day; college: the Saturday before Labor Day)."""
    first = date(season, 9, 1)
    labor_day = first + timedelta(days=(0 - first.weekday()) % 7)
    kickoff = labor_day + timedelta(days=3) if league == "nfl" else labor_day - timedelta(days=2)
    return datetime.combine(kickoff, time(20, 20), NEW_YORK).astimezone(UTC)


def current_week(league: League, now: datetime) -> tuple[int, int]:
    season = now.year if now.month >= 3 else now.year - 1
    start = season_start(league, season)
    max_week = 18 if league == "nfl" else 15
    if now < start:
        return season, 1
    week = int((now - start).days // 7) + 1
    return season, min(max(week, 1), max_week)


def _strength(league: League, season: int, team_id: str) -> float:
    return _normal("strength", league, season, team_id) * (6.0 if league == "nfl" else 12.0)


def _kickoff_offset(league: League, index: int, n_games: int) -> float:
    """Hours after the week's opening kickoff, mimicking a real slate."""
    if league == "nfl":
        # Thursday night, Sunday 1:00 / 4:25 pm windows, Sunday night, Monday night (relative to Thu 8:20 pm ET)
        if index == 0:
            return 0.0
        if index == n_games - 1:
            return 96.0
        if index == n_games - 2:
            return 72.0
        return 64.67 if index % 2 else 68.08
    # College: Saturday noon, 3:30 pm, 7:00 pm, 10:30 pm ET (relative to Sat 8:20 pm ET the week before)
    return [15.67, 19.17, 22.67, 26.17][index % 4] + 24.0


def _week_games(league: League, season: int, week: int, now: datetime) -> list[Game]:
    teams = _teams(league)
    rng = np.random.default_rng(_seed("pairing", league, season, week))
    order = rng.permutation(len(teams))
    start = season_start(league, season) + timedelta(weeks=week - 1)
    games: list[Game] = []
    n_games = len(order) // 2
    for i in range(0, len(order) - 1, 2):
        home, away = teams[int(order[i])], teams[int(order[i + 1])]
        kickoff = start + timedelta(hours=_kickoff_offset(league, i // 2, n_games))
        edge = _strength(league, season, home.id) - _strength(league, season, away.id) + 2.0
        sigma = 13.45 if league == "nfl" else 15.5
        margin = edge + _normal("margin", league, season, week, home.id) * sigma
        total = max(20.0, 44.0 + _normal("total", league, season, week, home.id) * 9.0)
        home_final = max(0, round((total + margin) / 2))
        away_final = max(0, round((total - margin) / 2))
        if home_final == away_final:
            home_final += 3
        elapsed = (now - kickoff).total_seconds()
        game_len = 3.25 * 3600
        if elapsed < 0:
            state, completed, period, clock, detail = (
                "pre",
                False,
                0,
                None,
                kickoff.astimezone(NEW_YORK).strftime("%a %I:%M %p ET"),
            )
            hs = as_ = None
        elif elapsed >= game_len:
            state, completed, period, clock, detail = "post", True, 4, "0:00", "Final"
            hs, as_ = home_final, away_final
        else:
            frac = elapsed / game_len
            game_seconds = frac * 3600
            period = min(4, int(game_seconds // 900) + 1)
            remaining = max(0, int(900 - (game_seconds - (period - 1) * 900)))
            clock = f"{remaining // 60}:{remaining % 60:02d}"
            state, completed, detail = "in", False, f"{clock} - {period}Q"
            hs, as_ = round(home_final * frac), round(away_final * frac)
        games.append(
            Game(
                event_id=f"syn-{league}-{season}-{week}-{i // 2}",
                league=league,
                season=season,
                season_type=2,
                week=week,
                start_time=kickoff,
                name=f"{away.name} at {home.name}",
                state=state,
                completed=completed,
                status_detail=detail,
                period=period,
                display_clock=clock,
                home=TeamScore(team=home, score=hs),
                away=TeamScore(team=away, score=as_),
                market=MarketLine(
                    source="synthetic", spread_home=-round(edge * 2) / 2, over_under=round(total * 2) / 2
                ),
            )
        )
    return games


def synthetic_scoreboard(league: League, now: datetime) -> list[Game]:
    season, week = current_week(league, now)
    return _week_games(league, season, week, now)


def synthetic_season_results(league: League, now: datetime) -> list[Game]:
    season, week = current_week(league, now)
    games: list[Game] = []
    for w in range(1, week + 1):
        games.extend(g for g in _week_games(league, season, w, now) if g.completed)
    return games
