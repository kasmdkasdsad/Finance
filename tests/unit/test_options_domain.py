"""The options domain, computed exactly: contracts, quotes, pricing, structures, analytics, liquidity, the book,
attribution and expiration — including the invariants no structure may break (property-based)."""

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quantpulse.options import analytics as an
from quantpulse.options import attribution as att
from quantpulse.options import expiration as exp
from quantpulse.options import liquidity as liq
from quantpulse.options import portfolio as book
from quantpulse.options import pricing as px
from quantpulse.options import structures as s
from quantpulse.options.contracts import (
    ContractError,
    OptionContract,
    days_to_expiration,
    expiration_time,
    is_option_symbol,
    occ_symbol,
    parse_occ,
    trading_days_to_expiration,
    years_to_expiration,
)
from quantpulse.options.quotes import Greeks, OptionQuote, QuoteRules, validate

NOW = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)  # Friday 10:00 New York, market open
EXP = date(2026, 10, 16)


def call(k, e=EXP, u="AAPL"):
    return OptionContract(u, e, "call", k)


def put(k, e=EXP, u="AAPL"):
    return OptionContract(u, e, "put", k)


def quote(c, bid=2.0, ask=2.1, *, at=NOW, feed="opra", greeks=None, spot=100.0, spot_at=NOW, **kw):
    return OptionQuote(c, bid, ask, at, feed, "test", greeks=greeks or Greeks(delta=0.5 if c.is_call else -0.5),
                       underlying_price=spot, underlying_at=spot_at, **kw)  # fmt: skip


# --------------------------------------------------------------------------- contracts
def test_occ_symbols_round_trip():
    c = parse_occ("AAPL261016C00210000")
    assert (c.underlying, c.expiration, c.kind, c.strike) == ("AAPL", date(2026, 10, 16), "call", 210.0)
    assert c.symbol == "AAPL261016C00210000" and occ_symbol("SPY", EXP, "put", 432.5) == "SPY261016P00432500"
    assert parse_occ("O:SPY261016P00432500").strike == 432.5 and is_option_symbol("SPY261016P00432500")
    assert not is_option_symbol("AAPL")


@pytest.mark.parametrize("bad", ["AAPL", "AAPL261316C00210000", "AAPL261016X00210000", "261016C00210000", ""])
def test_bad_symbols_are_refused(bad):
    with pytest.raises(ContractError):
        parse_occ(bad)


def test_contracts_that_cannot_exist_are_refused():
    for args in (("AAPL", EXP, "call", 0.0), ("AAPL", EXP, "call", -5.0), ("AAPL", EXP, "fwd", 10.0),
                 ("", EXP, "call", 10.0), ("AAPL", EXP, "call", math.nan)):  # fmt: skip
        with pytest.raises(ContractError):
            OptionContract(*args)


def test_days_and_years_to_expiration():
    c = call(100)
    assert c.dte(NOW) == 21 and days_to_expiration(EXP, NOW) == 21
    assert expiration_time(EXP).hour == 16  # the close, New York
    assert expiration_time(date(2026, 11, 27)).hour == 13  # the day after Thanksgiving closes early
    assert years_to_expiration(EXP, NOW) == pytest.approx(
        (expiration_time(EXP) - NOW).total_seconds() / (365 * 86400)
    )
    after = expiration_time(EXP) + timedelta(minutes=1)
    assert years_to_expiration(EXP, after) == 0.0 and c.expired(after) and not c.expired(NOW)
    assert trading_days_to_expiration(EXP, NOW) == 16  # Fri 25 Sep … Fri 16 Oct, trading days only


# --------------------------------------------------------------------------- quotes
def test_a_good_quote_is_usable_for_execution():
    v = validate(quote(call(100)), NOW)
    assert v.usable_for_execution and v.usable_for_research and v.blockers == ()


@pytest.mark.parametrize(
    ("kw", "reason", "research_ok"),
    [
        ({"at": NOW - timedelta(minutes=5)}, "stale quote", True),
        ({"at": None}, "no timestamp", True),
        ({"at": NOW + timedelta(minutes=1)}, "in the future", True),
        ({"spot_at": NOW - timedelta(minutes=5)}, "stale underlying", True),
        ({"bid": 2.2, "ask": 2.1}, "crossed market", False),
        ({"bid": 2.1, "ask": 2.1}, "locked market", True),
        ({"bid": 0.0}, "zero or missing bid", True),
        ({"ask": None}, "no valid ask", False),
        ({"bid": 1.0, "ask": 3.0}, "spread", True),
        ({"greeks": Greeks()}, "missing Greeks", True),
        ({"greeks": Greeks(delta=1.7)}, "impossible Greeks", False),
        ({"feed": "recorded"}, "research only", True),
        ({"feed": "model"}, "research only", True),
        ({"iv": 9.0}, "impossible implied volatility", False),
    ],
)
def test_a_bad_quote_is_refused_for_execution_with_the_reason(kw, reason, research_ok):
    v = validate(quote(call(100), **kw), NOW)
    assert not v.usable_for_execution and any(reason in b for b in v.blockers), v.blockers
    assert v.usable_for_research is research_ok


def test_an_expired_contract_a_closed_market_and_intrinsic_arbitrage():
    old = call(100, e=date(2026, 9, 18))
    assert any("expired" in b for b in validate(quote(old), NOW).blockers)
    night = datetime(2026, 9, 26, 2, 0, tzinfo=UTC)
    v = validate(quote(call(100), at=night, spot_at=night), night)
    assert not v.usable_for_execution and "the market is closed" in v.blockers and v.usable_for_research
    deep = validate(quote(call(80), bid=15.0, ask=15.5), NOW)  # intrinsic 20 > ask
    assert any("below intrinsic" in b for b in deep.blockers) and not deep.usable_for_research


def test_the_indicative_feed_is_labelled_and_limits_come_from_the_rules():
    v = validate(quote(call(100), feed="indicative"), NOW)
    assert v.usable_for_execution and any("not firm" in w for w in v.warnings)
    strict = QuoteRules(max_quote_age_seconds=5, execution_feeds=("opra",))
    v = validate(quote(call(100), feed="indicative", at=NOW - timedelta(seconds=10)), NOW, strict)
    assert any("stale" in b for b in v.blockers) and any("research only" in b for b in v.blockers)


# --------------------------------------------------------------------------- pricing
def test_put_call_parity_and_iv_round_trip():
    spot, k, t, vol, r = 100.0, 105.0, 0.25, 0.3, 0.04
    c, p = px.greeks("call", spot, k, t, vol, r), px.greeks("put", spot, k, t, vol, r)
    assert c.price - p.price == pytest.approx(spot - k * math.exp(-r * t), abs=1e-9)
    assert px.implied_vol("call", c.price, spot, k, t, r) == pytest.approx(vol, abs=1e-6)
    assert (
        px.implied_vol("call", 0.0, spot, k, t, r) is None
        and px.implied_vol("call", 200.0, spot, k, t, r) is None
    )
    assert c.theta < 0 and c.vega > 0 and 0 < c.delta < 1 and -1 < p.delta < 0
    assert c.intrinsic == 0.0 and c.extrinsic == pytest.approx(c.price)


def test_american_value_is_never_below_european():
    euro_call = px.greeks("call", 100, 100, 0.5, 0.25, 0.04).price
    assert px.american_price("call", 100, 100, 0.5, 0.25, 0.04) == pytest.approx(
        euro_call, rel=5e-3
    )  # no dividend
    euro_put = px.greeks("put", 80, 100, 0.5, 0.25, 0.04).price
    american_put = px.american_price("put", 80, 100, 0.5, 0.25, 0.04)
    assert american_put >= euro_put and american_put >= 20.0  # deep ITM put: worth at least exercise
    assert px.early_exercise_premium("put", 80, 100, 0.5, 0.25, 0.04) > 0


def test_model_probabilities():
    p_itm = px.prob_itm("call", 100, 110, 0.25, 0.3, 0.0)
    p_touch = px.prob_touch(100, 110, 0.25, 0.3, 0.0)
    assert 0 < p_itm < p_touch < 1 and p_touch == pytest.approx(2 * p_itm, rel=0.15)
    assert px.prob_touch(100, 90, 0.25, 0.3, 0.0) > px.prob_itm("put", 100, 90, 0.25, 0.3, 0.0)
    assert px.expected_move(100, 0.3, 0.25) == pytest.approx(15.0)


# --------------------------------------------------------------------------- structures
def test_long_call_and_put():
    lc = s.long_call(call(100), 3.0)
    assert lc.max_loss() == 300 and math.isinf(lc.max_profit()) and lc.breakevens() == [103.0]
    assert lc.defined_risk and lc.capital_required() == 300
    lp = s.long_put(put(100), 2.0)
    assert lp.max_loss() == 200 and lp.max_profit() == 9800 and lp.breakevens() == [98.0]


def test_verticals():
    bcs = s.bull_call_spread(call(100), 3.0, call(105), 1.0)
    assert (
        bcs.debit() == 200
        and bcs.max_loss() == 200
        and bcs.max_profit() == 300
        and bcs.breakevens() == [102.0]
    )
    bps = s.bull_put_spread(put(100), 2.5, put(95), 1.0)
    assert bps.debit() == -150 and bps.max_profit() == 150 and bps.max_loss() == 350
    assert bps.capital_required() == 350 and bps.breakevens() == [98.5] and bps.naked_legs() == []
    bear = s.bear_call_spread(call(100), 2.0, call(110), 0.5)
    assert bear.max_loss() == 850 and bear.max_profit() == 150
    bpd = s.bear_put_spread(put(100), 3.0, put(90), 1.0)
    assert bpd.max_loss() == 200 and bpd.max_profit() == 800
    with pytest.raises(s.StructureError):
        s.bull_call_spread(call(105), 1.0, call(100), 3.0)  # strikes the wrong way round
    with pytest.raises(s.StructureError):
        s.bull_call_spread(call(100), 3.0, call(105, e=date(2026, 11, 20)), 1.0)  # two expirations


def test_straddle_strangle_condor_butterfly():
    st_ = s.long_straddle(call(100), 3.0, put(100), 2.5)
    assert st_.max_loss() == 550 and st_.breakevens() == [94.5, 105.5] and math.isinf(st_.max_profit())
    sg = s.long_strangle(call(105), 1.0, put(95), 1.0)
    assert sg.max_loss() == 200 and sg.breakevens() == [93.0, 107.0]
    ic = s.iron_condor(put(90), 0.5, put(95), 1.5, call(105), 1.5, call(110), 0.5)
    assert ic.max_profit() == 200 and ic.max_loss() == 300 and ic.breakevens() == [93.0, 107.0]
    assert ic.naked_legs() == [] and ic.capital_required() == 300
    fly = s.call_butterfly(call(95), 6.0, call(100), 3.0, call(105), 1.0)
    assert fly.debit() == 100 and fly.max_loss() == 100 and fly.max_profit() == 400
    assert fly.breakevens() == [96.0, 104.0]


def test_stock_structures_and_cash_secured_put():
    cc = s.covered_call(100.0, call(105), 2.0)
    assert cc.max_profit() == 700 and cc.max_loss() == 9800 and cc.naked_legs() == []
    csp = s.cash_secured_put(put(95), 2.0)
    assert (
        csp.max_loss() == 9300 and csp.capital_required() == 9300 and csp.naked_legs(cash_secured=True) == []
    )
    assert s.classify_risk(csp)["defined_risk"]
    pp = s.protective_put(100.0, put(95), 2.0)
    assert pp.max_loss() == 700 and math.isinf(pp.max_profit())
    collar = s.collar(100.0, put(95), 2.0, call(110), 2.0)
    assert collar.max_loss() == 500 and collar.max_profit() == 1000


def test_naked_short_options_are_detected_and_never_executable():
    naked_call = s.Structure("naked_call", "AAPL", (s.Leg("short", 1, 2.0, call(100)),))
    assert math.isinf(naked_call.max_loss()) and not naked_call.defined_risk
    info = s.classify_risk(naked_call)
    assert info["never_executable"] and info["naked_legs"]
    naked_put = s.Structure("naked_put", "AAPL", (s.Leg("short", 1, 2.0, put(100)),))
    assert s.classify_risk(naked_put)["never_executable"]  # bounded, but uncovered without cash: refused
    ratio = s.Structure(
        "ratio", "AAPL", (s.Leg("long", 1, 3.0, call(100)), s.Leg("short", 2, 1.0, call(105)))
    )
    assert math.isinf(ratio.max_loss()) and len(ratio.naked_legs()) == 1 and ratio.naked_legs()[0].ratio == 1
    earlier_long = s.Structure("x", "AAPL", (s.Leg("long", 1, 1.0, call(110, e=date(2026, 10, 2))),
                                             s.Leg("short", 1, 2.0, call(100))))  # fmt: skip
    assert earlier_long.naked_legs()  # a long that expires first does not cover the short


def test_calendar_uses_the_model_for_the_later_leg():
    cal = s.calendar(call(100), 2.0, call(100, e=date(2026, 11, 20)), 3.5)
    assert not cal.single_expiry and cal.max_loss() > 0 and math.isfinite(cal.max_loss())
    with pytest.raises(s.StructureError):
        cal.pnl_at_expiry(100.0)


def test_net_greeks_add_legs_and_unknowns_stay_unknown():
    bcs = s.bull_call_spread(call(100), 3.0, call(105), 1.0)
    g = bcs.greeks([{"delta": 0.55, "gamma": 0.04, "theta": -0.05, "vega": 0.12, "rho": 0.03},
                    {"delta": 0.35, "gamma": 0.03, "theta": -0.04, "vega": 0.10, "rho": 0.02}])  # fmt: skip
    assert (
        g["delta"] == pytest.approx(20.0)
        and g["vega"] == pytest.approx(2.0)
        and g["theta"] == pytest.approx(-1.0)
    )
    partial = bcs.greeks([{"delta": 0.55}, {"delta": 0.35}])
    assert partial["delta"] == pytest.approx(20.0) and partial["gamma"] is None
    cc = s.covered_call(100.0, call(105), 2.0)
    assert cc.greeks([None, {"delta": 0.3, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}])[
        "delta"
    ] == pytest.approx(70.0)


# --------------------------------------------------------------------------- invariants (property-based)
strikes = st.floats(min_value=5, max_value=500, allow_nan=False)
prices = st.floats(min_value=0.01, max_value=50, allow_nan=False)


@settings(max_examples=200, deadline=None)
@given(k=strikes, width=st.floats(min_value=0.5, max_value=50), a=prices, b=prices)
def test_a_defined_risk_vertical_can_never_lose_more_than_its_width(k, width, a, b):
    hi, lo = max(a, b), min(a, b)
    credit = s.bull_put_spread(put(round(k + width, 2)), hi, put(round(k, 2)), lo)
    debit = s.bull_call_spread(call(round(k, 2)), hi, call(round(k + width, 2)), lo)
    w = (round(k + width, 2) - round(k, 2)) * 100
    for spread in (credit, debit):
        assert math.isfinite(spread.max_loss()) and spread.max_loss() <= w + abs(spread.debit()) + 1e-6
        grid = np.linspace(0.01, (k + width) * 3, 200)
        assert (spread.pnl_at_expiry(grid) >= -spread.max_loss() - 1e-6).all()
        assert (spread.pnl_at_expiry(grid) <= spread.max_profit() + 1e-6).all()


@settings(max_examples=200, deadline=None)
@given(k=strikes, premium=prices, kind=st.sampled_from(["call", "put"]))
def test_a_long_option_never_loses_more_than_its_premium(k, premium, kind):
    c = OptionContract("AAPL", EXP, kind, round(k, 2))
    lo = s.long_call(c, premium) if kind == "call" else s.long_put(c, premium)
    assert lo.max_loss() == pytest.approx(premium * 100)
    grid = np.linspace(0.01, k * 4, 300)
    assert (lo.pnl_at_expiry(grid) >= -premium * 100 - 1e-6).all()


@settings(max_examples=100, deadline=None)
@given(
    k=strikes,
    w1=st.floats(min_value=1, max_value=20),
    extra=st.floats(min_value=0.5, max_value=20),
    credit=prices,
)
def test_widening_a_credit_spread_raises_its_max_loss_by_exactly_the_extra_width(k, w1, extra, credit):
    k, w1, extra = round(k, 2), round(w1, 2), round(extra, 2)
    assume(credit < w1)  # a credit above the width is an arbitrage no market offers
    narrow = s.bull_put_spread(put(k + w1), credit, put(k), 0.0)
    wide = s.bull_put_spread(put(k + w1), credit, put(round(k - extra, 2)), 0.0) if k - extra > 0 else None
    if wide is None:
        return
    assert wide.max_loss() - narrow.max_loss() == pytest.approx(extra * 100, abs=1e-4)


@settings(max_examples=100, deadline=None)
@given(days=st.integers(min_value=0, max_value=800))
def test_dte_is_never_negative_for_a_contract_still_trading(days):
    e = date(2026, 9, 25) + timedelta(days=days)
    c = call(100, e=e)
    if not c.expired(NOW):
        assert c.dte(NOW) >= 0 and c.years(NOW) >= 0


@settings(max_examples=200, deadline=None)
@given(bid=st.floats(min_value=0.01, max_value=20), ask=st.floats(min_value=0.01, max_value=20))
def test_a_quote_with_the_bid_above_the_ask_is_never_usable(bid, ask):
    v = validate(quote(call(100), bid=bid, ask=ask), NOW)
    if bid > ask:
        assert not v.usable_for_research and not v.usable_for_execution


@settings(max_examples=50, deadline=None)
@given(deltas=st.lists(st.floats(min_value=-150, max_value=150), min_size=1, max_size=8),
       vegas=st.lists(st.floats(min_value=-50, max_value=50), min_size=8, max_size=8))  # fmt: skip
def test_portfolio_greeks_are_the_sum_of_the_positions(deltas, vegas):
    positions = [
        book.BookPosition(f"p{i}", "AAPL", "long_call", "x", "bullish", EXP.isoformat(), 21,
                          {"delta": d, "gamma": 0.0, "theta": -1.0, "vega": vegas[i], "rho": 0.0}, 100.0, 100.0, 50.0, d * 100)
        for i, d in enumerate(deltas)
    ]  # fmt: skip
    rep = book.aggregate(positions, 100_000)
    assert rep.totals["delta"] == pytest.approx(sum(deltas)) and rep.totals["vega"] == pytest.approx(
        sum(vegas[: len(deltas)])
    )
    assert rep.max_loss == pytest.approx(100.0 * len(deltas))


# --------------------------------------------------------------------------- analytics
def test_iv_rank_and_percentile_need_history():
    hist = [0.2 + 0.01 * i for i in range(30)]  # 0.20 … 0.49
    standing = an.iv_rank(0.35, hist)
    assert standing.rank == pytest.approx((0.35 - 0.2) / 0.29 * 100) and standing.percentile == pytest.approx(
        50.0
    )
    assert an.iv_rank(0.35, hist[:5]).rank is None and an.iv_rank(0.35, hist[:5]).days == 5
    assert an.iv_rank(0.6, hist).rank == 100.0 and an.iv_rank(0.1, hist).percentile == 0.0


def test_realized_volatility_windows_and_iv_rv():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    rv = an.realized_vols(closes)
    assert set(rv) == set(an.RV_WINDOWS) and 0.10 < rv[252] < 0.22  # 1% a day ≈ 16% a year
    assert an.realized_vol(closes[:5], 20) is None
    assert an.iv_rv(0.3, 0.2) == {"spread": pytest.approx(0.1), "ratio": pytest.approx(1.5)}


def test_term_structure_skew_and_implied_move():
    spot = 100.0
    quotes = []
    for e, iv in ((date(2026, 10, 16), 0.40), (date(2026, 11, 20), 0.30), (date(2026, 12, 18), 0.28)):
        for k in (90, 95, 100, 105, 110):
            for kind in ("call", "put"):
                c = OptionContract("AAPL", e, kind, k)
                skew = 0.05 if kind == "put" and k < 100 else 0.0
                v = px.greeks(kind, spot, k, c.years(NOW), iv + skew)
                quotes.append(OptionQuote(c, v.price * 0.98, v.price * 1.02, NOW, "opra", "t", iv=iv + skew,
                                          greeks=Greeks(delta=v.delta), underlying_price=spot, underlying_at=NOW))  # fmt: skip
    exps = an.by_expiry(quotes, spot, NOW)
    ts = an.term_structure(exps)
    assert ts["shape"] == "backwardation" and ts["slope_per_30d"] < 0
    front = exps[0]
    assert front.atm_iv == pytest.approx(0.40) and front.implied_move_pct and front.implied_move_pct > 0
    assert front.skew is not None and front.skew > 0  # puts carry the premium
    cm = an.constant_maturity_iv(exps, 30)
    assert 0.28 < cm < 0.40


def test_empirical_probabilities_count_what_happened():
    closes = list(100 * 1.005 ** np.arange(300))  # a steady rise: +2.5% every five days
    up = an.empirical_move_probability(closes, 5, 0.01)
    assert up["probability"] == 1.0 and up["effective_n"] == up["n"] // 5
    down = an.empirical_move_probability(closes, 5, -0.01)
    assert down["probability"] == 0.0
    touch = an.empirical_touch_probability(closes, closes, closes, 5, 0.01)
    assert touch["probability"] == 1.0


# --------------------------------------------------------------------------- liquidity
def test_liquidity_rejects_thin_contracts_and_prices_the_round_trip():
    good = liq.assess(quote(call(100), bid=2.0, ask=2.1, volume=500, open_interest=5000))
    assert good.ok and good.round_trip_cost == pytest.approx(10.0) and good.score > 0.5
    thin = liq.assess(quote(call(100), bid=1.0, ask=1.6, volume=2, open_interest=10))
    assert not thin.ok and len(thin.reasons) >= 3
    assert liq.collapsed([quote(call(100), bid=2.0, ask=2.1)], [quote(call(100), bid=1.8, ask=2.3)]) == [
        call(100).symbol
    ]


# --------------------------------------------------------------------------- the book
def test_three_bullish_tech_calls_are_one_bet():
    ps = [
        book.BookPosition(f"{u}-c", u, "long_call", "momentum", "bullish", EXP.isoformat(), 21,
                          {"delta": 50, "gamma": 2, "theta": -5, "vega": 10, "rho": 1}, 100.0, 300, 300, 5000,
                          sector="Information Technology")
        for u in ("AAPL", "MSFT", "NVDA")
    ]  # fmt: skip
    rep = book.aggregate(ps, 100_000)
    assert rep.totals["delta"] == 150 and rep.dollar_delta == 15000
    assert any("one bet" in w for w in rep.warnings)
    assert rep.concentration["sector"]["hhi"] == 1.0
    assert book.dte_bucket(0) == "0DTE" and book.dte_bucket(30) == "22-45" and book.herfindahl([1, 1]) == 0.5


# --------------------------------------------------------------------------- attribution
def test_attribution_splits_the_change_by_greek():
    start = att.Mark(spot=100, iv=0.30, value=500, days=0, delta=50, gamma=2, theta=-10, vega=20)
    end = att.Mark(spot=102, iv=0.28, value=554, days=1, delta=60, gamma=2, theta=-10, vega=20)
    a = att.attribute(start, end, execution=-5, fees=-1)
    assert a.delta == 100 and a.gamma == 4 and a.theta == -10 and a.vega == pytest.approx(-40)
    assert a.residual == pytest.approx(54 - 54) and a.total == pytest.approx(48)
    v = att.verdicts(a, direction="bullish")
    assert v["direction_correct"] and not v["volatility_helped"] and v["dominant_driver"] == "delta"
    path = att.attribute_path([start, end, att.Mark(101, 0.29, 530, 2, 55, 2, -10, 20)])
    assert path.underlying_move == 1 and path.iv_change == pytest.approx(-1.0)
    assert att.execution_cost(1, 2.05, 2.0, 100) == pytest.approx(-5.0)


# --------------------------------------------------------------------------- expiration
def test_the_expiration_state_machine():
    c = call(100)
    assert exp.assess([c], 100, NOW).state is exp.ExpiryState.OPEN
    near = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    assert exp.assess([c], 120, near).state is exp.ExpiryState.NEAR_EXPIRATION
    risk = datetime(2026, 10, 14, 14, 0, tzinfo=UTC)
    a = exp.assess([c], 100.5, risk)
    assert a.state is exp.ExpiryState.EXPIRATION_RISK and a.must_close
    today = datetime(2026, 10, 16, 14, 0, tzinfo=UTC)
    assert exp.assess([c], 100, today).state is exp.ExpiryState.EXPIRING_TODAY
    after = datetime(2026, 10, 16, 21, 0, tzinfo=UTC)
    assert exp.assess([c], 100, after).state is exp.ExpiryState.EXPIRED
    assert exp.transition(exp.ExpiryState.OPEN, exp.ExpiryState.CLOSED) is exp.ExpiryState.CLOSED
    with pytest.raises(exp.TransitionError):
        exp.transition(exp.ExpiryState.CLOSED, exp.ExpiryState.OPEN)
    with pytest.raises(exp.TransitionError):
        exp.transition(exp.ExpiryState.EXPIRATION_RISK, exp.ExpiryState.OPEN)


def test_settlement_and_assignment_are_modelled_not_assumed():
    assert exp.settlement(call(100), "long", 2, 105)["state"] == "EXERCISED"
    assert exp.settlement(call(100), "long", 2, 105)["share_delivery"] == 200
    short = exp.settlement(put(100), "short", 1, 90)
    assert short["state"] == "ASSIGNED" and short["share_delivery"] == 100 and short["cash_flow"] == -10000
    assert exp.settlement(call(100), "long", 1, 99.995)["state"] == "EXPIRED"
    risky = exp.assignment_risk(call(100), 110, 10.05, NOW, ex_dividend=date(2026, 10, 1), dividend=0.5)
    assert risky["level"] == "high"
    assert exp.assignment_risk(call(120), 110, 0.5, NOW)["level"] == "none"
    assert exp.assignment_risk(put(120), 100, 20.01, NOW)["level"] == "high"
