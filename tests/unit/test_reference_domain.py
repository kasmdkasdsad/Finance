from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantpulse.domain import earnings as earn
from quantpulse.domain import fundamental_factors as ff
from quantpulse.domain.sectors import FF12_NAMES, ff12
from quantpulse.domain.universe import Constituent, IndexChange, Membership, build_intervals
from quantpulse.providers import sp500


def _c(symbol: str) -> Constituent:
    return Constituent(symbol, symbol, "Industrials", None, None, None)


def _ch(d: str, added: str | None, removed: str | None) -> IndexChange:
    return IndexChange(date.fromisoformat(d), added, None, removed, None, None)


# ----------------------------------------------------------------------------- membership
def test_membership_replays_changes_backwards():
    current = ["A", "B", "NEW"]
    changes = [
        _ch("2020-01-10", "B", "OLD1"),  # B joined, OLD1 left
        _ch("2022-06-01", "NEW", "OLD2"),
        _ch("2023-03-01", "X", "X"),  # merger that kept the ticker: no-op
        _ch("2019-05-01", "OLD2", None),
    ]
    m = Membership([_c(s) for s in current], changes)
    assert m.members_on(date(2019, 1, 1)) == {"A", "OLD1"}
    assert m.members_on(date(2019, 6, 1)) == {"A", "OLD1", "OLD2"}
    assert m.members_on(date(2020, 1, 10)) == {"A", "B", "OLD2"}
    assert m.members_on(date(2022, 6, 1)) == {"A", "B", "NEW"}
    assert m.ever_members(date(2021, 1, 1), date(2023, 1, 1)) == {"A", "B", "OLD2", "NEW"}
    mask = m.mask(pd.bdate_range("2022-05-30", "2022-06-02"), ["NEW", "OLD2", "CUSTOM"])
    assert mask["NEW"].tolist() == [False, False, True, True]
    assert mask["OLD2"].tolist() == [True, True, False, False]
    assert mask["CUSTOM"].all()  # unknown tickers are always eligible


def test_ticker_that_leaves_and_returns():
    iv = build_intervals(["R"], [_ch("2015-01-01", None, "R"), _ch("2020-01-01", "R", None)])
    assert [(i.start, i.end) for i in iv["R"]] == [(None, date(2015, 1, 1)), (date(2020, 1, 1), None)]


def test_packaged_snapshot_is_consistent():
    cons, changes = sp500.packaged_snapshot()
    m = Membership(cons, changes)
    assert len(cons) >= 495 and all(c.cik and c.sector for c in cons)
    for year in range(2014, 2026):
        assert 495 <= len(m.members_on(date(year, 6, 30))) <= 510, year
    assert "BRK-B" in m.current and all("." not in s for s in m.current)


def test_wikipedia_table_parser_handles_spans():
    html = """<table id="changes"><tr><th>Date</th><th colspan=2>Added</th><th colspan=2>Removed</th>
    <th>Reason</th></tr>
    <tr><td rowspan=2>March 4, 2024</td><td>AAA</td><td>Alpha</td><td>ZZZ[3]</td><td>Zed</td><td>Cap</td></tr>
    <tr><td>BRK.B</td><td>Berk</td><td></td><td></td><td>Spin-off</td></tr></table>"""
    parser = sp500._TableParser("changes")
    parser.feed(html)
    rows = parser.rows()
    assert rows[1][:4] == ["March 4, 2024", "AAA", "Alpha", "ZZZ"]
    assert rows[2][:2] == ["March 4, 2024", "BRK.B"]
    assert sp500.normalise_ticker("BRK.B[a]") == "BRK-B"


# ----------------------------------------------------------------------------- sectors
def test_ff12_mapping():
    assert ff12("3571") == "BusEq"  # Apple: electronic computers
    assert ff12(6021) == "Money" and ff12("2834") == "Hlth" and ff12("1311") == "Enrgy"
    assert ff12("4911") == "Utils" and ff12("5331") == "Shops" and ff12(None) == "Other"
    assert set(FF12_NAMES) == {ff12(c) for c in range(0, 10000)} | {"Other"}


# ----------------------------------------------------------------------------- earnings
def _ny(d: str, hh: int, mm: int = 0) -> datetime:
    return datetime.fromisoformat(f"{d}T{hh:02d}:{mm:02d}:00").replace(tzinfo=earn.NEW_YORK).astimezone(UTC)


def test_reaction_day_timing():
    assert earn.reaction_day(_ny("2026-07-30", 16, 30)) == date(2026, 7, 31)  # after the close
    assert earn.reaction_day(_ny("2026-07-30", 7, 0)) == date(2026, 7, 30)  # before the open
    assert earn.reaction_day(_ny("2026-07-30", 12, 0)) == date(2026, 7, 30)  # during the session
    assert earn.reaction_day(_ny("2026-09-26", 9, 0)) == date(2026, 9, 28)  # Saturday → Monday


def test_reactions_moves_and_next_estimate():
    days = pd.bdate_range("2025-01-01", "2026-09-25")
    closes = dict.fromkeys((d.date() for d in days), 100.0)
    bench = dict.fromkeys(closes, 50.0)
    events = [
        _ny("2025-02-03", 16, 30),
        _ny("2025-05-05", 16, 30),
        _ny("2025-08-04", 16, 30),
        _ny("2025-11-03", 16, 30),
    ]
    for e, move in zip(events, (0.05, -0.08, 0.02, -0.04), strict=True):
        d = earn.reaction_day(e)
        for k in closes:
            if k >= d:
                closes[k] *= 1 + move
    rs = earn.reactions(events, closes, bench)
    assert [round(r.stock_return, 6) for r in rs] == [0.05, -0.08, 0.02, -0.04]
    assert rs[0].abnormal == pytest.approx(0.05)
    assert earn.typical_move(rs) == pytest.approx(np.sqrt(np.mean(np.square([0.05, 0.08, 0.02, 0.04]))))
    nxt = earn.estimate_next(events, date(2025, 11, 10))
    assert nxt == date(2026, 2, 3)  # median ~91-day gap after the Nov 4 reaction
    assert earn.estimate_next(events, date(2026, 9, 1)) >= date(2026, 9, 1)  # rolled forward, never past
    assert earn.estimate_next(events[:1], date(2025, 3, 1)) is None


# ----------------------------------------------------------------------------- fundamentals
def test_point_in_time_lag_and_staleness():
    dates = pd.bdate_range("2023-01-02", "2025-12-31")
    s = ff.point_in_time({date(2023, 12, 31): 10.0}, ff.ANNUAL_LAG, dates)
    assert s[:"2024-03-29"].isna().all()  # not yet published (Dec 31 + 90 days)
    assert s["2024-04-01"] == 10.0
    assert s.loc["2025-10-15":].isna().all()  # older than MAX_AGE_DAYS since it became available


def test_market_value_is_split_immune():
    dates = pd.bdate_range("2023-01-02", "2025-06-30")
    close = pd.Series(np.linspace(100, 200, len(dates)), index=dates)  # split-adjusted prices
    float_facts = [ff.Fact(end=date(2023, 6, 30), value=1_000_000.0)]
    mv = ff.market_value(float_facts, close, dates)
    d0 = close.index[close.index.searchsorted(pd.Timestamp("2023-06-30"), side="right") - 1]
    t = pd.Timestamp("2024-06-03")
    assert mv[t] == pytest.approx(1_000_000.0 * close[t] / close[d0])
    assert mv[:"2024-03-25"].isna().all()  # float report not yet available (270-day lag)


def test_company_factors_definitions():
    dates = pd.bdate_range("2022-01-03", "2025-06-30")
    close = pd.Series(50.0, index=dates)
    facts = ff.CompanyFacts(
        net_income=[
            ff.Fact(date(2023, 12, 31), 80.0, date(2023, 1, 1)),
            ff.Fact(date(2023, 9, 30), 5.0, date(2023, 7, 1)),
        ],
        operating_cash_flow=[ff.Fact(date(2023, 12, 31), 100.0, date(2023, 1, 1))],
        capex=[ff.Fact(date(2023, 12, 31), 30.0, date(2023, 1, 1))],
        gross_profit=[ff.Fact(date(2023, 12, 31), 400.0, date(2023, 1, 1))],
        assets=[ff.Fact(date(2022, 12, 31), 1600.0), ff.Fact(date(2023, 12, 31), 2000.0)],
        equity=[ff.Fact(date(2023, 12, 31), 500.0)],
        public_float=[ff.Fact(date(2023, 6, 30), 1000.0)],
    )
    f = ff.company_factors(facts, close, pd.DatetimeIndex(dates))
    t = pd.Timestamp("2024-04-15")
    assert f["earnings_yield"][t] == pytest.approx(80 / 1000)  # the quarterly 5.0 is ignored
    assert f["fcf_yield"][t] == pytest.approx(70 / 1000)
    assert f["book_to_market"][t] == pytest.approx(500 / 1000)
    assert f["gross_profitability"][t] == pytest.approx(0.2)
    assert f["roe"][t] == pytest.approx(80 / 500)
    assert f["asset_growth"][t] == pytest.approx(0.25)
    assert f["accruals"][t] == pytest.approx(-20 / 2000)
    panel = ff.fundamental_features({"X": facts}, pd.DataFrame({"X": close, "Y": close}))
    assert panel["roe"]["Y"].isna().all() and panel["roe"]["X"][t] == pytest.approx(0.16)


# ----------------------------------------------------------------------------- settings
def test_model_universe_setting_is_validated():
    from pydantic import ValidationError

    from quantpulse.config import Settings

    assert Settings(_env_file=None, model_universe="SP500").model_universe == "sp500"
    assert (
        Settings(_env_file=None, model_universe=" aapl, msft,nvda,aapl ").model_universe == "AAPL,MSFT,NVDA"
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_universe="AAPL,MSFT")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_type="forest")


def test_synthetic_fundamentals_do_not_see_the_future():
    """Simulated market value must not encode where the simulated price path ends (a look-ahead leak that
    would make value factors 'predict' returns on demo data)."""
    from quantpulse.domain import features as feat
    from quantpulse.providers import synthetic
    from quantpulse.services.market import history_frame

    now = datetime(2026, 9, 25, 20, 30, tzinfo=UTC)
    syms = [f"LEAK{i:02d}" for i in range(40)]
    close = pd.DataFrame(
        {
            s: history_frame(synthetic.synthetic_history(s, "1d", now - timedelta(days=1500), now, now))[
                "close"
            ]
            for s in syms
        }
    )
    factors = ff.fundamental_features({s: synthetic.synthetic_company_facts(s, now) for s in syms}, close)
    fwd = feat.forward_returns(close, 21)
    for name in ("earnings_yield", "book_to_market", "fcf_yield"):
        ic = feat.row_spearman(factors[name], fwd).dropna()
        assert len(ic) > 300 and abs(ic.mean()) < 0.08, (name, ic.mean())
