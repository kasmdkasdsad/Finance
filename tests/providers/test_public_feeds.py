"""Treasury, SEC EDGAR, fueleconomy.gov and ESPN parsers, validated against real captured payloads."""

from datetime import date

import httpx
import pytest
import respx

from quantpulse.core.errors import ProviderNoData, ProviderParseError
from quantpulse.providers.espn import ESPN, parse_scoreboard
from quantpulse.providers.fueleconomy import FuelEconomyGov
from quantpulse.providers.sec_edgar import SecEdgar, build_statements
from quantpulse.providers.treasury import Treasury, parse_curve_csv

from .conftest import load_json, load_text


# ----------------------------------------------------------------------------- Treasury
def test_treasury_csv_parses_latest_curve():
    curve = parse_curve_csv(load_text("treasury_202609.csv"))
    assert curve.as_of == date(2026, 9, 24)
    assert len(curve.points) == 14
    by_tenor = {p.tenor: p for p in curve.points}
    assert by_tenor["1 Mo"].rate == pytest.approx(0.0401)
    assert by_tenor["1.5 Month"].years == pytest.approx(0.125)
    assert by_tenor["10 Yr"].rate == pytest.approx(0.0518)
    assert [p.years for p in curve.points] == sorted(p.years for p in curve.points)


def test_treasury_csv_handles_blank_cells_and_rejects_bad_header():
    text = 'Date,"1 Mo","2 Mo"\n09/24/2026,4.01,\n09/23/2026,3.99,4.10\n'
    curve = parse_curve_csv(text)
    assert curve.as_of == date(2026, 9, 24) and len(curve.points) == 1
    with pytest.raises(ProviderParseError):
        parse_curve_csv("Foo,Bar\n1,2\n")
    with pytest.raises(ProviderNoData):
        parse_curve_csv('Date,"1 Mo"\n')


@respx.mock
async def test_treasury_falls_back_to_previous_month_early_in_month(http):
    route = respx.get(host="home.treasury.gov").mock(
        side_effect=[
            httpx.Response(200, text='Date,"1 Mo"\n'),
            httpx.Response(200, text=load_text("treasury_202609.csv")),
        ]
    )
    curve = await Treasury(http).curve(today=date(2026, 10, 1))
    assert curve.as_of == date(2026, 9, 24)
    urls = [str(c.request.url) for c in route.calls]
    assert "/all/202610" in urls[0] and "/all/202609" in urls[1]


# ----------------------------------------------------------------------------- SEC EDGAR
def test_sec_statements_use_primary_period_fiscal_years():
    name, statements, shares = build_statements(load_json("sec_companyfacts_aapl.json"))
    assert name == "Apple Inc."
    by_fy = {s.fiscal_year: s for s in statements}
    # Values as reported in Apple's 10-K filings.
    assert by_fy[2024].period_end == date(2024, 9, 28)
    assert by_fy[2024].revenue == 391_035_000_000
    assert by_fy[2024].net_income == 93_736_000_000
    assert by_fy[2024].diluted_eps == pytest.approx(6.08)
    assert by_fy[2023].revenue == 383_285_000_000
    assert by_fy[2025].revenue == 416_161_000_000
    # Total debt = LongTermDebt (incl. current portion) + commercial paper
    assert by_fy[2025].total_debt == 90_678_000_000 + 7_979_000_000
    assert by_fy[2025].capital_expenditure > 0
    assert shares == (14_594_180_000, date(2026, 7, 17))


def test_sec_multi_class_issuer_uses_balance_sheet_share_count():
    name, statements, shares = build_statements(load_json("sec_companyfacts_googl.json"))
    assert name == "Alphabet Inc."
    by_fy = {s.fiscal_year: s for s in statements}
    assert by_fy[2023].revenue == 307_394_000_000
    assert by_fy[2023].diluted_eps == pytest.approx(5.80)
    assert shares is not None and shares[0] == 12_230_000_000


def test_sec_rejects_payload_without_us_gaap():
    with pytest.raises(ProviderNoData):
        build_statements({"cik": 1, "entityName": "X", "facts": {"ifrs-full": {}}})
    with pytest.raises(ProviderParseError):
        build_statements({"entityName": "missing cik"})


@respx.mock
async def test_sec_end_to_end_with_user_agent(http):
    tickers = respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=load_json("sec_company_tickers.json"))
    )
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=load_json("sec_companyfacts_aapl.json"))
    )
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=load_json("sec_submissions_aapl.json"))
    )
    sec = SecEdgar(http, "Test Suite tests@example.com")
    data = await sec.fundamentals("AAPL")
    assert data.cik == "0000320193" and data.statements[-1].fiscal_year == 2025
    assert data.recent_filings and data.recent_filings[0].url.startswith(
        "https://www.sec.gov/Archives/edgar/data/320193/"
    )
    assert tickers.calls[0].request.headers["User-Agent"] == "Test Suite tests@example.com"
    assert await sec.resolve_cik("BRK.B") == ("0001067983", "BERKSHIRE HATHAWAY INC")
    with pytest.raises(ProviderNoData):
        await sec.resolve_cik("ZZZZ")
    assert tickers.call_count == 1  # ticker map cached


# ----------------------------------------------------------------------------- fueleconomy.gov
@respx.mock
async def test_fueleconomy_limited_trim_ratings(http):
    respx.get("https://www.fueleconomy.gov/ws/rest/vehicle/48019").mock(
        return_value=httpx.Response(200, json=load_json("fueleconomy_vehicle_48019.json"))
    )
    rating = await FuelEconomyGov(http).epa_rating(48019)
    assert (rating.city_mpg, rating.highway_mpg, rating.combined_mpg) == (30, 39, 34)
    assert "2025 Hyundai Elantra" in rating.source


# ----------------------------------------------------------------------------- ESPN
def test_espn_live_scoreboard_parsing():
    games = {g.event_id: g for g in parse_scoreboard("nfl", load_json("espn_nfl_scoreboard.json"))}
    live = games["401872948"]  # ATL at GB, 4th quarter
    assert live.state == "in" and live.period == 4 and live.display_clock == "14:17"
    assert (live.home.team.abbreviation, live.home.score, live.away.score) == ("GB", 7, 24)
    assert live.espn_home_win_prob is None or 0 <= live.espn_home_win_prob <= 1
    buf = games["401872953"]  # "BUF -7": home favourite
    assert buf.state == "pre" and buf.home.score is None
    assert buf.market.spread_home == -7.0
    cle = games["401872949"]  # "CAR -2.5": away favourite -> home +2.5
    assert cle.market.spread_home == 2.5
    assert cle.market.home_implied_prob is not None and cle.market.home_implied_prob < 0.5


def test_espn_completed_week_and_neutral_site():
    games = {g.event_id: g for g in parse_scoreboard("nfl", load_json("espn_nfl_week1_2026.json"))}
    sea = games["401872656"]
    assert sea.completed and sea.state == "post" and (sea.home.score, sea.away.score) == (13, 10)
    assert games["401872657"].neutral_site is True


def test_espn_college_fixture_and_rankings():
    games = parse_scoreboard("college-football", load_json("espn_cfb_scoreboard.json"))
    rut = next(g for g in games if g.home.team.abbreviation == "RUTG")
    assert rut.market.spread_home == -43.5
    assert all(g.league == "college-football" for g in games)


@respx.mock
async def test_espn_scoreboard_request_params_and_fbs_list(http):
    route = respx.get(host="site.api.espn.com").mock(
        return_value=httpx.Response(200, json=load_json("espn_cfb_scoreboard.json"))
    )
    espn = ESPN(http)
    games, meta = await espn.scoreboard("college-football", season=2026, season_type=2, week=4)
    params = route.calls[0].request.url.params
    assert params["groups"] == "80" and params["week"] == "4" and params["seasontype"] == "2"
    assert games and meta["season"] == 2026
    refs = {
        "items": [
            {"$ref": f"http://sports.core.api.espn.com/v2/x/teams/{i}?lang=en"} for i in range(100, 240)
        ]
    }
    respx.get(host="sports.core.api.espn.com").mock(return_value=httpx.Response(200, json=refs))
    ids = await espn.fbs_team_ids(2026)
    assert "164" in ids and len(ids) == 140
    respx.get(host="sports.core.api.espn.com").mock(return_value=httpx.Response(200, json={"items": []}))
    with pytest.raises(ProviderNoData):
        await espn.fbs_team_ids(2026)


@respx.mock
async def test_espn_cdn_block_is_a_provider_error(http):
    respx.get(host="site.api.espn.com").mock(
        return_value=httpx.Response(403, text="<HTML>Access Denied</HTML>")
    )
    from quantpulse.core.errors import ProviderHTTPError

    with pytest.raises(ProviderHTTPError):
        await ESPN(http).scoreboard("nfl")
