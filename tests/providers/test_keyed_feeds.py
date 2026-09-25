"""FMP, EIA and The Odds API adapters."""

from datetime import date

import httpx
import pytest
import respx

from quantpulse.core.errors import DomainError, ProviderNoData, ProviderNotConfigured
from quantpulse.providers.eia import EIA
from quantpulse.providers.fmp import FinancialModelingPrep
from quantpulse.providers.odds_api import OddsAPI


@respx.mock
@pytest.mark.parametrize(
    "row",
    [
        {
            "date": "2027-09-30",
            "revenueAvg": 4.5e11,
            "revenueLow": 4.3e11,
            "revenueHigh": 4.7e11,
            "epsAvg": 8.5,
            "numAnalystsRevenue": 20,
        },
        {
            "date": "2027-09-30",
            "estimatedRevenueAvg": 4.5e11,
            "estimatedEpsAvg": 8.5,
            "numberAnalystEstimatedRevenue": 20,
        },
    ],
)
async def test_fmp_accepts_stable_and_legacy_field_names(http, row):
    respx.get("https://financialmodelingprep.com/stable/analyst-estimates").mock(
        return_value=httpx.Response(200, json=[row])
    )
    respx.get("https://financialmodelingprep.com/stable/price-target-consensus").mock(
        return_value=httpx.Response(
            200, json=[{"symbol": "AAPL", "targetConsensus": 265.0, "targetHigh": 300, "targetLow": 200}]
        )
    )
    est = await FinancialModelingPrep(http, "KEY").estimates("AAPL")
    assert est.periods[0].revenue_avg == 4.5e11 and est.periods[0].end_date == date(2027, 9, 30)
    assert est.periods[0].analysts == 20 and est.target_mean_price == 265.0


async def test_fmp_not_configured(http):
    with pytest.raises(ProviderNotConfigured):
        await FinancialModelingPrep(http, None).estimates("AAPL")


@respx.mock
async def test_eia_weekly_prices(http):
    route = respx.get("https://api.eia.gov/v2/petroleum/pri/gnd/data/").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": {
                    "total": 3,
                    "data": [
                        {
                            "period": "2026-09-21",
                            "duoarea": "SFL",
                            "area-name": "FLORIDA",
                            "product": "EPMR",
                            "series": "EMM_EPMR_PTE_SFL_DPG",
                            "value": "3.102",
                            "units": "$/GAL",
                        },
                        {
                            "period": "2026-09-14",
                            "duoarea": "SFL",
                            "area-name": "FLORIDA",
                            "product": "EPMR",
                            "series": "EMM_EPMR_PTE_SFL_DPG",
                            "value": 3.15,
                            "units": "$/GAL",
                        },
                        {
                            "period": "2026-09-07",
                            "duoarea": "SFL",
                            "area-name": "FLORIDA",
                            "product": "EPMR",
                            "series": "EMM_EPMR_PTE_SFL_DPG",
                            "value": None,
                            "units": "$/GAL",
                        },
                    ],
                },
                "apiVersion": "2.1.8",
            },
        )
    )
    series = await EIA(http, "KEY").fuel_prices("sfl", "regular")
    assert series.latest.price == pytest.approx(3.102) and series.latest.period == date(2026, 9, 21)
    assert [p.period for p in series.history] == [date(2026, 9, 14), date(2026, 9, 21)]
    params = route.calls[0].request.url.params
    assert (
        params["facets[duoarea][]"] == "SFL"
        and params["facets[product][]"] == "EPMR"
        and params["api_key"] == "KEY"
    )


@respx.mock
async def test_eia_validation_and_empty(http):
    with pytest.raises(DomainError):
        await EIA(http, "KEY").fuel_prices("MARS", "regular")
    respx.get("https://api.eia.gov/v2/petroleum/pri/gnd/data/").mock(
        return_value=httpx.Response(200, json={"response": {"data": []}})
    )
    with pytest.raises(ProviderNoData):
        await EIA(http, "KEY").fuel_prices("NUS", "regular")
    with pytest.raises(ProviderNotConfigured):
        await EIA(http, None).fuel_prices("NUS", "regular")


@respx.mock
async def test_odds_api_consensus_and_quota_headers(http):
    def book(key, spread, home_ml, away_ml):
        return {
            "key": key,
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": home_ml},
                        {"name": "Los Angeles Chargers", "price": away_ml},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": -110, "point": spread},
                        {"name": "Los Angeles Chargers", "price": -110, "point": -spread},
                    ],
                },
            ],
        }

    respx.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds").mock(
        return_value=httpx.Response(
            200,
            headers={"x-requests-remaining": "480", "x-requests-used": "20"},
            json=[
                {
                    "id": "e1",
                    "commence_time": "2026-09-27T17:00:00Z",
                    "home_team": "Buffalo Bills",
                    "away_team": "Los Angeles Chargers",
                    "bookmakers": [
                        book("dk", -7.0, -320, 260),
                        book("fd", -6.5, -300, 245),
                        book("mgm", -7.5, -340, 270),
                    ],
                }
            ],
        )
    )
    odds = OddsAPI(http, "KEY")
    lines = await odds.lines("nfl")
    _, line = lines[("buffalobills", "losangeleschargers")]
    assert line.spread_home == -7.0 and line.bookmakers == 3
    assert 0.7 < line.home_implied_prob < 0.8
    assert odds.requests_remaining == "480"
