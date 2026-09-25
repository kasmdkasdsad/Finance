"""Valuation, portfolio, vehicle, sports, picks/email and poller behaviour through the API."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from quantpulse.schemas.fundamentals import (
    AnalystEstimates,
    AnalystPeriodEstimate,
    CompanyFundamentals,
    FinancialStatement,
)
from quantpulse.services.valuation import consensus_growth

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def fx(name):
    return json.loads((FIXTURES / name).read_text())


# ----------------------------------------------------------------------------- valuation
def test_consensus_growth_aligns_annual_estimates_after_last_fiscal_year():
    fund = CompanyFundamentals(
        symbol="X",
        statements=[FinancialStatement(fiscal_year=2025, period_end=date(2025, 9, 27), revenue=100.0)],
    )
    est = AnalystEstimates(
        symbol="X",
        periods=[
            AnalystPeriodEstimate(
                period="0q", end_date=date(2025, 12, 31), revenue_avg=30.0
            ),  # quarterly: ignored
            AnalystPeriodEstimate(
                period="-1y", end_date=date(2024, 9, 28), revenue_avg=90.0
            ),  # already reported
            AnalystPeriodEstimate(period="0y", end_date=date(2026, 9, 30), revenue_avg=110.0),
            AnalystPeriodEstimate(period="+1y", end_date=date(2027, 9, 30), revenue_avg=121.0),
        ],
    )
    assert consensus_growth(fund, est) == pytest.approx([0.10, 0.10])
    misaligned = AnalystEstimates(
        symbol="X",
        periods=[AnalystPeriodEstimate(period="2027", end_date=date(2027, 12, 31), revenue_avg=150.0)],
    )
    assert consensus_growth(fund, misaligned) == []


async def test_dcf_offline_report_is_complete_and_flagged(api):
    r = await api.post(
        "/api/v1/valuation/MSFT/dcf", json={"years": 5, "monte_carlo": {"paths": 2000, "seed": 7}}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    d = body["data"]
    assert body["meta"]["status"] == "synthetic"
    assert any("synthetic" in w.lower() for w in d["warnings"])
    assert len(d["dcf"]["projections"]) == 5
    assert d["inputs"]["terminal_growth"] == 0.025 and d["inputs"]["wacc"] == d["wacc"]["wacc"]
    assert d["wacc"]["weight_equity"] + d["wacc"]["weight_debt"] == pytest.approx(1.0)
    assert d["monte_carlo"]["seed"] == 7 and d["monte_carlo"]["valid_paths"] == 2000
    assert {a["field"] for a in d["assumptions"]} >= {"revenue_growth_y1", "ebit_margin_start", "tax_rate"}
    grid = d["sensitivity"]["values_per_share"]
    assert len(grid) == 5 and len(grid[0]) == 5


async def test_dcf_overrides_and_domain_errors(api):
    ok = await api.post(
        "/api/v1/valuation/MSFT/dcf",
        json={
            "years": 3,
            "revenue_growth": [0.1, 0.08, 0.06],
            "wacc": 0.09,
            "beta": 1.1,
            "monte_carlo": None,
        },
    )
    d = ok.json()["data"]
    assert d["inputs"]["revenue_growth"] == [0.1, 0.08, 0.06] and d["inputs"]["wacc"] == 0.09
    assert d["monte_carlo"] is None and d["wacc"]["beta_source"] == "user override"
    too_close = await api.post("/api/v1/valuation/MSFT/dcf", json={"wacc": 0.03, "terminal_growth": 0.028})
    assert too_close.status_code == 422 and "terminal growth" in too_close.json()["detail"]
    mismatch = await api.post("/api/v1/valuation/MSFT/dcf", json={"years": 5, "revenue_growth": [0.1]})
    assert mismatch.status_code == 422


async def test_dcf_uses_live_sec_fundamentals(live_api, mock_net):
    mock_net.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=fx("sec_company_tickers.json"))
    )
    mock_net.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=fx("sec_companyfacts_aapl.json"))
    )
    mock_net.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=fx("sec_submissions_aapl.json"))
    )
    fund = (await live_api.get("/api/v1/fundamentals/AAPL")).json()
    assert fund["meta"]["status"] == "live" and fund["meta"]["provider"] == "sec_edgar"
    r = await live_api.post("/api/v1/valuation/AAPL/dcf", json={"monte_carlo": {"paths": 1000, "seed": 1}})
    d = r.json()["data"]
    assert r.json()["meta"]["sources"]["fundamentals"]["status"] == "cached"
    assert d["inputs"]["base_revenue"] == 416_161_000_000
    assert d["inputs"]["shares_outstanding"] == 14_594_180_000
    assert d["inputs"]["debt"] == 98_657_000_000


# ----------------------------------------------------------------------------- portfolio
async def test_portfolio_crud_and_risk(api):
    payload = {
        "name": "Core",
        "holdings": [
            {"symbol": "aapl", "quantity": 10, "cost_basis": 150},
            {"symbol": "MSFT", "quantity": 5},
            {"symbol": "XOM", "quantity": 20},
        ],
    }
    created = await api.post("/api/v1/portfolios", json=payload)
    assert created.status_code == 201
    pid = created.json()["id"]
    assert [h["symbol"] for h in created.json()["holdings"]] == ["AAPL", "MSFT", "XOM"]
    assert (await api.post("/api/v1/portfolios", json=payload)).status_code == 422  # duplicate name
    dup = {"name": "Dup", "holdings": [{"symbol": "AAPL", "quantity": 1}, {"symbol": "aapl", "quantity": 2}]}
    assert (await api.post("/api/v1/portfolios", json=dup)).status_code == 422

    risk = await api.post(
        f"/api/v1/portfolios/{pid}/risk",
        json={"lookback_days": 365, "confidence": 0.99, "seed": 3, "monte_carlo_paths": 5000},
    )
    assert risk.status_code == 200, risk.text
    rep = risk.json()["data"]
    assert sum(p["weight"] for p in rep["positions"]) == pytest.approx(1.0)
    assert sum(p["risk_contribution"] for p in rep["positions"]) == pytest.approx(1.0)
    methods = {v["method"]: v for v in rep["var"]}
    assert set(methods) == {"historical", "parametric", "cornish_fisher", "monte_carlo"}
    for v in methods.values():
        assert v["var_pct"] > 0 and v["var_amount"] == pytest.approx(v["var_pct"] * rep["portfolio_value"])
        if v["cvar_pct"] is not None:
            assert v["cvar_pct"] >= v["var_pct"]
    front = rep["frontier"]
    assert front["max_sharpe"]["sharpe"] >= front["current"]["sharpe"] - 1e-9
    assert front["min_variance"]["volatility"] <= front["equal_weight"]["volatility"] + 1e-9
    assert rep["metrics"]["max_drawdown"] <= 0 and rep["metrics"]["observations"] > 200
    assert rep["positions"][0]["unrealized_pnl"] == pytest.approx((rep["positions"][0]["price"] - 150) * 10)

    updated = await api.put(
        f"/api/v1/portfolios/{pid}", json={"name": "Core", "holdings": [{"symbol": "SPY", "quantity": 1}]}
    )
    assert updated.status_code == 200 and updated.json()["holdings"][0]["symbol"] == "SPY"
    single = (await api.post(f"/api/v1/portfolios/{pid}/risk", json={})).json()["data"]
    assert single["frontier"] is None and any("two holdings" in w for w in single["warnings"])
    assert (await api.delete(f"/api/v1/portfolios/{pid}")).status_code == 204
    assert (await api.get(f"/api/v1/portfolios/{pid}")).status_code == 404


async def test_adhoc_analysis_respects_max_weight(api):
    r = await api.post(
        "/api/v1/portfolio/analyze",
        json={
            "holdings": [{"symbol": s, "quantity": 10} for s in ("AAPL", "MSFT", "NVDA", "JPM")],
            "max_weight": 0.4,
        },
    )
    f = r.json()["data"]["frontier"]
    for point in f["points"] + [f["max_sharpe"], f["min_variance"]]:
        assert max(point["weights"].values()) <= 0.4 + 1e-6
    infeasible = await api.post(
        "/api/v1/portfolio/analyze",
        json={
            "holdings": [{"symbol": "AAPL", "quantity": 1}, {"symbol": "MSFT", "quantity": 1}],
            "max_weight": 0.3,
        },
    )
    assert any("frontier unavailable" in w.lower() for w in infeasible.json()["data"]["warnings"])


# ----------------------------------------------------------------------------- vehicle
async def test_vehicle_lifecycle(api):
    profile = (await api.get("/api/v1/vehicle/profiles/hyundai-elantra-2025-limited")).json()
    assert profile["epa"]["combined_mpg"] == 34 and profile["pricing"]["base_msrp"] == 26525
    epa = (await api.get("/api/v1/vehicle/profiles/hyundai-elantra-2025-limited/epa")).json()
    assert epa["meta"]["status"] == "stale" and epa["meta"]["provider"] == "packaged-profile"

    v = (
        await api.post(
            "/api/v1/vehicles",
            json={"nickname": "Daily", "purchase_date": "2025-06-01", "fuel_region": "sfl"},
        )
    ).json()
    vid = v["id"]
    assert v["fuel_region"] == "SFL"
    empty = (await api.get(f"/api/v1/vehicles/{vid}/dashboard")).json()["data"]
    assert empty["telemetry_source"] == "synthetic" and empty["realized_mpg"] is None
    assert empty["effective_mpg"] == pytest.approx(34.0, abs=0.01)  # 55% city mix reproduces EPA combined
    assert empty["cost_per_mile"]["total"] == pytest.approx(
        sum(empty["cost_per_mile"][k] for k in ("fuel", "depreciation", "maintenance"))
    )

    t0 = datetime(2026, 8, 1, 12, tzinfo=UTC)
    for i, odo in enumerate((12000.0, 12350.0, 12690.0)):
        r = await api.post(
            f"/api/v1/vehicles/{vid}/fuel-logs",
            json={
                "filled_at": (t0 + timedelta(days=10 * i)).isoformat(),
                "odometer": odo,
                "gallons": 10.5,
                "price_per_gallon": 3.19,
                "full_tank": True,
            },
        )
        assert r.status_code == 201
    tel = await api.post(
        f"/api/v1/vehicles/{vid}/telemetry", json={"recorded_at": "2026-09-20T12:00:00Z", "odometer": 13100}
    )
    assert tel.status_code == 201
    m = await api.post(
        f"/api/v1/vehicles/{vid}/maintenance",
        json={"service_code": "engine_oil", "performed_on": "2026-06-01", "odometer": 9000, "cost": 79.5},
    )
    assert m.status_code == 201
    bad = await api.post(
        f"/api/v1/vehicles/{vid}/maintenance",
        json={"service_code": "flux_capacitor", "performed_on": "2026-06-01", "odometer": 1},
    )
    assert bad.status_code == 422

    dash = (await api.get(f"/api/v1/vehicles/{vid}/dashboard")).json()
    d = dash["data"]
    assert d["telemetry_source"] == "recorded" and d["odometer"] == 13100
    assert d["realized_mpg"] == pytest.approx(690 / 21, abs=0.01)
    assert d["fuel_spend_to_date"] == pytest.approx(3 * 10.5 * 3.19, abs=0.01)
    oil = next(x for x in d["maintenance"] if x["code"] == "engine_oil")
    assert oil["last_service_odometer"] == 9000 and oil["next_due_odometer"] == 16500
    assert d["current_value"] < 27675 and d["depreciation_per_day"] > 0
    assert set(dash["meta"]["sources"]) == {"epa", "fuel_prices"}

    logs = (await api.get(f"/api/v1/vehicles/{vid}/fuel-logs")).json()
    assert (await api.delete(f"/api/v1/vehicles/{vid}/fuel-logs/{logs[0]['id']}")).status_code == 204
    assert (await api.delete(f"/api/v1/vehicles/{vid}/fuel-logs/{logs[0]['id']}")).status_code == 404
    assert (await api.get("/api/v1/vehicles/999/dashboard")).status_code == 404
    assert (
        await api.post(
            "/api/v1/vehicles", json={"nickname": "x", "purchase_date": "2025-01-01", "fuel_region": "MARS"}
        )
    ).status_code == 422
    assert (await api.get("/api/v1/fuel/prices", params={"region": "R1Z", "grade": "premium"})).json()[
        "data"
    ]["grade"] == "premium"


async def test_vehicle_live_epa_and_eia(tmp_path, clock, mock_net):
    from .conftest import _client, make_settings

    mock_net.get("https://www.fueleconomy.gov/ws/rest/vehicle/48019").mock(
        return_value=httpx.Response(200, json=fx("fueleconomy_vehicle_48019.json"))
    )
    mock_net.get("https://api.eia.gov/v2/petroleum/pri/gnd/data/").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": {
                    "data": [
                        {
                            "period": "2026-09-21",
                            "duoarea": "NUS",
                            "product": "EPMR",
                            "series": "EMM_EPMR_PTE_NUS_DPG",
                            "value": 3.123,
                            "units": "$/GAL",
                        },
                    ]
                }
            },
        )
    )
    async for c in _client(make_settings(tmp_path, enable_live_data=True, eia_api_key="k"), clock):
        epa = (await c.get("/api/v1/vehicle/profiles/hyundai-elantra-2025-limited/epa")).json()
        assert epa["meta"]["status"] == "live" and epa["data"]["combined_mpg"] == 34
        fuel = (await c.get("/api/v1/fuel/prices")).json()
        assert fuel["meta"]["status"] == "live" and fuel["data"]["latest"]["price"] == 3.123


# ----------------------------------------------------------------------------- sports
async def test_sports_live_scoreboard_and_ratings(live_api, mock_net):
    week1 = fx("espn_nfl_week1_2026.json")
    current = fx("espn_nfl_scoreboard.json")

    def espn(request):
        if "week" in request.url.params:
            return httpx.Response(200, json=week1)
        return httpx.Response(200, json=current)

    mock_net.get(host="site.api.espn.com").mock(side_effect=espn)
    board = (await live_api.get("/api/v1/sports/nfl/scoreboard")).json()
    assert board["meta"]["sources"]["scoreboard"]["status"] == "live"
    assert board["meta"]["sources"]["season_results"]["status"] == "live"
    games = {g["game"]["event_id"]: g for g in board["data"]["games"]}
    live = games["401872948"]["win_probability"]
    assert live["fraction_remaining"] == pytest.approx((14 * 60 + 17) / 3600, abs=1e-3)
    assert live["home"] < 0.05  # GB trails by 17 early in the 4th
    buf = games["401872953"]["win_probability"]
    assert buf["home"] == pytest.approx(buf["pregame_home"]) and buf["home"] > 0.5  # BUF -7 at home
    ratings = (await live_api.get("/api/v1/sports/nfl/ratings")).json()["data"]
    assert ratings["games_processed"] == 4  # the four completed week-1 fixture games (deduplicated)
    top = ratings["ratings"][0]
    assert top["wins"] >= 1 and top["rating"] > 1500


async def test_sports_espn_blocked_is_fully_synthetic(live_api, mock_net):
    mock_net.get(host="site.api.espn.com").mock(return_value=httpx.Response(403, text="Access Denied"))
    body = (await live_api.get("/api/v1/sports/college-football/scoreboard")).json()
    assert body["meta"]["status"] == "synthetic"
    assert all(g["game"]["event_id"].startswith("syn-") for g in body["data"]["games"])
    for g in body["data"]["games"]:
        wp = g["win_probability"]
        assert wp["home"] + wp["away"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- picks & email
async def test_daily_picks_ratings(api):
    body = (await api.get("/api/v1/picks/daily", params={"top_n": 30})).json()
    d = body["data"]
    assert d["universe_size"] == 30 and d["screened"] == 30 and len(d["picks"]) == 30
    ratings = [p["rating"] for p in d["picks"]]
    assert all(1 <= r <= 10 for r in ratings)
    assert ratings == sorted(ratings, reverse=True)
    assert d["top_pick"]["symbol"] == d["picks"][0]["symbol"] and d["picks"][0]["rank"] == 1
    assert max(ratings) >= 8 and min(ratings) <= 3  # the scale is actually used
    assert d["data_status"] == "synthetic" and "not investment advice" in d["disclaimer"]
    assert d["trading_day"] == "2026-09-25"  # 10:00 ET on a trading day -> today's session


async def test_picks_email_policies(tmp_path, clock, monkeypatch):
    from .conftest import _client, make_settings

    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            self.host, self.port = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            self.tls = True

        def login(self, user, password):
            self.user = user

        def send_message(self, msg):
            sent.append(msg)
            return {}

    monkeypatch.setattr("quantpulse.services.notifications.smtplib.SMTP", FakeSMTP)

    async for c in _client(make_settings(tmp_path), clock):
        not_configured = await c.post(
            "/api/v1/picks/email", json={"recipients": ["a@example.com"], "allow_synthetic": True}
        )
        assert not_configured.status_code == 503

    settings = make_settings(
        tmp_path,
        smtp_host="smtp.example.com",
        email_from="bot@example.com",
        smtp_username="u",
        smtp_password="p",
        picks_recipients="team@example.com",
    )
    async for c in _client(settings, clock):
        refused = await c.post("/api/v1/picks/email", json={})
        assert refused.status_code == 409 and refused.json()["error"] == "synthetic_data"
        assert not sent
        bad_addr = await c.post("/api/v1/picks/email", json={"recipients": ["not-an-email"]})
        assert bad_addr.status_code == 422
        ok = await c.post("/api/v1/picks/email", json={"allow_synthetic": True, "top_n": 5})
        assert ok.status_code == 200, ok.text
        result = ok.json()
        assert result["sent_to"] == ["team@example.com"] and result["subject"].startswith("[SYNTHETIC DATA]")
    msg = sent[0]
    assert msg["To"] == "team@example.com" and msg["From"] == "bot@example.com"
    text = msg.get_body(preferencelist=("plain",)).get_content()
    assert "SYNTHETIC" in text and "/10" in text and "not investment advice" in text
    assert msg.get_body(preferencelist=("html",)) is not None


async def test_poller_sends_picks_once_per_trading_day(tmp_path, monkeypatch):
    from quantpulse.core.clock import FakeClock

    from .conftest import _client, make_settings

    deliveries = []

    async def fake_send(self, recipients, subject, text, html):
        deliveries.append((recipients, subject))

    monkeypatch.setattr("quantpulse.services.notifications.EmailNotifier.send", fake_send)
    clock = FakeClock(datetime(2026, 9, 25, 12, 40, tzinfo=UTC))  # 08:40 ET, before 08:45 send time
    settings = make_settings(
        tmp_path,
        picks_email_enabled=True,
        smtp_host="smtp.example.com",
        email_from="bot@example.com",
        picks_recipients="team@example.com",
        picks_allow_synthetic_email=True,
    )
    async for c in _client(settings, clock):
        poller = c.container.poller
        assert await poller.maybe_send_picks() == "waiting"
        clock.advance(10 * 60)  # 08:50 ET
        assert (await poller.maybe_send_picks()).startswith("sent for 2026-09-25 to 1")
        assert (await poller.maybe_send_picks()) == "sent for 2026-09-25"
        assert len(deliveries) == 1
        clock.advance(24 * 3600)  # Saturday
        assert await poller.maybe_send_picks() == "waiting"

    strict = make_settings(
        tmp_path,
        picks_email_enabled=True,
        smtp_host="smtp.example.com",
        email_from="bot@example.com",
        picks_recipients="team@example.com",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'strict.db'}",
    )
    async for c in _client(strict, FakeClock(datetime(2026, 9, 28, 13, 0, tzinfo=UTC))):
        assert "synthetic data" in await c.container.poller.maybe_send_picks()
    assert len(deliveries) == 1  # synthetic picks are never mailed unless explicitly allowed
