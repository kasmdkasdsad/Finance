from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from quantpulse.core.errors import NotFoundError
from quantpulse.db import repositories as repo
from quantpulse.db.models import FuelLogRow, TelemetryRow
from quantpulse.schemas.fundamentals import AnalystEstimates, CompanyFundamentals, Filing, FinancialStatement
from quantpulse.schemas.market import Bar, PriceHistory, Quote
from quantpulse.schemas.options import OptionChain, OptionContract, YieldCurve, YieldPoint
from quantpulse.schemas.sports import Game, PowerRating, TeamRef, TeamScore

T0 = datetime(2026, 9, 1, 20, 0, tzinfo=UTC)


def _history(n, close=100.0):
    return PriceHistory(
        symbol="AAPL",
        interval="1d",
        bars=[
            Bar(
                timestamp=T0 + timedelta(days=i),
                open=close,
                high=close + i + 1,
                low=close - 1,
                close=close + i,
            )
            for i in range(n)
        ],
    )


async def test_bars_upsert_is_idempotent_and_timezone_safe(database):
    async with database.session() as s:
        assert await repo.upsert_bars(s, _history(3), "yahoo") == 3
    async with database.session() as s:
        await repo.upsert_bars(s, _history(5, close=200.0), "polygon")
    async with database.session() as s:
        history, last_ts, provider = await repo.load_bars(s, "AAPL", "1d")
    assert len(history.bars) == 5
    assert history.bars[0].close == 200.0 and provider == "polygon"
    assert last_ts == T0 + timedelta(days=4) and last_ts.tzinfo is not None
    async with database.session() as s:
        assert await repo.load_bars(s, "MSFT", "1d") is None


async def test_a_longer_download_backfills_older_bars(database):
    """Regression: a short window stored first must not stop a later, longer download from adding the
    older history (the revision-window shortcut used to drop every bar before it)."""
    full = _history(40)
    recent = full.model_copy(update={"bars": full.bars[25:]})
    async with database.session() as s:
        assert await repo.upsert_bars(s, recent, "alpaca") == 15
    async with database.session() as s:
        assert await repo.upsert_bars(s, full, "alpaca") == 25 + 8  # 25 older bars + the revision window
    async with database.session() as s:
        history, _, _ = await repo.load_bars(s, "AAPL", "1d")
    assert len(history.bars) == 40 and history.bars[0].timestamp == T0


async def test_bars_upsert_writes_only_new_bars_unless_history_was_readjusted(database):
    async with database.session() as s:
        assert await repo.upsert_bars(s, _history(30), "yahoo") == 30
    async with (
        database.session() as s
    ):  # same history plus two new days: only the revision window is rewritten
        assert await repo.upsert_bars(s, _history(32), "yahoo") == 10  # days 22..31 (7-day overlap + new)
    readjusted = _history(32)
    readjusted = readjusted.model_copy(
        update={
            "bars": [
                b.model_copy(update={"close": b.close * 0.98, "low": b.low * 0.97}) for b in readjusted.bars
            ]
        }
    )
    async with (
        database.session() as s
    ):  # a split/dividend re-adjustment changes old closes: rewrite everything
        assert await repo.upsert_bars(s, readjusted, "yahoo") == 32
    async with database.session() as s:
        history, _, _ = await repo.load_bars(s, "AAPL", "1d")
    assert len(history.bars) == 32 and history.bars[0].close == pytest.approx(98.0)


async def test_quote_and_curve_round_trip(database):
    q = Quote(symbol="AAPL", price=210.0, previous_close=200.0, timestamp=T0)
    curve = YieldCurve(
        as_of=date(2026, 9, 24),
        points=[
            YieldPoint(tenor="1 Yr", years=1, rate=0.045),
            YieldPoint(tenor="3 Mo", years=0.25, rate=0.042),
        ],
    )
    async with database.session() as s:
        await repo.insert_quote(s, q, "yahoo")
        await repo.upsert_curve(s, curve, "treasury")
        await repo.upsert_curve(s, curve, "treasury")  # upsert, no duplicate error
        await repo.record_ingestion(s, "quote", "AAPL", "yahoo", 1)
    async with database.session() as s:
        got, at, provider = await repo.latest_quote(s, "AAPL")
        c, _, cprov = await repo.latest_curve(s)
        events = await repo.recent_ingestions(s)
    assert got.change_percent == pytest.approx(5.0) and provider == "yahoo" and at == T0
    assert [p.tenor for p in c.points] == ["3 Mo", "1 Yr"] and cprov == "treasury"
    assert events[0].dataset == "quote"


async def test_option_snapshot_round_trip(database):
    chain = OptionChain(
        underlying="AAPL",
        underlying_price=210.0,
        as_of=T0,
        expirations=[date(2026, 10, 16), date(2026, 11, 20)],
        contracts=[
            OptionContract(
                contract_symbol="AAPL261016C00210000",
                kind="call",
                strike=210,
                expiration=date(2026, 10, 16),
                bid=5,
                ask=5.2,
            ),
            OptionContract(
                contract_symbol="AAPL261120P00200000",
                kind="put",
                strike=200,
                expiration=date(2026, 11, 20),
                bid=4,
                ask=4.3,
            ),
        ],
    )
    async with database.session() as s:
        assert await repo.insert_option_snapshot(s, chain, "yahoo") == 2
        await repo.insert_option_snapshot(s, chain, "yahoo")  # duplicate snapshot ignored
    async with database.session() as s:
        got, at, provider = await repo.latest_option_snapshot(s, "AAPL", [date(2026, 10, 16)])
    assert len(got.contracts) == 1 and got.expirations == chain.expirations and at == T0
    assert provider == "yahoo"


async def test_fundamentals_round_trip(database):
    data = CompanyFundamentals(
        symbol="AAPL",
        cik="0000320193",
        name="Apple Inc.",
        shares_outstanding=1.5e10,
        shares_as_of=date(2026, 7, 17),
        statements=[
            FinancialStatement(fiscal_year=2025, period_end=date(2025, 9, 27), revenue=416e9, total_debt=98e9)
        ],
        recent_filings=[
            Filing(form="10-K", filing_date=date(2025, 10, 31), accession="0000320193-25-000079")
        ],
    )
    async with database.session() as s:
        await repo.upsert_fundamentals(s, data, "sec_edgar")
        await repo.insert_estimates(s, AnalystEstimates(symbol="AAPL", target_mean_price=250.0), "yahoo")
    updated = data.model_copy(
        update={"statements": [data.statements[0].model_copy(update={"revenue": 420e9})]}
    )
    async with database.session() as s:
        await repo.upsert_fundamentals(s, updated, "sec_edgar")
    async with database.session() as s:
        got, _, provider = await repo.load_fundamentals(s, "AAPL")
        est, _, _ = await repo.latest_estimates(s, "AAPL")
    assert got.statements[0].revenue == 420e9 and got.recent_filings[0].form == "10-K"
    assert got.shares_outstanding == 1.5e10 and provider == "sec_edgar"
    assert est.target_mean_price == 250.0


async def test_portfolio_crud(database):
    async with database.session() as s:
        p = await repo.create_portfolio(s, "Core", [("AAPL", 10, 150.0), ("MSFT", 5, None)])
        pid = p.id
    async with database.session() as s:
        assert await repo.portfolio_name_exists(s, "core")
        await repo.replace_portfolio(s, pid, "Core 2", [("NVDA", 3, None)])
    async with database.session() as s:
        row = await repo.get_portfolio(s, pid)
        assert row.name == "Core 2" and [h.symbol for h in row.holdings] == ["NVDA"]
        await repo.delete_portfolio(s, pid)
    async with database.session() as s:
        with pytest.raises(NotFoundError):
            await repo.get_portfolio(s, pid)


async def test_portfolio_duplicate_symbol_rejected_by_constraint(database):
    with pytest.raises(IntegrityError):
        async with database.session() as s:
            await repo.create_portfolio(s, "Dup", [("AAPL", 1, None), ("AAPL", 2, None)])


async def test_vehicle_children_cascade(database):
    async with database.session() as s:
        v = await repo.create_vehicle(
            s, nickname="Daily", profile_id="hyundai-elantra-2025-limited", purchase_date=date(2025, 6, 1)
        )
        vid = v.id
        await repo.add_row(s, TelemetryRow(vehicle_id=vid, recorded_at=T0, odometer=1000.0, source="manual"))
        log = await repo.add_row(
            s,
            FuelLogRow(
                vehicle_id=vid, filled_at=T0, odometer=1000, gallons=10, price_per_gallon=3.2, full_tank=True
            ),
        )
        await repo.upsert_fuel_prices(
            s, "NUS", "U.S.", "regular", [(date(2026, 9, 21), 3.1, "EMM_EPMR_PTE_NUS_DPG")], "eia"
        )
    async with database.session() as s:
        assert len(await repo.telemetry(s, vid)) == 1
        await repo.delete_child(s, FuelLogRow, vid, log.id)
        with pytest.raises(NotFoundError):
            await repo.delete_child(s, FuelLogRow, vid, log.id)
        prices = await repo.fuel_price_history(s, "NUS", "regular")
        assert prices[0].price == 3.1
        await repo.delete_vehicle(s, vid)
    async with database.session() as s:
        assert await repo.telemetry(s, vid) == []  # ON DELETE CASCADE with foreign_keys=ON


async def test_games_and_ratings(database):
    team = lambda i: TeamRef(id=str(i), abbreviation=f"T{i}", name=f"Team {i}")
    game = Game(
        event_id="401",
        league="nfl",
        season=2026,
        season_type=2,
        week=1,
        start_time=T0,
        name="T2 at T1",
        state="post",
        completed=True,
        status_detail="Final",
        home=TeamScore(team=team(1), score=24),
        away=TeamScore(team=team(2), score=17),
    )
    async with database.session() as s:
        await repo.upsert_games(s, [game])
        await repo.upsert_games(s, [game.model_copy(update={"home": TeamScore(team=team(1), score=27)})])
        await repo.save_ratings(
            s,
            "nfl",
            2026,
            [
                PowerRating(
                    rank=1,
                    team=team(1),
                    rating=1520,
                    games=1,
                    wins=1,
                    losses=0,
                    ties=0,
                    points_for=27,
                    points_against=17,
                    avg_margin=10,
                    last_change=20,
                )
            ],
        )
    async with database.session() as s:
        games = await repo.completed_games(s, "nfl", [2026])
    assert len(games) == 1 and games[0].home_score == 27


async def test_reference_data_round_trips(database):
    from quantpulse.schemas.reference import CompanyEvents, CompanyProfile, FrameFact

    profile = CompanyProfile(
        symbol="AAPL",
        cik="0000320193",
        name="Apple Inc.",
        sic="3571",
        sector="BusEq",
        sector_label="Business",
    )
    first = datetime(2025, 1, 30, 21, 30, tzinfo=UTC)
    events = CompanyEvents(
        profile=profile, earnings=[first, first + timedelta(days=91)], earnings_since=date(2020, 9, 1)
    )
    async with database.session() as s:
        await repo.save_company_events(s, events, "sec_edgar")
    later = events.model_copy(update={"earnings": [first + timedelta(days=91), first + timedelta(days=182)]})
    async with database.session() as s:
        await repo.save_company_events(s, later, "sec_edgar")  # overlapping releases are stored once
    async with database.session() as s:
        loaded, updated_at, provider = await repo.load_company_events(s, "AAPL")
        assert await repo.load_company_events(s, "MSFT") is None
    assert provider == "sec_edgar" and updated_at.tzinfo is not None
    assert loaded.earnings == [first, first + timedelta(days=91), first + timedelta(days=182)]
    assert loaded.profile.sector == "BusEq" and loaded.earnings_since == date(2020, 9, 1)

    async with database.session() as s:
        await repo.put_blob(s, "k", {"a": 1}, "wikipedia")
        await repo.put_blob(s, "k", {"a": 2}, "wikipedia")
        assert (await repo.get_blob(s, "k"))[0] == {"a": 2} and await repo.get_blob(s, "none") is None

    facts = [
        FrameFact(cik=320193, start=date(2023, 10, 1), end=date(2024, 9, 28), value=93.7e9, accn="a1"),
        FrameFact(cik=789019, start=date(2023, 7, 1), end=date(2024, 6, 30), value=88.1e9, accn="a2"),
    ]
    async with database.session() as s:
        assert await repo.save_frame(s, "NetIncomeLoss", "CY2024", facts) == 2
        restated = [facts[0].model_copy(update={"value": 94.0e9, "accn": "a3"})]
        await repo.save_frame(s, "NetIncomeLoss", "CY2024", restated)  # a restatement replaces the value
    async with database.session() as s:
        frame = {f.cik: f for f in await repo.load_frame(s, "NetIncomeLoss", "CY2024")}
        rows = await repo.facts_for(s, [320193], ["NetIncomeLoss", "Assets"])
    assert frame[320193].value == 94.0e9 and frame[789019].value == 88.1e9
    assert [(r.cik, r.value) for r in rows] == [(320193, 94.0e9)]


async def test_bulk_bar_reads(database):
    async with database.session() as s:
        await repo.upsert_bars(s, _history(4), "alpaca")
        msft = _history(2).model_copy(update={"symbol": "MSFT"})
        await repo.upsert_bars(s, msft, "alpaca")
        rows = await repo.bar_frame(s, ["AAPL", "MSFT", "NONE"], "1d", T0 + timedelta(days=1))
    assert [(r[0], r[1]) for r in rows] == [
        ("AAPL", T0 + timedelta(days=1)),
        ("AAPL", T0 + timedelta(days=2)),
        ("AAPL", T0 + timedelta(days=3)),
        ("MSFT", T0 + timedelta(days=1)),
    ]
    assert rows[0][5] == 101.0 and rows[0][7] == "alpaca" and len(rows[0]) == len(repo.BAR_COLUMNS)
