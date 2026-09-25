"""Async repositories: all SQL lives here. Services never build queries themselves.

Convention: every "latest/load" helper returns ``(value, observed_at, provider)`` — exactly the tuple the
data gateway's archive fallback expects — or ``None`` when nothing is stored.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any, cast

from sqlalchemy import Table, delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from quantpulse.core.clock import utcnow
from quantpulse.core.errors import NotFoundError
from quantpulse.db.models import (
    CompanyRow,
    EstimateSnapshotRow,
    FinancialStatementRow,
    FuelLogRow,
    FuelPriceRow,
    HoldingRow,
    IngestionEventRow,
    MaintenanceRecordRow,
    OptionSnapshotRow,
    PortfolioRow,
    PriceBarRow,
    QuoteSnapshotRow,
    SecFilingRow,
    SportsGameRow,
    TeamRatingRow,
    TelemetryRow,
    VehicleRow,
    YieldCurvePointRow,
)
from quantpulse.schemas.fundamentals import (
    AnalystEstimates,
    CompanyFundamentals,
    Filing,
    FinancialStatement,
)
from quantpulse.schemas.market import Bar, PriceHistory, Quote
from quantpulse.schemas.options import OptionChain, OptionContract, YieldCurve, YieldPoint
from quantpulse.schemas.sports import Game, PowerRating

STATEMENT_FIELDS = [
    name
    for name in FinancialStatement.model_fields
    if name not in {"fiscal_year", "period_end", "form", "filed", "accession"}
]


async def record_ingestion(session: AsyncSession, dataset: str, key: str, provider: str, rows: int) -> None:
    session.add(IngestionEventRow(dataset=dataset, key=key[:160], provider=provider, rows=rows))


async def recent_ingestions(session: AsyncSession, limit: int = 50) -> list[IngestionEventRow]:
    result = await session.execute(
        select(IngestionEventRow).order_by(IngestionEventRow.id.desc()).limit(limit)
    )
    return list(result.scalars())


# ----------------------------------------------------------------------------- market
async def upsert_bars(session: AsyncSession, history: PriceHistory, provider: str) -> int:
    if not history.bars:
        return 0
    table = cast(Table, PriceBarRow.__table__)
    now = utcnow()
    rows = [
        {
            "symbol": history.symbol,
            "interval": history.interval,
            "ts": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "provider": provider,
            "ingested_at": now,
        }
        for bar in history.bars
    ]
    stmt = sqlite_insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "interval", "ts"],
        set_={
            c: stmt.excluded[c] for c in ("open", "high", "low", "close", "volume", "provider", "ingested_at")
        },
    )
    await session.execute(stmt, rows)
    return len(rows)


async def load_bars(
    session: AsyncSession, symbol: str, interval: str, since: datetime | None = None
) -> tuple[PriceHistory, datetime, str] | None:
    query = select(PriceBarRow).where(PriceBarRow.symbol == symbol, PriceBarRow.interval == interval)
    if since is not None:
        query = query.where(PriceBarRow.ts >= since)
    rows = list((await session.execute(query.order_by(PriceBarRow.ts))).scalars())
    if not rows:
        return None
    bars = [
        Bar(timestamp=r.ts, open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume) for r in rows
    ]
    latest = max(rows, key=lambda r: r.ingested_at)
    return PriceHistory(symbol=symbol, interval=interval, bars=bars), rows[-1].ts, latest.provider


async def insert_quote(session: AsyncSession, quote: Quote, provider: str) -> None:
    session.add(
        QuoteSnapshotRow(
            symbol=quote.symbol,
            price=quote.price,
            previous_close=quote.previous_close,
            bid=quote.bid,
            ask=quote.ask,
            volume=quote.volume,
            payload=quote.model_dump(mode="json"),
            provider=provider,
            quoted_at=quote.timestamp,
        )
    )


async def latest_quote(session: AsyncSession, symbol: str) -> tuple[Quote, datetime, str] | None:
    row = (
        await session.execute(
            select(QuoteSnapshotRow)
            .where(QuoteSnapshotRow.symbol == symbol)
            .order_by(QuoteSnapshotRow.quoted_at.desc(), QuoteSnapshotRow.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return Quote.model_validate(row.payload), row.quoted_at, row.provider


# ----------------------------------------------------------------------------- rates & options
async def upsert_curve(session: AsyncSession, curve: YieldCurve, provider: str) -> int:
    table = cast(Table, YieldCurvePointRow.__table__)
    now = utcnow()
    rows = [
        {
            "curve_date": curve.as_of,
            "tenor": p.tenor,
            "years": p.years,
            "rate": p.rate,
            "provider": provider,
            "ingested_at": now,
        }
        for p in curve.points
    ]
    stmt = sqlite_insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["curve_date", "tenor"],
        set_={c: stmt.excluded[c] for c in ("years", "rate", "provider", "ingested_at")},
    )
    await session.execute(stmt, rows)
    return len(rows)


async def latest_curve(session: AsyncSession) -> tuple[YieldCurve, datetime, str] | None:
    latest_date = (
        await session.execute(
            select(YieldCurvePointRow.curve_date).order_by(YieldCurvePointRow.curve_date.desc()).limit(1)
        )
    ).scalar_one_or_none()
    if latest_date is None:
        return None
    rows = list(
        (
            await session.execute(
                select(YieldCurvePointRow).where(YieldCurvePointRow.curve_date == latest_date)
            )
        ).scalars()
    )
    curve = YieldCurve(
        as_of=latest_date, points=[YieldPoint(tenor=r.tenor, years=r.years, rate=r.rate) for r in rows]
    )
    return curve, max(r.ingested_at for r in rows), rows[0].provider


async def insert_option_snapshot(session: AsyncSession, chain: OptionChain, provider: str) -> int:
    if not chain.contracts:
        return 0
    table = cast(Table, OptionSnapshotRow.__table__)
    rows = [
        {
            "underlying": chain.underlying,
            "contract_symbol": c.contract_symbol[:40],
            "kind": c.kind,
            "strike": c.strike,
            "expiration": c.expiration,
            "bid": c.bid,
            "ask": c.ask,
            "last": c.last,
            "volume": c.volume,
            "open_interest": c.open_interest,
            "implied_volatility": c.implied_volatility,
            "underlying_price": chain.underlying_price,
            "provider": provider,
            "snapshot_at": chain.as_of,
        }
        for c in chain.contracts
    ]
    await session.execute(sqlite_insert(table).on_conflict_do_nothing(), rows)
    return len(rows)


async def latest_option_snapshot(
    session: AsyncSession, underlying: str, expirations: Sequence[date] | None = None
) -> tuple[OptionChain, datetime, str] | None:
    snap_at = (
        await session.execute(
            select(OptionSnapshotRow.snapshot_at)
            .where(OptionSnapshotRow.underlying == underlying)
            .order_by(OptionSnapshotRow.snapshot_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if snap_at is None:
        return None
    query = select(OptionSnapshotRow).where(
        OptionSnapshotRow.underlying == underlying, OptionSnapshotRow.snapshot_at == snap_at
    )
    rows = list((await session.execute(query)).scalars())
    if not rows:
        return None
    all_exp = sorted({r.expiration for r in rows})
    wanted = set(expirations) if expirations else set(all_exp)
    contracts = [
        OptionContract(
            contract_symbol=r.contract_symbol,
            kind=r.kind,
            strike=r.strike,
            expiration=r.expiration,
            bid=r.bid,
            ask=r.ask,
            last=r.last,
            volume=r.volume,
            open_interest=r.open_interest,
            implied_volatility=r.implied_volatility,
        )
        for r in rows
        if r.expiration in wanted
    ]
    chain = OptionChain(
        underlying=underlying,
        underlying_price=rows[0].underlying_price,
        as_of=rows[0].snapshot_at,
        expirations=all_exp,
        contracts=contracts,
    )
    return chain, rows[0].snapshot_at, rows[0].provider


# ----------------------------------------------------------------------------- fundamentals
async def upsert_fundamentals(session: AsyncSession, data: CompanyFundamentals, provider: str) -> int:
    company = (
        await session.execute(select(CompanyRow).where(CompanyRow.symbol == data.symbol))
    ).scalar_one_or_none()
    if company is None:
        company = CompanyRow(symbol=data.symbol)
        session.add(company)
    company.cik = data.cik
    company.name = data.name
    company.shares_outstanding = data.shares_outstanding
    company.shares_as_of = data.shares_as_of
    company.updated_at = utcnow()

    if data.statements:
        table = cast(Table, FinancialStatementRow.__table__)
        now = utcnow()
        rows: list[dict[str, Any]] = []
        for st in data.statements:
            row = st.model_dump()
            row.update({"symbol": data.symbol, "provider": provider, "ingested_at": now})
            rows.append(row)
        stmt = sqlite_insert(table)
        update_cols = [
            *STATEMENT_FIELDS,
            "period_end",
            "form",
            "filed",
            "accession",
            "provider",
            "ingested_at",
        ]
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "fiscal_year"], set_={c: stmt.excluded[c] for c in update_cols}
        )
        await session.execute(stmt, rows)

    if data.recent_filings and data.cik:
        ftable = cast(Table, SecFilingRow.__table__)
        frows = [
            {
                "symbol": data.symbol,
                "cik": data.cik,
                "accession": f.accession,
                "form": f.form,
                "filing_date": f.filing_date,
                "report_date": f.report_date,
                "primary_document": f.primary_document,
                "url": f.url,
            }
            for f in data.recent_filings
        ]
        await session.execute(
            sqlite_insert(ftable).on_conflict_do_nothing(index_elements=["accession"]), frows
        )
    return len(data.statements)


async def load_fundamentals(
    session: AsyncSession, symbol: str
) -> tuple[CompanyFundamentals, datetime, str] | None:
    company = (
        await session.execute(select(CompanyRow).where(CompanyRow.symbol == symbol))
    ).scalar_one_or_none()
    rows = list(
        (
            await session.execute(
                select(FinancialStatementRow)
                .where(FinancialStatementRow.symbol == symbol)
                .order_by(FinancialStatementRow.fiscal_year)
            )
        ).scalars()
    )
    if company is None or not rows:
        return None
    statements = [
        FinancialStatement(
            fiscal_year=r.fiscal_year,
            period_end=r.period_end,
            form=r.form,
            filed=r.filed,
            accession=r.accession,
            **{f: getattr(r, f) for f in STATEMENT_FIELDS},
        )
        for r in rows
    ]
    filings = [
        Filing(
            form=f.form,
            filing_date=f.filing_date,
            report_date=f.report_date,
            accession=f.accession,
            primary_document=f.primary_document,
            url=f.url,
        )
        for f in (
            await session.execute(
                select(SecFilingRow)
                .where(SecFilingRow.symbol == symbol)
                .order_by(SecFilingRow.filing_date.desc())
                .limit(20)
            )
        ).scalars()
    ]
    data = CompanyFundamentals(
        symbol=symbol,
        cik=company.cik,
        name=company.name,
        shares_outstanding=company.shares_outstanding,
        shares_as_of=company.shares_as_of,
        statements=statements,
        recent_filings=filings,
    )
    return data, max(r.ingested_at for r in rows), rows[-1].provider


async def insert_estimates(session: AsyncSession, estimates: AnalystEstimates, provider: str) -> None:
    session.add(
        EstimateSnapshotRow(
            symbol=estimates.symbol, payload=estimates.model_dump(mode="json"), provider=provider
        )
    )


async def latest_estimates(
    session: AsyncSession, symbol: str
) -> tuple[AnalystEstimates, datetime, str] | None:
    row = (
        await session.execute(
            select(EstimateSnapshotRow)
            .where(EstimateSnapshotRow.symbol == symbol)
            .order_by(EstimateSnapshotRow.captured_at.desc(), EstimateSnapshotRow.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return AnalystEstimates.model_validate(row.payload), row.captured_at, row.provider


# ----------------------------------------------------------------------------- portfolio
async def list_portfolios(session: AsyncSession) -> list[PortfolioRow]:
    return list((await session.execute(select(PortfolioRow).order_by(PortfolioRow.id))).scalars())


async def get_portfolio(session: AsyncSession, portfolio_id: int) -> PortfolioRow:
    row = await session.get(PortfolioRow, portfolio_id)
    if row is None:
        raise NotFoundError(f"portfolio {portfolio_id} not found")
    return row


async def portfolio_name_exists(session: AsyncSession, name: str, exclude_id: int | None = None) -> bool:
    query = select(PortfolioRow.id).where(func.lower(PortfolioRow.name) == name.lower())
    if exclude_id is not None:
        query = query.where(PortfolioRow.id != exclude_id)
    return (await session.execute(query)).first() is not None


async def create_portfolio(
    session: AsyncSession, name: str, holdings: Iterable[tuple[str, float, float | None]]
) -> PortfolioRow:
    row = PortfolioRow(name=name)
    row.holdings = [HoldingRow(symbol=s, quantity=q, cost_basis=c) for s, q, c in holdings]
    session.add(row)
    await session.flush()
    return row


async def replace_portfolio(
    session: AsyncSession, portfolio_id: int, name: str, holdings: Iterable[tuple[str, float, float | None]]
) -> PortfolioRow:
    row = await get_portfolio(session, portfolio_id)
    row.name = name
    row.holdings.clear()
    await session.flush()
    row.holdings.extend(HoldingRow(symbol=s, quantity=q, cost_basis=c) for s, q, c in holdings)
    row.updated_at = utcnow()
    await session.flush()
    return row


async def delete_portfolio(session: AsyncSession, portfolio_id: int) -> None:
    row = await get_portfolio(session, portfolio_id)
    await session.delete(row)


# ----------------------------------------------------------------------------- vehicle
async def create_vehicle(session: AsyncSession, **fields: Any) -> VehicleRow:
    row = VehicleRow(**fields)
    session.add(row)
    await session.flush()
    return row


async def get_vehicle(session: AsyncSession, vehicle_id: int) -> VehicleRow:
    row = await session.get(VehicleRow, vehicle_id)
    if row is None:
        raise NotFoundError(f"vehicle {vehicle_id} not found")
    return row


async def list_vehicles(session: AsyncSession) -> list[VehicleRow]:
    return list((await session.execute(select(VehicleRow).order_by(VehicleRow.id))).scalars())


async def delete_vehicle(session: AsyncSession, vehicle_id: int) -> None:
    await session.delete(await get_vehicle(session, vehicle_id))


async def add_row(session: AsyncSession, row: Any) -> Any:
    session.add(row)
    await session.flush()
    return row


async def telemetry(session: AsyncSession, vehicle_id: int) -> list[TelemetryRow]:
    return list(
        (
            await session.execute(
                select(TelemetryRow)
                .where(TelemetryRow.vehicle_id == vehicle_id)
                .order_by(TelemetryRow.recorded_at)
            )
        ).scalars()
    )


async def fuel_logs(session: AsyncSession, vehicle_id: int) -> list[FuelLogRow]:
    return list(
        (
            await session.execute(
                select(FuelLogRow).where(FuelLogRow.vehicle_id == vehicle_id).order_by(FuelLogRow.filled_at)
            )
        ).scalars()
    )


async def maintenance_records(session: AsyncSession, vehicle_id: int) -> list[MaintenanceRecordRow]:
    return list(
        (
            await session.execute(
                select(MaintenanceRecordRow)
                .where(MaintenanceRecordRow.vehicle_id == vehicle_id)
                .order_by(MaintenanceRecordRow.performed_on)
            )
        ).scalars()
    )


async def delete_child(session: AsyncSession, model: Any, vehicle_id: int, row_id: int) -> None:
    result = await session.execute(delete(model).where(model.id == row_id, model.vehicle_id == vehicle_id))
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise NotFoundError(f"record {row_id} not found for vehicle {vehicle_id}")


async def upsert_fuel_prices(
    session: AsyncSession,
    region: str,
    region_name: str,
    grade: str,
    points: Iterable[tuple[date, float, str | None]],
    provider: str,
) -> int:
    table = cast(Table, FuelPriceRow.__table__)
    now = utcnow()
    rows = [
        {
            "region": region,
            "region_name": region_name,
            "grade": grade,
            "period": period,
            "price": price,
            "series_id": series,
            "provider": provider,
            "ingested_at": now,
        }
        for period, price, series in points
    ]
    if not rows:
        return 0
    stmt = sqlite_insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["region", "grade", "period"],
        set_={c: stmt.excluded[c] for c in ("price", "region_name", "series_id", "provider", "ingested_at")},
    )
    await session.execute(stmt, rows)
    return len(rows)


async def fuel_price_history(
    session: AsyncSession, region: str, grade: str, limit: int = 104
) -> list[FuelPriceRow]:
    rows = (
        await session.execute(
            select(FuelPriceRow)
            .where(FuelPriceRow.region == region, FuelPriceRow.grade == grade)
            .order_by(FuelPriceRow.period.desc())
            .limit(limit)
        )
    ).scalars()
    return list(reversed(list(rows)))


# ----------------------------------------------------------------------------- sports
async def upsert_games(session: AsyncSession, games: Sequence[Game]) -> int:
    if not games:
        return 0
    table = cast(Table, SportsGameRow.__table__)
    now = utcnow()
    rows = [
        {
            "event_id": g.event_id,
            "league": g.league,
            "season": g.season,
            "season_type": g.season_type,
            "week": g.week,
            "start_time": g.start_time,
            "home_team_id": g.home.team.id,
            "away_team_id": g.away.team.id,
            "home_name": g.home.team.name[:80],
            "away_name": g.away.team.name[:80],
            "home_score": g.home.score,
            "away_score": g.away.score,
            "state": g.state,
            "completed": g.completed,
            "neutral_site": g.neutral_site,
            "updated_at": now,
        }
        for g in games
    ]
    stmt = sqlite_insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["event_id"],
        set_={
            c: stmt.excluded[c]
            for c in ("home_score", "away_score", "state", "completed", "start_time", "week", "updated_at")
        },
    )
    await session.execute(stmt, rows)
    return len(rows)


async def completed_games(session: AsyncSession, league: str, seasons: Sequence[int]) -> list[SportsGameRow]:
    return list(
        (
            await session.execute(
                select(SportsGameRow)
                .where(
                    SportsGameRow.league == league,
                    SportsGameRow.season.in_(list(seasons)),
                    SportsGameRow.completed.is_(True),
                )
                .order_by(SportsGameRow.start_time)
            )
        ).scalars()
    )


async def save_ratings(
    session: AsyncSession, league: str, season: int, ratings: Sequence[PowerRating]
) -> int:
    if not ratings:
        return 0
    table = cast(Table, TeamRatingRow.__table__)
    now = utcnow()
    rows = [
        {
            "league": league,
            "season": season,
            "team_id": r.team.id,
            "team_name": r.team.name[:80],
            "rating": r.rating,
            "games": r.games,
            "wins": r.wins,
            "losses": r.losses,
            "ties": r.ties,
            "computed_at": now,
        }
        for r in ratings
    ]
    stmt = sqlite_insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["league", "season", "team_id"],
        set_={
            c: stmt.excluded[c]
            for c in ("team_name", "rating", "games", "wins", "losses", "ties", "computed_at")
        },
    )
    await session.execute(stmt, rows)
    return len(rows)
