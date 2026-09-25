"""Async repositories: all SQL lives here. Services never build queries themselves.

Convention: every "latest/load" helper returns ``(value, observed_at, provider)`` — exactly the tuple the
data gateway's archive fallback expects — or ``None`` when nothing is stored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast

from sqlalchemy import Table, delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from quantpulse.core.clock import utcnow
from quantpulse.core.errors import NotFoundError
from quantpulse.db.models import (
    BrokerOrderRow,
    CompanyProfileRow,
    CompanyRow,
    EarningsEventRow,
    EstimateSnapshotRow,
    FinancialStatementRow,
    FuelLogRow,
    FuelPriceRow,
    FundamentalFactRow,
    HoldingRow,
    IngestionEventRow,
    MaintenanceRecordRow,
    OptionSnapshotRow,
    PortfolioRow,
    PredictionRow,
    PriceBarRow,
    QuoteSnapshotRow,
    ReferenceBlobRow,
    SandboxAccountRow,
    SandboxEquityRow,
    SandboxJournalRow,
    SandboxPositionRow,
    SandboxTradeRow,
    SecFilingRow,
    SportsGameRow,
    TeamRatingRow,
    TelemetryRow,
    TradingCycleRow,
    TradingEventRow,
    TradingStateRow,
    VehicleRow,
    YieldCurvePointRow,
)
from quantpulse.domain.sectors import FF12_NAMES
from quantpulse.schemas.fundamentals import (
    AnalystEstimates,
    CompanyFundamentals,
    Filing,
    FinancialStatement,
)
from quantpulse.schemas.market import Bar, PriceHistory, Quote
from quantpulse.schemas.options import OptionChain, OptionContract, YieldCurve, YieldPoint
from quantpulse.schemas.reference import CompanyEvents, CompanyProfile, FrameFact
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
REVISION_WINDOW = timedelta(days=7)


async def upsert_bars(session: AsyncSession, history: PriceHistory, provider: str) -> int:
    """Store bars, writing only what is new.

    Bars from ``REVISION_WINDOW`` before the latest stored bar onwards are always rewritten (vendors revise
    recent bars), and so are bars older than the earliest stored one (a longer download backfilling
    history). If the stored closes in the overlap no longer match the vendor's, the history has been
    re-adjusted (a split or dividend adjustment) and every bar is rewritten."""
    if not history.bars:
        return 0
    table = cast(Table, PriceBarRow.__table__)
    now = utcnow()
    bars = history.bars
    key = (PriceBarRow.symbol == history.symbol, PriceBarRow.interval == history.interval)
    span = (
        await session.execute(select(func.min(PriceBarRow.ts), func.max(PriceBarRow.ts)).where(*key))
    ).one()
    earliest, latest = span[0], span[1]
    if latest is not None:
        cutoff = latest - REVISION_WINDOW
        recent = await session.execute(
            select(PriceBarRow.ts, PriceBarRow.close).where(*key, PriceBarRow.ts >= cutoff)
        )
        stored: dict[datetime, float] = {row[0]: row[1] for row in recent.all()}
        overlap = [b for b in bars if b.timestamp in stored]
        unchanged = all(abs(b.close - stored[b.timestamp]) <= 1e-9 * max(1.0, abs(b.close)) for b in overlap)
        if overlap and unchanged:
            bars = [b for b in bars if b.timestamp >= cutoff or b.timestamp < earliest]
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
        for bar in bars
    ]
    if not rows:
        return 0
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


BAR_COLUMNS = ("symbol", "ts", "open", "high", "low", "close", "volume", "provider", "ingested_at")
_IN_CHUNK = 500  # SQLite limits the number of bound parameters per statement


async def bar_frame(
    session: AsyncSession, symbols: Sequence[str], interval: str, since: datetime
) -> list[tuple[Any, ...]]:
    """Stored bars for many symbols as plain tuples in :data:`BAR_COLUMNS` order (fast bulk read)."""
    out: list[tuple[Any, ...]] = []
    cols = [getattr(PriceBarRow, c) for c in BAR_COLUMNS]
    for i in range(0, len(symbols), _IN_CHUNK):
        chunk = list(symbols[i : i + _IN_CHUNK])
        result = await session.execute(
            select(*cols)
            .where(PriceBarRow.symbol.in_(chunk), PriceBarRow.interval == interval, PriceBarRow.ts >= since)
            .order_by(PriceBarRow.symbol, PriceBarRow.ts)
        )
        out.extend(tuple(r) for r in result.all())
    return out


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


# ----------------------------------------------------------------------------- trading sandbox
async def list_sandbox_accounts(session: AsyncSession) -> list[SandboxAccountRow]:
    return list((await session.execute(select(SandboxAccountRow).order_by(SandboxAccountRow.id))).scalars())


async def get_sandbox_account(session: AsyncSession, account_id: int) -> SandboxAccountRow:
    row = await session.get(SandboxAccountRow, account_id)
    if row is None:
        raise NotFoundError(f"sandbox account {account_id} not found")
    return row


async def sandbox_name_exists(session: AsyncSession, name: str, exclude_id: int | None = None) -> bool:
    query = select(SandboxAccountRow.id).where(func.lower(SandboxAccountRow.name) == name.lower())
    if exclude_id is not None:
        query = query.where(SandboxAccountRow.id != exclude_id)
    return (await session.execute(query)).first() is not None


async def sandbox_positions(session: AsyncSession, account_id: int) -> list[SandboxPositionRow]:
    return list(
        (
            await session.execute(
                select(SandboxPositionRow)
                .where(SandboxPositionRow.account_id == account_id)
                .order_by(SandboxPositionRow.symbol)
            )
        ).scalars()
    )


async def replace_sandbox_positions(
    session: AsyncSession, account_id: int, positions: Mapping[str, tuple[float, float]]
) -> None:
    """Replace an account's holdings with ``{symbol: (quantity, avg_cost)}``."""
    await session.execute(delete(SandboxPositionRow).where(SandboxPositionRow.account_id == account_id))
    session.add_all(
        SandboxPositionRow(account_id=account_id, symbol=s, quantity=q, avg_cost=c)
        for s, (q, c) in sorted(positions.items())
    )
    await session.flush()


async def sandbox_trades(session: AsyncSession, account_id: int, limit: int = 200) -> list[SandboxTradeRow]:
    """Most recent trades first."""
    return list(
        (
            await session.execute(
                select(SandboxTradeRow)
                .where(SandboxTradeRow.account_id == account_id)
                .order_by(SandboxTradeRow.executed_at.desc(), SandboxTradeRow.id.desc())
                .limit(limit)
            )
        ).scalars()
    )


async def sandbox_trade_totals(session: AsyncSession, account_id: int) -> tuple[int, float, float]:
    """``(number of trades, realised P&L, commissions paid)``."""
    row = (
        await session.execute(
            select(
                func.count(SandboxTradeRow.id),
                func.coalesce(func.sum(SandboxTradeRow.realized_pnl), 0.0),
                func.coalesce(func.sum(SandboxTradeRow.commission), 0.0),
            ).where(SandboxTradeRow.account_id == account_id)
        )
    ).one()
    return int(row[0]), float(row[1]), float(row[2])


async def sandbox_equity(session: AsyncSession, account_id: int, limit: int = 2000) -> list[SandboxEquityRow]:
    """The latest ``limit`` equity snapshots, oldest first."""
    rows = list(
        (
            await session.execute(
                select(SandboxEquityRow)
                .where(SandboxEquityRow.account_id == account_id)
                .order_by(SandboxEquityRow.recorded_at.desc(), SandboxEquityRow.id.desc())
                .limit(limit)
            )
        ).scalars()
    )
    return rows[::-1]


async def first_sandbox_benchmark(session: AsyncSession, account_id: int) -> SandboxEquityRow | None:
    """The earliest snapshot that recorded a benchmark price (the base for benchmark returns)."""
    return (
        await session.execute(
            select(SandboxEquityRow)
            .where(SandboxEquityRow.account_id == account_id, SandboxEquityRow.benchmark_price.is_not(None))
            .order_by(SandboxEquityRow.recorded_at, SandboxEquityRow.id)
            .limit(1)
        )
    ).scalar_one_or_none()


async def latest_sandbox_equity(session: AsyncSession, account_id: int) -> SandboxEquityRow | None:
    return (
        await session.execute(
            select(SandboxEquityRow)
            .where(SandboxEquityRow.account_id == account_id)
            .order_by(SandboxEquityRow.recorded_at.desc(), SandboxEquityRow.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def sandbox_journal(
    session: AsyncSession, account_id: int, limit: int = 100
) -> list[SandboxJournalRow]:
    """Most recent entries first."""
    return list(
        (
            await session.execute(
                select(SandboxJournalRow)
                .where(SandboxJournalRow.account_id == account_id)
                .order_by(SandboxJournalRow.created_at.desc(), SandboxJournalRow.id.desc())
                .limit(limit)
            )
        ).scalars()
    )


async def add_sandbox_journal(
    session: AsyncSession,
    account_id: int,
    kind: str,
    summary: str,
    details: Mapping[str, Any],
    created_at: datetime,
) -> SandboxJournalRow:
    row = SandboxJournalRow(
        account_id=account_id, kind=kind, summary=summary, details=dict(details), created_at=created_at
    )
    session.add(row)
    await session.flush()
    return row


async def clear_sandbox_history(session: AsyncSession, account_id: int) -> None:
    """Delete positions, trades, equity snapshots and journal entries (the account row itself stays)."""
    for model in (SandboxPositionRow, SandboxTradeRow, SandboxEquityRow, SandboxJournalRow):
        await session.execute(delete(model).where(model.account_id == account_id))


async def delete_sandbox_account(session: AsyncSession, account_id: int) -> None:
    row = await get_sandbox_account(session, account_id)
    await clear_sandbox_history(session, account_id)  # explicit: do not depend on SQLite FK enforcement
    await session.delete(row)


# ----------------------------------------------------------------------------- prediction ledger
async def insert_predictions(session: AsyncSession, rows: Sequence[Mapping[str, Any]]) -> int:
    """Insert predictions, ignoring any already logged for the same (symbol, source, horizon, day)."""
    if not rows:
        return 0
    table = cast(Table, PredictionRow.__table__)
    stmt = sqlite_insert(table).on_conflict_do_nothing(
        index_elements=["symbol", "source", "horizon_days", "made_on"]
    )
    inserted = 0
    for row in rows:  # per-row so the count reflects rows actually inserted
        result = await session.execute(stmt, [dict(row)])
        inserted += max(0, int(result.rowcount or 0))  # type: ignore[attr-defined]
    return inserted


async def insert_predictions_bulk(session: AsyncSession, rows: Sequence[Mapping[str, Any]]) -> int:
    """Insert many predictions at once (backfills), keeping any row already stored for the same key.

    Returns the number of rows actually inserted (``RETURNING`` reports only the rows written, which is
    exact even while other sessions write concurrently)."""
    if not rows:
        return 0
    table = cast(Table, PredictionRow.__table__)
    stmt = (
        sqlite_insert(table)
        .on_conflict_do_nothing(index_elements=["symbol", "source", "horizon_days", "made_on"])
        .returning(table.c.id)
    )
    inserted = 0
    for i in range(0, len(rows), 2000):
        result = await session.execute(stmt, [dict(r) for r in rows[i : i + 2000]])
        inserted += len(result.all())
    return inserted


async def delete_backfill(session: AsyncSession, source: str | None = None) -> int:
    query = delete(PredictionRow).where(PredictionRow.origin == "backfill")
    if source is not None:
        query = query.where(PredictionRow.source == source)
    result = await session.execute(query)
    return int(result.rowcount or 0)  # type: ignore[attr-defined]


async def prediction_counts(session: AsyncSession) -> dict[tuple[str, str, str], int]:
    """(origin, source, status) -> rows."""
    query = select(
        PredictionRow.origin, PredictionRow.source, PredictionRow.status, func.count(PredictionRow.id)
    ).group_by(PredictionRow.origin, PredictionRow.source, PredictionRow.status)
    return {(o, src, st): int(n) for o, src, st, n in (await session.execute(query)).all()}


async def predictions_logged_on(session: AsyncSession, made_on: date) -> int:
    """Live predictions already logged for ``made_on`` (backfilled history does not count)."""
    query = select(func.count(PredictionRow.id)).where(
        PredictionRow.made_on == made_on, PredictionRow.origin == "live"
    )
    return int((await session.execute(query)).scalar_one())


async def due_predictions(session: AsyncSession, on_or_before: date) -> list[PredictionRow]:
    query = (
        select(PredictionRow)
        .where(PredictionRow.status == "open", PredictionRow.target_date <= on_or_before)
        .order_by(PredictionRow.target_date, PredictionRow.id)
    )
    return list((await session.execute(query)).scalars())


async def score_rows(
    session: AsyncSession, *, symbol: str | None = None, origin: str | None = None
) -> list[Any]:
    """Only the columns the scorecard needs, as light rows (fast over hundreds of thousands of predictions)."""
    p = PredictionRow
    query = select(
        p.source,
        p.horizon_days,
        p.status,
        p.prob_up,
        p.outcome_up,
        p.in_50,
        p.in_90,
        p.prob_outperform,
        p.outcome_outperform,
        p.rank,
        p.realized_return,
        p.benchmark_return,
    )
    if symbol is not None:
        query = query.where(p.symbol == symbol)
    if origin is not None:
        query = query.where(p.origin == origin)
    return list((await session.execute(query)).all())


async def recent_predictions(
    session: AsyncSession, *, symbol: str | None = None, origin: str | None = None, limit: int = 60
) -> list[PredictionRow]:
    """Most recently resolved or logged first."""
    query = select(PredictionRow)
    if symbol is not None:
        query = query.where(PredictionRow.symbol == symbol)
    if origin is not None:
        query = query.where(PredictionRow.origin == origin)
    query = query.order_by(
        func.coalesce(PredictionRow.resolved_at, PredictionRow.created_at).desc(), PredictionRow.id.desc()
    ).limit(limit)
    return list((await session.execute(query)).scalars())


async def list_predictions(
    session: AsyncSession,
    *,
    symbol: str | None = None,
    status: str | None = None,
    source: str | None = None,
    horizon: int | None = None,
    origin: str | None = None,
    limit: int | None = 200,
) -> list[PredictionRow]:
    """Newest first."""
    query = select(PredictionRow)
    if origin is not None:
        query = query.where(PredictionRow.origin == origin)
    if symbol is not None:
        query = query.where(PredictionRow.symbol == symbol)
    if status is not None:
        query = query.where(PredictionRow.status == status)
    if source is not None:
        query = query.where(PredictionRow.source == source)
    if horizon is not None:
        query = query.where(PredictionRow.horizon_days == horizon)
    query = query.order_by(PredictionRow.made_on.desc(), PredictionRow.id.desc())
    if limit is not None:
        query = query.limit(limit)
    return list((await session.execute(query)).scalars())


# ----------------------------------------------------------------------------- reference data
async def save_company_events(session: AsyncSession, events: CompanyEvents, provider: str) -> None:
    p = events.profile
    row = (
        await session.execute(select(CompanyProfileRow).where(CompanyProfileRow.symbol == p.symbol))
    ).scalar_one_or_none()
    if row is None:
        row = CompanyProfileRow(symbol=p.symbol)
        session.add(row)
    row.cik, row.name, row.sic, row.sic_description = p.cik, p.name[:200], p.sic, p.sic_description
    row.sector, row.earnings_since, row.provider, row.updated_at = (
        p.sector,
        events.earnings_since,
        provider,
        utcnow(),
    )
    table = cast(Table, EarningsEventRow.__table__)
    if events.earnings:
        stmt = sqlite_insert(table).on_conflict_do_nothing(index_elements=["symbol", "announced_at"])
        await session.execute(stmt, [{"symbol": p.symbol, "announced_at": at} for at in events.earnings])
    await session.flush()


async def load_company_events(
    session: AsyncSession, symbol: str
) -> tuple[CompanyEvents, datetime, str] | None:
    row = (
        await session.execute(select(CompanyProfileRow).where(CompanyProfileRow.symbol == symbol))
    ).scalar_one_or_none()
    if row is None:
        return None
    times = (
        await session.execute(
            select(EarningsEventRow.announced_at)
            .where(
                EarningsEventRow.symbol == symbol,
                EarningsEventRow.announced_at >= datetime.combine(row.earnings_since, time.min, UTC),
            )
            .order_by(EarningsEventRow.announced_at)
        )
    ).scalars()
    events = CompanyEvents(
        profile=CompanyProfile(
            symbol=row.symbol,
            cik=row.cik,
            name=row.name,
            sic=row.sic,
            sic_description=row.sic_description,
            sector=row.sector,
            sector_label=FF12_NAMES.get(row.sector, row.sector),
        ),
        earnings=list(times),
        earnings_since=row.earnings_since,
    )
    return events, row.updated_at, row.provider


async def put_blob(session: AsyncSession, key: str, payload: Mapping[str, Any], provider: str) -> None:
    row = await session.get(ReferenceBlobRow, key)
    if row is None:
        session.add(ReferenceBlobRow(key=key, payload=dict(payload), provider=provider, fetched_at=utcnow()))
    else:
        row.payload, row.provider, row.fetched_at = dict(payload), provider, utcnow()
    await session.flush()


async def get_blob(session: AsyncSession, key: str) -> tuple[dict[str, Any], datetime, str] | None:
    row = await session.get(ReferenceBlobRow, key)
    return None if row is None else (row.payload, row.fetched_at, row.provider)


async def save_frame(session: AsyncSession, tag: str, frame: str, facts: Sequence[FrameFact]) -> int:
    if not facts:
        return 0
    table = cast(Table, FundamentalFactRow.__table__)
    rows = [
        {
            "tag": tag,
            "frame": frame,
            "cik": f.cik,
            "period_start": f.start,
            "period_end": f.end,
            "value": f.value,
            "accn": f.accn[:25],
        }
        for f in facts
    ]
    stmt = sqlite_insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["tag", "frame", "cik"],
        set_={c: stmt.excluded[c] for c in ("period_start", "period_end", "value", "accn")},
    )
    await session.execute(stmt, rows)
    return len(rows)


async def load_frame(session: AsyncSession, tag: str, frame: str) -> list[FrameFact]:
    rows = (
        await session.execute(
            select(FundamentalFactRow).where(FundamentalFactRow.tag == tag, FundamentalFactRow.frame == frame)
        )
    ).scalars()
    return [
        FrameFact(cik=r.cik, start=r.period_start, end=r.period_end, value=r.value, accn=r.accn) for r in rows
    ]


async def facts_for(
    session: AsyncSession, ciks: Iterable[int], tags: Iterable[str]
) -> list[FundamentalFactRow]:
    ids, names = list(ciks), list(tags)
    if not ids or not names:
        return []
    out: list[FundamentalFactRow] = []
    for i in range(0, len(ids), 500):  # keep SQL parameter lists small
        chunk = ids[i : i + 500]
        out.extend(
            (
                await session.execute(
                    select(FundamentalFactRow).where(
                        FundamentalFactRow.cik.in_(chunk), FundamentalFactRow.tag.in_(names)
                    )
                )
            ).scalars()
        )
    return out


# ----------------------------------------------------------------------------- Alpaca paper trading
async def get_trading_state(session: AsyncSession, key: str) -> dict[str, Any] | None:
    row = await session.get(TradingStateRow, key)
    return dict(row.value) if row is not None else None


async def put_trading_state(session: AsyncSession, key: str, value: Mapping[str, Any], now: datetime) -> None:
    stmt = sqlite_insert(TradingStateRow).values(key=key, value=dict(value), updated_at=now)
    stmt = stmt.on_conflict_do_update(
        index_elements=["key"], set_={"value": stmt.excluded.value, "updated_at": stmt.excluded.updated_at}
    )
    await session.execute(stmt)


async def add_trading_event(
    session: AsyncSession,
    kind: str,
    message: str,
    now: datetime,
    *,
    cycle_id: int | None = None,
    symbol: str | None = None,
    client_order_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> TradingEventRow:
    row = TradingEventRow(
        created_at=now,
        cycle_id=cycle_id,
        kind=kind,
        symbol=symbol,
        client_order_id=client_order_id,
        message=message,
        details=dict(details or {}),
    )
    session.add(row)
    return row


async def trading_events(
    session: AsyncSession,
    limit: int = 200,
    *,
    kinds: Sequence[str] | None = None,
    cycle_id: int | None = None,
) -> list[TradingEventRow]:
    stmt = select(TradingEventRow).order_by(TradingEventRow.id.desc()).limit(limit)
    if kinds:
        stmt = stmt.where(TradingEventRow.kind.in_(list(kinds)))
    if cycle_id is not None:
        stmt = stmt.where(TradingEventRow.cycle_id == cycle_id)
    return list((await session.scalars(stmt)).all())


async def get_broker_order(session: AsyncSession, client_order_id: str) -> BrokerOrderRow | None:
    return (
        await session.scalars(select(BrokerOrderRow).where(BrokerOrderRow.client_order_id == client_order_id))
    ).first()


async def broker_orders(
    session: AsyncSession,
    limit: int = 200,
    *,
    statuses: Sequence[str] | None = None,
    exclude_statuses: Iterable[str] | None = None,
    symbol: str | None = None,
    since: datetime | None = None,
) -> list[BrokerOrderRow]:
    stmt = select(BrokerOrderRow).order_by(BrokerOrderRow.id.desc()).limit(limit)
    if statuses:
        stmt = stmt.where(BrokerOrderRow.status.in_(list(statuses)))
    if exclude_statuses is not None:
        stmt = stmt.where(BrokerOrderRow.status.not_in(list(exclude_statuses)))
    if symbol:
        stmt = stmt.where(BrokerOrderRow.symbol == symbol)
    if since is not None:
        stmt = stmt.where(BrokerOrderRow.created_at >= since)
    return list((await session.scalars(stmt)).all())


async def filled_broker_orders(session: AsyncSession) -> list[BrokerOrderRow]:
    """Every order with at least one fill, oldest first (for round-trip performance)."""
    stmt = (
        select(BrokerOrderRow)
        .where(BrokerOrderRow.filled_quantity > 0)
        .order_by(BrokerOrderRow.filled_at, BrokerOrderRow.id)
    )
    return list((await session.scalars(stmt)).all())


async def create_trading_cycle(session: AsyncSession, **fields: Any) -> TradingCycleRow:
    row = TradingCycleRow(**fields)
    session.add(row)
    await session.flush()
    return row


async def get_trading_cycle(session: AsyncSession, cycle_id: int) -> TradingCycleRow:
    row = await session.get(TradingCycleRow, cycle_id)
    if row is None:
        raise NotFoundError(f"trading cycle {cycle_id} not found")
    return row


async def trading_cycle_by_key(session: AsyncSession, cycle_key: str) -> TradingCycleRow | None:
    return (
        await session.scalars(select(TradingCycleRow).where(TradingCycleRow.cycle_key == cycle_key))
    ).first()


async def trading_cycles(
    session: AsyncSession,
    limit: int = 50,
    *,
    statuses: Sequence[str] | None = None,
    since: datetime | None = None,
    oldest_first: bool = False,
) -> list[TradingCycleRow]:
    order = TradingCycleRow.started_at.asc() if oldest_first else TradingCycleRow.started_at.desc()
    stmt = select(TradingCycleRow).order_by(order).limit(limit)
    if statuses:
        stmt = stmt.where(TradingCycleRow.status.in_(list(statuses)))
    if since is not None:
        stmt = stmt.where(TradingCycleRow.started_at >= since)
    return list((await session.scalars(stmt)).all())
