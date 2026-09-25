"""Quotes, OHLCV history and real-time streaming (SSE + WebSocket)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from quantpulse.api.deps import ContainerDep, websocket_authorized
from quantpulse.api.params import parse_symbols, symbol_path
from quantpulse.core.errors import DomainError
from quantpulse.schemas.common import Envelope, StrictModel
from quantpulse.schemas.market import Interval, PriceHistory, Quote
from quantpulse.services.container import Container

router = APIRouter(prefix="/market", tags=["market"])


class QuotesOut(StrictModel):
    quotes: dict[str, Envelope[Quote]]


@router.get("/quote/{symbol}", response_model=Envelope[Quote], summary="Live quote with provenance")
async def quote(symbol: str = Depends(symbol_path), c: Container = ContainerDep) -> Envelope[Quote]:
    r = await c.market.quote(symbol)
    return Envelope[Quote](data=r.value, meta=r.provenance)


@router.get("/quotes", response_model=QuotesOut, summary="Batch quotes")
async def quotes(
    symbols: str = Query(..., description="Comma-separated tickers", examples=["AAPL,MSFT,SPY"]),
    c: Container = ContainerDep,
) -> QuotesOut:
    resolved = await c.market.quotes(parse_symbols(symbols))
    return QuotesOut(
        quotes={s: Envelope[Quote](data=r.value, meta=r.provenance) for s, r in resolved.items()}
    )


@router.get("/history/{symbol}", response_model=Envelope[PriceHistory], summary="OHLCV bars")
async def history(
    symbol: str = Depends(symbol_path),
    interval: Interval = Query("1d"),
    lookback_days: int = Query(365, ge=1, le=3650),
    c: Container = ContainerDep,
) -> Envelope[PriceHistory]:
    r = await c.market.history(symbol, interval, lookback_days)
    return Envelope[PriceHistory](data=r.value, meta=r.provenance)


async def _quote_events(
    c: Container, symbols: list[str], interval: float, max_events: int | None, is_disconnected
) -> AsyncIterator[str]:
    sent = 0
    yield "retry: 5000\n\n"
    while max_events is None or sent < max_events:
        if await is_disconnected():
            break
        resolved = await c.market.quotes(symbols)
        for symbol, r in resolved.items():
            payload = Envelope[Quote](data=r.value, meta=r.provenance).model_dump_json()
            yield f"event: quote\nid: {symbol}-{r.provenance.fetched_at.timestamp():.0f}\ndata: {payload}\n\n"
        sent += 1
        if max_events is not None and sent >= max_events:
            break
        await asyncio.sleep(interval)


@router.get(
    "/stream",
    summary="Server-Sent Events stream of live quotes",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream(
    request: Request,
    symbols: str = Query(..., examples=["AAPL,MSFT"]),
    interval: float = Query(5.0, ge=1.0, le=300.0, description="Seconds between pushes"),
    max_events: int | None = Query(
        None, ge=1, le=100000, description="Stop after N pushes (default: unbounded)"
    ),
    c: Container = ContainerDep,
) -> StreamingResponse:
    parsed = parse_symbols(symbols, limit=25)
    return StreamingResponse(
        _quote_events(c, parsed, interval, max_events, request.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.websocket("/ws")
async def quotes_ws(websocket: WebSocket) -> None:
    """WebSocket quote stream. Query: ``symbols=AAPL,MSFT&interval=5`` (+ ``api_key`` when auth is on)."""
    if not websocket_authorized(websocket):
        await websocket.close(code=4401, reason="missing or invalid api key")
        return
    try:
        symbols = parse_symbols(websocket.query_params.get("symbols", ""), limit=25)
        interval = float(websocket.query_params.get("interval", "5"))
        if not 1.0 <= interval <= 300.0:
            raise DomainError("interval must be between 1 and 300 seconds")
    except (DomainError, ValueError) as exc:
        await websocket.close(code=4422, reason=str(exc)[:120])
        return
    c: Container = websocket.app.state.container
    await websocket.accept()
    try:
        while True:
            resolved = await c.market.quotes(symbols)
            await websocket.send_text(
                json.dumps(
                    {
                        s: json.loads(Envelope[Quote](data=r.value, meta=r.provenance).model_dump_json())
                        for s, r in resolved.items()
                    }
                )
            )
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
