"""A stateful stand-in for Alpaca's *paper* trading API, mounted under the real ``alpaca-py`` SDK.

It is a ``requests`` transport adapter: the SDK builds real HTTP requests (URL, headers, JSON body) and
parses real JSON responses, so tests exercise the SDK and QuantPulse's provider end to end without the
network. It refuses any host other than ``paper-api.alpaca.markets``.

Behaviour per symbol (``fill_mode``): ``fill`` (immediately, at the limit price or the current price),
``partial`` (half the quantity), ``accept`` (rests as an open order), ``reject`` (403 insufficient buying
power) and ``timeout`` (the order IS recorded, then the connection times out — an ambiguous outcome).
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import BaseAdapter

PAPER_HOST = "paper-api.alpaca.markets"


def _iso(d: datetime) -> str:
    return d.astimezone(UTC).isoformat().replace("+00:00", "Z")


class FakeAlpacaPaper(BaseAdapter):
    def __init__(self, equity: float = 100_000.0, clock: Any = None) -> None:
        super().__init__()
        self.clock = clock
        self.cash = equity
        self.last_equity = equity
        self.positions: dict[str, dict[str, float]] = {}  # symbol -> {"qty", "avg"}
        self.prices: dict[str, float] = {}
        self.lastday: dict[str, float] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.by_client: dict[str, str] = {}
        self.market_open = True
        self.fill_mode: dict[str, str] = {}
        self.default_mode = "fill"
        self.log: list[tuple[str, str]] = []
        self.auth_headers_seen: list[bool] = []
        self.blocked = False
        self.fail_status: int | None = None  # next request answers this HTTP status

    # ------------------------------------------------------------------ helpers
    def now(self) -> datetime:
        return self.clock.now() if self.clock is not None else datetime.now(UTC)

    def price(self, symbol: str) -> float:
        if symbol in self.prices:
            return self.prices[symbol]
        pos = self.positions.get(symbol)
        return pos["avg"] if pos else 100.0

    def equity(self) -> float:
        return self.cash + sum(p["qty"] * self.price(s) for s, p in self.positions.items())

    def hold(self, symbol: str, qty: float, avg: float, price: float | None = None) -> None:
        """Seed a position (as if bought earlier)."""
        self.positions[symbol] = {"qty": qty, "avg": avg}
        self.prices[symbol] = price if price is not None else avg
        self.cash -= qty * avg
        self.last_equity = self.equity()

    def submitted(self) -> list[dict[str, Any]]:
        return sorted(self.orders.values(), key=lambda o: o["created_at"])

    def _response(
        self, request: requests.PreparedRequest, status: int, body: Any = None
    ) -> requests.Response:
        resp = requests.Response()
        resp.status_code = status
        resp._content = b"" if body is None else json.dumps(body).encode()
        resp.headers["Content-Type"] = "application/json"
        resp.url = request.url or ""
        resp.request = request
        resp.encoding = "utf-8"
        return resp

    def _error(
        self, request: requests.PreparedRequest, status: int, code: int, message: str
    ) -> requests.Response:
        return self._response(request, status, {"code": code, "message": message})

    # ------------------------------------------------------------------ JSON shapes
    def _account(self) -> dict[str, Any]:
        long_mv = sum(p["qty"] * self.price(s) for s, p in self.positions.items())
        equity = self.cash + long_mv
        return {
            "id": "904837e3-3b76-47ec-b432-046db621571b",
            "account_number": "PA3TESTPAPER1",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": str(max(self.cash, 0.0) * 2),
            "regt_buying_power": str(max(self.cash, 0.0) * 2),
            "cash": str(self.cash),
            "portfolio_value": str(equity),
            "equity": str(equity),
            "last_equity": str(self.last_equity),
            "long_market_value": str(long_mv),
            "short_market_value": "0",
            "pattern_day_trader": False,
            "trading_blocked": self.blocked,
            "transfers_blocked": False,
            "account_blocked": False,
            "trade_suspended_by_user": False,
            "shorting_enabled": False,
            "multiplier": "2",
            "daytrade_count": 0,
            "created_at": "2026-01-02T15:00:00Z",
        }

    def _position(self, symbol: str) -> dict[str, Any]:
        p = self.positions[symbol]
        price = self.price(symbol)
        mv = p["qty"] * price
        cost = p["qty"] * p["avg"]
        last = self.lastday.get(symbol, price)
        return {
            "asset_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, symbol)),
            "symbol": symbol,
            "exchange": "NASDAQ",
            "asset_class": "us_equity",
            "avg_entry_price": str(p["avg"]),
            "qty": str(p["qty"]),
            "qty_available": str(p["qty"]),
            "side": "long",
            "market_value": str(mv),
            "cost_basis": str(cost),
            "unrealized_pl": str(mv - cost),
            "unrealized_plpc": str(mv / cost - 1 if cost else 0.0),
            "unrealized_intraday_pl": str((price - last) * p["qty"]),
            "unrealized_intraday_plpc": str(price / last - 1 if last else 0.0),
            "current_price": str(price),
            "lastday_price": str(last),
            "change_today": str(price / last - 1 if last else 0.0),
        }

    def _order(self, body: dict[str, Any], status: str) -> dict[str, Any]:
        now = _iso(self.now())
        kind = body.get("type", "market")
        return {
            "id": str(uuid.uuid4()),
            "client_order_id": body.get("client_order_id") or str(uuid.uuid4()),
            "created_at": now,
            "updated_at": now,
            "submitted_at": now,
            "filled_at": None,
            "expired_at": None,
            "canceled_at": None,
            "failed_at": None,
            "replaced_at": None,
            "replaced_by": None,
            "replaces": None,
            "asset_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, body["symbol"])),
            "symbol": body["symbol"],
            "asset_class": "us_equity",
            "notional": None,
            "qty": str(body.get("qty")),
            "filled_qty": "0",
            "filled_avg_price": None,
            "order_class": "simple",
            "order_type": kind,
            "type": kind,
            "side": body["side"],
            "time_in_force": body.get("time_in_force", "day"),
            "limit_price": str(body["limit_price"]) if body.get("limit_price") is not None else None,
            "stop_price": None,
            "status": status,
            "extended_hours": False,
            "legs": None,
            "trail_percent": None,
            "trail_price": None,
            "hwm": None,
        }

    def _fill(self, order: dict[str, Any], qty: float) -> None:
        symbol, side = order["symbol"], order["side"]
        price = float(order["limit_price"]) if order["limit_price"] else self.price(symbol)
        if side == "buy":
            pos = self.positions.setdefault(symbol, {"qty": 0.0, "avg": price})
            total = pos["qty"] + qty
            pos["avg"] = (pos["qty"] * pos["avg"] + qty * price) / total
            pos["qty"] = total
            self.cash -= qty * price
        else:
            pos = self.positions[symbol]
            pos["qty"] -= qty
            self.cash += qty * price
            if pos["qty"] <= 1e-9:
                del self.positions[symbol]
        self.prices.setdefault(symbol, price)
        filled = float(order["filled_qty"]) + qty
        prev_avg = float(order["filled_avg_price"] or 0.0)
        order["filled_avg_price"] = str((prev_avg * float(order["filled_qty"]) + price * qty) / filled)
        order["filled_qty"] = str(filled)
        order["status"] = "filled" if math.isclose(filled, float(order["qty"])) else "partially_filled"
        stamp = _iso(self.now())
        order["updated_at"] = stamp
        if order["status"] == "filled":
            order["filled_at"] = stamp

    def complete(self, client_order_id: str) -> None:
        """Fill whatever is left of an open order (e.g. a resting or partially filled one)."""
        order = self.orders[self.by_client[client_order_id]]
        left = float(order["qty"]) - float(order["filled_qty"])
        if left > 0:
            self._fill(order, left)

    def _submit(self, request: requests.PreparedRequest, body: dict[str, Any]) -> requests.Response:
        cid = body.get("client_order_id")
        if cid and cid in self.by_client:
            return self._error(request, 422, 40010001, "client_order_id must be unique")
        symbol, side, qty = body["symbol"], body["side"], float(body["qty"])
        mode = self.fill_mode.get(symbol, self.default_mode)
        if mode == "reject":
            return self._error(request, 403, 40310000, "insufficient buying power")
        if side == "sell" and qty > self.positions.get(symbol, {"qty": 0.0})["qty"] + 1e-9:
            return self._error(request, 403, 40310000, "insufficient qty available for order")
        order = self._order(body, "accepted")
        self.orders[order["id"]] = order
        self.by_client[order["client_order_id"]] = order["id"]
        if mode in ("fill", "timeout"):
            self._fill(order, qty)
        elif mode == "partial":
            self._fill(order, math.floor(qty / 2) or qty)
        if mode == "timeout":
            raise requests.exceptions.ReadTimeout("simulated timeout after Alpaca accepted the order")
        return self._response(request, 200, order)

    def _list_orders(self, query: dict[str, list[str]]) -> list[dict[str, Any]]:
        status = (query.get("status") or ["open"])[0]
        limit = int((query.get("limit") or ["50"])[0])
        after = query.get("after")
        symbols = set((query.get("symbols") or [""])[0].split(",")) - {""}
        terminal = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}
        rows = []
        for o in sorted(self.orders.values(), key=lambda o: o["submitted_at"], reverse=True):
            is_open = o["status"] not in terminal
            if (status == "open" and not is_open) or (status == "closed" and is_open):
                continue
            if after and o["submitted_at"] <= _iso(datetime.fromisoformat(after[0].replace("Z", "+00:00"))):
                continue
            if symbols and o["symbol"] not in symbols:
                continue
            rows.append(o)
        return rows[:limit]

    def _cancel(self, order: dict[str, Any]) -> bool:
        if order["status"] in ("filled", "canceled", "expired", "rejected"):
            return False
        order["status"] = "canceled"
        order["canceled_at"] = order["updated_at"] = _iso(self.now())
        return True

    # ------------------------------------------------------------------ transport
    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        url = urlparse(request.url or "")
        if url.netloc != PAPER_HOST:
            raise AssertionError(f"QuantPulse must only talk to Alpaca PAPER, not {url.netloc}")
        self.log.append((request.method or "", url.path))
        self.auth_headers_seen.append(
            bool(request.headers.get("APCA-API-KEY-ID")) and bool(request.headers.get("APCA-API-SECRET-KEY"))
        )
        if self.fail_status is not None:
            status, self.fail_status = self.fail_status, None
            return self._error(request, status, 50010000, "internal server error")
        query = parse_qs(url.query)
        path = url.path.removeprefix("/v2")
        method = request.method
        body = json.loads(request.body) if request.body else {}
        if method == "GET" and path == "/account":
            return self._response(request, 200, self._account())
        if method == "GET" and path == "/clock":
            now = self.now()
            return self._response(
                request,
                200,
                {
                    "timestamp": _iso(now),
                    "is_open": self.market_open,
                    "next_open": _iso(now + timedelta(hours=18)),
                    "next_close": _iso(now + timedelta(hours=6)),
                },
            )
        if method == "GET" and path == "/positions":
            return self._response(request, 200, [self._position(s) for s in sorted(self.positions)])
        if method == "GET" and path == "/orders":
            return self._response(request, 200, self._list_orders(query))
        if method == "POST" and path == "/orders":
            return self._submit(request, body)
        if method == "GET" and path == "/orders:by_client_order_id":
            cid = (query.get("client_order_id") or [""])[0]
            if cid not in self.by_client:
                return self._error(request, 404, 40410000, "order not found")
            return self._response(request, 200, self.orders[self.by_client[cid]])
        if path.startswith("/orders/"):
            oid = path.split("/")[-1]
            if oid not in self.orders:
                return self._error(request, 404, 40410000, "order not found")
            if method == "GET":
                return self._response(request, 200, self.orders[oid])
            if method == "DELETE":
                if not self._cancel(self.orders[oid]):
                    return self._error(request, 422, 42210000, "order is not cancelable")
                return self._response(request, 204)
        if method == "DELETE" and path == "/orders":
            out = [
                {"id": o["id"], "status": 200, "body": None}
                for o in list(self.orders.values())
                if self._cancel(o)
            ]
            return self._response(request, 207, out)
        if method == "DELETE" and path.startswith("/positions/"):
            symbol = path.split("/")[-1]
            if symbol not in self.positions:
                return self._error(request, 404, 40410000, "position not found")
            qty = float((query.get("qty") or [self.positions[symbol]["qty"]])[0])
            order = self._order({"symbol": symbol, "side": "sell", "qty": qty, "type": "market"}, "accepted")
            self.orders[order["id"]] = order
            self.by_client[order["client_order_id"]] = order["id"]
            self._fill(order, qty)
            return self._response(request, 200, order)
        if method == "DELETE" and path == "/positions":
            out = []
            for symbol in list(self.positions):
                qty = self.positions[symbol]["qty"]
                order = self._order(
                    {"symbol": symbol, "side": "sell", "qty": qty, "type": "market"}, "accepted"
                )
                self.orders[order["id"]] = order
                self.by_client[order["client_order_id"]] = order["id"]
                self._fill(order, qty)
                out.append({"order_id": order["id"], "status": 200, "symbol": symbol, "body": order})
            return self._response(request, 207, out)
        return self._error(request, 404, 40400000, f"no fake route for {method} {path}")

    def close(self) -> None:
        return None
