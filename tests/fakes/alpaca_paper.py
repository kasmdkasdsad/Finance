"""A stateful stand-in for Alpaca's *paper* trading API, mounted under the real ``alpaca-py`` SDK.

It is a ``requests`` transport adapter: the SDK builds real HTTP requests (URL, headers, JSON body) and
parses real JSON responses, so tests exercise the SDK and QuantPulse's provider end to end without the
network. It refuses any host other than ``paper-api.alpaca.markets``.

Behaviour per symbol (``fill_mode``): ``fill`` (immediately, at the limit price or the current price),
``partial`` (half the quantity), ``accept`` (rests as an open order), ``reject`` (403 insufficient buying
power) and ``timeout`` (the order IS recorded, then the connection times out — an ambiguous outcome).

Options (OCC symbols) are traded in contracts of 100 shares: single-leg orders with a position intent and
multi-leg (``mleg``) orders whose legs fill together at the net limit price. Like Alpaca, it enforces the
account's options level (1 covered calls / cash-secured puts, 2 long options, 3 spreads), refuses naked
short calls and uncovered puts, and :meth:`FakeAlpacaPaper.expire` settles contracts at expiration (in the
money: exercised or assigned into shares; out of the money: removed worthless).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from requests.adapters import BaseAdapter

PAPER_HOST = "paper-api.alpaca.markets"
OCC = re.compile(r"^(?P<root>[A-Z0-9.]{1,6})(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def occ(symbol: str) -> dict[str, Any] | None:
    m = OCC.match(symbol)
    if m is None:
        return None
    y = m["ymd"]
    return {
        "root": m["root"],
        "expiration": date(2000 + int(y[:2]), int(y[2:4]), int(y[4:])),
        "kind": "call" if m["cp"] == "C" else "put",
        "strike": int(m["strike"]) / 1000,
    }


def multiplier(symbol: str) -> int:
    return 100 if OCC.match(symbol) else 1


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
        self.outage = False  # every request fails as if the network (or Alpaca) were down
        self.buying_power_override: float | None = None  # e.g. a margin account's buying power
        self.bodies: list[dict[str, Any]] = []  # JSON body of every POST /v2/orders, as the SDK sent it
        self.options_level = 3  # 0 none, 1 covered calls / cash-secured puts, 2 long options, 3 spreads
        self.settlements: list[dict[str, Any]] = []  # what expiration did (exercises, assignments, worthless)

    # ------------------------------------------------------------------ helpers
    def now(self) -> datetime:
        return self.clock.now() if self.clock is not None else datetime.now(UTC)

    def price(self, symbol: str) -> float:
        if symbol in self.prices:
            return self.prices[symbol]
        pos = self.positions.get(symbol)
        return pos["avg"] if pos else 100.0

    def value(self, symbol: str) -> float:
        return self.positions[symbol]["qty"] * self.price(symbol) * multiplier(symbol)

    def equity(self) -> float:
        return self.cash + sum(self.value(s) for s in self.positions)

    def hold(self, symbol: str, qty: float, avg: float, price: float | None = None) -> None:
        """Seed a position (as if bought earlier; a negative quantity is a short option)."""
        self.positions[symbol] = {"qty": qty, "avg": avg}
        self.prices[symbol] = price if price is not None else avg
        self.cash -= qty * avg * multiplier(symbol)
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
        long_mv = sum(max(self.value(s), 0.0) for s in self.positions)
        short_mv = sum(min(self.value(s), 0.0) for s in self.positions)
        equity = self.cash + long_mv + short_mv
        return {
            "id": "904837e3-3b76-47ec-b432-046db621571b",
            "account_number": "PA3TESTPAPER1",
            "status": "ACTIVE",
            "currency": "USD",
            "buying_power": str(
                self.buying_power_override
                if self.buying_power_override is not None
                else max(self.cash, 0.0) * 2
            ),
            "regt_buying_power": str(max(self.cash, 0.0) * 2),
            "cash": str(self.cash),
            "portfolio_value": str(equity),
            "equity": str(equity),
            "last_equity": str(self.last_equity),
            "long_market_value": str(long_mv),
            "short_market_value": str(short_mv),
            "pattern_day_trader": False,
            "trading_blocked": self.blocked,
            "transfers_blocked": False,
            "account_blocked": False,
            "trade_suspended_by_user": False,
            "shorting_enabled": False,
            "multiplier": "2",
            "daytrade_count": 0,
            "created_at": "2026-01-02T15:00:00Z",
            "options_approved_level": self.options_level,
            "options_trading_level": self.options_level,
            "options_buying_power": str(max(self.cash - self._reserved(), 0.0)),
        }

    def _reserved(self) -> float:
        """Cash held against short puts (cash-secured) and the width of credit spreads."""
        out = 0.0
        for sym, p in self.positions.items():
            c = occ(sym)
            if c is not None and p["qty"] < 0 and c["kind"] == "put":
                out += c["strike"] * 100 * -p["qty"]
        return out

    def _position(self, symbol: str) -> dict[str, Any]:
        p = self.positions[symbol]
        price = self.price(symbol)
        m = multiplier(symbol)
        mv = p["qty"] * price * m
        cost = p["qty"] * p["avg"] * m
        last = self.lastday.get(symbol, price)
        return {
            "asset_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, symbol)),
            "symbol": symbol,
            "exchange": "" if m > 1 else "NASDAQ",
            "asset_class": "us_option" if m > 1 else "us_equity",
            "avg_entry_price": str(p["avg"]),
            "qty": str(p["qty"]),
            "qty_available": str(p["qty"]),
            "side": "long" if p["qty"] >= 0 else "short",
            "market_value": str(mv),
            "cost_basis": str(cost),
            "unrealized_pl": str(mv - cost),
            "unrealized_plpc": str((mv - cost) / abs(cost) if cost else 0.0),
            "unrealized_intraday_pl": str((price - last) * p["qty"] * m),
            "unrealized_intraday_plpc": str(price / last - 1 if last else 0.0),
            "current_price": str(price),
            "lastday_price": str(last),
            "change_today": str(price / last - 1 if last else 0.0),
        }

    def _order(self, body: dict[str, Any], status: str) -> dict[str, Any]:
        if body.get("order_class") == "mleg":
            return self._mleg_order(body, status)
        now = _iso(self.now())
        kind = body.get("type", "market")
        option = occ(body["symbol"]) is not None
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
            "asset_class": "us_option" if option else "us_equity",
            "position_intent": body.get("position_intent"),
            "notional": str(body["notional"]) if body.get("notional") is not None else None,
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

    def _mleg_order(self, body: dict[str, Any], status: str) -> dict[str, Any]:
        """A multi-leg order as Alpaca reports it: no symbol, side or asset class of its own; legs nested."""
        now = _iso(self.now())
        parent = str(uuid.uuid4())
        legs = [
            {
                **self._order(
                    {
                        "symbol": leg["symbol"],
                        "side": leg["side"],
                        "qty": float(leg["ratio_qty"]) * float(body["qty"]),
                        "type": "limit",
                        "position_intent": leg.get("position_intent"),
                    },
                    status,
                ),
                "order_class": "mleg",
                "type": None,
                "order_type": None,
                "limit_price": None,
                "ratio_qty": str(leg["ratio_qty"]),
            }
            for leg in body["legs"]
        ]
        return {
            "id": parent,
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
            "asset_id": "",
            "symbol": "",
            "asset_class": "",
            "notional": None,
            "qty": str(body["qty"]),
            "filled_qty": "0",
            "filled_avg_price": None,
            "order_class": "mleg",
            "order_type": "limit",
            "type": "limit",
            "side": "",
            "position_intent": "",
            "time_in_force": body.get("time_in_force", "day"),
            "limit_price": str(body["limit_price"]),
            "stop_price": None,
            "status": status,
            "extended_hours": False,
            "legs": legs,
            "trail_percent": None,
            "trail_price": None,
            "hwm": None,
        }

    def _move(self, symbol: str, signed_qty: float, price: float) -> None:
        """Change a position by ``signed_qty`` (buy +, sell −) at ``price``; cash moves the other way."""
        pos = self.positions.setdefault(symbol, {"qty": 0.0, "avg": price})
        before = pos["qty"]
        after = before + signed_qty
        if before == 0 or ((before > 0) != (after > 0) and after != 0):
            pos["avg"] = price  # opened, or flipped through zero
        elif abs(after) > abs(before):
            pos["avg"] = (abs(before) * pos["avg"] + abs(signed_qty) * price) / abs(after)
        pos["qty"] = after
        self.cash -= signed_qty * price * multiplier(symbol)
        if abs(pos["qty"]) <= 1e-9:
            del self.positions[symbol]
        self.prices.setdefault(symbol, price)

    def _fill(self, order: dict[str, Any], qty: float) -> None:
        if order.get("order_class") == "mleg":
            self._fill_mleg(order, qty)
            return
        symbol, side = order["symbol"], order["side"]
        price = float(order["limit_price"]) if order["limit_price"] else self.price(symbol)
        self._move(symbol, qty if side == "buy" else -qty, price)
        filled = float(order["filled_qty"]) + qty
        prev_avg = float(order["filled_avg_price"] or 0.0)
        order["filled_avg_price"] = str((prev_avg * float(order["filled_qty"]) + price * qty) / filled)
        order["filled_qty"] = str(filled)
        order["status"] = "filled" if math.isclose(filled, float(order["qty"])) else "partially_filled"
        stamp = _iso(self.now())
        order["updated_at"] = stamp
        if order["status"] == "filled":
            order["filled_at"] = stamp

    def _fill_mleg(self, order: dict[str, Any], qty: float) -> None:
        """Fill ``qty`` units of a multi-leg order at its net limit price: every leg at its current price,
        the first leg adjusted so the legs' net equals the limit (never below a cent)."""
        legs = order["legs"]
        sign = [1 if leg["side"] == "buy" else -1 for leg in legs]
        ratio = [float(leg["ratio_qty"]) for leg in legs]
        prices = [self.price(leg["symbol"]) for leg in legs]
        net = sum(s * r * p for s, r, p in zip(sign, ratio, prices, strict=True))
        prices[0] = max(0.01, prices[0] + (float(order["limit_price"]) - net) / (sign[0] * ratio[0]))
        stamp = _iso(self.now())
        for leg, s_, r, p in zip(legs, sign, ratio, prices, strict=True):
            self._move(leg["symbol"], s_ * r * qty, p)
            done = float(leg["filled_qty"]) + r * qty
            leg["filled_avg_price"] = str(p)
            leg["filled_qty"] = str(done)
            leg["status"] = "filled" if math.isclose(done, float(leg["qty"])) else "partially_filled"
            leg["updated_at"] = stamp
            leg["filled_at"] = stamp if leg["status"] == "filled" else None
        filled = float(order["filled_qty"]) + qty
        net_fill = sum(s * r * p for s, r, p in zip(sign, ratio, prices, strict=True))
        order["filled_avg_price"] = str(round(net_fill, 4))
        order["filled_qty"] = str(filled)
        order["status"] = "filled" if math.isclose(filled, float(order["qty"])) else "partially_filled"
        order["updated_at"] = stamp
        if order["status"] == "filled":
            order["filled_at"] = stamp

    def _option_refusal(self, body: dict[str, Any]) -> str | None:
        """What Alpaca would refuse about an option order (level, coverage, closing what is not held)."""
        legs = body.get("legs") or [
            {"symbol": body["symbol"], "side": body["side"], "ratio_qty": 1,
             "position_intent": body.get("position_intent")}
        ]  # fmt: skip
        qty = float(body["qty"])
        if qty != int(qty):
            return "options are traded in whole contracts"
        if body.get("type") != "limit" and body.get("order_class") == "mleg":
            return "multi-leg orders must be limit orders"
        need = 3 if body.get("order_class") == "mleg" else 1
        for leg in legs:
            n = float(leg["ratio_qty"]) * qty
            c = occ(leg["symbol"])
            if c is None:
                return f"{leg['symbol']} is not an option contract"
            held = self.positions.get(leg["symbol"], {"qty": 0.0})["qty"]
            intent = leg.get("position_intent") or (
                "buy_to_open" if leg["side"] == "buy" else "sell_to_close"
            )
            if intent == "sell_to_close" and held < n - 1e-9:
                return f"insufficient qty available for order ({leg['symbol']})"
            if intent == "buy_to_close" and -held < n - 1e-9:
                return f"no short position to close ({leg['symbol']})"
            if intent == "buy_to_open" and body.get("order_class") != "mleg":
                need = max(need, 2)
        if body.get("order_class") == "mleg":
            for kind in ("call", "put"):
                of_kind = [x for x in legs if (occ(x["symbol"]) or {}).get("kind") == kind]
                short = sum(
                    float(x["ratio_qty"]) for x in of_kind if x.get("position_intent") == "sell_to_open"
                )
                long = sum(
                    float(x["ratio_qty"]) for x in of_kind if x.get("position_intent") == "buy_to_open"
                )
                if short > long:
                    return f"uncovered short {kind}s are not permitted"
        elif (body.get("position_intent") or "") == "sell_to_open":
            c = occ(body["symbol"])
            assert c is not None
            if c["kind"] == "call":
                shares = self.positions.get(c["root"], {"qty": 0.0})["qty"]
                covered = 0.0
                for sym, pos in self.positions.items():
                    o = occ(sym)
                    if o is not None and pos["qty"] < 0 and o["root"] == c["root"] and o["kind"] == "call":
                        covered -= pos["qty"]
                if shares < 100 * (covered + qty) - 1e-9:
                    return "uncovered short calls are not permitted (not enough shares)"
            elif self.cash - self._reserved() < c["strike"] * 100 * qty:
                return "insufficient options buying power for cash-secured put"
        if self.options_level < need:
            return f"account is not approved for this options strategy (level {self.options_level} < {need})"
        return None

    def expire(self, day: date, spots: dict[str, float]) -> list[dict[str, Any]]:
        """Settle every contract expiring on or before ``day`` against the underlying's closing price, as
        the OCC would: in the money by a cent or more is exercised (long) or assigned (short) into shares
        at the strike; anything else expires worthless."""
        out = []
        for sym in list(self.positions):
            c = occ(sym)
            if c is None or c["expiration"] > day:
                continue
            qty = self.positions[sym]["qty"]
            spot = spots[c["root"]]
            itm = (spot - c["strike"] if c["kind"] == "call" else c["strike"] - spot) >= 0.01
            del self.positions[sym]
            event = {"symbol": sym, "qty": qty, "spot": spot, "itm": itm}
            if itm:
                shares = 100 * qty * (1 if c["kind"] == "call" else -1)
                self.prices.setdefault(c["root"], spot)
                self._move(c["root"], shares, c["strike"])
                event["shares"] = shares
                event["kind"] = "exercised" if qty > 0 else "assigned"
            else:
                event["kind"] = "expired_worthless"
            out.append(event)
        self.settlements += out
        return out

    def assign(self, symbol: str, spot: float) -> dict[str, Any]:
        """Early assignment of a short option (American style: any day, usually deep in the money or before an
        ex-dividend date): the contracts disappear and the shares move at the strike."""
        c = occ(symbol)
        qty = self.positions[symbol]["qty"]
        assert c is not None and qty < 0, "only a short option can be assigned"
        del self.positions[symbol]
        shares = (
            100 * qty * (1 if c["kind"] == "call" else -1)
        )  # short call: shares delivered; short put: taken
        self.prices.setdefault(c["root"], spot)
        self._move(c["root"], shares, c["strike"])
        event = {
            "symbol": symbol,
            "qty": qty,
            "spot": spot,
            "itm": True,
            "shares": shares,
            "kind": "assigned",
        }
        self.settlements.append(event)
        return event

    def complete(self, client_order_id: str) -> None:
        """Fill whatever is left of an open order (e.g. a resting or partially filled one)."""
        order = self.orders[self.by_client[client_order_id]]
        left = float(order["qty"]) - float(order["filled_qty"])
        if left > 0:
            self._fill(order, left)

    def _submit(self, request: requests.PreparedRequest, body: dict[str, Any]) -> requests.Response:
        self.bodies.append(dict(body))
        cid = body.get("client_order_id")
        if cid and cid in self.by_client:
            return self._error(request, 422, 40010001, "client_order_id must be unique")
        if (body.get("qty") is None) == (body.get("notional") is None):
            return self._error(request, 422, 40010001, "qty or notional is required")
        if body.get("order_class") == "mleg" or occ(body.get("symbol") or "") is not None:
            return self._submit_option(request, body)
        symbol, side = body["symbol"], body["side"]
        if body.get("notional") is not None:
            body = {**body, "qty": round(float(body["notional"]) / self.price(symbol), 9)}
        qty = float(body["qty"])
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

    def _submit_option(self, request: requests.PreparedRequest, body: dict[str, Any]) -> requests.Response:
        if body.get("notional") is not None:
            return self._error(request, 422, 40010001, "options cannot be ordered by notional")
        refusal = self._option_refusal(body)
        if refusal is not None:
            return self._error(request, 403, 40310000, refusal)
        key = body.get("symbol") or body["legs"][0]["symbol"]
        mode = self.fill_mode.get(key, self.default_mode)
        if mode == "reject":
            return self._error(request, 403, 40310000, "insufficient options buying power")
        order = self._order(body, "accepted")
        self.orders[order["id"]] = order
        self.by_client[order["client_order_id"]] = order["id"]
        qty = float(body["qty"])
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
        for leg in order.get("legs") or []:
            if leg["status"] not in ("filled", "canceled", "expired", "rejected"):
                leg["status"] = "canceled"
                leg["canceled_at"] = leg["updated_at"] = order["canceled_at"]
        return True

    def _close(self, symbol: str, qty: float | None = None) -> dict[str, Any]:
        """Alpaca's close-position: a market order the other way (buy to close a short option)."""
        held = self.positions[symbol]["qty"]
        n = abs(qty) if qty is not None else abs(held)
        option = occ(symbol) is not None
        side = "sell" if held > 0 else "buy"
        intent = ("sell_to_close" if held > 0 else "buy_to_close") if option else None
        order = self._order(
            {"symbol": symbol, "side": side, "qty": n, "type": "market", "position_intent": intent},
            "accepted",
        )
        self.orders[order["id"]] = order
        self.by_client[order["client_order_id"]] = order["id"]
        self._fill(order, n)
        return order

    # ------------------------------------------------------------------ transport
    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        url = urlparse(request.url or "")
        if url.netloc != PAPER_HOST:
            raise AssertionError(f"QuantPulse must only talk to Alpaca PAPER, not {url.netloc}")
        self.log.append((request.method or "", url.path))
        self.auth_headers_seen.append(
            bool(request.headers.get("APCA-API-KEY-ID")) and bool(request.headers.get("APCA-API-SECRET-KEY"))
        )
        if self.outage:
            raise requests.exceptions.ConnectionError("simulated network interruption: Alpaca unreachable")
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
            symbol = unquote(path.split("/")[-1])
            if symbol not in self.positions:
                return self._error(request, 404, 40410000, "position not found")
            qty = (query.get("qty") or [None])[0]
            order = self._close(symbol, float(qty) if qty is not None else None)
            return self._response(request, 200, order)
        if method == "DELETE" and path == "/positions":
            out = []
            # short option legs first (buy them back), then everything else, as a careful broker would
            for symbol in sorted(self.positions, key=lambda x: self.positions[x]["qty"] >= 0):
                if symbol not in self.positions:
                    continue
                order = self._close(symbol)
                out.append({"order_id": order["id"], "status": 200, "symbol": symbol, "body": order})
            return self._response(request, 207, out)
        return self._error(request, 404, 40400000, f"no fake route for {method} {path}")

    def close(self) -> None:
        return None
