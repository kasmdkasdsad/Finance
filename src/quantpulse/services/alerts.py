"""Alerts to a phone, at no cost: ntfy push notifications, a webhook, and a dead-man's-switch heartbeat.

Every channel is optional; with none configured, alerts are only recorded (the dashboard's event log shows
them). Nothing here can change what the Brain does — alerts report, they never decide.

* **ntfy** (``QP_ALERT_NTFY_URL``) — the free ntfy app on the iPhone subscribes to a long random topic;
* **webhook** (``QP_ALERT_WEBHOOK_URL``) — a JSON POST that Slack and Discord incoming webhooks accept;
* **heartbeat** (``QP_HEARTBEAT_URL``, e.g. healthchecks.io's free plan) — pinged while QuantPulse is
  healthy and ``<url>/fail`` when it is not. When the server itself dies the pings stop, and the heartbeat
  service sends the alert: the one failure a server cannot report about itself.

The same alert (kind and subject) is not repeated within ``QP_ALERT_COOLDOWN_MINUTES``. A failed delivery
is logged and never raised: alerting must not be able to disturb trading or its safeguards. Messages never
contain a secret (and every log line is masked anyway).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import httpx

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database

logger = logging.getLogger(__name__)
Severity = Literal["info", "warning", "critical"]
TIMEOUT = 8.0
HEARTBEAT_EVERY = timedelta(minutes=5)
PRIORITY = {"info": "default", "warning": "high", "critical": "urgent"}
TAGS = {"info": "information_source", "warning": "warning", "critical": "rotating_light"}


@dataclass(frozen=True, slots=True)
class Alert:
    kind: str
    title: str
    message: str
    severity: Severity = "warning"
    subject: str = ""


class AlertService:
    def __init__(self, settings: Settings, clock: Clock, db: Database, http: httpx.AsyncClient) -> None:
        self._s = settings
        self._clock = clock
        self._db = db
        self._http = http
        self._last: dict[tuple[str, str], datetime] = {}
        self._last_beat: tuple[datetime, bool] | None = None
        # the last heartbeat ping and the last one that reported "healthy" and was delivered (for the status)
        self.last_heartbeat: dict[str, Any] | None = None
        self.last_healthy_heartbeat_at: datetime | None = None
        self.sent: list[dict[str, Any]] = []  # the most recent alerts (for the dashboard), newest last

    @property
    def channels(self) -> list[str]:
        return [
            name
            for name, value in (("ntfy", self._s.alert_ntfy_url), ("webhook", self._s.alert_webhook_url))
            if value is not None
        ]

    async def send(self, alert: Alert, *, force: bool = False) -> bool:
        """Deliver ``alert`` unless the same one went out within the cool-down; ``True`` when it was sent."""
        now = self._clock.now()
        key = (alert.kind, alert.subject)
        cooldown = timedelta(minutes=self._s.alert_cooldown_minutes)
        if not force and key in self._last and now - self._last[key] < cooldown:
            return False
        self._last[key] = now
        delivered = []
        if self._s.alert_ntfy_url is not None:
            delivered.append(("ntfy", await self._ntfy(alert)))
        if self._s.alert_webhook_url is not None:
            delivered.append(("webhook", await self._webhook(alert)))
        record = {
            "at": now.isoformat(),
            "kind": alert.kind,
            "severity": alert.severity,
            "title": alert.title,
            "message": alert.message,
            "delivered": dict(delivered),
        }
        self.sent = [*self.sent[-49:], record]
        try:  # recorded with the trading events: the dashboard's history of what was reported
            async with self._db.session() as s:
                await repo.add_trading_event(
                    s,
                    "alert",
                    f"{alert.title}: {alert.message}"[:1000],
                    now,
                    symbol=alert.subject[:16] or None,
                    details={
                        "kind": alert.kind,
                        "severity": alert.severity,
                        "delivered": record["delivered"],
                    },
                )
        except Exception:  # the database may be the very thing that is down
            logger.warning("alert %s could not be recorded in the database", alert.kind)
        logger.warning("ALERT [%s] %s: %s", alert.severity, alert.title, alert.message)
        return True

    async def heartbeat(self, healthy: bool) -> bool | None:
        """Ping the heartbeat URL (``<url>/fail`` when unhealthy): every few minutes, and at once when health
        changes. ``None`` when not due or not configured."""
        if self._s.heartbeat_url is None:
            return None
        now = self._clock.now()
        if self._last_beat is not None:
            at, was = self._last_beat
            if was == healthy and now - at < HEARTBEAT_EVERY:
                return None
        url = self._s.heartbeat_url.get_secret_value().rstrip("/")
        self._last_beat = (now, healthy)
        try:
            response = await self._http.get(url if healthy else f"{url}/fail", timeout=TIMEOUT)
            delivered = response.status_code < 400
        except httpx.HTTPError as exc:
            logger.warning("heartbeat ping failed (%s)", type(exc).__name__)
            delivered = False
        if delivered and healthy:
            self.last_healthy_heartbeat_at = now
        self.last_heartbeat = {
            "at": now.isoformat(),
            "healthy": healthy,
            "delivered": delivered,
            "last_healthy_delivered_at": self.last_healthy_heartbeat_at.isoformat()
            if self.last_healthy_heartbeat_at
            else None,
        }
        return delivered

    async def _ntfy(self, alert: Alert) -> bool:
        assert self._s.alert_ntfy_url is not None
        headers = {
            "Title": f"QuantPulse: {alert.title}"[:200].encode("ascii", "replace").decode(),
            "Priority": PRIORITY[alert.severity],
            "Tags": TAGS[alert.severity],
        }
        try:
            response = await self._http.post(
                self._s.alert_ntfy_url.get_secret_value(),
                content=alert.message.encode("utf-8"),
                headers=headers,
                timeout=TIMEOUT,
            )
            return response.status_code < 400
        except httpx.HTTPError as exc:
            logger.warning("ntfy alert failed (%s)", type(exc).__name__)
            return False

    async def _webhook(self, alert: Alert) -> bool:
        assert self._s.alert_webhook_url is not None
        text = f"QuantPulse [{alert.severity.upper()}] {alert.title}: {alert.message}"
        body = {"text": text, "content": text[:1900], "kind": alert.kind, "severity": alert.severity}
        try:
            response = await self._http.post(
                self._s.alert_webhook_url.get_secret_value(),
                content=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT,
            )
            return response.status_code < 400
        except httpx.HTTPError as exc:
            logger.warning("webhook alert failed (%s)", type(exc).__name__)
            return False
