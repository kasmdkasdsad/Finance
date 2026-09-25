"""Email delivery (SMTP) and rendering of the daily picks digest."""

from __future__ import annotations

import asyncio
import html
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from quantpulse.config import Settings
from quantpulse.core.errors import QuantPulseError
from quantpulse.schemas.common import DataStatus
from quantpulse.schemas.picks import DailyPicks


class NotificationError(QuantPulseError):
    """Email could not be sent (maps to HTTP 502)."""


class NotConfiguredError(QuantPulseError):
    """SMTP settings are missing (maps to HTTP 503)."""


class SyntheticDataRefused(QuantPulseError):
    """Refused to email picks computed from synthetic prices (maps to HTTP 409)."""


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:+.1f}%"


def _factor_pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:+.1f}%"


def render_picks_email(picks: DailyPicks) -> tuple[str, str, str]:
    """Return ``(subject, plain_text, html)`` for a picks digest."""
    synthetic = picks.data_status is DataStatus.SYNTHETIC
    prefix = "[SYNTHETIC DATA] " if synthetic else ""
    top = picks.top_pick
    headline = f"{top.symbol} {top.rating}/10" if top else "no picks"
    subject = f"{prefix}QuantPulse daily picks for {picks.trading_day}: top pick {headline}"

    lines = [
        f"QuantPulse daily picks — {picks.trading_day}",
        f"Data status: {picks.data_status.value.upper()}   Screened {picks.screened} of {picks.universe_size} symbols",
        "",
    ]
    if synthetic:
        lines += ["WARNING: live market data was unavailable; these numbers are SYNTHETIC and not real.", ""]
    lines.append(f"{'#':>2}  {'Symbol':<7} {'Rating':>6}  {'Price':>10} {'Day':>7}  Drivers")
    for p in picks.picks:
        lines.append(
            f"{p.rank:>2}  {p.symbol:<7} {p.rating:>4}/10  {p.price:>10,.2f} {_pct(p.change_percent):>7}  "
            f"{', '.join(p.drivers) or '—'}"
        )
    lines += ["", "Methodology: " + picks.methodology, "", picks.disclaimer]
    text = "\n".join(lines)

    rows = "".join(
        f"<tr><td>{p.rank}</td><td><b>{html.escape(p.symbol)}</b><br><small>{html.escape(p.name or '')}</small></td>"
        f"<td style='text-align:center'><b>{p.rating}</b>/10</td><td style='text-align:right'>{p.price:,.2f}</td>"
        f"<td style='text-align:right'>{_pct(p.change_percent)}</td>"
        f"<td style='text-align:right'>{_factor_pct(p.factors.momentum_12_1)}</td>"
        f"<td>{html.escape(', '.join(p.drivers) or '—')}</td><td>{p.data_status.value}</td></tr>"
        for p in picks.picks
    )
    banner = (
        "<p style='background:#fde2e1;color:#8a1c1c;padding:8px;border-radius:6px'><b>Synthetic data:</b> live "
        "market data was unavailable, so these numbers are simulated and not real.</p>"
        if synthetic
        else ""
    )
    body_html = f"""<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2328">
<h2 style="margin-bottom:4px">QuantPulse daily picks — {html.escape(picks.trading_day)}</h2>
<p style="margin-top:0;color:#57606a">Data status: <b>{picks.data_status.value.upper()}</b> · screened {picks.screened} of {picks.universe_size} symbols</p>
{banner}
<table cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:14px" border="1" bordercolor="#d0d7de">
<thead style="background:#f6f8fa"><tr><th>#</th><th>Symbol</th><th>Rating</th><th>Price</th><th>Day</th><th>12-1 mom.</th><th>Drivers</th><th>Data</th></tr></thead>
<tbody>{rows}</tbody></table>
<p style="font-size:12px;color:#57606a">{html.escape(picks.methodology)}</p>
<p style="font-size:12px;color:#57606a"><i>{html.escape(picks.disclaimer)}</i></p>
</body></html>"""
    return subject, text, body_html


class EmailNotifier:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    @property
    def configured(self) -> bool:
        return self._s.smtp_configured

    def _build(self, recipients: list[str], subject: str, text: str, body_html: str) -> EmailMessage:
        msg = EmailMessage()
        msg["From"] = self._s.email_from or ""
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg["Date"] = formatdate(localtime=False)
        msg["Message-ID"] = make_msgid(domain=(self._s.email_from or "quantpulse.local").split("@")[-1])
        msg.set_content(text)
        msg.add_alternative(body_html, subtype="html")
        return msg

    def _send_sync(self, msg: EmailMessage) -> None:
        host, port = self._s.smtp_host or "", self._s.smtp_port
        password = self._s.smtp_password.get_secret_value() if self._s.smtp_password else None
        context = ssl.create_default_context()
        try:
            if self._s.smtp_security == "ssl":
                server: smtplib.SMTP = smtplib.SMTP_SSL(host, port, timeout=30, context=context)
            else:
                server = smtplib.SMTP(host, port, timeout=30)
            with server:
                if self._s.smtp_security == "starttls":
                    server.starttls(context=context)
                if self._s.smtp_username and password:
                    server.login(self._s.smtp_username, password)
                refused = server.send_message(msg)
            if refused:
                raise NotificationError(f"recipients refused: {', '.join(refused)}")
        except (smtplib.SMTPException, OSError) as exc:
            raise NotificationError(f"SMTP delivery failed: {type(exc).__name__}: {exc}") from exc

    async def send(self, recipients: list[str], subject: str, text: str, body_html: str) -> None:
        if not self.configured:
            raise NotConfiguredError(
                "email is not configured: set QP_SMTP_HOST and QP_EMAIL_FROM (and credentials)"
            )
        if not recipients:
            raise NotConfiguredError("no recipients: pass 'recipients' or set QP_PICKS_RECIPIENTS")
        msg = self._build(recipients, subject, text, body_html)
        await asyncio.to_thread(self._send_sync, msg)
