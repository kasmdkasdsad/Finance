"""Catalyst agent: earnings — the scheduled event that moves single stocks most.

From the earnings calendar and past reactions gathered in :mod:`quantpulse.brain.research_data` (SEC 8-K
item 2.02 filings, the vendor calendar when configured, reactions measured on the cycle's own closes):

* **Event risk** — days to the next release and the stock's typical reaction (root-mean-square of its
  last releases). It is posted to working memory so the decision step does not open a position just
  before a release (the ``QP_TRADING_EARNINGS_BLACKOUT_DAYS`` policy, at least ``QP_BRAIN_EARNINGS_CAUTION_DAYS``).
* **Post-earnings drift** — a large abnormal reaction in the last two months, measured against the stock's
  typical move, tends to continue; the view decays as the release ages. Without a recent surprise the
  agent says so and abstains from a direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion
from .base import Agent, symbols_only
from .common import opinion, sgn, squash, symbol_quality

DRIFT_WINDOW = 60  # calendar days a post-earnings drift view lasts


class CatalystAgent(Agent):
    spec = AgentSpec(
        id="catalyst",
        source="events",
        failure="skips without an earnings calendar; the earnings caution still uses the trading data's dates",
        name="Catalyst (earnings)",
        description="Next earnings date and typical reaction (event risk), and post-earnings drift after a "
        "large surprise.",
        family=AgentFamily.SPECIALIST,
        capabilities=("earnings_calendar", "event_risk", "post_earnings_drift"),
        inputs=("events",),
        subjects=("symbol",),
        priority=30,
        horizon_days=21,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        has_feature = ctx.model is not None and "earn_reaction" in ctx.model.features.columns
        return None if ctx.events or has_feature else "no earnings calendar or reaction history this cycle"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        out = [self._one(ctx, s) for s in symbols_only(subjects)]
        ctx.working.post(
            "event_risk",
            {
                o.subject: {
                    "days_to_earnings": o.meta["days_to_next"],
                    "typical_move": o.meta.get("typical_move"),
                }
                for o in out
                if o.meta.get("days_to_next") is not None
            },
        )
        return out

    def _one(self, ctx: BrainContext, s: str) -> Opinion:
        e = ctx.events.get(s) or {}
        q = symbol_quality(ctx, s)
        ev: list[Evidence] = []
        days_to, typical = e.get("days_to_next"), e.get("typical_move")
        if e.get("next") and days_to is None:
            days_to = (date.fromisoformat(e["next"]) - ctx.as_of.date()).days
        meta = {
            "days_to_next": days_to,
            "typical_move": typical,
            "next": e.get("next"),
            "next_source": e.get("next_source"),
        }
        if days_to is not None:
            ev.append(
                Evidence(
                    "days_to_earnings",
                    days_to,
                    f"earnings in {days_to} days ({e.get('next')}, {e.get('next_source')})"
                    + (f"; typical reaction ±{typical:.1%}" if typical else ""),
                    0,
                    0.8 if days_to <= 10 else 0.4,
                    source="sec_edgar" if e.get("next_source") == "estimated" else "calendar",
                    quality=q,
                )
            )
        score, confidence = 0.0, 0.0
        since, abnormal = e.get("days_since_last"), e.get("last_abnormal")
        surprise = None
        if since is not None and abnormal is not None and since <= DRIFT_WINDOW:
            daily = ctx.ind(s, "rv63")
            unit = typical or ((daily / 15.87) if daily else None)  # annual → daily volatility
            if unit:
                surprise = abnormal / unit
                decay = 1.0 - since / DRIFT_WINDOW
                score = 0.7 * squash(surprise, 2.0) * decay if abs(surprise) >= 1.0 else 0.0
                confidence = (0.2 + 0.35 * min(abs(surprise) / 3, 1.0)) * (0.4 + 0.6 * decay)
                ev.append(
                    Evidence(
                        "earnings_surprise",
                        round(abnormal, 4),
                        f"last release {since} days ago: {abnormal:+.1%} beyond the market ({surprise:+.1f}× its typical move)",
                        sgn(score),
                        0.8,
                        quality=q,
                    )
                )
        elif ctx.feature(s, "earn_reaction") is not None:
            er = ctx.feature(s, "earn_reaction")
            assert er is not None
            ev.append(
                Evidence(
                    "earn_reaction",
                    round(er, 2),
                    f"latest earnings reaction {er:+.1f} volatility units (stock model)",
                    sgn(er),
                    0.3,
                    source="stock_model",
                    quality=q,
                )
            )
        if not ev:
            return self.abstain(s, "no earnings calendar or reaction history", ["earnings events"])
        meta["surprise"] = round(surprise, 2) if surprise is not None else None
        thesis = "; ".join(x.detail for x in ev)
        if score:
            thesis = f"{s} post-earnings drift {'up' if score > 0 else 'down'}: " + thesis
        else:
            thesis = f"{s} no drift signal: " + thesis
        return opinion(
            self,
            s,
            score,
            confidence,
            thesis,
            ev,
            quality=q,
            used=["earnings events", "daily closes"],
            meta=meta,
            directional=bool(score),  # event risk without a drift signal is context, not a vote
        )
