"""Options agent: what the listed options market is pricing and trading for each focus symbol.

Reads the option-chain metrics gathered in :mod:`quantpulse.brain.research_data` (live or cached chains
only): at-the-money implied volatility and its premium over realised volatility, the term structure (an
inverted curve prices near-term stress or an event), put/call skew (demand for downside protection),
put/call volume and open-interest ratios, and volume-to-open-interest turnover (unusual activity). The
implied move over the horizon is reported for sizing and event risk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion
from .base import Agent, symbols_only
from .common import opinion, sgn, squash, symbol_quality

HORIZON = 21


class OptionsAgent(Agent):
    spec = AgentSpec(
        id="options",
        source="options",
        failure="skips without live option chains (never synthetic); its vote is listed as missing",
        name="Options",
        description="Implied volatility level and term structure, put/call skew, put/call flow and unusual "
        "options turnover from live option chains.",
        family=AgentFamily.SPECIALIST,
        capabilities=("implied_volatility", "term_structure", "skew", "put_call_ratio", "unusual_activity"),
        inputs=("options",),
        subjects=("symbol",),
        priority=35,
        horizon_days=HORIZON,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return None if ctx.options else "no live option chains this cycle"

    def subjects(self, ctx: BrainContext) -> list[str]:
        return [s for s in ctx.focus if s in ctx.options]

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        return [self._one(ctx, s) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str) -> Opinion:
        m = ctx.options.get(s)
        if not m:
            return self.abstain(s, "no live option chain", ["option chain"])
        q = symbol_quality(ctx, s)
        ev: list[Evidence] = []
        parts: dict[str, float] = {}
        iv, rv = m.get("atm_iv"), ctx.ind(s, "rv63")
        if iv is not None:
            premium = iv / rv - 1 if rv else None
            ev.append(
                Evidence(
                    "atm_iv",
                    round(iv, 4),
                    f"at-the-money implied volatility {iv:.0%}"
                    + (f" ({premium:+.0%} vs 3-month realised)" if premium is not None else ""),
                    0,
                    0.6,
                    source="options",
                    quality=q,
                )
            )
        slope = m.get("term_slope")
        if slope is not None and slope < -0.03:
            parts["term"] = -0.3
            ev.append(
                Evidence(
                    "term_slope",
                    round(slope, 4),
                    f"inverted term structure ({slope:+.1%}): near-term stress or an event is priced",
                    -1,
                    0.6,
                    source="options",
                    quality=q,
                )
            )
        skew = m.get("skew")
        if skew is not None:
            parts["skew"] = -squash(skew - 0.04, 0.06)
            ev.append(
                Evidence(
                    "skew",
                    round(skew, 4),
                    f"put-call skew {skew:+.1%} ({'puts rich: hedging demand' if skew > 0.08 else 'normal' if skew > 0 else 'calls rich'})",
                    sgn(parts["skew"]),
                    0.5,
                    source="options",
                    quality=q,
                )
            )
        pcv = m.get("pc_volume")
        turnover = m.get("volume_oi")
        if pcv is not None:
            flow = -squash(math.log(max(pcv, 1e-3)), 0.7)  # log ratio: 1.0 is balanced
            unusual = turnover is not None and turnover > 1.0
            parts["flow"] = flow * (1.5 if unusual else 1.0)
            ev.append(
                Evidence(
                    "pc_volume",
                    round(pcv, 2),
                    f"put/call volume {pcv:.2f}"
                    + (f" on unusual turnover ({turnover:.1f}× open interest)" if unusual else ""),
                    sgn(flow),
                    0.7 if unusual else 0.4,
                    source="options",
                    quality=q,
                )
            )
        pco = m.get("pc_oi")
        if pco is not None:
            ev.append(
                Evidence(
                    "pc_oi",
                    round(pco, 2),
                    f"put/call open interest {pco:.2f}",
                    0,
                    0.3,
                    source="options",
                    quality=q,
                )
            )
        if not parts and iv is None:
            return self.abstain(s, "the option chain has no usable quotes", ["implied volatility", "volume"])
        score = squash(sum(parts.values()), 1.0) if parts else 0.0
        implied_move = iv * math.sqrt(HORIZON / 252) if iv is not None else None
        confidence = 0.2 + 0.3 * min(len(parts) / 3, 1.0)
        label = (
            "bearish positioning"
            if score < -0.15
            else "bullish positioning"
            if score > 0.15
            else "balanced positioning"
        )
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s} options show {label}: "
            + ", ".join(e.detail for e in sorted(ev, key=lambda e: -e.strength)[:3]),
            ev,
            quality=q,
            used=[f"option chain ({m.get('source')}, {m.get('status')})"],
            missing=[k for k in ("atm_iv", "skew", "pc_volume", "term_slope") if m.get(k) is None],
            meta={
                "implied_move": round(implied_move, 4) if implied_move is not None else None,
                "atm_iv": iv,
                "unusual_activity": bool(turnover is not None and turnover > 1.0),
            },
        )
