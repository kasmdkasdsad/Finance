"""Fundamental and valuation agents, both built on the stock model's point-in-time fundamentals (SEC XBRL
frames with a publication lag; see :mod:`quantpulse.domain.fundamental_factors`) — nothing is re-fetched or
re-derived here. Each metric is ranked against the model's whole universe (and, for valuation, against the
company's sector peers when there are enough of them).

* **Fundamental** (quality, about a quarter): gross profitability, return on equity, low accruals (earnings
  backed by cash), conservative asset growth, and the latest earnings reaction.
* **Valuation** (about a quarter): earnings yield, free-cash-flow yield and book-to-market; cheapness is
  tempered by a value-trap check (cheap, weak momentum and poor quality) and expensive names with strong
  momentum are not called bearish on price alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion
from .base import Agent, symbols_only
from .common import agreement, cross_section_z, opinion, sgn, squash, symbol_quality

# metric -> (sign: +1 higher is better, label)
QUALITY = {
    "gross_profitability": (1, "gross profit / assets"),
    "roe": (1, "return on equity"),
    "accruals": (-1, "accruals (lower = earnings backed by cash)"),
    "asset_growth": (-1, "asset growth (lower = conservative)"),
    "earn_reaction": (1, "latest earnings reaction"),
}
VALUE = {
    "earnings_yield": (1, "earnings yield"),
    "fcf_yield": (1, "free-cash-flow yield"),
    "book_to_market": (1, "book-to-market"),
}
MIN_PEERS = 5


def _model_unavailable(ctx: BrainContext, cols: Sequence[str]) -> str | None:
    if ctx.model is None:
        return "the stock model (and its point-in-time fundamentals) is not available this cycle"
    have = [c for c in cols if c in ctx.model.features.columns]
    if len(have) < 2:
        return "the stock model has no fundamental features"
    return None


def _peer_sector(ctx: BrainContext, symbol: str) -> str | None:
    score = ctx.model.live.get(symbol) if ctx.model is not None else None
    return (score.sector_label or score.sector if score else None) or ctx.sectors.get(symbol)


class FundamentalAgent(Agent):
    spec = AgentSpec(
        id="fundamental",
        source="fundamentals",
        failure="skips without a completed stock-model run; its vote is listed as missing",
        name="Fundamental quality",
        description="Profitability, earnings quality (accruals), balance-sheet growth and the latest earnings "
        "reaction, ranked against the model universe (point-in-time SEC data).",
        family=AgentFamily.SPECIALIST,
        capabilities=("profitability", "earnings_quality", "asset_growth", "earnings_reaction"),
        inputs=("model",),
        subjects=("symbol",),
        priority=40,
        horizon_days=63,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return _model_unavailable(ctx, list(QUALITY))

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        assert ctx.model is not None
        cols = [c for c in QUALITY if c in ctx.model.features.columns]
        z = cross_section_z(ctx.model.features[cols])
        return [self._one(ctx, s, z) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str, z: pd.DataFrame) -> Opinion:
        if s not in z.index:
            return self.abstain(s, "not covered by the stock model", ["fundamentals"])
        row = z.loc[s].dropna()
        if len(row) < 2:
            return self.abstain(s, "fewer than two fundamental metrics on file", list(QUALITY))
        q = symbol_quality(ctx, s)
        ev: list[Evidence] = []
        parts: dict[str, float] = {}
        for col, zval in row.items():
            sign, label = QUALITY[str(col)]
            parts[str(col)] = sign * float(zval)
            raw = ctx.feature(s, str(col))
            ev.append(
                Evidence(
                    str(col),
                    round(raw, 4) if raw is not None else None,
                    f"{label} {raw:+.3f} ({sign * float(zval):+.1f}σ vs the universe)"
                    if raw is not None
                    else label,
                    sgn(sign * float(zval)),
                    min(abs(float(zval)) / 3, 1.0),
                    source="sec_xbrl" if col != "earn_reaction" else "computed",
                    quality=q,
                )
            )
        composite = sum(parts.values()) / len(parts)
        score = squash(composite, 1.0)
        coverage = len(parts) / len(QUALITY)
        confidence = (0.2 + 0.45 * coverage) * (0.5 + 0.5 * agreement(parts.values(), score))
        label = "strong" if score > 0.15 else "weak" if score < -0.15 else "average"
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s} fundamental quality {label}: "
            + ", ".join(e.detail for e in sorted(ev, key=lambda e: -e.strength)[:3]),
            ev,
            quality=q,
            used=["SEC XBRL fundamentals (annual, point-in-time)", "stock model features"],
            missing=[c for c in QUALITY if c not in row.index],
            invalidation="a new annual filing that reverses the profitability or accrual picture",
            meta={
                "composite_z": round(composite, 3),
                "as_of": ctx.model.as_of.isoformat() if ctx.model else None,
            },
        )


class ValuationAgent(Agent):
    spec = AgentSpec(
        id="valuation",
        source="fundamentals",
        failure="skips without a completed stock-model run; its vote is listed as missing",
        name="Valuation",
        description="Earnings, free-cash-flow and book yields against the universe and sector peers, with a "
        "value-trap check.",
        family=AgentFamily.SPECIALIST,
        capabilities=("relative_value", "sector_relative_value", "value_trap"),
        inputs=("model",),
        subjects=("symbol",),
        priority=40,
        horizon_days=63,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return _model_unavailable(ctx, list(VALUE))

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        assert ctx.model is not None
        feats = ctx.model.features
        cols = [c for c in VALUE if c in feats.columns]
        z = cross_section_z(feats[cols])
        sectors = {sym: _peer_sector(ctx, str(sym)) for sym in feats.index}
        return [self._one(ctx, s, z, feats[cols], sectors) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str, z: pd.DataFrame, raw: pd.DataFrame, sectors: dict) -> Opinion:
        if s not in z.index:
            return self.abstain(s, "not covered by the stock model", ["fundamentals"])
        row = z.loc[s].dropna()
        ey = ctx.feature(s, "earnings_yield")
        notes: list[str] = []
        if ey is not None and ey <= 0 and "earnings_yield" in row.index:
            row = row.drop("earnings_yield")  # loss-making: an earnings yield says nothing about value
            notes.append("loss-making: earnings yield ignored")
        if len(row) < 2:
            return self.abstain(s, "fewer than two valuation metrics on file", list(VALUE))
        q = symbol_quality(ctx, s)
        sector = sectors.get(s)
        peers = [p for p, sec in sectors.items() if sector and sec == sector]
        sector_z: pd.Series | None = None
        if len(peers) >= MIN_PEERS:
            sector_z = cross_section_z(raw.loc[peers]).loc[s]
        ev: list[Evidence] = []
        parts: dict[str, float] = {}
        for col, zval in row.items():
            c = str(col)
            rel = float(sector_z[c]) if sector_z is not None and pd.notna(sector_z.get(c)) else None
            blended = 0.5 * float(zval) + 0.5 * rel if rel is not None else float(zval)
            parts[c] = blended
            value = ctx.feature(s, c)
            detail = f"{VALUE[c][1]} {value:.3f}" if value is not None else VALUE[c][1]
            detail += (
                f" ({float(zval):+.1f}σ universe"
                + (f", {rel:+.1f}σ vs {sector}" if rel is not None else "")
                + ")"
            )
            ev.append(
                Evidence(
                    c, value, detail, sgn(blended), min(abs(blended) / 3, 1.0), source="sec_xbrl", quality=q
                )
            )
        cheap = sum(parts.values()) / len(parts)
        score = squash(cheap, 1.2)
        mom = ctx.ind(s, "mom_12_1")
        roe = ctx.feature(s, "roe")
        if score > 0.2 and mom is not None and mom < -0.15 and (roe is None or roe < 0.05):
            score *= 0.4
            notes.append(
                f"possible value trap: 12-1 momentum {mom:+.0%}, ROE {roe:.0%}"
                if roe is not None
                else "possible value trap: weak momentum"
            )
        if score < -0.2 and mom is not None and mom > 0.25:
            score *= 0.5
            notes.append(f"expensive, but 12-1 momentum {mom:+.0%}: expensive can stay expensive")
        for n in notes:
            ev.append(Evidence("valuation_check", None, n, 0, 0.6, quality=q))
        confidence = (0.2 + 0.4 * len(parts) / len(VALUE)) * (0.5 + 0.5 * agreement(parts.values(), score))
        if notes:
            confidence *= 0.8
        label = "cheap" if score > 0.15 else "expensive" if score < -0.15 else "fairly valued"
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s} looks {label}: "
            + ", ".join(e.detail for e in ev[:3])
            + ("; " + "; ".join(notes) if notes else ""),
            ev,
            quality=q,
            used=[
                "SEC XBRL fundamentals (annual, point-in-time)",
                "sector peers" if sector_z is not None else "universe",
            ],
            missing=[c for c in VALUE if c not in row.index],
            invalidation="earnings or cash flow falling enough to erase the yield advantage",
            meta={"cheapness_z": round(cheap, 3), "sector": sector, "peers": len(peers), "checks": notes},
        )
