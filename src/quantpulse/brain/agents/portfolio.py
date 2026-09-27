"""Portfolio agent: the Alpaca paper portfolio as a whole — concentration, sector exposure, beta,
correlation, cash (margin), and positions at their stop — and whether each holding still fits.

It is a *constraint*: it does not forecast prices. Its findings (``meta['constraints']``) tell the
decision step what must be reduced or closed, and what the portfolio cannot absorb, using the same limits
the deterministic risk engine enforces (:class:`~quantpulse.services.trading_risk.RiskLimits`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..context import BrainContext
from ..types import PORTFOLIO, AgentFamily, AgentSpec, DataState, Evidence, Opinion, Stance
from .base import Agent, symbols_only

SECTOR_LIMIT = 0.45  # a single sector above this share of equity is flagged as an unintended bet
BETA_LIMIT = 1.4


class PortfolioAgent(Agent):
    role = "constraint"
    spec = AgentSpec(
        id="portfolio",
        source="account",
        failure="no portfolio hints; the decision step still checks portfolio fit and the risk engine still checks every limit",
        name="Portfolio",
        description="Concentration, sector exposure, beta, correlation, cash and stop-losses of the paper "
        "portfolio; whether each holding still fits.",
        family=AgentFamily.PORTFOLIO,
        capabilities=("concentration", "sector_exposure", "beta", "correlation", "cash", "stop_loss"),
        inputs=("portfolio", "limits"),
        subjects=("portfolio", "symbol"),
        priority=30,
        horizon_days=0,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        if not ctx.portfolio.available:
            return f"the paper account could not be read ({ctx.portfolio.error})"
        return None

    def subjects(self, ctx: BrainContext) -> list[str]:
        return [PORTFOLIO, *ctx.held]

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        overview = self._portfolio(ctx)
        out = [overview]
        for s in symbols_only(subjects):
            out.append(self._holding(ctx, s))
        ctx.working.post("portfolio_constraints", overview.meta["constraints"])
        return out

    def _opinion(
        self, subject: str, thesis: str, ev: list[Evidence], meta: dict, score: float = 0.0
    ) -> Opinion:
        return Opinion(
            agent_id=self.spec.id,
            agent_version=self.spec.version,
            subject=subject,
            stance=Stance.NEUTRAL if abs(score) < 0.15 else (Stance.BEARISH if score < 0 else Stance.BULLISH),
            score=score,
            confidence=1.0,
            horizon_days=0,
            thesis=thesis,
            evidence=ev,
            data_used=["Alpaca paper account"],
            data_quality=DataState.LIVE,
            meta={"gradeable": False, **meta},
        )

    def _portfolio(self, ctx: BrainContext) -> Opinion:
        p, L = ctx.portfolio, ctx.limits
        eq = p.equity
        a = p.account
        assert a is not None
        weights = {s: p.weight(s) for s in ctx.held}
        ev: list[Evidence] = [
            Evidence("equity", round(eq, 2), f"equity ${eq:,.2f}"),
            Evidence("positions", len(weights), f"{len(weights)} of {L.max_positions} positions"),
        ]
        flags: list[str] = []
        cash_pct = a.cash / eq if eq > 0 else 0.0
        if a.cash < 0:
            flags.append(
                f"cash is negative (${a.cash:,.0f}): the account is on margin; no new buys until it is not"
            )
        ev.append(
            Evidence(
                "cash_pct",
                round(cash_pct, 4),
                f"cash {cash_pct:.1%} of equity (reserve {L.cash_buffer_pct:.0%})",
                direction=-1 if a.cash < 0 else 0,
            )
        )
        overweight = {s: w - L.max_position_pct for s, w in weights.items() if w > L.max_position_pct + 1e-9}
        for s in overweight:
            flags.append(f"{s} is {weights[s]:.1%} of equity (limit {L.max_position_pct:.0%})")
        hhi = sum(w * w for w in weights.values())
        ev.append(
            Evidence(
                "hhi", round(hhi, 4), f"concentration index {hhi:.2f} (1/n = {1 / max(len(weights), 1):.2f})"
            )
        )
        exposure = a.long_market_value / eq if eq > 0 else 0.0
        ev.append(
            Evidence(
                "exposure",
                round(exposure, 4),
                f"long exposure {exposure:.0%} (limit {L.max_total_exposure_pct:.0%})",
                direction=-1 if exposure > L.max_total_exposure_pct else 0,
            )
        )
        if exposure > L.max_total_exposure_pct + 1e-9:
            flags.append(f"long exposure {exposure:.0%} exceeds {L.max_total_exposure_pct:.0%}")

        sectors: dict[str, float] = {}
        for s, w in weights.items():
            sectors[ctx.sectors.get(s, "unknown")] = sectors.get(ctx.sectors.get(s, "unknown"), 0.0) + w
        for sec, w in sorted(sectors.items(), key=lambda kv: -kv[1]):
            ev.append(Evidence(f"sector:{sec}", round(w, 4), f"{sec} {w:.0%} of equity"))
            if w > SECTOR_LIMIT and sec not in ("unknown",):
                flags.append(f"sector {sec} is {w:.0%} of equity (flag above {SECTOR_LIMIT:.0%})")
        betas = {s: ctx.ind(s, "beta") for s in weights}
        known = {s: b for s, b in betas.items() if b is not None}
        beta = sum(weights[s] * b for s, b in known.items()) if known else None
        if beta is not None:
            ev.append(
                Evidence("portfolio_beta", round(beta, 3), f"portfolio beta {beta:.2f} (equity-weighted)")
            )
            if beta > BETA_LIMIT:
                flags.append(f"portfolio beta {beta:.2f} above {BETA_LIMIT}")
        held_cols = [s for s in weights if s in ctx.close.columns]
        avg_corr = None
        if len(held_cols) >= 2:
            m = ctx.close[held_cols].pct_change(fill_method=None).iloc[-63:].corr().to_numpy()
            n = m.shape[0]
            avg_corr = float((np.nansum(m) - n) / (n * (n - 1)))
            ev.append(
                Evidence(
                    "avg_correlation",
                    round(avg_corr, 3),
                    f"average pairwise correlation of holdings {avg_corr:.2f} (63 days)",
                )
            )
        at_stop = {
            s: pos.unrealized_plpc
            for s, pos in ctx.portfolio.positions.items()
            if pos.qty > 0 and pos.unrealized_plpc <= -L.max_position_loss_pct
        }
        for s, pl in at_stop.items():
            flags.append(f"{s} is at its stop ({pl:+.1%} vs −{L.max_position_loss_pct:.0%})")
        free_slots = max(
            L.max_positions
            - len(weights)
            - len({o.symbol for o in ctx.portfolio.open_orders if o.side == "buy"} - set(weights)),
            0,
        )
        spendable = min(a.buying_power, a.cash - L.cash_buffer_pct * eq)
        constraints = {
            "overweight": {s: round(x, 4) for s, x in overweight.items()},
            "at_stop": {s: round(x, 4) for s, x in at_stop.items()},
            "margin": a.cash < 0,
            "spendable_cash": round(max(spendable, 0.0), 2),
            "free_slots": free_slots,
            "sector_weights": {k: round(v, 4) for k, v in sectors.items()},
            "beta": beta,
            "avg_correlation": avg_corr,
            "exposure": round(exposure, 4),
        }
        thesis = (
            "; ".join(flags) if flags else "within limits"
        ) + f" — {len(weights)} positions, exposure {exposure:.0%}"
        return self._opinion(PORTFOLIO, thesis, ev, {"constraints": constraints, "flags": flags})

    def _holding(self, ctx: BrainContext, s: str) -> Opinion:
        L = ctx.limits
        pos = ctx.portfolio.positions[s]
        w = ctx.portfolio.weight(s)
        ev = [
            Evidence(
                "weight",
                round(w, 4),
                f"{w:.1%} of equity (limit {L.max_position_pct:.0%})",
                direction=-1 if w > L.max_position_pct else 0,
            ),
            Evidence(
                "unrealized_plpc",
                round(pos.unrealized_plpc, 4),
                f"unrealised {pos.unrealized_plpc:+.1%} (stop −{L.max_position_loss_pct:.0%})",
                direction=-1 if pos.unrealized_plpc <= -L.max_position_loss_pct else 0,
            ),
        ]
        hint, score, why = "hold", 0.0, "fits the portfolio limits"
        if pos.unrealized_plpc <= -L.max_position_loss_pct:
            hint, score, why = "close", -1.0, f"at its stop ({pos.unrealized_plpc:+.1%})"
        elif w > L.max_position_pct + 1e-9:
            hint, score, why = "reduce", -0.6, f"overweight ({w:.1%} vs {L.max_position_pct:.0%})"
        return self._opinion(
            s,
            f"{s}: {why}",
            ev,
            {"action_hint": hint, "weight": round(w, 4), "target_max_weight": L.max_position_pct},
            score,
        )
