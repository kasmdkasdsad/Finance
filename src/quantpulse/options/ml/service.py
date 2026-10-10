"""The options edge model as a service: training (a research job), the model registry, persistence, and the
live assessments the ``OptionsMLAgent`` reads.

**Training** (:meth:`OptionsMLService.train`, the ``options_ml`` research job while the market is closed):

1. *model* rows — probe candidates on model-priced chains over the core universe's real daily prices;
2. *recorded* rows — the same probes on the chains QuantPulse recorded from Alpaca (one snapshot per underlying
   and day, near the close), one underlying at a time so memory stays small;
3. *shadow* and *paper* rows — every closed position the Options Brain opened while the model was scoring, with
   the feature vector recorded at the time and its realised return per dollar at risk;
4. the model is fitted and validated out of sample (:class:`~.model.OptionsEdgeModel`), registered in the
   model registry as a **challenger** to the rule (slot ``options_candidate_score``) with its out-of-sample,
   walk-forward and stress evidence, and its **live record** — how well the model's predictions ranked the
   realised outcomes of the positions it scored, against the rule's ranking of the same positions. The record
   belongs to the procedure, not one fit: each daily retrain is the same model on more data, so it inherits the
   predictions its predecessors made (every one recorded with the model version that made it);
5. the registry advances it as far as the evidence allows. Only at AUTHORITATIVE (offline gates passed *and* a
   live record beating the rule on at least 30 positions) does the agent vote; until then it is recorded, so its
   live record can build.

**Live** (:meth:`assess`): the same feature vector the training rows have, from the cycle's own view of the
underlying (its chain, the SVI surface fitted once per chain, the HAR forecast, the day's features) — and the
prediction, the drivers and the vector itself are recorded with the candidate.

Nothing here sends an order or changes a limit.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, cast

import numpy as np
from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db.models import BrainStateRow
from quantpulse.db.options_models import (
    OptionsChainSnapshotRow,
    OptionsPositionRow,
    OptionsQuoteRow,
    OptionsTradeCandidateRow,
)
from quantpulse.db.session import Database
from quantpulse.options.contracts import ContractError, Kind, OptionContract
from quantpulse.options.data import ChainSnapshot
from quantpulse.options.lab.chains import ModelChains, RecordedChains
from quantpulse.options.quotes import OptionQuote
from quantpulse.options.selection import Candidate
from quantpulse.schemas.common import DataStatus

from .dataset import Dataset, Row, build_rows
from .features import FEATURES, candidate_features, vector
from .model import MODEL_VERSION, OptionsEdgeModel, _spearman
from .surface import Surface, fit_surface

logger = logging.getLogger(__name__)

STATE_KEY = "options_ml_model"
SLOT = "options_candidate_score"  # the rule's slot: the model is its challenger
AGENT = "OptionsMLAgent"
HISTORY_DAYS = 1300
RECORDED_DTE = (5, 130)
RECORDED_MONEYNESS = 0.30


def _clean(x: Any) -> Any:
    """JSON-safe: numpy scalars to numbers, non-finite floats to ``None``."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, date | datetime):
        return x.isoformat()
    if hasattr(x, "item") and not isinstance(x, str):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class OptionsMLService:
    def __init__(self, settings: Settings, db: Database, clock: Clock, market: Any = None,
                 registry: Any = None) -> None:  # fmt: skip
        self._s = settings
        self._db = db
        self._clock = clock
        self._market = market
        self._registry = registry
        self.model: OptionsEdgeModel | None = None
        self.meta: dict[str, Any] = {}
        self.last_train: dict[str, Any] | None = None
        self._loaded = False
        self._running = asyncio.Lock()
        self._surfaces: dict[tuple[str, str], tuple[Surface, dict[str, float | None]]] = {}

    # ------------------------------------------------------------------ state
    @property
    def stage(self) -> str | None:
        return self.meta.get("stage")

    @property
    def authoritative(self) -> bool:
        return self.stage == "AUTHORITATIVE"

    async def ensure_loaded(self) -> None:
        """Load the saved model once per process (never fatal: without one, the agent abstains)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            await self.load()
        except Exception:
            logger.exception("loading the options edge model failed (the rule decides until it is retrained)")

    async def load(self) -> bool:
        async with self._db.session() as s:
            row = await s.get(BrainStateRow, STATE_KEY)
            value = dict(row.value) if row is not None else {}
        blob = value.pop("blob", None)
        if not blob:
            return False
        model = await asyncio.to_thread(OptionsEdgeModel.from_bytes, base64.b64decode(blob))
        if model is None:
            self.meta = {**value, "note": "the saved model was made for other features or another scikit-learn "
                                          "version: it is retrained at the next research run"}  # fmt: skip
            return False
        self.model, self.meta = model, value
        await self.refresh_stage()
        return True

    async def refresh_stage(self) -> None:
        """The registry decides the model's standing (a person or the gates may have moved it)."""
        mid = self.meta.get("model_id")
        if self._registry is None or mid is None:
            return
        for m in await self._registry.models(SLOT):
            if m["id"] == mid:
                self.meta["stage"] = m["stage"]
                self.meta["next_gate"] = m.get("next_gate")

    # ------------------------------------------------------------------ live
    def _surface(self, view: Any) -> tuple[Surface, dict[str, float | None]]:
        key = (view.underlying, view.chain.fetched_at.isoformat())
        hit = self._surfaces.get(key)
        if hit is None:
            if len(self._surfaces) > 256:
                self._surfaces.clear()
            surf = fit_surface(view.chain.quotes, view.spot, view.chain.fetched_at, max_slices=8)
            hit = (surf, surf.features())
            self._surfaces[key] = hit
        return hit

    def assess(
        self, view: Any, cand: Candidate, stock_view: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """The model's view of one candidate (``None`` without a model, or when the view lacks a chain). Never
        raises: a failure is logged and the candidate goes on without it."""
        if (
            not self._s.options_ml_enabled
            or self.model is None
            or view is None
            or not getattr(view, "usable", False)
        ):
            return None
        try:
            surface, sf = self._surface(view)
            closes = [view.closes[d] for d in sorted(view.closes)]
            score = (stock_view or {}).get("score_signed")
            x = candidate_features(cand, float(view.spot), view.chain.fetched_at, day=view.features, closes=closes,
                                   surface=surface, stock_score=score, grade="shadow", surface_features=sf)  # fmt: skip
            vec = vector(x)
            pred = self.model.predict(vec[None, :])[0]
        except Exception:
            logger.exception("the options edge model could not assess %s", getattr(view, "underlying", "?"))
            return None
        return {"model_id": self.meta.get("model_id"), "version": MODEL_VERSION, "stage": self.stage,
                "authoritative": self.authoritative, "prediction": pred.as_dict(),
                "x": [None if not math.isfinite(v) else round(float(v), 6) for v in vec]}  # fmt: skip

    # ------------------------------------------------------------------ data
    async def _prices(self, symbols: Sequence[str]) -> dict[str, dict[date, float]]:
        closes: dict[str, dict[date, float]] = {}
        if self._market is None:
            return closes
        for sym in symbols:
            try:
                r = await self._market.history(sym, "1d", lookback_days=HISTORY_DAYS)
            except Exception:
                continue
            if r.status is DataStatus.SYNTHETIC:
                continue  # never trained on made-up prices
            closes[sym] = {b.timestamp.astimezone(NEW_YORK).date(): float(b.close) for b in r.value.bars}
        return closes

    async def _recorded_source(self, underlying: str, since: datetime, closes: Mapping[date, float]
                               ) -> RecordedChains | None:  # fmt: skip
        """One recorded chain per day for ``underlying`` (the snapshot nearest 15:45 New York), contracts within
        the DTE window and ±30% of the price only."""
        async with self._db.session() as s:
            snaps = (await s.execute(
                select(OptionsChainSnapshotRow.id, OptionsChainSnapshotRow.fetched_at,
                       OptionsChainSnapshotRow.underlying_price, OptionsChainSnapshotRow.feed)
                .where(OptionsChainSnapshotRow.underlying == underlying, OptionsChainSnapshotRow.fetched_at >= since)
            )).all()  # fmt: skip
            best: dict[date, tuple[float, Any]] = {}
            for sid, at, px, feed in snaps:
                local = at.astimezone(NEW_YORK)
                gap = abs((local.hour * 60 + local.minute) - (15 * 60 + 45))
                if px and (local.date() not in best or gap < best[local.date()][0]):
                    best[local.date()] = (gap, (sid, at, float(px), feed))
            if len(best) < 3:
                return None
            chains: dict[tuple[str, date], ChainSnapshot] = {}
            for day, (_, (sid, at, px, feed)) in sorted(best.items()):
                rows = (await s.execute(
                    select(OptionsQuoteRow.expiration, OptionsQuoteRow.kind, OptionsQuoteRow.strike,
                           OptionsQuoteRow.bid, OptionsQuoteRow.ask, OptionsQuoteRow.iv, OptionsQuoteRow.open_interest,
                           OptionsQuoteRow.volume, OptionsQuoteRow.quote_at)
                    .where(OptionsQuoteRow.snapshot_id == sid,
                           OptionsQuoteRow.strike >= px * (1 - RECORDED_MONEYNESS),
                           OptionsQuoteRow.strike <= px * (1 + RECORDED_MONEYNESS),
                           OptionsQuoteRow.expiration >= day + timedelta(days=RECORDED_DTE[0]),
                           OptionsQuoteRow.expiration <= day + timedelta(days=RECORDED_DTE[1]))
                )).all()  # fmt: skip
                quotes = []
                for exp, kind, strike, bid, ask, iv, oi, vol, qat in rows:
                    try:
                        c = OptionContract(underlying, exp, cast(Kind, kind), float(strike))
                    except ContractError:
                        continue
                    quotes.append(OptionQuote(c, bid, ask, qat or at, "recorded", "recorded", iv=iv,
                                              open_interest=oi, volume=vol, underlying_price=px, underlying_at=at))  # fmt: skip
                if quotes:
                    chains[(underlying, day)] = ChainSnapshot(
                        underlying, px, at, at, feed or "recorded", "recorded", quotes
                    )
        return RecordedChains(chains, {underlying: dict(closes)}) if len(chains) >= 3 else None

    async def _live_rows(self) -> tuple[list[Row], dict[str, Any]]:
        """Closed shadow and paper positions the model scored: training rows (with the vector recorded at the
        time) and the model's live record against the rule's on the same positions."""
        async with self._db.session() as s:
            res = (await s.execute(
                select(OptionsPositionRow.mode, OptionsPositionRow.underlying, OptionsPositionRow.family,
                       OptionsPositionRow.opened_at, OptionsPositionRow.closed_at, OptionsPositionRow.realized_pnl,
                       OptionsPositionRow.max_loss, OptionsTradeCandidateRow.audit)
                .join(OptionsTradeCandidateRow, OptionsTradeCandidateRow.id == OptionsPositionRow.candidate_id)
                .where(OptionsPositionRow.status == "closed", OptionsPositionRow.realized_pnl.is_not(None))
            )).all()  # fmt: skip
        rows: list[Row] = []
        pred, rule, outcome = [], [], []
        for mode, und, fam, opened, closed, pnl, loss, audit in res:
            if not loss or loss <= 0 or opened is None or closed is None:
                continue
            ror = float(pnl) / float(loss)
            op = next(
                (
                    o
                    for o in ((audit or {}).get("verdict") or {}).get("opinions") or []
                    if o.get("agent") == AGENT
                ),
                None,
            )
            data = (op or {}).get("data") or {}
            x = data.get("x")
            if not isinstance(x, list) or len(x) != len(FEATURES):
                continue
            feats = {k: (math.nan if v is None else float(v)) for k, v in zip(FEATURES, x, strict=True)}
            grade = "paper" if mode == "paper" else "shadow"
            rows.append(
                Row(feats, ror, opened.date().toordinal(), closed.date().toordinal(), und, fam, grade, "live")
            )
            p = (data.get("prediction") or {}).get("expected")
            eor = ((audit or {}).get("metrics") or {}).get("expected_on_risk")
            if p is not None and eor is not None:
                pred.append(float(p))
                rule.append(float(eor))
                outcome.append(ror)
        y = np.array(outcome)
        record = {"n": len(outcome), "score": round(_spearman(np.array(pred), y), 4) if len(y) >= 10 else None,
                  "champion_score": round(_spearman(np.array(rule), y), 4) if len(y) >= 10 else None,
                  "metric": "rank correlation of the predicted and realised return per $ at risk (closed positions)"}  # fmt: skip
        return rows, record

    # ------------------------------------------------------------------ training
    async def train(self, *, budget_seconds: float | None = None,
                    closes: Mapping[str, Mapping[date, float]] | None = None) -> dict[str, Any]:  # fmt: skip
        """One budgeted training run (see the module docstring). One at a time."""
        if not self._s.options_ml_enabled:
            return {"skipped": "the options edge model is switched off (QP_OPTIONS_ML_ENABLED=false)"}
        if self._running.locked():
            return {"skipped": "an options model training run is already in progress"}
        async with self._running:
            return await self._train(budget_seconds or self._s.options_ml_budget_seconds, closes)

    async def _train(self, budget: float, given: Mapping[str, Mapping[date, float]] | None) -> dict[str, Any]:
        started = time.monotonic()
        now = self._clock.now()
        data_deadline = started + 0.6 * budget  # the rest is the fit and the validation
        universe = list(self._s.options_universe)
        closes = {u: dict(v) for u, v in (given or await self._prices(universe)).items() if v}
        usable = [u for u in universe if u in closes]
        counts: dict[str, int] = defaultdict(int)
        rows: list[Row] = []
        if usable:
            model_rows = await asyncio.to_thread(
                build_rows, ModelChains(closes), usable, closes, grade="model", step=self._s.options_ml_step_days,
                deadline=started + 0.4 * budget, max_rows=self._s.options_ml_max_rows,
            )  # fmt: skip
            rows += model_rows
            counts["model"] = len(model_rows)
        since = now - timedelta(days=self._s.options_ml_recorded_days)
        for u in usable:
            if time.monotonic() > data_deadline:
                break
            src = await self._recorded_source(u, since, closes.get(u, {}))
            if src is None:
                continue
            rec = await asyncio.to_thread(build_rows, src, [u], closes, grade="recorded", step=1,
                                          deadline=data_deadline)  # fmt: skip
            rows += rec
            counts["recorded"] += len(rec)
        live, record = await self._live_rows()
        rows += live
        counts["live"] = len(live)
        ds = Dataset.from_rows(rows)
        out: dict[str, Any]
        if ds.n < self._s.options_ml_min_rows:
            out = {
                "skipped": f"{ds.n} labelled rows (minimum {self._s.options_ml_min_rows})",
                "rows": dict(counts),
            }
            self.last_train = {"at": now.isoformat(), **out}
            return out
        model = OptionsEdgeModel(max_iter=self._s.options_ml_trees)
        try:
            report = await asyncio.to_thread(model.fit, ds)
        except ValueError as exc:
            out = {"skipped": str(exc), "rows": dict(counts)}
            self.last_train = {"at": now.isoformat(), **out}
            return out
        report = _clean(report)
        reg: dict[str, Any] = {}
        if self._registry is not None:
            reg = await self._registry.register(
                SLOT, "ml", f"Options edge model v{MODEL_VERSION} (gradient boosting + conformal)",
                description="Expected return per $ at risk of an option candidate under a standard exit policy: "
                "quantile gradient boosting stacked with ridge, calibrated probability of profit, conformal intervals.",
                params=dict(model.params), data=report.get("data"),
            )  # fmt: skip
            oos = report["oos"]
            await self._registry.record(
                reg["id"], oos={"score": oos["ic"], "baseline": oos["rule_ic"], "n": oos["n"]},
                walk_forward=report["walk_forward"], stress=report["stress"], shadow=record,
            )  # fmt: skip
            reg = await self._registry.advance(reg["id"])
        meta = {"model_id": reg.get("id"), "stage": reg.get("stage", "CANDIDATE"), "next_gate": reg.get("next_gate"),
                "trained_at": now.isoformat(), "rows": dict(counts), "live_record": record, "card": _clean(model.card())}  # fmt: skip
        blob = base64.b64encode(await asyncio.to_thread(model.to_bytes)).decode()
        async with self._db.session() as s:
            row = await s.get(BrainStateRow, STATE_KEY)
            value = {**meta, "blob": blob}
            if row is None:
                s.add(BrainStateRow(key=STATE_KEY, value=value, updated_at=now))
            else:
                row.value, row.updated_at = value, now
        self.model, self.meta, self._loaded = model, meta, True
        out = {"rows": dict(counts), "model_id": meta["model_id"], "stage": meta["stage"], "oos": report["oos"],
               "walk_forward": report["walk_forward"], "stress": report["stress"], "live_record": record,
               "seconds": round(time.monotonic() - started, 1)}  # fmt: skip
        self.last_train = {"at": now.isoformat(), **_clean(out)}
        return out

    # ------------------------------------------------------------------ reporting
    async def status(self) -> dict[str, Any]:
        await self.ensure_loaded()
        await self.refresh_stage()
        card = self.meta.get("card") or {}
        return _clean({
            "enabled": self._s.options_ml_enabled,
            "model": None if self.model is None else {
                "model_id": self.meta.get("model_id"), "stage": self.stage, "authoritative": self.authoritative,
                "next_gate": self.meta.get("next_gate"), "trained_at": self.meta.get("trained_at"),
                "rows": self.meta.get("rows"), "data": card.get("data"), "oos": card.get("oos"),
                "walk_forward": card.get("walk_forward"), "stress": card.get("stress"),
                "conformal": card.get("conformal"), "importance": card.get("importance"),
                "live_record": self.meta.get("live_record"),
            },
            "note": self.meta.get("note"),
            "last_train": self.last_train,
            "how_it_decides": "recorded with every candidate; votes only once the model registry makes it "
            "AUTHORITATIVE (out-of-sample, walk-forward and stress evidence, and a live record beating the rule)",
        })  # fmt: skip
