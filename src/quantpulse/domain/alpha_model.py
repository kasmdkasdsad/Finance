"""Cross-sectional return model trained walk-forward, with an honest out-of-sample track record.

Target
    For every date, the ``h``-day forward returns of the universe are ranked and mapped to normal scores
    (rank-Gauss). The model therefore predicts *relative* performance: which stocks will do better than
    the others, not where the market goes.

Model
    Ridge regression on per-date z-scored features (:mod:`quantpulse.domain.features`). The ridge penalty
    is picked inside each training window by a purged time-series split (the last 25% of the window is
    the validation set, and training labels that overlap it are dropped).

Walk-forward protocol (no look-ahead)
    To predict at the close of day *t*, the model may only use labels already realised by *t*: a sample
    from day *s* has a label ending at *s + h*, so training uses days *s ≤ t − h* (purging), within a
    rolling ``train_window``. The model is refitted every ``retrain_every`` days and the coefficients are
    frozen in between. Every prediction reported as out-of-sample was made this way.

Evaluation (all out-of-sample)
    * information coefficient (IC): per-date Spearman correlation of prediction vs realised return,
      its mean, and a t-statistic on non-overlapping dates;
    * hit rate: how often a stock predicted above the median actually finished above it (and vice versa);
    * bucket returns: average realised return of each prediction quintile, and the top-minus-bottom spread;
    * a top-``k`` long-only portfolio rebalanced every ``h`` days, net of trading costs, against an
      equal-weight universe and the benchmark;
    * the same IC statistics for the platform's hand-set factor rule, as a baseline to beat.

Probabilities
    Out-of-sample predictions are bucketed; in each bucket the observed frequency of beating the benchmark
    over ``h`` days (and the mean excess return) is shrunk towards the overall base rate with
    ``prior_strength`` pseudo-observations and made monotone (pool-adjacent-violators). Live predictions
    are mapped through that table. A model without skill therefore reports probabilities close to the base
    rate instead of confident-sounding numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantpulse.core.errors import DomainError
from quantpulse.domain import features as feat

BASELINE_WEIGHTS: dict[str, float] = {  # the Daily Picks rule, expressed on the feature library
    "mom_12_1": 0.20,
    "mom_3m": 0.10,
    "trend_50_200": 0.25,
    "sharpe_126": 0.25,
    "vol_63": -0.10,
    "rsi_14": -0.10,
}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    horizon: int = 21
    train_window: int = 756
    min_train: int = 252
    retrain_every: int = 21
    lambdas: tuple[float, ...] = (0.3, 3.0, 30.0, 300.0)
    top_k: int = 5
    buckets: int = 5
    cost_bps: float = 10.0
    prior_strength: float = 50.0
    features: tuple[str, ...] = tuple(feat.FEATURES)

    def __post_init__(self) -> None:
        if not 1 <= self.horizon <= 126:
            raise DomainError("horizon must be between 1 and 126 trading days")
        if self.min_train < 60 or self.train_window < self.min_train:
            raise DomainError("need min_train >= 60 and train_window >= min_train")
        if self.retrain_every < 1 or self.top_k < 1 or self.buckets < 2:
            raise DomainError("retrain_every and top_k must be >= 1 and buckets >= 2")
        if not self.lambdas or any(lam <= 0 for lam in self.lambdas):
            raise DomainError("lambdas must be positive")
        unknown = [f for f in self.features if f not in feat.FEATURES]
        if unknown:
            raise DomainError(f"unknown features: {unknown}")


# ----------------------------------------------------------------------------- data plumbing
@dataclass
class ModelData:
    """Aligned inputs: long feature matrix, rank-Gauss labels, raw forward returns and date positions.

    Everything the walk-forward loop touches is precomputed as NumPy arrays indexed by row, with each row's
    position in the trading calendar, so selecting "all rows known by day t" is a vector comparison."""

    X: pd.DataFrame  # (date, symbol) -> feature z-scores
    fwd: pd.DataFrame  # wide forward returns (dates x symbols), NaN where not yet realised
    bench_fwd: pd.Series  # benchmark forward return per date
    positions: pd.Series  # date -> integer position in the trading calendar
    horizon: int
    y: pd.Series = field(init=False)
    y_wide: pd.DataFrame = field(init=False)
    X_np: np.ndarray = field(init=False)
    y_np: np.ndarray = field(init=False)
    row_pos: np.ndarray = field(init=False)
    date_pos: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.X = self.X.sort_index()
        target = feat.rank_gauss(self.fwd).stack(future_stack=True)
        target.index.names = ["date", "symbol"]
        self.y = target.reindex(self.X.index)
        self.y_wide = self.y.unstack()
        self.X_np = self.X.to_numpy(dtype=float)
        self.y_np = self.y.to_numpy(dtype=float)
        self.row_pos = self.positions.reindex(self.X.index.get_level_values("date")).to_numpy(dtype=int)
        self.date_pos = np.unique(self.row_pos)

    @property
    def calendar(self) -> pd.Index:
        return self.positions.index

    @property
    def dates(self) -> list[pd.Timestamp]:
        return [self.calendar[p] for p in self.date_pos]

    def pos(self, d: pd.Timestamp) -> int:
        return int(self.positions[d])

    def predict(self, mask: np.ndarray, coef: np.ndarray) -> pd.Series:
        return pd.Series(self.X_np[mask] @ coef, index=self.X.index[mask])


def build_data(
    panel: feat.Panel, horizon: int, names: Sequence[str] | None = None, min_coverage: float = 0.9
) -> tuple[ModelData, dict[str, pd.DataFrame]]:
    raw = feat.compute_features(panel)
    X = feat.feature_matrix(raw, list(names or feat.FEATURES), min_coverage)
    if X.empty:
        raise DomainError(f"no dates with enough feature history (the first {feat.WARMUP} days are warm-up)")
    positions = pd.Series(np.arange(len(panel.close.index)), index=panel.close.index)
    fwd = feat.forward_returns(panel.close, horizon)
    bench_fwd = panel.benchmark.shift(-horizon) / panel.benchmark - 1
    return ModelData(X=X, fwd=fwd, bench_fwd=bench_fwd, positions=positions, horizon=horizon), raw


# ----------------------------------------------------------------------------- fitting
def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge without intercept on standardised inputs: ``(XᵀX/n + λI)⁻¹ Xᵀy/n``."""
    n, p = X.shape
    if n == 0:
        raise DomainError("no training samples")
    return np.linalg.solve(X.T @ X / n + lam * np.eye(p), X.T @ y / n)


def _mean_ic(pred: pd.Series, target_wide: pd.DataFrame) -> float:
    if pred.empty:
        return float("nan")
    wide = pred.unstack()
    ic = feat.row_spearman(wide, target_wide.reindex(index=wide.index, columns=wide.columns)).dropna()
    return float(ic.mean()) if len(ic) else float("nan")


def _train(data: ModelData, lo: int, hi: int, config: ModelConfig) -> tuple[np.ndarray, float]:
    """Fit on sample days with calendar position in ``(lo, hi]``; λ is chosen on a purged hold-out made
    of the last 25% of those days, then the model is refitted on all of them."""
    labelled = np.isfinite(data.y_np)
    rows = (data.row_pos > lo) & (data.row_pos <= hi) & labelled
    days = data.date_pos[(data.date_pos > lo) & (data.date_pos <= hi)]
    lam = sorted(config.lambdas)[len(config.lambdas) // 2]
    split = int(len(days) * 0.75)
    if split >= 60 and len(days) - split >= 20:
        val_start = int(days[split])
        inner = rows & (data.row_pos + data.horizon <= val_start)
        val = rows & (data.row_pos >= val_start)
        if np.unique(data.row_pos[inner]).size >= 40:
            Xi, yi = data.X_np[inner], data.y_np[inner]
            scores = {
                c: _mean_ic(data.predict(val, fit_ridge(Xi, yi, c)), data.y_wide) for c in config.lambdas
            }
            finite = {k: v for k, v in scores.items() if math.isfinite(v)}
            if finite:  # ties go to the stronger penalty
                lam = max(finite, key=lambda k: (round(finite[k], 6), k))
    return fit_ridge(data.X_np[rows], data.y_np[rows], lam), lam


@dataclass
class WalkForward:
    predictions: pd.Series  # out-of-sample, (date, symbol)
    fits: list[tuple[pd.Timestamp, float, np.ndarray]]  # (fit date, λ, coefficients)


def walk_forward(data: ModelData, config: ModelConfig) -> WalkForward:
    dp = data.date_pos
    h = data.horizon
    fits: list[tuple[pd.Timestamp, float, np.ndarray]] = []
    segments: list[tuple[int, np.ndarray]] = []  # (index into dp where the coefficients take over, coef)
    last_fit: int | None = None
    for k, p in enumerate(dp):
        if last_fit is not None and k - last_fit < config.retrain_every:
            continue
        cutoff = int(p) - h  # only labels realised by day p
        n_days = int(np.count_nonzero((dp <= cutoff) & (dp > cutoff - config.train_window)))
        if n_days < config.min_train:
            continue
        coef, lam = _train(data, cutoff - config.train_window, cutoff, config)
        fits.append((data.calendar[p], lam, coef))
        segments.append((k, coef))
        last_fit = k
    if not fits:
        raise DomainError(
            f"not enough history: the model needs {config.min_train} labelled days after the "
            f"{feat.WARMUP}-day warm-up, plus the {h}-day label horizon"
        )
    preds = []
    for i, (k0, coef) in enumerate(segments):
        k1 = segments[i + 1][0] if i + 1 < len(segments) else len(dp)
        mask = (data.row_pos >= dp[k0]) & (data.row_pos <= dp[k1 - 1])
        preds.append(data.predict(mask, coef))
    out = pd.concat(preds)
    out.index.names = ["date", "symbol"]
    return WalkForward(predictions=out, fits=fits)


def fit_final(data: ModelData, config: ModelConfig) -> tuple[np.ndarray, float, pd.Timestamp]:
    """The live model: trained on every label realised by the last date (within the training window)."""
    last = int(data.date_pos[-1])
    cutoff = last - data.horizon
    n_days = int(np.count_nonzero((data.date_pos <= cutoff) & (data.date_pos > cutoff - config.train_window)))
    if n_days < config.min_train:
        raise DomainError("not enough labelled history to fit the live model")
    coef, lam = _train(data, cutoff - config.train_window, cutoff, config)
    return coef, lam, data.calendar[last]


# ----------------------------------------------------------------------------- evaluation
@dataclass(frozen=True, slots=True)
class ICStats:
    mean_ic: float
    ic_std: float
    t_stat: float | None
    positive_share: float
    hit_rate: float | None
    n_dates: int


def ic_stats(pred_wide: pd.DataFrame, realized: pd.DataFrame, horizon: int) -> tuple[ICStats, pd.Series]:
    ic = feat.row_spearman(pred_wide, realized).dropna()
    if ic.empty:
        raise DomainError("no out-of-sample dates with realised returns yet")
    sub = ic.iloc[::horizon]  # non-overlapping label windows
    t = (
        float(sub.mean() / sub.std(ddof=1) * math.sqrt(len(sub)))
        if len(sub) > 2 and sub.std(ddof=1) > 0
        else None
    )
    rel_p = pred_wide.sub(pred_wide.median(axis=1), axis=0)
    rel_r = realized.sub(realized.median(axis=1), axis=0)
    mask = rel_p.notna() & rel_r.notna() & (rel_p != 0) & (rel_r != 0)
    agree = (np.sign(rel_p) == np.sign(rel_r)).astype(float).where(mask)
    hits = agree.stack(future_stack=True).dropna()
    return (
        ICStats(
            mean_ic=float(ic.mean()),
            ic_std=float(ic.std(ddof=1)) if len(ic) > 1 else 0.0,
            t_stat=t,
            positive_share=float((ic > 0).mean()),
            hit_rate=float(hits.mean()) if len(hits) else None,
            n_dates=len(ic),
        ),
        ic,
    )


def bucket_returns(pred_wide: pd.DataFrame, realized: pd.DataFrame, buckets: int) -> list[float | None]:
    """Mean realised return per prediction bucket (0 = lowest predictions), averaged over dates."""
    ranks = pred_wide.where(realized.notna()).rank(axis=1, pct=True)
    idx = np.ceil(ranks * buckets).clip(1, buckets) - 1
    out: list[float | None] = []
    for b in range(buckets):
        per_date = realized.where(idx == b).mean(axis=1).dropna()
        out.append(float(per_date.mean()) if len(per_date) else None)
    return out


@dataclass
class Backtest:
    dates: list[pd.Timestamp]  # start of the first period, then the end of every period
    strategy: list[float]  # growth of $1
    universe: list[float]
    benchmark: list[float]
    period_returns: pd.DataFrame  # strategy / universe / benchmark per holding period
    turnover: float

    def metrics(self, periods_per_year: float, risk_free: float = 0.0) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        for name in ("strategy", "universe", "benchmark"):
            r = self.period_returns[name].to_numpy(dtype=float)
            n = r.size
            growth = float(np.prod(1 + r))
            ann = growth ** (periods_per_year / n) - 1 if n and growth > 0 else None
            vol = float(r.std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else None
            rf_p = (1 + risk_free) ** (1 / periods_per_year) - 1
            ex = r - rf_p
            sharpe = (
                float(ex.mean() / ex.std(ddof=1) * math.sqrt(periods_per_year))
                if n > 1 and ex.std(ddof=1) > 0
                else None
            )
            curve = np.concatenate([[1.0], np.cumprod(1 + r)])
            dd = float(np.min(curve / np.maximum.accumulate(curve) - 1)) if n else None
            out[name] = {
                "total_return": growth - 1,
                "annual_return": ann,
                "annual_volatility": vol,
                "sharpe": sharpe,
                "max_drawdown": dd,
            }
        pr = self.period_returns
        out["strategy"]["beat_universe_share"] = (
            float((pr["strategy"] > pr["universe"]).mean()) if len(pr) else None
        )
        return out


def top_k_backtest(
    pred_wide: pd.DataFrame,
    realized: pd.DataFrame,
    bench: pd.Series,
    config: ModelConfig,
    calendar: pd.DatetimeIndex,
) -> Backtest:
    """Hold the top ``k`` predictions in equal weight for ``h`` days, rebalancing on non-overlapping dates."""
    usable = [
        d for d in pred_wide.index if realized.loc[d].notna().sum() >= config.top_k and pd.notna(bench.get(d))
    ]
    rebalance = usable[:: config.horizon]
    if len(rebalance) < 2:
        raise DomainError("not enough realised out-of-sample periods for a backtest")
    held: set[str] = set()
    rows = []
    turnover_total = 0.0
    for d in rebalance:
        scores = pred_wide.loc[d][realized.loc[d].notna()].dropna().sort_values(ascending=False)
        chosen = set(scores.index[: config.top_k])
        traded = len(chosen ^ held) / max(1, config.top_k)  # fraction of the book bought + sold
        turnover_total += traded / 2
        cost = traded * config.cost_bps / 10_000
        rows.append(
            {
                "date": d,
                "strategy": float(realized.loc[d, list(chosen)].mean()) - cost,
                "universe": float(realized.loc[d].dropna().mean()),
                "benchmark": float(bench[d]),
            }
        )
        held = chosen
    pr = pd.DataFrame(rows).set_index("date")
    curves = (1 + pr).cumprod()
    first = [1.0]
    where = {d: i for i, d in enumerate(calendar)}
    ends = [calendar[min(where[d] + config.horizon, len(calendar) - 1)] for d in pr.index]
    return Backtest(
        dates=[rebalance[0], *ends],
        strategy=first + curves["strategy"].tolist(),
        universe=first + curves["universe"].tolist(),
        benchmark=first + curves["benchmark"].tolist(),
        period_returns=pr,
        turnover=turnover_total / len(rebalance),
    )


# ----------------------------------------------------------------------------- probability calibration
@dataclass(frozen=True, slots=True)
class CalibrationBin:
    z_mid: float
    n: int
    observed: float  # raw frequency of beating the benchmark
    probability: float  # shrunk + monotone
    mean_excess: float  # shrunk mean return in excess of the benchmark


def _pav(values: list[float], weights: list[float]) -> list[float]:
    """Pool-adjacent-violators: the closest non-decreasing sequence (weighted least squares)."""
    blocks = [[v, w, 1] for v, w in zip(values, weights, strict=True)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-15:
            v1, w1, c1 = blocks[i]
            v2, w2, c2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out: list[float] = []
    for v, _, c in blocks:
        out.extend([v] * int(c))
    return out


@dataclass
class Calibration:
    bins: list[CalibrationBin]
    base_rate: float
    mean_excess: float

    def probability(self, z: float) -> float:
        xs = [b.z_mid for b in self.bins]
        return float(np.interp(z, xs, [b.probability for b in self.bins]))

    def expected_excess(self, z: float) -> float:
        xs = [b.z_mid for b in self.bins]
        return float(np.interp(z, xs, [b.mean_excess for b in self.bins]))


def calibrate(
    pred_wide: pd.DataFrame, realized: pd.DataFrame, bench: pd.Series, config: ModelConfig
) -> Calibration:
    z = feat.cross_sectional_z(pred_wide)
    excess = realized.sub(bench.reindex(realized.index), axis=0)
    pairs = pd.DataFrame(
        {"z": z.stack(future_stack=True), "excess": excess.reindex_like(z).stack(future_stack=True)}
    ).dropna()
    if len(pairs) < config.buckets * 10:
        raise DomainError("not enough realised out-of-sample predictions to calibrate probabilities")
    base = float((pairs["excess"] > 0).mean())
    mean_ex = float(pairs["excess"].mean())
    edges = np.quantile(pairs["z"], np.linspace(0, 1, config.buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pairs["bin"] = np.clip(np.searchsorted(edges, pairs["z"], side="right") - 1, 0, config.buckets - 1)
    k = config.prior_strength
    raw: list[tuple[float, int, float, float, float]] = []
    for b in range(config.buckets):
        g = pairs[pairs["bin"] == b]
        if g.empty:
            continue
        wins = float((g["excess"] > 0).sum())
        n = len(g)
        raw.append(
            (
                float(g["z"].mean()),
                n,
                wins / n,
                (wins + k * base) / (n + k),
                (float(g["excess"].sum()) + k * mean_ex) / (n + k),
            )
        )
    weights = [float(r[1]) for r in raw]
    probs = _pav([r[3] for r in raw], weights)
    excess_mono = _pav([r[4] for r in raw], weights)
    bins = [
        CalibrationBin(z_mid=r[0], n=r[1], observed=r[2], probability=p, mean_excess=e)
        for r, p, e in zip(raw, probs, excess_mono, strict=True)
    ]
    return Calibration(bins=bins, base_rate=base, mean_excess=mean_ex)


# ----------------------------------------------------------------------------- the full run
@dataclass
class LivePrediction:
    symbol: str
    score: float
    z: float
    rank: int
    prob_outperform: float
    expected_excess: float


@dataclass
class AlphaRun:
    config: ModelConfig
    oos: ICStats
    ic_series: pd.Series
    baseline: ICStats
    baseline_ic_series: pd.Series
    buckets: list[float | None]
    backtest: Backtest
    calibration: Calibration
    importance: dict[str, tuple[float, float]]  # feature -> (mean coefficient, sign consistency)
    fits: list[tuple[pd.Timestamp, float, np.ndarray]]
    live_date: pd.Timestamp
    live: list[LivePrediction]
    live_lambda: float
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def baseline_scores(data: ModelData) -> pd.Series:
    cols = [c for c in BASELINE_WEIGHTS if c in data.X.columns]
    if not cols:
        raise DomainError("baseline features are missing from the feature set")
    w = np.array([BASELINE_WEIGHTS[c] for c in cols])
    return pd.Series(data.X[cols].to_numpy(dtype=float) @ w, index=data.X.index)


def run(data: ModelData, config: ModelConfig) -> AlphaRun:
    wf = walk_forward(data, config)
    pred_wide = wf.predictions.unstack()
    realized = data.fwd.reindex(index=pred_wide.index, columns=pred_wide.columns)
    oos, ic_series = ic_stats(pred_wide, realized, data.horizon)
    base_wide = baseline_scores(data).unstack().reindex(index=pred_wide.index, columns=pred_wide.columns)
    baseline, baseline_ic = ic_stats(base_wide, realized, data.horizon)
    bench = data.bench_fwd.reindex(pred_wide.index)
    calibration = calibrate(pred_wide, realized, bench, config)
    backtest = top_k_backtest(pred_wide, realized, bench, config, pd.DatetimeIndex(data.positions.index))
    coefs = np.array([c for _, _, c in wf.fits])
    mean_coef = coefs.mean(axis=0)
    consistency = (np.sign(coefs) == np.sign(mean_coef)).mean(axis=0)
    importance = {name: (float(mean_coef[i]), float(consistency[i])) for i, name in enumerate(data.X.columns)}
    coef, lam, last = fit_final(data, config)
    latest = data.predict(data.row_pos == data.pos(last), coef)
    scores = pd.Series(latest.to_numpy(), index=latest.index.get_level_values("symbol"))
    sd = float(scores.std(ddof=0))
    zs = (scores - scores.mean()) / sd if sd > 0 else scores * 0.0
    order = zs.sort_values(ascending=False)
    live = [
        LivePrediction(
            symbol=s,
            score=float(scores[s]),
            z=float(order[s]),
            rank=i + 1,
            prob_outperform=calibration.probability(float(order[s])),
            expected_excess=calibration.expected_excess(float(order[s])),
        )
        for i, s in enumerate(order.index)
    ]
    labelled = ic_series.index
    return AlphaRun(
        config=config,
        oos=oos,
        ic_series=ic_series,
        baseline=baseline,
        baseline_ic_series=baseline_ic,
        buckets=bucket_returns(pred_wide, realized, config.buckets),
        backtest=backtest,
        calibration=calibration,
        importance=importance,
        fits=wf.fits,
        live_date=last,
        live=live,
        live_lambda=lam,
        oos_start=labelled[0],
        oos_end=labelled[-1],
    )
