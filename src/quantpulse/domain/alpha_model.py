"""Cross-sectional return models trained walk-forward, with an honest out-of-sample track record.

Target
    For every date, the ``h``-day forward returns of the universe are ranked and mapped to normal scores
    (rank-Gauss). The models therefore predict *relative* performance: which stocks will do better than
    the others, not where the market goes. With a point-in-time universe (the S&P 500 as it was on each
    date) only the stocks that were members on a date are ranked, trained on and evaluated there, and a
    stock that is delisted inside the horizon keeps its last price (cash-out) instead of vanishing.

Models
    * ``ridge``: linear regression with an L2 penalty on per-date z-scored features, refitted monthly;
    * ``gbm``: gradient-boosted trees (histogram GBM) that can learn interactions and non-linear effects
      (e.g. value only working among profitable firms), refitted quarterly on at most ``gbm_max_rows``
      randomly sampled rows;
    * ``ensemble``: the average of the two models' per-date z-scores (the default).

    Hyper-parameters (the ridge penalty, the tree size) are chosen inside each training window on a purged
    hold-out: the last 25% of the window is the validation set and training labels that overlap it are
    dropped.

Walk-forward protocol (no look-ahead)
    To predict at the close of day *t*, a model may only use labels already realised by *t*: a sample
    from day *s* has a label ending at *s + h*, so training uses days *s ≤ t − h* (purging), within a
    rolling ``train_window``. Coefficients or trees are frozen between refits. Every prediction reported
    as out-of-sample was made this way, and every model type is evaluated on the same dates.

Evaluation (all out-of-sample)
    * information coefficient (IC): per-date Spearman correlation of prediction vs realised return,
      its mean, and a t-statistic on non-overlapping dates; also the IC *within industries* (realised
      returns minus the industry average), which isolates stock picking from industry bets;
    * hit rate: how often a stock predicted above the median actually finished above it (and vice versa);
    * bucket returns: average realised return of each prediction quintile, and the top-minus-bottom spread;
    * a top-``k`` long-only portfolio rebalanced every ``h`` days, net of trading costs, against an
      equal-weight universe and the benchmark;
    * the same statistics for the platform's hand-set factor rule, as a baseline to beat.

Probabilities
    Out-of-sample predictions are bucketed; in each bucket the observed frequency of beating the benchmark
    over ``h`` days (and the mean excess return) is shrunk towards the overall base rate with
    ``prior_strength`` pseudo-observations and made monotone (pool-adjacent-violators). Live predictions
    are mapped through that table. A model without skill therefore reports probabilities close to the base
    rate instead of confident-sounding numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
MODEL_TYPES: tuple[str, ...] = ("ridge", "gbm", "ensemble")
MODEL_LABELS: dict[str, str] = {
    "ridge": "Ridge regression",
    "gbm": "Gradient-boosted trees",
    "ensemble": "Ensemble (ridge + trees)",
    "baseline": "Hand-set factor rule",
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
    features: tuple[str, ...] | None = None  # None: every column of the feature matrix
    model_type: str = "ensemble"
    compare: bool = True  # also evaluate the other model types on the same dates
    gbm_retrain_every: int = 63
    gbm_max_rows: int = 150_000
    gbm_grid: tuple[tuple[int, int], ...] = ((7, 100), (31, 60))  # (leaves per tree, trees)
    gbm_tune_every: int = 4  # re-choose the tree size every this many refits
    seed: int = 7

    def __post_init__(self) -> None:
        if not 1 <= self.horizon <= 126:
            raise DomainError("horizon must be between 1 and 126 trading days")
        if self.min_train < 60 or self.train_window < self.min_train:
            raise DomainError("need min_train >= 60 and train_window >= min_train")
        if self.retrain_every < 1 or self.gbm_retrain_every < 1 or self.top_k < 1 or self.buckets < 2:
            raise DomainError("retrain intervals and top_k must be >= 1 and buckets >= 2")
        if not self.lambdas or any(lam <= 0 for lam in self.lambdas):
            raise DomainError("lambdas must be positive")
        if self.model_type not in MODEL_TYPES:
            raise DomainError(f"model_type must be one of {', '.join(MODEL_TYPES)}")
        if not self.gbm_grid or any(leaves < 2 or trees < 1 for leaves, trees in self.gbm_grid):
            raise DomainError("gbm_grid needs (leaves >= 2, trees >= 1) pairs")
        if self.features is not None:
            unknown = [f for f in self.features if f not in feat.all_features()]
            if unknown:
                raise DomainError(f"unknown features: {unknown}")

    @property
    def model_kinds(self) -> list[str]:
        """The base learners that must be trained (the ensemble needs both)."""
        if self.compare or self.model_type == "ensemble":
            return ["ridge", "gbm"]
        return [self.model_type]


# ----------------------------------------------------------------------------- data plumbing
@dataclass
class ModelData:
    """Aligned inputs: long feature matrix, rank-Gauss labels, raw forward returns and date positions.

    Everything the walk-forward loop touches is precomputed as NumPy arrays indexed by row, with each row's
    position in the trading calendar, so selecting "all rows known by day t" is a vector comparison."""

    X: pd.DataFrame  # (date, symbol) -> feature z-scores
    fwd: pd.DataFrame  # wide forward returns (dates x symbols), NaN where not yet realised / not eligible
    bench_fwd: pd.Series  # benchmark forward return per date
    positions: pd.Series  # date -> integer position in the trading calendar
    horizon: int
    sectors: pd.Series | None = None  # symbol -> industry, for within-industry evaluation
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
        self.X_np = self.X.to_numpy(dtype=np.float32)
        self.y_np = self.y.to_numpy(dtype=float)
        self.row_pos = self.positions.reindex(self.X.index.get_level_values("date")).to_numpy(dtype=int)
        self.date_pos = np.unique(self.row_pos)

    @property
    def calendar(self) -> pd.Index:
        return self.positions.index

    @property
    def dates(self) -> list[pd.Timestamp]:
        return [self.calendar[p] for p in self.date_pos]

    @property
    def feature_names(self) -> list[str]:
        return [str(c) for c in self.X.columns]

    def pos(self, d: pd.Timestamp) -> int:
        return int(self.positions[d])

    def predict(self, mask: np.ndarray, coef: np.ndarray) -> pd.Series:
        return pd.Series(self.X_np[mask].astype(np.float64) @ coef, index=self.X.index[mask])


def prepare_features(
    raw: Mapping[str, pd.DataFrame], sectors: pd.Series | None, neutral: bool
) -> dict[str, pd.DataFrame]:
    """Industry-neutralise every stock-level feature (sector features are industry averages already)."""
    if sectors is None or not neutral:
        return dict(raw)
    return {
        name: wide if name in feat.SECTOR_FEATURES else feat.neutralise(wide, sectors)
        for name, wide in raw.items()
    }


def default_names(raw: Mapping[str, pd.DataFrame]) -> list[str]:
    order = list(feat.all_features())
    return [n for n in order if n in raw]


def raw_features(
    panel: feat.Panel,
    *,
    extra: Mapping[str, pd.DataFrame] | None = None,
    sectors: pd.Series | None = None,
    eligible: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Every raw feature panel (price, extra, industry averages), blanked where a stock is not eligible.
    Returns the panels and the aligned eligibility mask."""
    close = panel.close
    raw = feat.compute_features(panel)
    for name, frame in (extra or {}).items():
        raw[name] = frame.reindex(index=close.index, columns=close.columns)
    mask = None
    if eligible is not None:
        mask = eligible.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        raw = {k: v.where(mask) for k, v in raw.items()}
    if sectors is not None:
        raw.update(feat.sector_features(raw, sectors))
    return raw, mask


def build_data(
    panel: feat.Panel,
    horizon: int,
    names: Sequence[str] | None = None,
    min_coverage: float = 0.9,
    *,
    extra: Mapping[str, pd.DataFrame] | None = None,
    sectors: pd.Series | None = None,
    eligible: pd.DataFrame | None = None,
    neutral: bool = True,
) -> tuple[ModelData, dict[str, pd.DataFrame]]:
    """Features, labels and positions for the walk-forward loop.

    ``extra``: additional raw feature panels (earnings, fundamentals). ``sectors``: symbol → industry,
    enabling industry features and industry-neutral features. ``eligible``: dates × symbols mask of
    index membership; ineligible cells are excluded from every cross-section, label and evaluation.
    Returns the data and the raw (un-neutralised, masked) feature panels."""
    close = panel.close
    raw, mask = raw_features(panel, extra=extra, sectors=sectors, eligible=eligible)
    chosen = list(names) if names is not None else default_names(raw)
    unknown = [n for n in chosen if n not in raw]
    if unknown:
        raise DomainError(f"features not available: {unknown}")
    prepared = prepare_features({n: raw[n] for n in chosen}, sectors, neutral)
    price_names = [n for n in chosen if n in feat.FEATURES]
    X = feat.feature_matrix(
        prepared, chosen, min_coverage, coverage_names=price_names or None, dtype=np.float32
    )
    if X.empty:
        raise DomainError(f"no dates with enough feature history (the first {feat.WARMUP} days are warm-up)")
    positions = pd.Series(np.arange(len(close.index)), index=close.index)
    # A listing that ends keeps its last price: the holder is cashed out rather than the stock vanishing
    # from the evaluation (which would flatter every backtest).
    fwd = feat.forward_returns(close.ffill(), horizon)
    if mask is not None:
        fwd = fwd.where(mask)
    bench_fwd = panel.benchmark.shift(-horizon) / panel.benchmark - 1
    data = ModelData(X=X, fwd=fwd, bench_fwd=bench_fwd, positions=positions, horizon=horizon, sectors=sectors)
    return data, raw


# ----------------------------------------------------------------------------- learners
def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge without intercept on standardised inputs: ``(XᵀX/n + λI)⁻¹ Xᵀy/n``."""
    n, p = X.shape
    if n == 0:
        raise DomainError("no training samples")
    X = X.astype(np.float64, copy=False)
    return np.linalg.solve(X.T @ X / n + lam * np.eye(p), X.T @ y / n)


@dataclass
class Fitted:
    """One trained learner: ridge coefficients or a tree ensemble, plus the chosen hyper-parameters."""

    kind: str
    params: tuple[float, ...]
    coef: np.ndarray | None = None
    model: Any = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef is not None:
            return X.astype(np.float64, copy=False) @ self.coef
        return np.asarray(self.model.predict(X), dtype=float)

    @property
    def label(self) -> str:
        if self.kind == "ridge":
            return f"λ={self.params[0]:g}"
        return f"{int(self.params[0])} leaves × {int(self.params[1])} trees"


def _mean_ic(pred: pd.Series, target_wide: pd.DataFrame) -> float:
    if pred.empty:
        return float("nan")
    wide = pred.unstack()
    ic = feat.row_spearman(wide, target_wide.reindex(index=wide.index, columns=wide.columns)).dropna()
    return float(ic.mean()) if len(ic) else float("nan")


def _window(data: ModelData, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    rows = (data.row_pos > lo) & (data.row_pos <= hi) & np.isfinite(data.y_np)
    days = data.date_pos[(data.date_pos > lo) & (data.date_pos <= hi)]
    return rows, days


def _holdout(data: ModelData, rows: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Purged split: the last 25% of the window validates; training labels overlapping it are dropped."""
    split = int(len(days) * 0.75)
    if split < 60 or len(days) - split < 20:
        return None
    val_start = int(days[split])
    inner = rows & (data.row_pos + data.horizon <= val_start)
    val = rows & (data.row_pos >= val_start)
    if np.unique(data.row_pos[inner]).size < 40:
        return None
    return inner, val


def _val_ic(data: ModelData, val: np.ndarray, pred: np.ndarray) -> float:
    return _mean_ic(pd.Series(pred, index=data.X.index[val]), data.y_wide)


def _best(scores: Mapping[Any, float], tie_break: Any) -> Any | None:
    finite = {k: v for k, v in scores.items() if math.isfinite(v)}
    if not finite:
        return None
    return max(finite, key=lambda k: (round(finite[k], 6), tie_break(k)))


def train_ridge(data: ModelData, lo: int, hi: int, config: ModelConfig) -> Fitted:
    """Fit on sample days with calendar position in ``(lo, hi]``; λ is chosen on the purged hold-out
    (ties go to the stronger penalty), then the model is refitted on all of them."""
    rows, days = _window(data, lo, hi)
    lam = sorted(config.lambdas)[len(config.lambdas) // 2]
    split = _holdout(data, rows, days)
    if split is not None:
        inner, val = split
        Xi, yi, Xv = data.X_np[inner], data.y_np[inner], data.X_np[val].astype(np.float64)
        scores = {c: _val_ic(data, val, Xv @ fit_ridge(Xi, yi, c)) for c in config.lambdas}
        lam = _best(scores, lambda c: c) or lam
    return Fitted("ridge", (lam,), coef=fit_ridge(data.X_np[rows], data.y_np[rows], lam))


def _tree_model(params: tuple[float, ...], n_rows: int, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    leaves, trees = params
    return HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_iter=int(trees),
        max_leaf_nodes=int(leaves),
        min_samples_leaf=max(20, n_rows // 500),
        l2_regularization=1.0,
        max_bins=63,
        early_stopping=False,
        random_state=seed,
    )


def _sample(mask: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size > max_rows:
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    return idx


def train_gbm(
    data: ModelData, lo: int, hi: int, config: ModelConfig, params: tuple[float, ...] | None = None
) -> Fitted:
    """Gradient-boosted trees on at most ``gbm_max_rows`` sampled rows. Without ``params`` the tree size
    is chosen from ``gbm_grid`` on the purged hold-out (ties go to the smaller model)."""
    rows, days = _window(data, lo, hi)
    if not rows.any():
        raise DomainError("no training samples")
    rng = np.random.default_rng(config.seed + max(hi, 0))
    if params is None:
        params = tuple(float(x) for x in config.gbm_grid[0])
        split = _holdout(data, rows, days) if len(config.gbm_grid) > 1 else None
        if split is not None:
            inner, val = split
            ii = _sample(inner, config.gbm_max_rows, rng)
            Xv = data.X_np[val]
            scores: dict[tuple[float, ...], float] = {}
            for g in config.gbm_grid:
                key = tuple(float(x) for x in g)
                m = _tree_model(key, ii.size, config.seed).fit(data.X_np[ii], data.y_np[ii])
                scores[key] = _val_ic(data, val, np.asarray(m.predict(Xv), dtype=float))
            params = _best(scores, lambda k: -k[0] * k[1]) or params
    idx = _sample(rows, config.gbm_max_rows, rng)
    model = _tree_model(params, idx.size, config.seed).fit(data.X_np[idx], data.y_np[idx])
    return Fitted("gbm", params, model=model)


def _zs(v: np.ndarray, clip: float = 3.0) -> np.ndarray:
    sd = float(np.std(v))
    return np.clip((v - v.mean()) / sd, -clip, clip) if sd > 0 else np.zeros_like(v)


@dataclass
class FinalModel:
    """The live model(s): trained on every label realised by the last date."""

    kind: str
    ridge: Fitted | None
    gbm: Fitted | None

    def score(self, X: np.ndarray) -> np.ndarray:
        """Scores for one cross-section (the ensemble z-scores each learner within it)."""
        if self.kind == "ridge" and self.ridge is not None:
            return self.ridge.predict(X)
        if self.kind == "gbm" and self.gbm is not None:
            return self.gbm.predict(X)
        if self.ridge is None or self.gbm is None:
            raise DomainError("the ensemble needs both learners")
        return (_zs(self.ridge.predict(X)) + _zs(self.gbm.predict(X))) / 2


def combine(predictions: Sequence[pd.Series]) -> pd.Series:
    """Average of per-date z-scores, on the dates every model predicted."""
    zs = [feat.cross_sectional_z(p.unstack()) for p in predictions]
    dates = zs[0].index
    for z in zs[1:]:
        dates = dates.intersection(z.index)
    cols = zs[0].columns
    total = sum(z.reindex(index=dates, columns=cols) for z in zs)
    out = (total / len(zs)).stack(future_stack=True).dropna()
    out.index.names = ["date", "symbol"]
    return out


@dataclass
class WalkForward:
    predictions: pd.Series  # out-of-sample, (date, symbol)
    fits: list[tuple[pd.Timestamp, Fitted]]  # (fit date, fitted learner)


Progress = Callable[[float, str], None]


def walk_forward(
    data: ModelData, config: ModelConfig, kind: str = "ridge", progress: Progress | None = None
) -> WalkForward:
    if kind not in ("ridge", "gbm"):
        raise DomainError("walk_forward trains 'ridge' or 'gbm' learners")
    dp = data.date_pos
    h = data.horizon
    every = config.retrain_every if kind == "ridge" else config.gbm_retrain_every
    fits: list[tuple[pd.Timestamp, Fitted]] = []
    segments: list[tuple[int, Fitted]] = []  # (index into dp where the learner takes over, learner)
    last_fit: int | None = None
    params: tuple[float, ...] | None = None
    for k, p in enumerate(dp):
        if last_fit is not None and k - last_fit < every:
            continue
        cutoff = int(p) - h  # only labels realised by day p
        n_days = int(np.count_nonzero((dp <= cutoff) & (dp > cutoff - config.train_window)))
        if n_days < config.min_train:
            continue
        lo = cutoff - config.train_window
        if kind == "ridge":
            fitted = train_ridge(data, lo, cutoff, config)
        else:
            tune = params is None or len(fits) % config.gbm_tune_every == 0
            fitted = train_gbm(data, lo, cutoff, config, None if tune else params)
            params = fitted.params
        fits.append((data.calendar[p], fitted))
        segments.append((k, fitted))
        last_fit = k
        if progress is not None:
            progress(k / len(dp), f"training {MODEL_LABELS[kind].lower()} (refit {len(fits)})")
    if not fits:
        raise DomainError(
            f"not enough history: the model needs {config.min_train} labelled days after the "
            f"{feat.WARMUP}-day warm-up, plus the {h}-day label horizon"
        )
    preds = []
    for i, (k0, fitted) in enumerate(segments):
        k1 = segments[i + 1][0] if i + 1 < len(segments) else len(dp)
        mask = (data.row_pos >= dp[k0]) & (data.row_pos <= dp[k1 - 1])
        preds.append(pd.Series(fitted.predict(data.X_np[mask]), index=data.X.index[mask]))
    out = pd.concat(preds)
    out.index.names = ["date", "symbol"]
    return WalkForward(predictions=out, fits=fits)


def fit_final(data: ModelData, config: ModelConfig) -> tuple[FinalModel, pd.Timestamp]:
    """The live model: every learner trained on each label realised by the last date (within the window)."""
    last = int(data.date_pos[-1])
    cutoff = last - data.horizon
    n_days = int(np.count_nonzero((data.date_pos <= cutoff) & (data.date_pos > cutoff - config.train_window)))
    if n_days < config.min_train:
        raise DomainError("not enough labelled history to fit the live model")
    lo = cutoff - config.train_window
    kinds = config.model_kinds
    ridge = train_ridge(data, lo, cutoff, config) if "ridge" in kinds else None
    gbm = train_gbm(data, lo, cutoff, config) if "gbm" in kinds else None
    return FinalModel(config.model_type, ridge, gbm), data.calendar[last]


def tree_importance(
    data: ModelData, final: FinalModel, config: ModelConfig, max_rows: int = 20_000
) -> dict[str, float]:
    """Permutation importance of the live tree model on its most recent training rows: how much the
    correlation between its predictions and the target drops when one feature is shuffled."""
    if final.gbm is None:
        return {}
    last = int(data.date_pos[-1])
    cutoff = last - data.horizon
    rows, _ = _window(data, cutoff - config.train_window // 4, cutoff)
    rng = np.random.default_rng(config.seed)
    idx = _sample(rows, max_rows, rng)
    if idx.size < 50:
        return {}
    X, y = data.X_np[idx], data.y_np[idx]
    base = float(np.corrcoef(final.gbm.predict(X), y)[0, 1])
    out: dict[str, float] = {}
    for j, name in enumerate(data.feature_names):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        out[name] = base - float(np.corrcoef(final.gbm.predict(Xp), y)[0, 1])
    return out


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
class ModelEval:
    """Out-of-sample results of one model type (or the baseline rule) on the common dates."""

    name: str
    predictions: pd.Series
    oos: ICStats
    ic_series: pd.Series
    within_sector: ICStats | None
    buckets: list[float | None]
    backtest: Backtest
    refits: int

    @property
    def spread(self) -> float | None:
        lo, hi = self.buckets[0], self.buckets[-1]
        return None if lo is None or hi is None else hi - lo


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
    importance: dict[str, tuple[float, float]]  # feature -> (mean ridge coefficient, sign consistency)
    tree_importance: dict[str, float]  # feature -> permutation importance in the live tree model
    fits: dict[str, list[tuple[pd.Timestamp, Fitted]]]  # learner -> walk-forward refits
    models: dict[str, ModelEval]  # every evaluated model type + "baseline"
    within_sector: ICStats | None
    final: FinalModel
    live_date: pd.Timestamp
    live: list[LivePrediction]
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    @property
    def chosen(self) -> str:
        return self.config.model_type


def baseline_scores(data: ModelData) -> pd.Series:
    cols = [c for c in BASELINE_WEIGHTS if c in data.X.columns]
    if not cols:
        raise DomainError("baseline features are missing from the feature set")
    w = np.array([BASELINE_WEIGHTS[c] for c in cols])
    return pd.Series(data.X[cols].to_numpy(dtype=float) @ w, index=data.X.index)


def evaluate(
    name: str, predictions: pd.Series, data: ModelData, config: ModelConfig, refits: int = 0
) -> ModelEval:
    pred_wide = predictions.unstack()
    realized = data.fwd.reindex(index=pred_wide.index, columns=pred_wide.columns)
    oos, ic_series = ic_stats(pred_wide, realized, data.horizon)
    within = None
    if data.sectors is not None:
        try:
            within, _ = ic_stats(pred_wide, feat.neutralise(realized, data.sectors), data.horizon)
        except DomainError:
            within = None
    bench = data.bench_fwd.reindex(pred_wide.index)
    return ModelEval(
        name=name,
        predictions=predictions,
        oos=oos,
        ic_series=ic_series,
        within_sector=within,
        buckets=bucket_returns(pred_wide, realized, config.buckets),
        backtest=top_k_backtest(pred_wide, realized, bench, config, pd.DatetimeIndex(data.positions.index)),
        refits=refits,
    )


def run(data: ModelData, config: ModelConfig, progress: Progress | None = None) -> AlphaRun:
    kinds = config.model_kinds
    share = {"ridge": 0.3, "gbm": 0.55} if len(kinds) == 2 else {kinds[0]: 0.85}
    walks: dict[str, WalkForward] = {}
    offset = 0.0
    for k in kinds:
        sub = None
        if progress is not None:
            lo, width = offset, share[k]

            def sub(f: float, stage: str, lo: float = lo, width: float = width) -> None:
                progress(lo + width * f, stage)

        walks[k] = walk_forward(data, config, k, sub)
        offset += share[k]
    if progress is not None:
        progress(offset, "evaluating out of sample")
    preds = {k: w.predictions for k, w in walks.items()}
    if "ridge" in preds and "gbm" in preds:
        preds["ensemble"] = combine([preds["ridge"], preds["gbm"]])
    chosen = config.model_type
    # Every model is judged on the same dates: those the chosen model predicted.
    dates = preds[chosen].index.get_level_values("date").unique()
    refits = {"ridge": len(walks["ridge"].fits) if "ridge" in walks else 0}
    refits["gbm"] = len(walks["gbm"].fits) if "gbm" in walks else 0
    refits["ensemble"] = refits["ridge"] + refits["gbm"]
    models: dict[str, ModelEval] = {}
    for name, p in preds.items():
        on = p[p.index.get_level_values("date").isin(dates)]
        models[name] = evaluate(name, on, data, config, refits[name])
    base = baseline_scores(data)
    base = base[base.index.get_level_values("date").isin(dates)]
    models["baseline"] = evaluate("baseline", base, data, config)
    main = models[chosen]

    pred_wide = main.predictions.unstack()
    realized = data.fwd.reindex(index=pred_wide.index, columns=pred_wide.columns)
    calibration = calibrate(pred_wide, realized, data.bench_fwd.reindex(pred_wide.index), config)

    importance: dict[str, tuple[float, float]] = {}
    if "ridge" in walks:
        coefs = np.array([f.coef for _, f in walks["ridge"].fits if f.coef is not None])
        mean_coef = coefs.mean(axis=0)
        consistency = (np.sign(coefs) == np.sign(mean_coef)).mean(axis=0)
        importance = {
            name: (float(mean_coef[i]), float(consistency[i])) for i, name in enumerate(data.feature_names)
        }

    if progress is not None:
        progress(0.9, "fitting the live model")
    final, last = fit_final(data, config)
    latest = data.row_pos == data.pos(last)
    symbols = data.X.index[latest].get_level_values("symbol")
    scores = pd.Series(final.score(data.X_np[latest]), index=symbols)
    sd = float(scores.std(ddof=0))
    zs = (scores - scores.mean()) / sd if sd > 0 else scores * 0.0
    order = zs.sort_values(ascending=False)
    live = [
        LivePrediction(
            symbol=str(s),
            score=float(scores[s]),
            z=float(order[s]),
            rank=i + 1,
            prob_outperform=calibration.probability(float(order[s])),
            expected_excess=calibration.expected_excess(float(order[s])),
        )
        for i, s in enumerate(order.index)
    ]
    labelled = main.ic_series.index
    return AlphaRun(
        config=config,
        oos=main.oos,
        ic_series=main.ic_series,
        baseline=models["baseline"].oos,
        baseline_ic_series=models["baseline"].ic_series,
        buckets=main.buckets,
        backtest=main.backtest,
        calibration=calibration,
        importance=importance,
        tree_importance=tree_importance(data, final, config),
        fits={k: w.fits for k, w in walks.items()},
        models=models,
        within_sector=main.within_sector,
        final=final,
        live_date=last,
        live=live,
        oos_start=labelled[0],
        oos_end=labelled[-1],
    )
