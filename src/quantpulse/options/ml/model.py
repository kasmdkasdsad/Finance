"""The edge model: what a candidate is likely to return per dollar at risk, how sure that is, and why.

Components (scikit-learn only; every one fitted with the rows' evidence weights):

* three **gradient-boosted quantile regressors** — the 10th, 50th and 90th percentile of the return per dollar
  at risk (nonlinear, interaction-aware, missing values routed natively; the family is a categorical feature and
  costs carry monotone constraints: paying more never helps);
* a gradient-boosted **mean** regressor, **stacked** with a ridge baseline (standardised, median-imputed, the
  family one-hot) by the weight that minimises out-of-sample error — the linear model steadies the trees where
  data is thin (Bali, Beckmeyer, Moerke and Weigert, 2023: nonlinearity helps most when combined with the
  linear signal);
* a gradient-boosted **classifier** for the probability of profit, **isotonic-calibrated** on out-of-sample
  predictions only (70% means 70%);
* **conformalized quantile regression** (Romano, Patterson and Candès, 2019): the 10–90 interval widened by the
  out-of-sample quantile of its own misses, so it covers about 80% of outcomes on data it has not seen — its
  lower bound is the model's conservative estimate of the edge.

Validation is :func:`.cv.walk_forward` (purged, embargoed, expanding) for every number that a gate reads, and
:func:`.cv.combinatorial` for the distribution of out-of-sample paths. The **champion** is the rule the Brain
already uses — the expected value per dollar at risk under the market's distribution (``eor_market``) — so the
model is compared with what it would replace on exactly the same rows. In-sample fit is never reported as
evidence.
"""

from __future__ import annotations

import io
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from .cv import combinatorial, walk_forward
from .dataset import Dataset
from .drift import out_of_range, reference
from .features import CATEGORICAL, FAMILY_CODES, FEATURES, MONOTONE

MODEL_VERSION = 1
QUANTILES = (0.1, 0.5, 0.9)
ALPHA = 0.2  # the conformal interval covers 1 − α of outcomes
STACK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPLAIN_TOP = 8


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return 0.0
    from scipy.stats import spearmanr

    r = spearmanr(a[ok], b[ok]).statistic
    return float(r) if math.isfinite(r) else 0.0


def _quintiles(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean outcome of the top and bottom fifth of rows ranked by ``pred``."""
    if len(y) < 10:
        return math.nan, math.nan
    order = np.argsort(pred)
    k = max(1, len(y) // 5)
    return float(y[order[-k:]].mean()), float(y[order[:k]].mean())


@dataclass
class EdgePrediction:
    expected: float  # stacked expected return per $ at risk
    median: float
    lower: float  # conformal lower bound (1 − α interval)
    upper: float
    p_win: float  # calibrated probability of profit
    out_of_range: float  # share of features outside the training range
    drivers: list[tuple[str, float]] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def as_dict(self) -> dict[str, Any]:
        return {"expected": round(self.expected, 4), "median": round(self.median, 4), "lower": round(self.lower, 4),
                "upper": round(self.upper, 4), "p_win": round(self.p_win, 4),
                "out_of_range": round(self.out_of_range, 3),
                "drivers": [[n, round(v, 4)] for n, v in self.drivers]}  # fmt: skip


@dataclass
class Components:
    dead: np.ndarray  # columns with no value at all in the training rows (filled with 0: they never split)
    quantile: dict[float, Any]
    mean: Any
    classifier: Any
    ridge: Any
    ridge_center: np.ndarray
    ridge_scale: np.ndarray


class OptionsEdgeModel:
    def __init__(self, *, max_iter: int = 200, learning_rate: float = 0.05, max_leaf_nodes: int = 31,
                 min_samples_leaf: int = 40, l2: float = 1.0, seed: int = 7, n_splits: int = 5,
                 embargo_days: int = 5) -> None:  # fmt: skip
        self.params: dict[str, Any] = {"max_iter": max_iter, "learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes,
                       "min_samples_leaf": min_samples_leaf, "l2": l2, "seed": seed, "n_splits": n_splits,
                       "embargo_days": embargo_days}  # fmt: skip
        self.names: tuple[str, ...] = FEATURES
        self.parts: Components | None = None
        self.stack_weight = 1.0
        self.calibration: Any = None
        self.conformal_q = 0.0
        self.medians = np.zeros(len(FEATURES))
        self.importance: list[tuple[str, float]] = []
        self.reference: dict[str, dict[str, Any]] = {}
        self.report: dict[str, Any] = {}
        self.sklearn_version = ""

    # ------------------------------------------------------------------ components
    def _hgb(self, kind: str, **kw: Any) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        p = self.params
        cat = [n in CATEGORICAL for n in self.names]
        mono = [0 if n in CATEGORICAL else MONOTONE.get(n, 0) for n in self.names]
        common = {"max_iter": p["max_iter"], "learning_rate": p["learning_rate"], "max_leaf_nodes": p["max_leaf_nodes"],
                  "min_samples_leaf": p["min_samples_leaf"], "l2_regularization": p["l2"], "random_state": p["seed"],
                  "categorical_features": cat, "early_stopping": False}  # fmt: skip
        if kind == "classifier":
            return HistGradientBoostingClassifier(monotonic_cst=mono, **common, **kw)
        return HistGradientBoostingRegressor(monotonic_cst=mono, **common, **kw)

    def _ridge_design(self, X: np.ndarray, center: np.ndarray | None = None, scale: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
        num = [j for j, n in enumerate(self.names) if n not in CATEGORICAL]
        Z = X[:, num].copy()
        if center is None:
            center = np.nanmedian(np.where(np.isfinite(Z), Z, np.nan), axis=0)
            center = np.where(np.isfinite(center), center, 0.0)
        Z = np.where(np.isfinite(Z), Z, center)
        if scale is None:
            scale = Z.std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
        Z = np.clip((Z - center) / scale, -6, 6)
        fam = X[:, self.names.index("family")]
        onehot = np.zeros((len(X), len(FAMILY_CODES)))
        ok = np.isfinite(fam)
        onehot[np.flatnonzero(ok), fam[ok].astype(int)] = 1.0
        return np.hstack([Z, onehot]), center, scale

    def _fit_parts(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Components:
        from sklearn.linear_model import Ridge

        dead = ~np.isfinite(X).any(axis=0)
        X = _fill_dead(X, dead)
        win = (y > 0).astype(int)
        quantile = {
            q: self._hgb("regressor", loss="quantile", quantile=q).fit(X, y, sample_weight=w)
            for q in QUANTILES
        }
        mean = self._hgb("regressor", loss="squared_error").fit(X, y, sample_weight=w)
        if len(set(win.tolist())) > 1:
            classifier = self._hgb("classifier").fit(X, win, sample_weight=w)
        else:
            classifier = float(win.mean())  # one class only: a constant
        Z, c, s = self._ridge_design(X)
        ridge = Ridge(alpha=10.0).fit(Z, y, sample_weight=w)
        return Components(dead, quantile, mean, classifier, ridge, c, s)

    def _raw(self, parts: Components, X: np.ndarray) -> dict[str, np.ndarray]:
        X = _fill_dead(X, parts.dead)
        q = {k: m.predict(X) for k, m in parts.quantile.items()}
        lo, med, hi = np.minimum(q[0.1], q[0.5]), q[0.5], np.maximum(q[0.9], q[0.5])
        p = (parts.classifier.predict_proba(X)[:, 1] if not isinstance(parts.classifier, float)
             else np.full(len(X), parts.classifier))  # fmt: skip
        Z, _, _ = self._ridge_design(X, parts.ridge_center, parts.ridge_scale)
        return {
            "lo": lo,
            "med": med,
            "hi": hi,
            "mean": parts.mean.predict(X),
            "ridge": parts.ridge.predict(Z),
            "p": p,
        }

    # ------------------------------------------------------------------ fitting
    def fit(self, ds: Dataset) -> dict[str, Any]:
        """Out-of-sample validation first (everything the report says comes from it), then the final fit on
        all rows. Returns the report; raises ``ValueError`` when there is too little data to validate."""
        import sklearn
        from sklearn.isotonic import IsotonicRegression

        self.sklearn_version = sklearn.__version__
        X, y, w = ds.X, ds.y, ds.weights()
        rule = ds.column("eor_market")
        p = self.params
        splits = walk_forward(ds.t0, ds.t1, n_splits=p["n_splits"], embargo_days=p["embargo_days"])
        if len(splits) < 2:
            raise ValueError(
                f"too little history to validate out of sample ({ds.n} rows, {len(splits)} splits)"
            )
        oos = {k: np.full(ds.n, np.nan) for k in ("lo", "med", "hi", "mean", "ridge", "p")}
        folds = []
        for train, test in splits:
            parts = self._fit_parts(X[train], y[train], w[train])
            raw = self._raw(parts, X[test])
            for k, v in raw.items():
                oos[k][test] = v
            folds.append((train, test))
        seen = np.isfinite(oos["mean"])
        yo = y[seen]
        # the stacking weight and the calibration: chosen on out-of-sample predictions only
        best = min(
            STACK_GRID,
            key=lambda a: float(np.mean((a * oos["mean"][seen] + (1 - a) * oos["ridge"][seen] - yo) ** 2)),
        )
        self.stack_weight = float(best)
        stacked = self.stack_weight * oos["mean"] + (1 - self.stack_weight) * oos["ridge"]
        win = (y > 0).astype(float)
        self.calibration = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            oos["p"][seen], win[seen]
        )
        p_cal = np.full(ds.n, np.nan)
        p_cal[seen] = self.calibration.predict(oos["p"][seen])
        # conformal: the quantile of the interval's misses, out of sample (and its coverage on the last fold,
        # with the quantile taken from the earlier folds only)
        miss = np.maximum(oos["lo"] - y, y - oos["hi"])
        self.conformal_q = _conformal_q(miss[seen])
        last_test = folds[-1][1]
        earlier = seen.copy()
        earlier[last_test] = False
        q_early = _conformal_q(miss[earlier]) if earlier.sum() >= 30 else self.conformal_q
        covered = (y[last_test] >= oos["lo"][last_test] - q_early) & (
            y[last_test] <= oos["hi"][last_test] + q_early
        )
        report = self._evaluate(ds, stacked, p_cal, rule, folds, seen)
        report["conformal"] = {"alpha": ALPHA, "q": round(self.conformal_q, 4),
                               "holdout_coverage": round(float(covered.mean()), 4) if len(covered) else None,
                               "mean_width": round(float(np.nanmean(oos["hi"][seen] - oos["lo"][seen]) + 2 * self.conformal_q), 4)}  # fmt: skip
        report["stress"] = self._stress(ds, stacked, rule, seen, w)
        report["walk_forward"] = _walk_forward_verdict(report["folds"])
        # global importance: permutation on the last fold, refitted parts (out of sample)
        self.importance = self._permutation(X, y, folds[-1], w)
        report["importance"] = [[n, round(v, 5)] for n, v in self.importance[:15]]
        # the final model on everything, and what "normal" looks like for drift checks
        self.parts = self._fit_parts(X, y, w)
        self.medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in X.T])
        self.reference = reference(X, self.names)
        report["data"] = ds.summary()
        report["stacking_weight"] = self.stack_weight
        report["params"] = dict(self.params)
        report["version"] = MODEL_VERSION
        self.report = report
        return report

    def _evaluate(self, ds: Dataset, stacked: np.ndarray, p_cal: np.ndarray, rule: np.ndarray,
                  folds: Sequence[tuple[np.ndarray, np.ndarray]], seen: np.ndarray) -> dict[str, Any]:  # fmt: skip
        y = ds.y
        per_fold = []
        for _, test in folds:
            top, _bottom = _quintiles(stacked[test], y[test])
            rule_top, _ = _quintiles(np.nan_to_num(rule[test], nan=-9.0), y[test])
            per_fold.append({"from": date.fromordinal(int(ds.t0[test].min())).isoformat(),
                             "to": date.fromordinal(int(ds.t0[test].max())).isoformat(), "n": len(test),
                             "ic": round(_spearman(stacked[test], y[test]), 4),
                             "rule_ic": round(_spearman(rule[test], y[test]), 4),
                             "top_quintile": round(top, 4), "rule_top_quintile": round(rule_top, 4),
                             "mean": round(float(y[test].mean()), 4)})  # fmt: skip
        ys, ss = y[seen], stacked[seen]
        ics = np.array([f["ic"] for f in per_fold])
        top, bottom = _quintiles(ss, ys)
        rule_top, _ = _quintiles(np.nan_to_num(rule[seen], nan=-9.0), ys)
        win = (ys > 0).astype(float)
        brier = float(np.mean((p_cal[seen] - win) ** 2))
        base = float(np.mean((win.mean() - win) ** 2))
        naive = float(np.mean((ys - ys.mean()) ** 2))
        by_grade = {}
        for g in sorted(set(ds.grade[seen].tolist())):
            m = ds.grade[seen] == g
            if m.sum() >= 30:
                by_grade[str(g)] = {"n": int(m.sum()), "ic": round(_spearman(ss[m], ys[m]), 4),
                                    "rule_ic": round(_spearman(rule[seen][m], ys[m]), 4)}  # fmt: skip
        return {
            "folds": per_fold,
            "oos": {
                "n": int(seen.sum()),
                "ic": round(_spearman(ss, ys), 4),
                "rule_ic": round(_spearman(rule[seen], ys), 4),
                "ic_mean": round(float(ics.mean()), 4),
                "ic_t": round(float(ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)))), 3) if len(ics) > 1 and ics.std(ddof=1) > 0 else None,
                "top_quintile": round(top, 4),
                "bottom_quintile": round(bottom, 4),
                "rule_top_quintile": round(rule_top, 4),
                "all": round(float(ys.mean()), 4),
                "brier": round(brier, 5),
                "brier_base": round(base, 5),
                "brier_skill": round(1 - brier / base, 4) if base > 0 else None,
                "r2": round(1 - float(np.mean((ss - ys) ** 2)) / naive, 4) if naive > 0 else None,
                "by_grade": by_grade,
            },
        }  # fmt: skip

    def _stress(self, ds: Dataset, stacked: np.ndarray, rule: np.ndarray, seen: np.ndarray, w: np.ndarray
                ) -> dict[str, Any]:  # fmt: skip
        """Does the edge survive what could go wrong? Costs doubled (one more full spread paid), each volatility
        regime alone, each underlying left out, and every combinatorial purged path."""
        y, ss = ds.y[seen], stacked[seen]
        cost = np.nan_to_num(ds.column("spread_cost_on_risk")[seen], nan=0.0)
        top_costly, _ = _quintiles(ss, y - 2 * cost)
        rv = ds.column("rv20")[seen]
        regimes = {}
        if np.isfinite(rv).sum() >= 90:
            cuts = np.nanpercentile(rv, [33.3, 66.7])
            for name, m in (
                ("low_vol", rv <= cuts[0]),
                ("mid_vol", (rv > cuts[0]) & (rv <= cuts[1])),
                ("high_vol", rv > cuts[1]),
            ):
                if m.sum() >= 30:
                    regimes[name] = round(_spearman(ss[m], y[m]), 4)
        loo = {}
        us = ds.underlying[seen]
        for u in sorted(set(us.tolist())):
            m = us != u
            if m.sum() >= 30 and (~m).sum() >= 10:
                loo[str(u)] = round(_spearman(ss[m], y[m]), 4)
        # combinatorial purged paths: the mean model alone, refitted per path (cheap and enough for a verdict)
        paths = []
        for train, test in combinatorial(
            ds.t0, ds.t1, groups=6, k_test=2, embargo_days=self.params["embargo_days"]
        ):
            dead = ~np.isfinite(ds.X[train]).any(axis=0)
            m = self._hgb("regressor", loss="squared_error").fit(
                _fill_dead(ds.X[train], dead), ds.y[train], sample_weight=w[train]
            )
            pred = m.predict(_fill_dead(ds.X[test], dead))
            paths.append((_spearman(pred, ds.y[test]), _spearman(rule[test], ds.y[test])))
        share = float(np.mean([a > b for a, b in paths])) if paths else 0.0
        positive = float(np.mean([a > 0 for a, _ in paths])) if paths else 0.0
        min_regime = min(regimes.values()) if regimes else None
        min_loo = min(loo.values()) if loo else None
        passed = bool(
            math.isfinite(top_costly) and top_costly > 0
            and paths and share >= 0.6 and positive >= 0.7
            and (min_regime is None or min_regime > -0.05)
            and (min_loo is None or min_loo > 0)
        )  # fmt: skip
        return {"top_quintile_costs_doubled": round(top_costly, 4) if math.isfinite(top_costly) else None,
                "regime_ic": regimes, "leave_one_underlying_out_min_ic": min_loo, "cpcv_paths": len(paths),
                "cpcv_share_beating_rule": round(share, 3), "cpcv_share_positive": round(positive, 3),
                "passed": passed}  # fmt: skip

    def _permutation(self, X: np.ndarray, y: np.ndarray, fold: tuple[np.ndarray, np.ndarray], w: np.ndarray
                     ) -> list[tuple[str, float]]:  # fmt: skip
        train, test = fold
        dead = ~np.isfinite(X[train]).any(axis=0)
        m = self._hgb("regressor", loss="squared_error").fit(
            _fill_dead(X[train], dead), y[train], sample_weight=w[train]
        )
        Xt, yt = _fill_dead(X[test], dead), y[test]
        base = _spearman(m.predict(Xt), yt)
        rng = np.random.default_rng(self.params["seed"])
        out = []
        for j, name in enumerate(self.names):
            drops = []
            for _ in range(3):
                Xp = Xt.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                drops.append(base - _spearman(m.predict(Xp), yt))
            out.append((name, float(np.mean(drops))))
        return sorted(out, key=lambda kv: kv[1], reverse=True)

    # ------------------------------------------------------------------ predicting
    @property
    def fitted(self) -> bool:
        return self.parts is not None

    def predict(self, X: np.ndarray, *, explain: bool = True) -> list[EdgePrediction]:
        if self.parts is None:
            raise RuntimeError("the model is not fitted")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        raw = self._raw(self.parts, X)
        stacked = self.stack_weight * raw["mean"] + (1 - self.stack_weight) * raw["ridge"]
        p = self.calibration.predict(raw["p"]) if self.calibration is not None else raw["p"]
        out = []
        for i in range(len(X)):
            oor, _ = out_of_range(self.reference, X[i], self.names)
            pred = EdgePrediction(float(stacked[i]), float(raw["med"][i]), float(raw["lo"][i] - self.conformal_q),
                                  float(raw["hi"][i] + self.conformal_q), float(p[i]), oor)  # fmt: skip
            if explain:
                pred.drivers = self._drivers(X[i], float(stacked[i]))
            out.append(pred)
        return out

    def _drivers(self, x: np.ndarray, base: float) -> list[tuple[str, float]]:
        """Local attribution: how the expected value changes when each of the globally most important features
        is set back to its typical (training median) value. Positive: the feature raises the estimate here."""
        assert self.parts is not None
        top = [n for n, v in self.importance[:EXPLAIN_TOP] if v > 0] or [
            n for n, _ in self.importance[:EXPLAIN_TOP]
        ]
        idx = [self.names.index(n) for n in top]
        if not idx:
            return []
        batch = np.repeat(x[None, :], len(idx), axis=0)
        for r, j in enumerate(idx):
            batch[r, j] = self.medians[j]
        raw = self._raw(self.parts, batch)
        alt = self.stack_weight * raw["mean"] + (1 - self.stack_weight) * raw["ridge"]
        contrib = [(self.names[j], base - float(a)) for j, a in zip(idx, alt, strict=True)]
        return sorted(contrib, key=lambda kv: abs(kv[1]), reverse=True)[:5]

    # ------------------------------------------------------------------ persistence
    def card(self) -> dict[str, Any]:
        """What the model is, what it was trained on and how it did out of sample (never in sample)."""
        return {"version": MODEL_VERSION, "features": list(self.names), "sklearn": self.sklearn_version,
                "stacking_weight": self.stack_weight, "conformal_q": self.conformal_q, **self.report}  # fmt: skip

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        pickle.dump(self, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return buf.getvalue()

    @staticmethod
    def from_bytes(blob: bytes) -> OptionsEdgeModel | None:
        """The saved model, or ``None`` when it was made for other features or another scikit-learn (retrain)."""
        import sklearn

        try:
            m = pickle.loads(blob)  # written by this service into its own database (never user input)
        except Exception:
            return None
        if (
            not isinstance(m, OptionsEdgeModel)
            or m.names != FEATURES
            or m.sklearn_version != sklearn.__version__
        ):
            return None
        return m


def _fill_dead(X: np.ndarray, dead: np.ndarray) -> np.ndarray:
    """A column the training rows never filled is set to 0 everywhere (scikit-learn cannot bin an all-missing
    column; a constant one is simply never split on)."""
    if not dead.any():
        return X
    X = X.copy()
    X[:, dead] = 0.0
    return X


def _conformal_q(miss: np.ndarray) -> float:
    m = miss[np.isfinite(miss)]
    if len(m) < 10:
        return 0.0
    level = min(1.0, (1 - ALPHA) * (1 + 1 / len(m)))
    return max(0.0, float(np.quantile(m, level)))


def _walk_forward_verdict(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Passed: the model ranks outcomes better than the rule in most held-out windows, and its top fifth earns
    more than the rule's top fifth on average."""
    n = len(folds)
    beats = sum(1 for f in folds if f["ic"] > f["rule_ic"])
    positive = sum(1 for f in folds if f["ic"] > 0)
    tops = [f["top_quintile"] - f["rule_top_quintile"] for f in folds
            if math.isfinite(f["top_quintile"]) and math.isfinite(f["rule_top_quintile"])]  # fmt: skip
    lift = float(np.mean(tops)) if tops else math.nan
    passed = (
        n >= 2
        and beats >= math.ceil(0.6 * n)
        and positive >= math.ceil(0.6 * n)
        and math.isfinite(lift)
        and lift > 0
    )
    return {"passed": bool(passed), "windows": n, "windows_positive": positive, "windows_beating_rule": beats,
            "top_quintile_lift_vs_rule": round(lift, 4) if math.isfinite(lift) else None}  # fmt: skip
