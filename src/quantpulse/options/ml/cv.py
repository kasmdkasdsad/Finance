"""Cross-validation for labels that span time (López de Prado, *Advances in Financial Machine Learning*, ch. 7, 12).

An option label opened on day ``t0`` is only known on ``t1`` (its exit). Ordinary k-fold leaks: a training row
whose label overlaps the test window has seen the test period's prices. So:

* **purging** — a training row is dropped when its ``[t0, t1]`` overlaps the test window;
* **embargo** — and when it starts within ``embargo_days`` after the test window ends (serial correlation);
* :func:`walk_forward` — expanding-window splits, training always strictly before testing (what the gates use);
* :func:`combinatorial` — combinatorial purged cross-validation: the time axis cut into ``groups`` blocks, every
  choice of ``k`` test blocks a split (training on what is left, purged on both sides). The many resulting
  out-of-sample paths give a *distribution* of performance — the share of paths on which the model beats the
  rule is a guard against one lucky split.

Times are integers (``date.toordinal()``), so the functions are pure numpy.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


def _blocks(t0: np.ndarray, n: int) -> list[tuple[int, int]]:
    """``n`` contiguous time blocks (by distinct entry days), as inclusive (first day, last day)."""
    days = np.unique(t0)
    if len(days) < n:
        return []
    cuts = np.array_split(days, n)
    return [(int(c[0]), int(c[-1])) for c in cuts if len(c)]


def _purged_train(t0: np.ndarray, t1: np.ndarray, tests: list[tuple[int, int]], embargo: int,
                  candidates: np.ndarray) -> np.ndarray:  # fmt: skip
    keep = candidates.copy()
    for lo, hi in tests:
        overlaps = (t0 <= hi) & (t1 >= lo)
        embargoed = (t0 > hi) & (t0 <= hi + embargo)
        keep &= ~(overlaps | embargoed)
    return keep


def walk_forward(t0: np.ndarray, t1: np.ndarray, n_splits: int = 5, embargo_days: int = 5,
                 min_train: int = 50) -> list[Split]:  # fmt: skip
    """Expanding-window splits: test block ``j`` (``j`` = 1 … ``n_splits``), training on rows that end before it
    (purged and embargoed). Blocks whose training set would be under ``min_train`` rows are skipped."""
    t0, t1 = np.asarray(t0), np.asarray(t1)
    blocks = _blocks(t0, n_splits + 1)
    out: list[Split] = []
    for lo, hi in blocks[1:]:
        test = (t0 >= lo) & (t0 <= hi)
        train = _purged_train(t0, t1, [(lo, hi)], embargo_days, t0 < lo)
        train &= t1 < lo  # strictly the past: a label still open at the test start is never used
        if train.sum() >= min_train and test.sum() > 0:
            out.append((np.flatnonzero(train), np.flatnonzero(test)))
    return out


def combinatorial(t0: np.ndarray, t1: np.ndarray, groups: int = 6, k_test: int = 2, embargo_days: int = 5,
                  min_train: int = 50) -> list[Split]:  # fmt: skip
    """Combinatorial purged splits: every ``k_test`` of ``groups`` time blocks as the test set."""
    t0, t1 = np.asarray(t0), np.asarray(t1)
    blocks = _blocks(t0, groups)
    everyone = np.ones(len(t0), dtype=bool)
    out: list[Split] = []
    for combo in combinations(range(len(blocks)), k_test):
        tests = [blocks[i] for i in combo]
        test = np.zeros(len(t0), dtype=bool)
        for lo, hi in tests:
            test |= (t0 >= lo) & (t0 <= hi)
        train = _purged_train(t0, t1, tests, embargo_days, everyone & ~test)
        if train.sum() >= min_train and test.sum() > 0:
            out.append((np.flatnonzero(train), np.flatnonzero(test)))
    return out
