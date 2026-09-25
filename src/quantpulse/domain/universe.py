"""Point-in-time index membership (for survivorship-bias-free research).

Backtesting only on *today's* index members inflates results: the stocks that shrank, were acquired
cheaply or went bankrupt are silently excluded. Given today's members and the dated list of additions
and removals, :class:`Membership` replays the changes backwards and answers "was X in the index on
day d?".

Replay rules (walking backwards from today):

* an addition of ``A`` effective on ``d`` means ``A`` was *not* a member before ``d``, so an interval
  ``[d, end)`` closes for ``A``;
* a removal of ``R`` effective on ``d`` means ``R`` *was* a member until ``d``, so an interval ending at
  ``d`` opens for ``R``;
* a change that removes and re-adds the same ticker (a merger that kept the symbol) is a no-op;
* anything still open when the log runs out was a member since before the log starts.

Membership intervals are half-open: a member on ``d`` when ``start <= d < end``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True, slots=True)
class Constituent:
    symbol: str
    name: str
    sector: str | None
    sub_industry: str | None
    date_added: date | None
    cik: str | None


@dataclass(frozen=True, slots=True)
class IndexChange:
    effective: date
    added: str | None
    added_name: str | None
    removed: str | None
    removed_name: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class Interval:
    start: date | None  # None: since before the change log begins
    end: date | None  # None: still a member

    def contains(self, d: date) -> bool:
        return (self.start is None or d >= self.start) and (self.end is None or d < self.end)

    def overlaps(self, lo: date, hi: date) -> bool:
        return (self.start is None or self.start <= hi) and (self.end is None or self.end > lo)


def build_intervals(current: Iterable[str], changes: Sequence[IndexChange]) -> dict[str, list[Interval]]:
    open_end: dict[str, date | None] = dict.fromkeys(current)
    out: dict[str, list[Interval]] = defaultdict(list)
    for ch in sorted(changes, key=lambda c: c.effective, reverse=True):
        if ch.added and ch.added == ch.removed:
            continue
        d = ch.effective
        if ch.added and ch.added in open_end:
            out[ch.added].append(Interval(d, open_end.pop(ch.added)))
        if ch.removed and ch.removed not in open_end:
            open_end[ch.removed] = d
    for symbol, end in open_end.items():
        out[symbol].append(Interval(None, end))
    return {s: sorted(v, key=lambda i: i.start or date.min) for s, v in out.items()}


class Membership:
    def __init__(
        self, constituents: Sequence[Constituent], changes: Sequence[IndexChange], as_of: date | None = None
    ) -> None:
        self.constituents = list(constituents)
        self.changes = sorted(changes, key=lambda c: c.effective)
        self.current = [c.symbol for c in self.constituents]
        self.intervals = build_intervals(self.current, self.changes)
        self.as_of = as_of or (self.changes[-1].effective if self.changes else date.today())
        self.log_start = self.changes[0].effective if self.changes else None
        self.gics = {c.symbol: c.sector for c in self.constituents if c.sector}
        self.cik = {c.symbol: c.cik for c in self.constituents if c.cik}

    def is_member(self, symbol: str, d: date) -> bool:
        return any(i.contains(d) for i in self.intervals.get(symbol, ()))

    def members_on(self, d: date) -> set[str]:
        return {s for s, iv in self.intervals.items() if any(i.contains(d) for i in iv)}

    def ever_members(self, start: date, end: date) -> set[str]:
        """Every symbol that was a member at some point in ``[start, end]``."""
        return {s for s, iv in self.intervals.items() if any(i.overlaps(start, end) for i in iv)}

    def mask(self, dates: pd.DatetimeIndex, symbols: Sequence[str]) -> pd.DataFrame:
        """Boolean frame (dates x symbols): True where the symbol was a member on that date.

        Symbols the log knows nothing about (e.g. a custom ticker) are treated as always eligible."""
        days = pd.DatetimeIndex(dates)
        out = {}
        for s in symbols:
            iv = self.intervals.get(s)
            if iv is None:
                out[s] = pd.Series(True, index=days)
                continue
            flags = pd.Series(False, index=days)
            for i in iv:
                lo = pd.Timestamp(i.start) if i.start else days.min()
                hi = pd.Timestamp(i.end) if i.end else days.max() + pd.Timedelta(days=1)
                flags |= (days >= lo) & (days < hi)
            out[s] = flags
        return pd.DataFrame(out, index=days)
