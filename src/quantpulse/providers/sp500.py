"""S&P 500 constituents and their dated history, from Wikipedia (keyless).

Two tables are used:

* ``List_of_S&P_500_companies`` (table ``#constituents``): today's members with GICS sector, date added
  and SEC CIK;
* ``Historical_components_of_the_S&P_500`` (table ``#changes``): every addition and removal with its
  effective date.

Together they let :mod:`quantpulse.domain.universe` rebuild membership on any past date, which is what a
survivorship-bias-free backtest needs. A snapshot of both tables ships with the package
(``data/sp500_*.csv``) so everything works offline; a live refresh replaces it when Wikipedia is
reachable. Tickers are normalised to the platform's convention (``BRK.B`` → ``BRK-B``).
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime
from html.parser import HTMLParser
from importlib import resources

from quantpulse.core.errors import ProviderParseError
from quantpulse.core.http import HttpClient
from quantpulse.domain.universe import Constituent, IndexChange

NAME = "wikipedia"
CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
SNAPSHOT_CONSTITUENTS = "sp500_constituents.csv"
SNAPSHOT_CHANGES = "sp500_changes.csv"
_REF = re.compile(r"\[\s*(?:\d+|[a-z]|note \d+)\s*\]")


def normalise_ticker(raw: str) -> str | None:
    t = _REF.sub("", raw).strip().upper().replace(".", "-")
    return t if t and re.fullmatch(r"[A-Z0-9\-]{1,10}", t) else None


class _TableParser(HTMLParser):
    """Extract one ``<table id=...>`` as rows of cell texts, expanding rowspan/colspan."""

    def __init__(self, table_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self._id = table_id
        self._depth = 0  # nesting depth inside the wanted table
        self._row: list[tuple[str, int, int]] | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self.raw_rows: list[list[tuple[str, int, int]]] = []
        self.found = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table":
            if self._depth:
                self._depth += 1
            elif a.get("id") == self._id:
                self._depth, self.found = 1, True
            return
        if self._depth != 1:
            return
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._span = (int(a.get("rowspan") or 1), int(a.get("colspan") or 1))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._depth:
            self._depth -= 1
            return
        if self._depth != 1:
            return
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = _REF.sub("", " ".join("".join(self._cell).split())).strip()
            self._row.append((text, *self._span))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.raw_rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._depth == 1 and self._cell is not None:
            self._cell.append(data)

    def rows(self) -> list[list[str]]:
        """The table as a rectangular grid (spanned cells repeated)."""
        grid: list[list[str]] = []
        carry: dict[int, tuple[str, int]] = {}  # column -> (text, rows still to fill)
        for raw in self.raw_rows:
            row: list[str] = []
            cells = list(raw)
            col = 0
            while cells or col in carry:
                if col in carry:
                    text, left = carry[col]
                    row.append(text)
                    if left <= 1:
                        del carry[col]
                    else:
                        carry[col] = (text, left - 1)
                    col += 1
                    continue
                text, rowspan, colspan = cells.pop(0)
                for _ in range(colspan):
                    row.append(text)
                    if rowspan > 1:
                        carry[col] = (text, rowspan - 1)
                    col += 1
            grid.append(row)
        return grid


def _table(html: str, table_id: str) -> list[list[str]]:
    parser = _TableParser(table_id)
    parser.feed(html)
    if not parser.found:
        raise ProviderParseError(NAME, f"table #{table_id} not found")
    return parser.rows()


def _date(text: str) -> date | None:
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return date.fromisoformat(m.group(0)) if m else None


def parse_constituents(html: str) -> list[Constituent]:
    rows = _table(html, "constituents")
    header = [h.lower() for h in rows[0]]

    def col(*names: str) -> int | None:
        for i, h in enumerate(header):
            if any(n in h for n in names):
                return i
        return None

    i_sym, i_name, i_sector = col("symbol"), col("security"), col("sector")
    i_sub, i_added, i_cik = col("sub-industry"), col("date added"), col("cik")
    if i_sym is None or i_name is None:
        raise ProviderParseError(NAME, "constituents table has no Symbol/Security columns")
    out: list[Constituent] = []
    for r in rows[1:]:
        if len(r) <= max(i_sym, i_name):
            continue
        symbol = normalise_ticker(r[i_sym])
        if symbol is None:
            continue
        out.append(
            Constituent(
                symbol=symbol,
                name=r[i_name],
                sector=r[i_sector] if i_sector is not None and i_sector < len(r) else None,
                sub_industry=r[i_sub] if i_sub is not None and i_sub < len(r) else None,
                date_added=_date(r[i_added]) if i_added is not None and i_added < len(r) else None,
                cik=r[i_cik].zfill(10)
                if i_cik is not None and i_cik < len(r) and r[i_cik].isdigit()
                else None,
            )
        )
    if len(out) < 400:
        raise ProviderParseError(NAME, f"only {len(out)} constituents parsed")
    return out


def parse_changes(html: str) -> list[IndexChange]:
    rows = _table(html, "changes")
    out: list[IndexChange] = []
    for r in rows:
        if len(r) < 6:
            continue
        effective = _date(r[0])
        if effective is None:
            continue  # header rows
        added, removed = normalise_ticker(r[1]), normalise_ticker(r[3])
        if added is None and removed is None:
            continue
        out.append(
            IndexChange(
                effective=effective,
                added=added,
                added_name=r[2] or None,
                removed=removed,
                removed_name=r[4] or None,
                reason=r[5] or None,
            )
        )
    if len(out) < 100:
        raise ProviderParseError(NAME, f"only {len(out)} index changes parsed")
    return sorted(out, key=lambda c: c.effective)


# ----------------------------------------------------------------------------- packaged snapshot
def constituents_to_csv(rows: list[Constituent]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["symbol", "name", "sector", "sub_industry", "date_added", "cik"])
    for c in rows:
        w.writerow([c.symbol, c.name, c.sector or "", c.sub_industry or "", c.date_added or "", c.cik or ""])
    return buf.getvalue()


def changes_to_csv(rows: list[IndexChange]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["effective", "added", "added_name", "removed", "removed_name", "reason"])
    for c in rows:
        w.writerow(
            [
                c.effective,
                c.added or "",
                c.added_name or "",
                c.removed or "",
                c.removed_name or "",
                c.reason or "",
            ]
        )
    return buf.getvalue()


def constituents_from_csv(text: str) -> list[Constituent]:
    return [
        Constituent(
            symbol=r["symbol"],
            name=r["name"],
            sector=r["sector"] or None,
            sub_industry=r["sub_industry"] or None,
            date_added=date.fromisoformat(r["date_added"]) if r["date_added"] else None,
            cik=r["cik"] or None,
        )
        for r in csv.DictReader(io.StringIO(text))
    ]


def changes_from_csv(text: str) -> list[IndexChange]:
    return [
        IndexChange(
            effective=date.fromisoformat(r["effective"]),
            added=r["added"] or None,
            added_name=r["added_name"] or None,
            removed=r["removed"] or None,
            removed_name=r["removed_name"] or None,
            reason=r["reason"] or None,
        )
        for r in csv.DictReader(io.StringIO(text))
    ]


def packaged_snapshot() -> tuple[list[Constituent], list[IndexChange]]:
    data = resources.files("quantpulse.data")
    return (
        constituents_from_csv(data.joinpath(SNAPSHOT_CONSTITUENTS).read_text(encoding="utf-8")),
        changes_from_csv(data.joinpath(SNAPSHOT_CHANGES).read_text(encoding="utf-8")),
    )


class SP500Wikipedia:
    name = NAME

    def __init__(self, http: HttpClient, user_agent: str) -> None:
        self._http = http
        self._headers = {"User-Agent": user_agent}

    def configured(self) -> bool:
        return True

    async def snapshot(self) -> tuple[list[Constituent], list[IndexChange]]:
        current = await self._http.get_text(NAME, CONSTITUENTS_URL, headers=self._headers, timeout=30.0)
        history = await self._http.get_text(NAME, CHANGES_URL, headers=self._headers, timeout=30.0)
        return parse_constituents(current), parse_changes(history)
