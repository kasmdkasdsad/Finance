"""Command-line entry points."""

from __future__ import annotations

import argparse


def run_api() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the QuantPulse API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "quantpulse.api.app:app_factory", factory=True, host=args.host, port=args.port, reload=args.reload
    )


def run_migrations() -> None:
    from quantpulse.config import get_settings
    from quantpulse.db import migrate

    parser = argparse.ArgumentParser(description="Apply or roll back database migrations")
    parser.add_argument("revision", nargs="?", default="head")
    parser.add_argument("--downgrade", action="store_true")
    args = parser.parse_args()
    url = get_settings().database_url
    if args.downgrade:
        migrate.downgrade(url, args.revision)
    else:
        migrate.upgrade(url, args.revision)
    print(f"database at revision {migrate.current_revision(url)} (head {migrate.head_revision()})")


def run_transfer() -> None:
    """Copy the whole database (e.g. the PC's SQLite file) into another, empty one (the cloud's PostgreSQL)."""
    from quantpulse.db.transfer import TransferError, transfer

    parser = argparse.ArgumentParser(description="Copy every QuantPulse table into an empty database")
    parser.add_argument("--source", required=True, help="e.g. sqlite+aiosqlite:///data/quantpulse.db")
    parser.add_argument(
        "--target", required=True, help="e.g. postgresql+asyncpg://user:password@host/quantpulse"
    )
    args = parser.parse_args()
    try:
        out = transfer(args.source, args.target)
    except TransferError as exc:
        raise SystemExit(f"transfer refused: {exc}") from None
    print(f"copied {out['rows']} row(s) across {len(out['counts'])} non-empty table(s); row counts verified")
