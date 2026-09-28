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


def run_preflight() -> None:
    """Check a cloud deployment's environment without starting anything (exit 0: pass, 2: fail).

    Prints masked values only: never a key, secret, token, password or database password."""
    import json

    from quantpulse.config import get_settings
    from quantpulse.services import preflight

    parser = argparse.ArgumentParser(description="Check the (cloud) environment before starting QuantPulse")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    try:
        settings = get_settings()
    except Exception as exc:  # a setting that does not even load (QP_ALPACA_PAPER=false, a typo, ...)
        errors = getattr(exc, "errors", None)
        names = sorted({str(e["loc"][0]) for e in errors() if e.get("loc")}) if callable(errors) else []
        print(
            f"QuantPulse preflight: FAIL: the settings do not load ({', '.join(names) or type(exc).__name__})"
        )
        raise SystemExit(2) from None
    report = preflight.run(settings)
    print(json.dumps(report.as_dict(), indent=2) if args.json else "\n".join(report.lines()))
    raise SystemExit(0 if report.ok else 2)


def run_hash_password() -> None:
    """Make the dashboard password hash for QP_DASHBOARD_PASSWORD_HASH (the password is typed, never shown)."""
    import getpass
    import sys

    from quantpulse.core.passwords import MIN_LENGTH, PasswordHashError, hash_password

    argparse.ArgumentParser(
        description=f"Hash a dashboard password (at least {MIN_LENGTH} characters) for QP_DASHBOARD_PASSWORD_HASH"
    ).parse_args()
    if sys.stdin.isatty():
        first = getpass.getpass("Dashboard password: ")
        if getpass.getpass("Again: ") != first:
            raise SystemExit("the two passwords differ: nothing was made")
    else:  # piped in (never as an argument: it would land in the shell history)
        first = sys.stdin.readline().rstrip("\n")
    try:
        print(hash_password(first))
    except PasswordHashError as exc:
        raise SystemExit(f"not hashed: {exc}") from None


def run_cloud_check() -> None:
    """Check a running deployment from anywhere (exit 0: healthy with one live supervisor, 1: not).

    Reads the API address and token from QP_CLOUD_URL and QP_API_TOKEN (never from the command line, so the
    token does not land in the shell history) and prints no secret."""
    import json
    import os
    import sys
    import time
    import urllib.error
    import urllib.request

    parser = argparse.ArgumentParser(
        description="Verify the cloud deployment: health, one supervisor, trading"
    )
    parser.add_argument("--commit", help="also require this commit to be the one running (after a deploy)")
    parser.add_argument("--wait", type=float, default=0, help="keep checking for up to this many seconds")
    args = parser.parse_args()
    base = os.environ.get("QP_CLOUD_URL", "").rstrip("/")
    token = os.environ.get("QP_API_TOKEN", "")
    if not base or not token:
        raise SystemExit("set QP_CLOUD_URL (e.g. https://quantpulse-api.onrender.com) and QP_API_TOKEN first")
    deadline = time.monotonic() + max(0.0, args.wait)
    while True:
        problems: list[str] = []
        try:
            req = urllib.request.Request(f"{base}/api/v1/brain/cloud-status", headers={"X-API-Key": token})
            with urllib.request.urlopen(req, timeout=30) as r:
                st = json.load(r)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            st, problems = None, [f"the API did not answer: {type(exc).__name__}"]
        if st is not None:
            sv, sup = st["service"], st["supervisor"]
            if args.commit and not (sv.get("commit") or "").startswith(args.commit):
                problems.append(f"running commit {sv.get('short_commit')}, expected {args.commit[:7]}")
            if st["database"]["status"] != "ok":
                problems.append(f"database: {st['database']['detail']}")
            if not st["alpaca"]["paper_endpoint_verified"]:
                problems.append("the Alpaca paper endpoint is not verified")
            if not (sup.get("leader") or {}).get("live"):
                problems.append("no live supervisor lease")
            age = sup.get("last_tick_age_seconds")
            if age is None or age > 300:
                problems.append(f"the supervisor last ticked {age} s ago")
            ae = st["autonomous_execution"]
            print(f"QuantPulse {sv['version']} @ {sv.get('short_commit') or '?'} on {sv.get('instance')}, up "
                  f"{sv['uptime_seconds']} s")  # fmt: skip
            print(f"  supervisor leader: {(sup.get('leader') or {}).get('holder')} (live "
                  f"{(sup.get('leader') or {}).get('live')}), standing by: {len(sup.get('standby_processes') or {})}, "
                  f"last tick {age} s ago: {sup.get('last_result')}")  # fmt: skip
            print(f"  Alpaca: paper endpoint verified {st['alpaca']['paper_endpoint_verified']}, "
                  f"{st['alpaca']['connectivity']['detail']}")  # fmt: skip
            print(f"  market {'OPEN' if st['market']['open'] else 'closed'}; autonomous paper execution "
                  f"{'PERMITTED' if ae['permitted'] else 'not permitted'}")  # fmt: skip
            for reason in ae["reasons"]:
                print(f"    - {reason}")
        if not problems or time.monotonic() >= deadline:
            break
        time.sleep(15)
    for p in problems:
        print(f"PROBLEM: {p}")
    print("OK" if not problems else "NOT OK")
    sys.exit(0 if not problems else 1)
