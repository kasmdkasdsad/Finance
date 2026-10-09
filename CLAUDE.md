# QuantPulse — notes for Claude sessions

QuantPulse runs a multi-agent Brain on an **Alpaca PAPER** account (stocks and options). There is no live-money
path anywhere in the code, and there must never be one.

## The owner's standing rules (every session)

- Paper only. Never add a live endpoint, live credential or live fallback.
- Never read, print, copy or ask for the Alpaca keys or any other secret. They live in `deploy/.env` and
  `deploy/ops.env` on the server: do not open those files.
- Never place, cancel or change an order by hand. Every order goes through the Brain, the trading service and the
  risk engine.
- Never weaken a trading or safety control (risk limits, kill switches, paper-only enforcement, the preflight's
  protected settings) unless the owner explicitly asks for that specific change.
- Label any conclusion UNPROVEN while the sample is too small.
- Code changes go on the branch `claude/keen-tesla-y7rion`. Before pushing, run `ruff check .`,
  `ruff format --check .`, `mypy src` and the full `pytest` suite, and push only when all pass. Never merge the PR.

## If you are running on the server (`/opt/quantpulse`)

This folder is the **live deployed copy**. The deploy tool replaces it on every deploy (`git checkout --detach`),
so local changes break the next automatic deploy and are lost anyway.

- Do not edit, create or delete tracked files here, and do not run `git checkout`, `reset`, `clean`, `commit` or
  `push`. To change code, write down the change (files, what and why) for a cloud session working on the branch;
  it ships through CI and deploys automatically after the market closes.
- Observe and operate with `./qp` from `/opt/quantpulse/deploy`:
  - `./qp status` — the server at a glance (Brain supervisor, last cycle, deployed commit and its CI result);
  - `timeout 20 ./qp logs api` (also `dashboard`, `db`) — the logs; `./qp logs` follows them until stopped, so
    always bound it with `timeout`;
  - `./qp ci-gate` — whether the newest commit would be deployed;
  - `./qp restart` — only outside market hours unless something is broken (the Brain recovers first);
  - `./qp update` — deploys mid-session: only when the owner asks;
  - `./qp stop-trading` — the Brain kill switch, for an emergency; tell the owner at once.
- Never run `./qp keys`, `./qp setup`, `./qp restore`, `./qp rollback` or `./qp down` without the owner asking.
- The dashboard and API are on the tailnet only; the API's monitoring pages are under `/api/v1/` (see
  `deploy/ORACLE.md`).

## Where things are

- `src/quantpulse/brain/` — the Brain (stock agents, consensus, the Options Brain in `brain/options/`).
- `src/quantpulse/options/` — option pricing, structures, the research lab (`options/lab/`).
- `src/quantpulse/services/trading_risk.py` — the risk engine; `trading.py` — the only order path.
- `deploy/` — the server tooling (`qp`, `qpops.py`, compose); `deploy/ORACLE.md` — how the server runs.
- `OPTIONS.md` — how option strategies are researched, promoted and traded.
