"""The Brain's 24/7 research subsystem: while the market is closed the Brain grades, analyses, researches, tests,
learns and prepares — and production trading stays protected from anything unproven.

* :mod:`.operating` — the market-open / pre-market / closed state machine and the execution-readiness gate;
* :mod:`.queue` — the persistent research queue (priority by expected information value; restartable);
* :mod:`.scheduler` — the closed-market scheduler (leader only, resource-limited, execution first);
* :mod:`.resources` — memory and CPU limits;
* :mod:`.catalog`, :mod:`.handlers` — the research jobs;
* :mod:`.ledger` — the learning ledger (no conclusion without its evidence; UNPROVEN until it suffices);
* :mod:`.lifecycle` — DISCOVERED → … → EVALUATION → (a person) → PRODUCTION.
"""
