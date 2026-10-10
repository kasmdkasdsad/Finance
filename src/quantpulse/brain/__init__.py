"""QuantPulse Brain: a coordinated team of specialist agents over shared memory.

The loop (one *cycle*)::

    perception → data quality → specialist agents → debate → consensus → opportunities
      → portfolio construction → deterministic risk review → (paper execution) → predictions
      → outcomes → reflection → learning → back into the next cycle

Principles, enforced in code:

* **Agents propose, deterministic code authorises.** No agent can reach the broker: they read a snapshot
  (:mod:`quantpulse.brain.context`); proposals go through the trading service's risk engine and order
  manager (the same gates as every strategy order). Paper only.
* **Evidence, not text.** Every agent returns structured :class:`~quantpulse.brain.types.Opinion` objects
  (stance, score, confidence, evidence, missing data), so they can be aggregated, stored and graded.
* **Learning from observations only.** Agent reliability is measured from matured predictions; with too
  few observations an agent is "unproven" and weighted neutrally — never given an invented score.
* **"I do not know" is an answer.** Thin coverage, poor data or strong disagreement produce NO_ACTION.
"""
