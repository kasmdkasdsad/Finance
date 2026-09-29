"""Market evolution as a first-class learning problem.

Markets change: volatility and its shape at short timescales, the microstructure (spreads, depth,
short-horizon reversal, the noise in prices), the way options are priced, correlations, liquidity, how orders
fill, and how strategies perform. This package detects such changes, tracks whether relationships QuantPulse
has learned still hold (weakened, disappeared, inverted — the old estimates are kept, never overwritten), and,
when a change is significant, writes down *competing* explanations with a test for each and asks the lab to
re-validate the strategies it touches.

It never assumes a cause. In particular it never concludes that AI (or any single actor) is behind a change:
a change is detected first, then every hypothesis that could explain it is listed with what it predicts, and
only evidence that separates them moves any of them.

* :mod:`.microstructure` — intraday metrics at several timescales (micro-volatility, noise, jumps, reversal,
  effective spread, illiquidity, the intraday volume profile);
* :mod:`.shifts` — distribution-shift and change-point tests;
* :mod:`.relationships` — the status of learned relationships over time;
* :mod:`.hypotheses` — competing explanations and their tests;
* :mod:`.monitor` — the Market Evolution Monitor that ties them together;
* :mod:`.registry` — the versioned model registry (no model becomes authoritative on in-sample results).
"""
