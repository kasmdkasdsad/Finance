"""Options as a first-class, deterministic domain: contracts, quotes, pricing, structures, analytics.

Everything in this package is plain calculation — payoffs, break-evens, maximum profit and loss, Greeks,
implied volatility and its rank and percentile, liquidity, portfolio exposure, P&L attribution, expiration
states. No model (and no language model) ever does this arithmetic; a language model may only explain it.

The research lab (:mod:`quantpulse.options.lab`) and the Options Brain (:mod:`quantpulse.brain.options`)
build on these; execution goes through the existing trading service, risk engine and order manager, paper
only.
"""

MULTIPLIER = 100  # standard US equity option: one contract is 100 shares
