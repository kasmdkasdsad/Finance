"""Options machine learning: a learned, out-of-sample-validated estimate of each candidate's edge.

The rule the Options Brain has always used to rank a candidate is its expected value per dollar at risk under
the market's own (lognormal) distribution or the underlying's past moves. This package learns, from outcomes,
what that rule cannot see — and has to prove it before it decides anything.

* :mod:`.surface` — the implied-volatility surface: an SVI smile fitted to every expiration (Gatheral's raw
  parameterisation), checked for butterfly and calendar arbitrage, and read as level, skew, curvature, risk
  reversal, butterfly and term slope; and how rich or cheap each listed contract is against the smooth surface.
* :mod:`.volatility` — realized-volatility forecasts (the HAR model of Corsi, fitted point in time on daily
  returns, with an EWMA fallback) and the forward-looking volatility risk premium: implied volatility at the
  structure's expiration against the volatility forecast for the same horizon.
* :mod:`.features` — one fixed, documented feature vector per candidate: the underlying (trend, momentum,
  realized and implied volatility, IV rank), the surface, the forecast, the structure itself (payoff, Greeks
  per dollar at risk, costs, the rule's own expected value) and the stock Brain's view. The same function serves
  research and the live Brain.
* :mod:`.labels` — triple-barrier outcomes (take profit, stop, time): a candidate's realised return per dollar at
  risk when held under one standard exit policy, filled at REALISTIC prices with fees.
* :mod:`.dataset` — labelled rows from model-priced chains over real underlying prices (many, cheap, biased),
  from chains QuantPulse recorded from the market (few, real) and from shadow and paper outcomes; every row
  carries its grade, its label's time span and a sample weight.
* :mod:`.cv` — purged, embargoed walk-forward and combinatorial purged cross-validation (López de Prado): no
  label that overlaps a test window is ever trained on.
* :mod:`.model` — the edge model: gradient-boosted quantile regressors (10th/50th/90th percentile of the return
  per dollar at risk), a gradient-boosted classifier for the probability of profit (isotonic-calibrated out of
  sample), a ridge baseline stacked by out-of-sample error, conformalized quantile intervals (CQR) for honest
  uncertainty, local explanations and a model card.
* :mod:`.drift` — population stability and out-of-distribution checks against the training data.
* :mod:`.service` — training (a research job), the model registry (the model is a challenger to the rule until
  it beats it on held-out and then live data), persistence and the live predictions the ``OptionsMLAgent``
  reads.

Nothing here sends an order or changes a limit. Until the registry makes the model AUTHORITATIVE its opinion is
recorded with every candidate (so it can be graded) and abstains from the vote.
"""
