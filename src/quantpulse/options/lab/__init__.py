"""The options research lab: strategies as explicit genomes, a realistic backtester with five execution
models, walk-forward validation, Monte Carlo and tail stress, overfitting defences, baselines, promotion
stages, decay detection, counterfactuals, learning and experiments.

Every strategy is a :class:`~quantpulse.options.lab.genome.Genome` — every parameter explicit — and every
change is a new version with its reason. Evidence is always labelled by where it came from: a backtest on
recorded chains, a backtest on *model-priced* chains (the logic and the underlying's path, not real option
prices), paper shadow trading, or paper execution. They are never pooled without their labels.
"""
