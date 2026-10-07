# Options intelligence, strategy research and market evolution

QuantPulse trades options **on the Alpaca paper account only** — simulated money, the same account the Brain
already manages. Options are not a second engine: they are part of the Brain's cycle, and every option order
goes through the same trading service, risk engine, last gate, order manager and kill switches as a stock
order. There is no live-money path, no exercise call, and no way to express a naked short option.

This document explains what the options layer does, how a strategy earns the right to trade, **how QuantPulse
learns**, how the Market Evolution Monitor watches the market change, and — honestly — what is and is not
established yet.

---

## 1. The safety model

| Rule | Where it is enforced |
|---|---|
| Paper only | the broker is built with `paper=True`, its URL is checked before every cycle and every order; no setting selects live trading |
| One order path | option orders leave only through `OrderManager` (`_send` is the single call to the broker), write-ahead recorded, never resent blindly |
| The risk engine is final | `RiskBook.evaluate_option` — the *same* book as stocks (kill switch, account, market hours, daily loss, cash, working orders) plus the option checks below |
| No naked short options | refused three times over: the order spec cannot express one (`naked_short_legs`), the risk engine recomputes coverage (a covered call needs 100 unencumbered shares per contract), the settings refuse undefined-risk families |
| Defined risk only | the maximum loss is computed from the payoff at expiration (never estimated by a model); `inf` is refused |
| Never 0DTE | at least one day to expiration for any order, whatever the settings (0DTE is research only) |
| Never exercise | QuantPulse closes positions `QP_OPTIONS_CLOSE_DTE` days before expiration; anything left is settled by the OCC and recorded as such |
| Early assignment | a leg gone while the rest is held (American options can be assigned any day) freezes the position: no exit orders (the remaining legs hedge the delivered shares — closing them alone could leave naked stock), no new trade on that underlying, the likely assignment recorded as inferred, and a critical alert asks a person to close the shares and the remaining legs together |
| Fresh quotes only | every leg is re-quoted just before the order; model-priced and recorded quotes can never be execution quotes; the indicative feed is labelled |
| Limits only tighten | every `QP_OPTIONS_*` limit is a protected control: the cloud preflight refuses a looser value and the self-improvement engine never proposes one |
| Human approval | adding a structure family, anything undefined-risk (never allowed anyway), an AI model becoming authoritative, going live (impossible) |

The option checks in the risk book: structure (1–4 distinct contracts, one underlying, intents matching sides),
no naked short, defined risk, allowed family, the account's Alpaca options level, expiration window, maximum
loss per trade (dollars and share of equity; lower for exploration), book loss per underlying and in total,
open structures, contracts per leg, fresh two-sided execution quotes, spreads and open interest, the book's net
delta and vega (fail closed when a Greek is unknown), the limit price never worse than the natural price, and
closing orders never exceeding what is held.

## 2. What happens in a Brain cycle

```
perceive     live chains; the day's point-in-time features (the backtester's own functions); IV history
manage       paper orders synced with Alpaca; marks; exits (strategy rules + stop/take-profit + close before
             expiration); OCC settlements recorded
candidates   only strategies that passed validation (PAPER_SHADOW and above); the lab's entry rules and
             contract selection
deliberate   eighteen agents: support, oppose, abstain (and say what is missing) or veto
decide       thesis, bull/bear/devil's advocate, a plain-words explanation; options versus shares
execute      shadow always; paper at PAPER_ACTIVE (one-contract exploration from VALIDATION) through the trading service
learn        attribution, critique, counterfactual, learning events; weights, lessons, graded misses (daily)
```

**Which underlyings.** The core list (`QP_OPTIONS_UNIVERSE`) and every underlying with an open position are
read every cycle. Besides them, the Options Brain scans the stock universe's 300 most liquid names (the S&P 500
and the trading ETFs, by 20-day dollar volume; `QP_OPTIONS_SCAN_SIZE`), 16 a cycle (`QP_OPTIONS_SCAN_PER_CYCLE`):
up to half are the names the stock Brain holds its strongest views on that cycle, where an option expression is
likeliest; the rest are the longest unread, so every name comes round — a session's 78 cycles read each about
four times a day, and each builds its own implied-volatility history from its first read (IV rank needs 60
days of it; until then strategies that filter on it do not enter there). Reading every name every cycle is not
possible on Alpaca's free data plan (about 200 requests a minute; a chain is one to several requests) and would
outrun a five-minute cycle. A scanned name passes exactly the checks a core one does: quote age, spread and open
interest per leg, the agents, the risk engine. Its daily prices come from the price warehouse the trading
loader keeps; its next earnings date is looked up once a day. A company whose date cannot be found is marked
unknown, never guessed: the earnings agent vetoes strategies there that avoid events or sell volatility. At most
60 candidates a cycle (the best by verdict) are deliberated and recorded; the research lab still backtests the
core list.

**Options versus shares.** For each candidate the option's verdict plus `QP_OPTIONS_PRIORITY_WEIGHT` is set
against the stock agents' consensus on the same underlying. The weight can tip a close call toward the option
(and then the stock entry on that name is marked "expressed through options"); it can never make a failing
option pass — vetoes and the validated edge are checked first.

**Sizing.** Paper orders use the strategy's risk budget, capped by the protected per-trade loss, the contract
limit and half the book's delta and vega limits. Limit prices are REALISTIC-level (a quarter of the spread
beyond the mid on each leg), never market orders.

## 3. How a strategy earns the right to trade

```
RESEARCH → EXTRACTED → BACKTESTING → VALIDATION → WALK_FORWARD → PAPER_SHADOW → PAPER_ACTIVE → PROVEN
                                                                                 (any) → RETIRED
```

One gate at a time, each recorded in the version's stage history with the evidence:

* **EXTRACTED** — the genome is explicit and valid (every parameter stated; assumptions recorded).
* **BACKTESTING / VALIDATION** — backtests under all five fill models; at least 30 trades; positive per dollar at
  risk under REALISTIC *and* PESSIMISTIC fills (an edge that exists only at the mid is no edge).
* **WALK_FORWARD** — parameters chosen on the past only; out-of-sample expectancy positive; overfit risk
  (parameters per trade, train/test divergence, concentration, suspicious Sharpe or win rate, the deflated
  Sharpe for everything tried) below 0.5.
* **PAPER_SHADOW** — Monte Carlo risk of ruin, tail stress within the maximum loss, beats the baselines (no
  trade, the simple option, random controls), survives the population-wide false-discovery control and the
  critic (costs, wider spreads, a day's delay, one ticker removed, one period removed, small samples).
* **PAPER_ACTIVE** — at least `QP_OPTIONS_MIN_SHADOW_TRADES` shadow trades on **live quotes**, over enough
  sessions, positive after costs; a default-executable defined-risk family (others need a person).
* **PROVEN** — 50+ real paper trades, positive with t ≥ 2, not decaying.

Decay is watched on live results (CUSUM and tests against what validation promised): DEGRADING drops a
strategy back to shadow, BROKEN retires it. Nothing is deleted; a change is always a new version.

**Exploration.** A system that never trades never learns, and the research evidence is model-priced (no
historical option quotes): live quotes are the real test. So with `QP_OPTIONS_EXPLORATION=true` (the default),
trading starts once a strategy passes **VALIDATION** (at least 30 backtest trades, positive per dollar at risk
after REALISTIC *and* PESSIMISTIC fills):
* it is shadow-traded on live quotes, and may trade *one* contract on paper (maximum loss ≤
  `QP_OPTIONS_EXPLORATION_MAX_LOSS`), so the lab learns from real paper fills while the evidence for the later
  gates (walk-forward, Monte Carlo, tail stress, baselines, false-discovery control, the critic) builds;
* its expected edge is the held-out (walk-forward) result when that is positive, else its backtest at
  PESSIMISTIC fills; each candidate records which;
* at most 20 strategies below PAPER_SHADOW explore at once: the furthest along first, then by an upper
  confidence bound (the expected edge plus a bonus that shrinks as a strategy is tried), so every candidate
  gets its turn instead of the same few taking every slot;
* the strategy critic counts them as less proven than a strategy at PAPER_SHADOW, and every other agent, the
  expected edge and the risk engine still decide each trade;
* decay is watched on their live trades: one that breaks with the edge it is traded on is retired;
* full-size paper trades still need PAPER_ACTIVE. Exploration trades are labelled as exploration everywhere.

With exploration off, nothing trades before PAPER_SHADOW, and then only in shadow until PAPER_ACTIVE.

**Research time.** Besides the daily run after the close (`QP_OPTIONS_RESEARCH_BUDGET_SECONDS`, 3 minutes), the
closed-market research queue runs the lab for **25 minutes every hour** (`options_research`), under the same
resource limits as all research, one run at a time. Each run evaluates never-tested and stale strategies,
applies the false-discovery control, moves strategies through their stages, runs experiments and breeds the
next generation.

**The search never stalls, and keeps trying new things.**
* Every generation brings random immigrants: brand-new strategies drawn from the whole searchable space of the
  default-executable families, each the most novel of eight draws against everything tried.
  * There are at least two per generation, plus any budget the generation's own rule leaves unused.
  * Anything that earns its stages can trade without a person enabling its family.
* While no strategy has passed VALIDATION yet, a run explores instead of waiting: wider (two-gene) variations
  of the best-scoring candidates, plus immigrants.

The more that is tried, the higher the population-wide false-discovery bar. Nothing reaches the paper account
without passing every stage above.

## 4. Evidence is never mixed

| Label | What it is | What it can justify |
|---|---|---|
| `model` | backtests on **model-priced** option chains over real underlying prices | shadow trading at most |
| `shadow` | simulated fills (REALISTIC) on **live** Alpaca option quotes | PAPER_ACTIVE |
| `paper` | real Alpaca **paper** orders and fills | PROVEN |

Free data has no historical option quotes, so research is model-priced — labelled on every backtest, score and
strategy page. Real evidence begins with shadow trading on live quotes.

## 5. How QuantPulse learns

1. **From research.** Documented strategies (papers, index methodologies, educational material) are claims, not
   facts: each claim starts `UNTESTED` and is tested on QuantPulse's own trades (`SUPPORTED`, `NOT_REPRODUCED`,
   `INCONCLUSIVE`).
2. **From failure.** A strategy stopped by a recognisable failure (only flattering fills, losses in low or high
   IV, a regime) yields *competing* hypotheses — each a child version, never an edit — queued by **expected
   information** (how much the answer would shrink our uncertainty), not expected profit.
3. **From search.** Generations of mutation, crossover, regime restriction and portfolio-aware variants, every
   child tested from scratch; the false-discovery control across the whole population keeps the search from
   manufacturing winners.
4. **From every closed position.** P&L attribution (delta, gamma, theta, vega, execution, fees), a critique
   (was the direction right? the structure? the timing? execution?), a counterfactual (the same view with
   shares), a learning event (predicted probability of profit vs the outcome, for calibration).
5. **From what it did not do.** Rejected candidates are graded after the market has spoken: a good rejection, a
   correctly avoided loser, a missed winner, a bad rejection.
6. **Weights by context.** Each strategy's weight in each context (market regime × structure × volatility state)
   is a posterior probability that its edge is positive, **shrunk** toward zero with little evidence and
   recency-weighted with a floor (old evidence fades, never vanishes).
7. **Lessons** become memory only when they replicate (at least three times), with their evidence, confidence
   and an expiry.
8. **Machine learning only where it helps out of sample** — and through the model registry (below).

## 6. The Market Evolution Monitor

Each trading day after the close it measures, per underlying and market-wide, at several timescales:

* **micro-volatility and microstructure** from 1-minute bars: realized volatility at 1/5/15/30/60 minutes (the
  volatility signature), the noise ratio, the jump share, the variance ratio, 1- and 5-minute autocorrelation,
  Roll and quoted spreads, Amihud illiquidity, the open/close volume shares, Parkinson volatility;
* daily volatility (5/20/60 days) and its volatility; the option market's own readings (30-day ATM IV, IV rank,
  IV/RV, skew, term slope, implied move); dollar volume; the universe's average correlation; QuantPulse's own
  execution quality; each strategy's live results.

It then compares each series' recent window with its reference window (shape, level and spread tests, PSI,
change points) with **effective sample sizes** for autocorrelated series, applies **one false-discovery-rate
control across everything**, and records a change only if it survives. Every change gets the full list of
**competing hypotheses** — chance, a data artifact, the market as a whole, events, liquidity, market structure,
composition, automated liquidity provision — each marked consistent, inconsistent or untestable. None is
assumed: automated (AI-driven) liquidity provision is explicitly *not identifiable from prices alone*.

Relationships the strategies rely on (does the volatility risk premium predict realized volatility? does
micro-volatility explain option pricing, spreads, slippage, strategy results?) are re-estimated and **appended**:
stable, strengthened, weakened, disappeared, inverted or emerged — the history is kept. A relationship counts as
explaining an outcome only when both halves of the data agree. Strategies a change touches are re-validated on
recent data; a failure is recorded as a WATCH (live results, not a backtest, decide a demotion).

## 7. The model registry

Every model (statistical, machine-learned, AI, rule) lives in a slot:
`CANDIDATE → OOS_VALIDATED → WALK_FORWARD_VALIDATED → STRESS_VALIDATED → PAPER_SHADOW → AUTHORITATIVE`.
In-sample results are recorded and **refused as evidence**; becoming authoritative needs out-of-sample,
walk-forward and stress evidence and a live shadow record that beats the champion — and a person's approval for
an AI model (`POST /registry/models/{id}/approve`, typed confirmation). The replaced champion keeps its history.

## 8. Settings

The account owner wants a high-risk, high-reward trader focused on options (paper only), so the limits below
were raised at their request: $2,000 or 2% per position (first $500 / 1%), 25% across all options (6%), 5% per
underlying (2%), 12 positions (6), 20 contracts (10), 50% net delta and 1% net vega (30% / 0.5%), $2,000 per
exploration contract ($250); a protective stop only at the whole maximum loss (was half of it), so positions have
time to work; options favoured over shares with a weight of 0.5 (0.15); twelve underlyings (eight). Each limit
can still be tightened in the server's environment, never loosened past these. Paper-only, the kill switches,
the account's 4% daily loss limit, defined risk only, no 0DTE, closing before expiration, the strategies' own
exits and the quote checks are unchanged.

| Setting | Default | Meaning |
|---|---|---|
| `QP_OPTIONS_ENABLED` | `true` | research, shadow trading and the Options Brain |
| `QP_OPTIONS_EXECUTION` | `true` | paper option orders (false: shadow only) |
| `QP_OPTIONS_PRIORITY_WEIGHT` | `0.5` | favour options over shares in close calls; never forces |
| `QP_OPTIONS_UNIVERSE` | SPY,QQQ,IWM,AAPL,MSFT,NVDA,AMZN,META,TSLA,AMD,GOOGL,NFLX | the core: chains read every cycle |
| `QP_OPTIONS_SCAN_SIZE` / `_SCAN_PER_CYCLE` | 300 / 16 | the scan: the most liquid names of the stock universe, a rotating batch a cycle (0: the core only) |
| `QP_OPTIONS_ALLOWED_STRUCTURES` | the seven defined-risk defaults | may drop families; adding needs a person |
| `QP_OPTIONS_MAX_LOSS_PER_TRADE` / `_PCT_PER_TRADE` | $2,000 / 2% | per new position |
| `QP_OPTIONS_MAX_TOTAL_RISK_PCT` / `_UNDERLYING_RISK_PCT` | 25% / 5% | book limits |
| `QP_OPTIONS_MAX_POSITIONS` / `_MAX_CONTRACTS` | 12 / 20 | |
| `QP_OPTIONS_MIN_DTE` / `_MAX_DTE` / `_CLOSE_DTE` | 7 / 60 / 2 | entry window; close before expiration |
| `QP_OPTIONS_MAX_SPREAD_PCT` / `_MAX_QUOTE_AGE_SECONDS` / `_MIN_OPEN_INTEREST` | 15% / 120 / 100 | per leg |
| `QP_OPTIONS_MAX_DELTA_PCT` / `_MAX_VEGA_PCT` | 50% / 1% | the book's net Greeks |
| `QP_OPTIONS_TAKE_PROFIT_PCT` / `_STOP_LOSS_PCT` | 50% / 100% | protective overlay on live positions (the stop only at the whole maximum loss) |
| `QP_OPTIONS_MIN_SHADOW_TRADES` | 10 | before PAPER_ACTIVE |
| `QP_OPTIONS_EXPLORATION` / `_EXPLORATION_MAX_LOSS` | true / $2,000 | one-contract exploration from VALIDATION (at most 20 strategies) |
| `QP_OPTIONS_RESEARCH_TIME` / `_RESEARCH_BUDGET_SECONDS` | 16:40 / 180 | the daily research run (the research queue adds 25 min every hour while the market is closed) |
| `QP_EVOLUTION_ENABLED` / `_TIME` / `_REFERENCE_DAYS` / `_RECENT_DAYS` | true / 16:50 / 120 / 20 | the monitor |

## 9. API and dashboard

`/api/v1/options/status | chains?underlying= | candidates | strategies | strategies/{id} | research |
experiments | learning | portfolio | positions | greeks | performance | counterfactuals |
missed-opportunities`, `POST /options/research/run`, `POST /options/learn`;
`/api/v1/evolution/status | changes | relationships`, `POST /evolution/run`; `/api/v1/registry/models`;
`/api/v1/brain/status` carries an options summary. Dashboard pages: **Options** and
*Research → Market changes*.

## 10. What is established — and what is not

* The code, its safety checks and its tests are implemented and pass (fake Alpaca paper API, fake options
  market, synthetic prices).
* No option strategy is *statistically established*: research evidence is model-priced; shadow and paper
  evidence accumulates only once this runs against live Alpaca paper data.
* IV rank needs 60 days of QuantPulse's own IV records before any strategy filtering on it can trade.
* The indicative options feed is not firm; fills on the paper account may differ from any real market.
