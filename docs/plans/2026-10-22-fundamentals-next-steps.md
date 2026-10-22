# Plan: fundamentals swing research, next steps

Hand-off plan for an agent working in a fresh session. Everything it needs is
in this file or in the repo on branch `claude/swing-strategy-fundamentals-v5mb8y`
(draft PR #7). Read `AGENTS.md` and `docs/STATUS.md` first, then this file.

## 0. Where things stand

Two research lines exist, both wired into the harness (registry, backtest
engine, walk-forward, FTMO sim, promotion gates in `src/research/promote.py`):

| Line | Idea | What exists | What is missing |
|------|------|-------------|-----------------|
| `event_drift` | Market prices an expected surprise before a scheduled release. Signal = surprise momentum (consensus anchoring) or nowcast minus consensus. Variants `pre` (exit before print), `post` (hold through), `fade` (against the reaction). | Calendar fetch (FXStreet), point-in-time surprises, gasoline CPI nowcast, external forecast hook, event table, strategy, research runner, 21 tests. Calendar-only result: surprise-direction hit rate 0.534 (t 1.70) for momentum, 0.483 for the gasoline nowcast. | The price-based study. Needs the local M15 parquet bars. |
| `macro_swing` | Per-currency macro score (2y-yield repricing, carry, risk-off beta, oil for CAD, copper for AUD, contrarian CFTC positioning) → pair bias = base − quote → trade only above a threshold, D1 trigger times it, ATR stop, exit on bias flip. | Loaders for FRED, ECB, Bank of Canada, RBA, Japan MOF, Bank of England, BIS, CFTC; scoring; strategy; 10 tests. Full real fetch works; live table in `docs/STATUS.md`. | Any evidence the score predicts returns. Needs either the local bars or the daily check in Phase A below. |

Hard constraints to respect:

- Market bars (`data/parquet`) live only on the owner's Windows machine
  (`D:\AI\trade\data\parquet`). Cloud sessions cannot backtest on bars. Do not
  download bars into a cloud container unless the owner explicitly asks.
- `data/` and `reports/` are gitignored. Calendar and macro caches are rebuilt
  with `scripts/download_calendar.py --gasoline` and `scripts/download_macro.py`.
- Point-in-time rule: anything used at a decision time must come from data
  published strictly before it. The macro panel already applies availability
  lags (market data next day, CFTC report + 3 days). Keep it that way.
- Component weights in `configs/strategy_macro_swing.yaml` are priors. Do not
  fit them to the IC check or to the backtest without a holdout, and if you do
  fit anything, say so in the ledger.
- `docs/STATUS.md` says: do not retune rejected lines. A NO-GO is a result.
- Commit messages: no model names, end with the attribution lines the session
  provides. Run the full test suite with a real exit code before every push
  (`set -o pipefail; python -m pytest -q | tail -2` or just `python -m pytest -q`).

## Phase A — cloud-runnable: does the macro bias predict returns at all?

Goal: an information-coefficient (IC) check of the macro pair bias against
forward FX returns using FRED daily exchange rates, which are free and
reachable from the cloud container. This decides whether the bar backtest is
worth running.

### A1. Data

FRED daily spot rates (noon buying rates, back decades):

| Pair | FRED id | Quote convention | Inversion needed |
|------|---------|------------------|------------------|
| EURUSD | `DEXUSEU` | USD per EUR | no |
| GBPUSD | `DEXUSUK` | USD per GBP | no |
| USDJPY | `DEXJPUS` | JPY per USD | no |
| AUDUSD | `DEXUSAL` | USD per AUD | no |
| USDCAD | `DEXCAUS` | CAD per USD | no |

Fetch with `src.data.macro.fetch_series({"source": "fred", "id": ...}, name)` and
cache as `data/macro/fx_{PAIR}.csv` via `save_series`. Add them to a new
`macro.series.fx` group in the config only if you want `download_macro.py` to
manage them; otherwise fetch inside the new script.

### A2. Build `src/research/macro_ic.py` + `scripts/run_macro_ic.py`

Inputs: the macro panel (`load_macro_panel`), the strategy config, the FX
series. Steps:

1. `build_currency_scores(panel, cfg["macro"], currencies)` then
   `pair_bias(scores, pairs)`. The bias index is calendar-daily with lags
   already applied: the value dated `t` uses data published on or before
   `t-1`. Align to the FX index with `bias.reindex(fx.index, method="ffill")`.
2. Forward return of the pair from FX close on day `t` to day `t+h` (by
   position in the FX business-day index), horizons `h ∈ {5, 10, 20, 40}`.
   Sign convention: a positive bias means long the pair, so the "signed"
   return is `sign(bias) * fwd_return`.
3. Per pair and pooled across pairs, per horizon:
   - Spearman rank IC between bias and forward return.
   - Mean signed return and hit rate when `|bias| >= min_abs_bias` (use the
     config value, 0.75) and when below it.
   - A t-statistic that respects overlap: subsample every `h`-th day (report
     the median t across the `h` offsets) or use a Newey-West HAC standard
     error with lag `h-1`. Do not report a naive t on overlapping returns.
4. Per-component IC: for each component frame in
   `build_currency_scores(...)["components"]`, form the pair difference
   `comp[base] - comp[quote]` and compute the same IC. This shows whether
   carry, rate momentum, risk or positioning carries the signal.
5. By year: pooled 20-day IC per calendar year.
6. Write `reports/macro_ic.json` and `reports/macro_ic.md` with those tables.

Tests (`tests/test_macro_ic.py`, synthetic data, fast): forward returns are
positionally correct and never use same-day or earlier prices; a bias built
from a known future return has IC near 1 and a random bias near 0; the
threshold split sums to the full sample; the non-overlap t-stat routine
returns ~0 on noise.

### A3. Decision rule for Phase A

| Pooled 20d Spearman IC | Non-overlap t | Verdict |
|---|---|---|
| ≥ 0.05 | ≥ 2.0, same sign in ≥ 5 of 8 years | Proceed to the bar backtest (Phase B) |
| 0.02 to 0.05 | any | Weak. Run Phase B only with `trigger: none`; expect a marginal result |
| < 0.02 or negative | any | NO-GO for this component set. Record in `docs/STATUS.md`. Do not retune weights; the next hypothesis needs a different input (e.g. rate expectations from futures), not new weights |

Record the outcome in `docs/STATUS.md` under the `macro_swing` row either way.

### A4. Optional, also cloud-runnable: trigger comparison runner

`scripts/run_macro_compare.py`: loads `strategy_macro_swing.yaml`, runs
backtest + walk-forward + FTMO sim for `trigger` in `none`, `pullback`,
`breakout`, writes one comparison table to `reports/macro_compare.md`. It can
be built and unit-tested with synthetic bars in the cloud; it only *runs* on
the owner's machine. Reuse `scripts/run_event_study.py` as the pattern.

## Phase B — owner's machine: the price-based results

Run on the Windows machine with `data/parquet` present:

```bash
python scripts/download_calendar.py --gasoline
python scripts/download_macro.py --show
python scripts/run_event_study.py                      # pre / post / fade
python scripts/run_backtest.py --strategy macro_swing
python scripts/run_walkforward.py --strategy macro_swing
# repeat the two macro commands with signal.trigger = none and = breakout
# (edit the yaml or copy it to configs/strategy_macro_swing_none.yaml and
# pass --config), or use scripts/run_macro_compare.py if Phase A4 exists
```

What to read, in order:

1. `reports/event_study_surprise_momentum.md`: forecast-quality block first
   (surprise hit rate, reaction in signal direction), then raw R per variant
   with t-stats, then the engine/FTMO rows. A `pre` variant with raw mean R
   near zero means no pricing-in drift for these series.
2. `reports/walkforward_summary_macro_swing.json` and
   `promotion_verdict_macro_swing.md`: the gates are expectancy > 0.15R,
   profit factor > 1.25, max DD < 8%, ≥ 3 profitable folds of ≥ 4, Monte Carlo
   success ≥ 0.80. Compare the three triggers on OOS expectancy before any
   other change.
3. Update `docs/STATUS.md`: move each line to Rejected or keep it Active with
   the numbers. Paste the two markdown reports into the chat for review.

## Phase C — decisions after B

- **Gold.** Add XAUUSD to the macro universe with two gold-specific
  components: US 10y real yield (`DFII10`, negative beta) and broad dollar
  (`DTWEXBGS`, negative beta). The loader already handles both as FRED
  globals; the strategy maps a pair's currencies, so XAU needs a small
  special case (score XAU as a pseudo-currency from those two components).
- **Event study forecasts.** The gasoline nowcast lost to consensus. The only
  remaining lever is an external point-in-time forecast file
  (`nowcast.external_csv`: columns `timestamp,currency,event,nowcast`).
  Candidates: Cleveland Fed nowcast vintages, Kalshi CPI bracket prices
  captured the day before each release. Park this unless Phase B's `pre`
  variant shows drift.
- **PR #7.** Draft, clean, no CI in the repo. Merge when the owner is happy,
  or keep stacking on the branch.

## Pitfalls already hit (save yourself the hour)

- pandas 3.x: `Series.tz_localize` is for the index; use `.dt.tz_localize` or
  a `DatetimeIndex`. Mixed timestamp layouts in one column need
  `format="ISO8601"` with a `format="mixed"` fallback (`to_utc_series` in
  `src/data/calendar.py`).
- FXStreet files member-state prints (DE, ES, ...) under currency EUR with the
  same event name, and lists flash and final releases under the same name.
  Series are keyed by (currency, country, event, preliminary). US CPI
  consensus equals the print exactly in 80% of 2026 rows; treat 2026 with
  suspicion.
- Bank of England IADB throttles bursts: every other request returns a
  redirect to an HTML error page. The fetcher retries with long waits; pace
  any new BoE calls.
- BIS policy rates: the dataflow is `WS_CBPOL`, not `WS_CBPOL_D`.
- FRED `SP500` only carries 10 years. IMF copper (`PCOPPUSDM`) is monthly with
  a ~3-month lag, so the AUD metals component is usually dropped as stale by
  the 45-day forward-fill limit; a daily copper CSV can be dropped in as a
  `csv` source.
- The dollar index is absent from the CFTC financial-futures file; USD
  positioning is derived as the negative mean of the other currencies.
- `python -m pytest | tail` hides the exit code. A push went out with a
  failing test that way. Use `set -o pipefail` or no pipe.

## Definition of done per phase

- Phase A: `scripts/run_macro_ic.py` runs in the cloud container end to end,
  tests pass, `reports/macro_ic.md` exists, `docs/STATUS.md` records the
  verdict from the A3 table, commit pushed to the branch.
- Phase B: the three reports exist locally, their key tables are pasted into
  the chat, `docs/STATUS.md` updated and pushed.
- Phase C: a decision per bullet written into `docs/STATUS.md` "Next".
