# Research status

Source data: Dukascopy BID OHLC in `data/parquet` (local only).
Detailed run artifacts stay in `reports/` (local only).

## Rejected (do not retune this cycle)

| Strategy | Notes | Verdict |
|----------|-------|---------|
| `cssm` | Session CS momentum ~-0.05R | NO-GO |
| `csmo` | CS momentum open ~-0.044R | NO-GO |
| `orb` | Pure session ORB | NO-GO |
| `htf_pullback` | Majors H4/M15 EMA pullback ~-0.045R | NO-GO |
| `htf_adx_pullback` | ADX-filtered HTF pullback ~-0.079R | NO-GO |
| `d1_mom` | D1 20d CS momentum ~-0.064R | NO-GO |
| `fail_fade` | London fail fade ~-0.111R | NO-GO |
| `vwap_fade` | Session VWAP fade ~-0.035R | NO-GO |
| `msnr` | H4 MSNR + M15 rejection ~-0.079R | NO-GO |
| `donchian_atr` | H1 Donchian + ATR ~-0.050R | NO-GO |
| `xau_htf_pullback` | XAUUSD H4/M15 EMA pullback; OOS expectancy ~-0.13R | NO-GO |
| `macro_swing` | Macro pair bias (rates, carry, risk, oil, copper, CFTC), `trigger: none`; OOS expectancy ~+0.054R, Monte Carlo success 0.18. Phase A IC was WEAK. Trailing-stop and correlation-gated basket variants (owner request) reach OOS ~+0.063R, PF 1.20; the basket passes Monte Carlo (0.93) and the FTMO path but not expectancy / PF. See "Macro swing" sections below. | NO-GO |

## Active

| Strategy | Hypothesis | Status |
|----------|------------|--------|
| `event_drift` | Market prices an expected surprise before scheduled releases. Pre-release signal = surprise momentum (consensus anchoring) or gasoline CPI nowcast gap. Variants: `pre` (exit before print), `post` (hold through), `fade` (against the reaction). | RESEARCH — harness + tests in place; needs local run on parquet bars |

Run: `python scripts/download_calendar.py --gasoline` then `python scripts/run_event_study.py`.
Read `reports/event_study_<mode>.md`: forecast quality first (surprise hit rate, reaction in signal direction), then raw R per variant, then engine/FTMO rows.

### Calendar-only results (2018-01 → 2026-10, FXStreet, no price bars yet)

`python scripts/run_event_study.py --calendar-only` answers one question: does the pre-release signal predict the *direction* of the realized surprise?

| Signal | Events | Surprise-direction hit rate | t vs 0.5 | Read |
|--------|--------|-----------------------------|----------|------|
| `surprise_momentum` (mean of last 2 standardized surprises, 23 G10 series) | 641 | 0.534 | 1.70 | Marginal. Consensus anchoring exists but is weak; not significant at conventional levels. Best series: EMU flash HICP 0.68 (n=28), AU unemployment 0.74 (n=27); worst: US core PCE 0.37, US retail sales 0.43. Small per-series samples, so treat the ranking as noise until the price study confirms it. |
| `nowcast_gap` (gasoline-based headline CPI nowcast vs consensus, US CPI MoM) | 60 | 0.483 | -0.26 | Fails. The nowcast's absolute error (0.16) is worse than consensus (0.09); economists already price gasoline. A better public forecast (Cleveland Fed vintages, prediction markets) must come from outside: see `nowcast.external_csv`. |

Data caveats found while building: FXStreet files member-state prints (DE, ES, ...) under currency EUR with the same name and lists flash/final under the same name, so series are keyed by (currency, country, event, preliminary); the config pins EMU flash HICP and US advance GDP. US CPI MoM consensus equals the print exactly in 80% of 2026 rows vs 17–50% in earlier years; verify consensus integrity before trusting 2026.

Verdict so far: the pricing-in thesis is not dead but the easy signals are weak. The price-based part (`pre` drift, `post` surprise, `fade`) still needs the local M15 bars.

### Macro swing: live score table (2026-10-21, real data, all sources fetched)

| CCY | rate mom | carry | risk | positioning | score |
|---|---|---|---|---|---|
| USD | 1.89 | 0.74 | -0.10 | 1.19 | 1.12 |
| AUD | -0.10 | 1.37 | 0.19 | -0.89 | 0.09 |
| GBP | 0.60 | 0.63 | 0.06 | -1.59 | 0.06 |
| CAD | -0.53 | -0.69 | 0.13 | 0.57 | -0.13 |
| EUR | -0.74 | -0.47 | -0.00 | -0.92 | -0.58 |
| JPY | -1.11 | -1.57 | -0.19 | -0.72 | -0.94 |

Pair bias: USDJPY +2.06, USDCAD +1.25, AUDUSD -1.03, GBPUSD -1.06, EURUSD -1.70 (CAD also carries oil +0.28). Caveats: the GBP rate input is the BoE 5y par yield (no free daily 2y); the BoE endpoint throttles bursts, so the fetcher retries with long waits; IMF copper is monthly with a ~3-month lag, so the AUD metals component is usually stale and dropped by the 45-day forward-fill limit.

### Macro swing: Phase A IC check (2026-10-22, FRED daily FX, no price bars)

`python scripts/run_macro_ic.py` → `reports/macro_ic.md`. Pair bias from the config as is (weights are priors; nothing fitted) vs forward log returns on FRED H.10 noon rates (`DEXUSEU`, `DEXUSUK`, `DEXJPUS`, `DEXUSAL`, `DEXCAUS`), 5 pairs, decision dates 2018-01-02 → 2026-09-17. Bias dated t uses data published by t−1; the return runs from the fix on t to the fix h business days later. t-stats are clustered by date (the five USD pairs are not independent) and corrected for overlap.

| h | pooled Spearman IC | t (non-overlap) | t (Newey-West, lag h−1) | IC with 1 extra day of lag |
|---|---|---|---|---|
| 5 | 0.054 | 1.82 | 2.19 | 0.050 |
| 10 | 0.062 | 1.35 | 1.84 | 0.059 |
| 20 | 0.083 | 1.41 | 1.74 | 0.077 |
| 40 | 0.128 | 1.29 | 2.06 | 0.128 |

- Threshold split (20d): |bias| ≥ 0.75 on 37% of pair-days, +27.3 bp mean in the bias direction, hit 0.559, t 1.01; below it +5.7 bp, hit 0.501.
- Per component (20d IC of base − quote): carry 0.118 (t 2.29), positioning 0.110 (t 1.55), metals 0.045 (AUDUSD only), rate momentum 0.036 (t 0.46; the largest weight), risk −0.095 (t −1.80), oil −0.154 (USDCAD only, t −1.58). Risk and oil point the wrong way over this sample. That is a finding, not a licence to flip or reweight them; any change needs a holdout and a ledger note.
- Per pair (20d IC): EURUSD 0.143, USDJPY 0.094, GBPUSD 0.050, AUDUSD 0.013, USDCAD −0.033.
- By year (20d IC): 2018 0.23, 2019 −0.20, 2020 0.10, 2021 −0.05, 2022 0.13, 2023 0.13, 2024 0.13, 2025 −0.06 (2026 partial 0.04). No year has |t| above 1.5.

**Verdict: WEAK.** The IC clears 0.05 and the year count (5 of 8), but t is 1.41 (Newey-West 1.74), below 2.0. The A3 table has no row for IC ≥ 0.05 with t < 2. That case fails PROCEED on t alone, and NO-GO is defined as IC < 0.02, so it is read as WEAK. Action per A3: run Phase B only with `trigger: none`, and expect a marginal result.

Context, not part of the verdict: from 2011 the pooled 20d IC is 0.020 (t 0.54) and carry drops to 0.021. Before 2018 that composite has no positioning (CFTC history starts in 2017 and needs a 378-day warm-up). So the 2018+ IC leans on the 2022–24 USD rate cycle. A naive t on the overlapping 20d returns would read 5.85; without the overlap correction this would have looked like a GO. Caveats: noon mid rates instead of 21:00 UTC BID bars; no costs. The one-extra-day-lag column shows no same-day leakage (e.g. CFTC data released Friday afternoon vs the Friday noon fix).

### Macro swing: Phase B bar backtest (2026-10-22, cloud run, `trigger: none`)

Inputs:
- Bars: the M5 BID parquet files on the repo's `data` branch (Git LFS), resampled to M15, covering 2018-01-01 → 2026-10-20.
- Calendar: refreshed from FXStreet, so the news blackout is on.
- Macro panel: as in Phase A.
- Config: `strategy_macro_swing.yaml` with only `signal.trigger` changed to `none`.

| Run | Trades | Expectancy | Profit factor | Win rate | Max DD |
|---|---|---|---|---|---|
| Full-sample backtest | 280 | +0.040R | 1.08 | 0.29 | 11.4% |
| Walk-forward OOS (14 folds, 18m train / 6m test) | | +0.054R | 1.30 | | 4.7% |

Promotion gates:

| Gate | Needed | Result | Pass |
|---|---|---|---|
| OOS expectancy | > 0.15R | 0.054R | no |
| Profit factor | > 1.25 | 1.30 | yes |
| Max drawdown | < 8% | 4.7% | yes |
| Profitable folds | ≥ 3 of ≥ 4 | 7 of 14 | yes |
| Monte Carlo success | ≥ 0.80 | 0.18 (ruin 0.20) | no |

**Verdict: NO-GO.** This matches the Phase A read: there is a small positive edge, but it is far too thin for the prop gates.
- Per the plan, `pullback` / `breakout` are compared only if `none` clears the gates. It does not, so they were not run.
- Weights were not retuned.
- Caveat: these M15 bars were built from M5, not the owner's native M15 files, so a local run can differ slightly.

### Macro swing: trailing stop and correlation-gated basket (2026-10-22, owner request after the NO-GO)

`python scripts/run_macro_compare.py` → `reports/macro_compare.md`. Same bars, panel and calendar as Phase B; variants are in `variants:` of `configs/strategy_macro_swing.yaml`. The values were fixed before the first run and not tuned:
- `trail`: no fixed target, and no 20-day max hold. After each bar, the stop ratchets to 1R (= the initial 2 ATR stop distance) behind the best price since entry. The bias-flip exit stays.
- `basket`: `trail`, plus up to 3 positions at once (one per pair). A new trade is skipped when its 60-day daily-return correlation with an open one, signed by both sides, is above 0.6. The correlation is point-in-time and ends the day before. Example: long EURUSD + long GBPUSD is the same short-USD bet and gets skipped.

| Variant | Trades | Exp R | PF | Win | Max DD | Return / yr | OOS exp R | OOS PF | Folds + | MC success | FTMO path | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fixed_tp` (Phase B) | 280 | 0.040 | 1.08 | 0.29 | 11.4% | 0.6% | 0.054 | 1.30 | 7/14 | 0.18 | fail | NO-GO |
| `trail` | 380 | 0.055 | 1.14 | 0.40 | 7.5% | 1.2% | 0.062 | 1.20 | 8/14 | 0.27 | fail | NO-GO |
| `basket` | 583 | 0.089 | 1.22 | 0.42 | 7.8% | 2.9% | 0.063 | 1.20 | 8/14 | 0.93 | pass | NO-GO |

What changed and what didn't:
- The trailing stop raises the win rate and cuts the drawdown.
- The basket roughly doubles the trades, so Monte Carlo (which runs through the whole trade list) and the FTMO path replay now pass.
- Neither moves the per-trade OOS edge much (0.054 → 0.063R vs the 0.15R gate), and OOS PF slips to 1.20 (gate 1.25).
- Still NO-GO.

Caveats:
- Three variants were tried after a NO-GO, so the best row is optimistic.
- The Monte Carlo shuffles trades as independent draws and the engine's drawdown is on closed trades, so with up to 3 positions open the real risk is higher than these numbers suggest.
- Stops that the next bar gapped through (fill at the stop, not the open) cost under 0.1R in total, so gaps do not change the picture.

Open question for the owner, not acted on: the expectancy gate is per trade. A basket can pass the prop-style checks (Monte Carlo, FTMO path) with a thin per-trade edge, so decide whether that gate is the right one for a basket before testing anything else here.

## Next

Hand-off plan with phases, decision rules and pitfalls: `docs/plans/2026-10-22-fundamentals-next-steps.md`.

1. Run the event study locally on parquet bars; judge `pre` vs `post` vs `fade` on raw t-stats before any knob.
2. Price drift can exist even with a weak surprise forecast (positioning, not forecasting). If `pre` is flat too, the pricing-in line is NO-GO for these series.
3. Plug in a stronger forecast via `nowcast.external_csv` (Cleveland Fed nowcast vintages, Kalshi CPI implied) and re-run `--calendar-only` before spending bars on it.
4. Macro swing is NO-GO (Phase A WEAK, Phase B fails expectancy and Monte Carlo). Do not retune weights or triggers. A next macro hypothesis needs a different input (e.g. rate expectations from futures / OIS), not new weights.
5. M5 / M1 BID bars for the majors are on the `data` branch via Git LFS (`git lfs pull --include=...`), so cloud sessions can backtest by resampling them to M15.
