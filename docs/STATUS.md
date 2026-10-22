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
| `macro_swing` | Macro pair bias (rates, carry, risk, oil, copper, CFTC), `trigger: none`; OOS expectancy ~+0.054R. Trailing-stop / correlation-gated basket variants reach OOS ~+0.063R but fail expectancy and PF. On the 2012–2017 holdout the signal itself is negative (pooled 20d IC −0.098; carry −0.137). An expected-hikes input and a wider universe (NZD, CHF, crosses) both fail the pre-registered holdout rule. See "Macro swing" sections below. | NO-GO |
| `event_drift` | Pricing-in before scheduled releases (23 G10 series), surprise-momentum or gasoline-nowcast signal. `pre` raw −0.006R (t −0.24, n=826), `post` −0.013R, `fade` −0.006R; every engine run negative. The pricing-in rule in the plan makes a flat `pre` a NO-GO. One weak lead (fading US CPI reactions, ~+0.07–0.13R, n 33–71) is parked; see "Event study on bars" below. | NO-GO |

| `tom` | Turn of the month on US index CFDs (long NAS100 from the close before the month's last trading day to the close of the 3rd trading day). The signal check passes on 1971–2017 but has faded since the 2010s. On 2018+ bars it does no better than the same rules on a mid-month window: OOS +0.061R vs placebo +0.112R. See "Turn of the month" below. | NO-GO |

## Active

None. Both fundamentals lines and the turn-of-month line are rejected (2026-10-22).

Event study run: `python scripts/download_calendar.py --gasoline` then `python scripts/run_event_study.py`.
Read `reports/event_study_<mode>.md`: forecast quality first (surprise hit rate, reaction in signal direction), then raw R per variant, then engine/FTMO rows.

### Calendar-only results (2018-01 → 2026-10, FXStreet, no price bars yet)

`python scripts/run_event_study.py --calendar-only` answers one question: does the pre-release signal predict the *direction* of the realized surprise?

| Signal | Events | Surprise-direction hit rate | t vs 0.5 | Read |
|--------|--------|-----------------------------|----------|------|
| `surprise_momentum` (mean of last 2 standardized surprises, 23 G10 series) | 641 | 0.534 | 1.70 | Marginal. Consensus anchoring exists but is weak; not significant at conventional levels. Best series: EMU flash HICP 0.68 (n=28), AU unemployment 0.74 (n=27); worst: US core PCE 0.37, US retail sales 0.43. Small per-series samples, so treat the ranking as noise until the price study confirms it. |
| `nowcast_gap` (gasoline-based headline CPI nowcast vs consensus, US CPI MoM) | 60 | 0.483 | -0.26 | Fails. The nowcast's absolute error (0.16) is worse than consensus (0.09); economists already price gasoline. A better public forecast (Cleveland Fed vintages, prediction markets) must come from outside: see `nowcast.external_csv`. |

Data caveats found while building: FXStreet files member-state prints (DE, ES, ...) under currency EUR with the same name and lists flash/final under the same name, so series are keyed by (currency, country, event, preliminary); the config pins EMU flash HICP and US advance GDP. US CPI MoM consensus equals the print exactly in 80% of 2026 rows vs 17–50% in earlier years; verify consensus integrity before trusting 2026.

Verdict at that stage: the pricing-in thesis was not dead, but the easy signals were weak. The price-based study below settles it.

### Event study on bars (2026-10-22, cloud run, M15 BID bars 2018-01 → 2026-10)

Bars come from the data branch (`scripts/bars_from_data_branch.py`), with the FXStreet calendar refreshed and FRED `GASREGW`. The config is unchanged.

Reading rule, fixed before the run (plan Phase B and "Next" 2 below): judge `pre` / `post` / `fade` on raw t-stats, and treat a flat `pre` as NO-GO for these series. 3 variants × 2 signals = 6 tests, so a single |t| ≈ 2 is weak evidence.

Raw returns (no stops, net of costs) and engine runs (stops, costs, one position):

| Signal | Variant | n | Raw mean R | Hit | t | Engine trades | Engine exp R | PF |
|---|---|---|---|---|---|---|---|---|
| `surprise_momentum` | `pre` | 826 | −0.006 | 0.48 | −0.24 | 349 | −0.017 | 0.93 |
| | `post` | 829 | −0.013 | 0.49 | −0.51 | 349 | −0.041 | 0.85 |
| | `fade` | 408 | −0.006 | 0.47 | −0.39 | 330 | −0.022 | 0.82 |
| `nowcast_gap` (US CPI) | `pre` | 71 | −0.079 | 0.35 | −1.26 | 66 | −0.123 | 0.54 |
| | `post` | 71 | −0.132 | 0.35 | −1.75 | 66 | −0.191 | 0.48 |
| | `fade` | 33 | +0.134 | 0.61 | 2.31 | 33 | +0.127 | 2.88 |

- Momentum signals in the strong half (|z| ≥ 0.85) are no better than the weak half.
- The print reaction goes against the momentum signal slightly more often than with it (hit 0.49, t −1.38).
- By year, `pre` swings between −0.07R and +0.09R with no trend.

**Verdict: NO-GO.** There is no pricing-in drift before these releases, holding through the print loses, and fading the reaction does nothing across the 23 series.

Parked lead: fading US CPI reactions (enter 15 min after the print, exit after 4 h). This is a post-hoc look, not a pre-set test:

| Signal | Fade when | n | Mean R | Hit | t |
|---|---|---|---|---|---|
| `nowcast_gap` | the reaction goes the signal's way (config) | 33 | +0.134 | 0.61 | 2.31 |
| `nowcast_gap` | every CPI reaction | 71 | +0.075 | 0.54 | 1.66 |
| `surprise_momentum` | the reaction goes the signal's way | 27 | +0.143 | 0.63 | 1.52 |
| `surprise_momentum` | every CPI reaction | 47 | +0.065 | 0.55 | 1.07 |

- The nowcast itself predicts the CPI surprise the wrong way (hit 0.43), so the edge, if any, is in the reaction, not the forecast.
- Conditioning roughly halves n and doubles the mean, which noise also does.
- At 4–8 trades a year it cannot carry a prop challenge alone.
- To test it honestly it needs out-of-sample data (pre-2018 intraday bars, which the data branch does not have) or a forward test.

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

### Macro swing: holdout check and three improvement ideas (2026-10-22, owner request)

Three ideas, tested under a rule fixed in `configs/strategy_macro_swing.yaml` (`validation:`) before any run. A signal change is kept only if its pooled 20d IC beats the base config's on both the dev window (2018-01 → 2026-09) and the holdout (2012-01 → 2017-12), and is positive on the holdout. Bars start in 2018, so the holdout uses FRED daily FX only.

The ideas:
1. **Longer history for an honest holdout.** CFTC positioning now starts in July 2010. Old TFF files name the date column differently, and the NZD market was renamed in 2022.
2. **Wider universe (`wide`).** Adds NZD and CHF:
   - CHF: policy rate (BIS), 2y yield (SNB), positioning.
   - NZD: policy rate and positioning; RBNZ blocks scripted downloads, so there is no 2y.
   - Pairs: NZDUSD, USDCHF, EURJPY, GBPJPY, AUDJPY and EURGBP (11 pairs, 8 currencies).
3. **Expected hikes (`hikes`).** A new input: 2y yield minus the policy rate, compared across currencies, at weight 1.0.

`python scripts/run_macro_ic.py --validate` → `reports/macro_ic_validation.md`:

| Variant | Pairs | Dev IC (t) | Holdout IC (t) | Dev IC, 5 base pairs | Holdout IC, 5 base pairs | Kept |
|---|---|---|---|---|---|---|
| base (`fixed_tp`) | 5 | 0.084 (1.44) | **−0.098 (−1.43)** | 0.084 | −0.098 | baseline |
| `hikes` | 5 | 0.052 (0.89) | −0.081 (−1.18) | 0.052 | −0.081 | no |
| `wide` | 11 | 0.038 (0.82) | −0.091 (−1.75) | 0.070 | −0.096 | no |
| `wide_hikes` | 11 | −0.004 (−0.09) | −0.077 (−1.49) | 0.039 | −0.077 | no |

The base signal is negative on the holdout, so its 2018+ edge does not carry over to the earlier regime.
- Holdout per component (20d IC):
  - carry −0.137 (t −2.11). On the dev window it was +0.118.
  - rate momentum −0.038, risk −0.055, positioning −0.017.
  - oil +0.114 (USDCAD only).
- By year, 2012, 2013 and 2017 are strongly negative.
- Coverage checked: AUD 2y starts mid-2013, and BIS has no Japan policy rate during QQE (2013–2016), so JPY carry is missing on 56% of holdout days. Otherwise every component is present. The risk component uses VIX only before 2016, because FRED SP500 has 10 years.

Bar backtests, 2018+ only, for information. The holdout already rejects every signal change, so these cannot override it. The macro panel now has the longer CFTC history, which changes 2018–19 positioning: the `fixed_tp` full-sample numbers move from +0.040R / MC 0.18 to +0.063R / MC 0.36. OOS is unchanged, because the folds start mid-2019.

| Variant | Trades | Exp R | Max DD | Return / yr | OOS exp R | OOS PF | Folds + | MC success | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `basket` (5 pairs) | 582 | 0.079 | 6.9% | 2.6% | 0.063 | 1.20 | 8/14 | 0.91 | NO-GO |
| `hikes` | 550 | 0.122 | 6.1% | 3.8% | 0.097 | 1.31 | 9/14 | 0.96 | NO-GO |
| `wide` (11 pairs) | 914 | 0.029 | 12.8% | 1.5% | 0.017 | 1.09 | 6/14 | 0.31 | NO-GO |
| `wide_hikes` | 946 | 0.040 | 23.0% | 2.1% | 0.025 | 1.12 | 6/14 | 0.45 | NO-GO |

- `wide` adds trades but each one is worse: the signal is weaker on the new pairs, as its lower dev IC already said.
- `hikes` looks best in this 2018+ backtest but fails the holdout (IC −0.081) and has a lower dev IC than the base, so treat its backtest as in-sample luck.
- The cross bars are built from the data branch's M1 legs (`scripts/bars_from_data_branch.py`). They sit within 0.03% of FRED's cross rates on a median day.
- The engine now converts cross P&L through the USD major. Before, a full EURJPY stop booked about −0.6R.

**Verdict: none of the three ideas is kept, and macro_swing stays NO-GO.** The composite's edge is regime-dependent: positive 2018–2026, negative 2012–2017. Reweighting would mean fitting one of the two regimes.

### Turn of the month (2026-10-22, new idea, cloud run)

Hypothesis: US equity indices rise from the last trading day of a month through the first three of the next (Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008). Everything below was fixed in `configs/strategy_tom.yaml` before the data was looked at.

**Signal check** on FRED daily closes (`python scripts/run_tom_check.py` → `reports/tom_check.md`). TOM days are the last trading day of each month and the first 3 of the next. The test compares the mean daily log return on TOM days with all other days (Welch t). Rule: t ≥ 2 on NASDAQ Composite over the 1971–2017 holdout, and TOM minus other > 0 on 2018+ for both series.

| Series | Window | TOM bp/day | Other bp/day | Diff | t | Month windows | Window hit |
|---|---|---|---|---|---|---|---|
| NASDAQ Composite | 1971–2017 | 12.1 | 1.6 | +10.5 | 3.58 | 561 | 0.63 |
| NASDAQ Composite | 2018+ | 9.0 | 5.2 | +3.9 | 0.48 | 104 | 0.60 |
| NASDAQ-100 | 1986–2017 | 13.3 | 2.8 | +10.5 | 2.16 | 382 | 0.61 |
| NASDAQ-100 | 2018+ | 10.0 | 6.0 | +4.0 | 0.48 | 104 | 0.57 |

PASS by the rule. By decade, though, the excess falls from +9 to +16 bp/day (1970s–1990s) to +2 bp (2010s, t 0.39) and +5 bp (2020s, t 0.51): consistent with decay after publication.

**Bar backtest** (`python scripts/run_tom_backtest.py` → `reports/tom_compare.md`).
- Bars: NAS100 / US500 M15 CFD bars from the data branch.
- Calendar: US trading days come from the FRED NASDAQ-100 dates.
- Entry and exit at the bar ending at the 16:00 New York close, with a 2× daily ATR protective stop and no target.
- Pre-registered before the bar run: TOM counts only if it clears the gates and its OOS expectancy beats a placebo with the same rules on trading days 9–12. The placebo measures plain index drift.
- The engine now sizes index CFDs at $1 per point per lot (`contracts:` in `configs/costs.yaml`), and `is_cross` no longer misreads NAS100 as an FX cross.

| Variant | Trades | Exp R | PF | Win | Max DD | Return / yr | OOS exp R | Folds + | MC success | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `tom` (NAS100) | 104 | 0.112 | 1.50 | 0.61 | 1.3% | 0.7% | 0.061 | 7/14 | 0.00 | NO-GO |
| `placebo` (days 9–12) | 105 | 0.093 | 1.43 | 0.55 | 1.8% | 0.6% | **0.112** | 9/14 | 0.00 | NO-GO |
| `tom_us500` (cross-check) | 104 | 0.008 | 1.03 | 0.54 | 3.3% | 0.0% | −0.077 | 4/14 | 0.00 | NO-GO |

**Verdict: NO-GO.** On 2018+ the TOM trade earns no more than holding the index on any other four days, so the profit is index drift, not a calendar effect. It also cannot pass a prop challenge: 104 trades of about 0.1R at 0.5% risk add up to about 6% in 8.7 years, so the Monte Carlo never reaches the +10% target.

## Next

Hand-off plan with phases, decision rules and pitfalls: `docs/plans/2026-10-22-fundamentals-next-steps.md`.

1. Done (2026-10-22): event study on bars. `pre` is flat, so event_drift is NO-GO.
2. The "price drift without a good forecast" idea is answered: `pre` is flat for these series.
3. Optional: a stronger external forecast via `nowcast.external_csv` (Cleveland Fed nowcast vintages, Kalshi CPI implied). Re-run `--calendar-only` first. Worth it only together with the parked CPI-fade lead, and that needs out-of-sample bars.
4. Macro swing is NO-GO: Phase A WEAK, Phase B fails expectancy, and the signal is negative on the 2012–2017 holdout. The trailing stop, basket, expected-hikes input and wider universe were all tried. Recommendation: stop this line rather than reweight it. A new macro hypothesis should come with its own holdout test (`run_macro_ic.py --validate`) before any bar run.
5. Turn of the month is NO-GO; the decade table shows the calendar effect has faded since the 2010s.
6. Cloud sessions can backtest: `python scripts/bars_from_data_branch.py` builds `data/parquet/{PAIR}_M15.parquet` from the `data` branch (Git LFS M5 BID; crosses from M1 legs).
