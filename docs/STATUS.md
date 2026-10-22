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

## Active

| Strategy | Hypothesis | Status |
|----------|------------|--------|
| `macro_swing` | Fundamentals pick pair + direction, technicals time it. Per-currency score = weighted mean of standardized components: 2y-yield repricing (60d), policy-rate carry, risk-off beta (VIX + equities), oil for CAD, copper for AUD, contrarian CFTC positioning. Pair bias = base − quote; trade only when |bias| ≥ threshold; D1 pullback / breakout / none trigger; ATR stop; exit on bias flip. | RESEARCH — loaders + scoring + tests in place; needs local run on parquet bars |
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

## Next

1. Run the event study locally on parquet bars; judge `pre` vs `post` vs `fade` on raw t-stats before any knob.
2. Price drift can exist even with a weak surprise forecast (positioning, not forecasting). If `pre` is flat too, the pricing-in line is NO-GO for these series.
3. Plug in a stronger forecast via `nowcast.external_csv` (Cleveland Fed nowcast vintages, Kalshi CPI implied) and re-run `--calendar-only` before spending bars on it.
4. Macro swing: `python scripts/download_macro.py` then `python scripts/run_backtest.py --strategy macro_swing` and `run_walkforward.py`. Compare `trigger: none` (always in the strongest pair) vs `pullback` vs `breakout` before any weight tuning; component weights are priors, not fitted.
