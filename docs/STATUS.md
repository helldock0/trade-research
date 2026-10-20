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

| Strategy | Notes | Verdict |
|----------|-------|---------|
| `nykz_breakout` | NAS100 M5: 07:00-10:00 NY range, body-close break, enter next close, stop beyond break bar +2pt, 4R, hold to SL/TP. 2023-01 → 2026-10 with real bid/ask: 800 trades, WR 19.6%, ~-0.044R (-34.8R, max DD 75.8R). Zero-spread ~+0.010R. Longs ~0R, shorts ~-0.093R. Variant sweep (`scripts/compare_nykz_breakout.py`, 5 break rules x 2 stop rules, 2023-26 plus a 2018-22 check): next-candle confirmation +0.085R (2023-26, max DD 30R) / +0.062R (2018-22); close >=0.1% beyond + prior-candle stop +0.053R / +0.189R; body-size filter alone no help. Best pooled t ~2.5 after 10 tries: promising, not proven | FTMO 2-step sim (`scripts/prop_sim_nykz_breakout.py`, a challenge started at every trade 2018-26, open-trade loss counted): v3 pass rate 73% at 0.5% risk (median 208 days), 52% at 1% (59 days), 30% at 2% (22 days), steady across 2018-22 / 2023-26 starts; v4 prop 75% at 0.5% but 60% / 99% by era and median 389 days, 49% at 1% (95 days). v4 does not beat v3 for FTMO. v1 negative. Two live versions: **v3 (normal accounts, `strategy_nykz_breakout.yaml`)** and **v4 prop (`strategy_nykz_breakout_prop.yaml`) = v3 + skip quietest 20% of days by daily ATR + 3R**: 613 trades, WR 29.0%, +99.8R, max DD 19.5R, streak 13 (2023-26); 2018-22 +80.2R, max DD 32.0R (v3 + ATR filter at 4R: +121.1R, DD 27.0R). **v3**: 753 trades, WR 23.0%, +102.4R, PF 1.17, max DD 36.3R, losing streak 18 (longs +91.4R, shorts +11.0R); 2018-22 check +0.153R. Target sweep on v3 (`--grid rr`, 2023-26 / 2018-22): 1-2R ~0 or negative; 3R +98.6R / +86.5R (64% up months, streak 12); 4R +102.4R / +159.8R; 6R +146.8R / +166.6R (streak 22, winners held ~45h); 8R +181.9R / +159.9R (WR 15%, streak 27, all from longs). Gains above 4R come from longs riding the 2023-26 uptrend and do not show in 2018-22. Weekday filter (`--grid weekday`): no day is weak in both periods; skip Mon +128.1R (2023-26) but +60.9R (2018-22), skip Thu +64.2R but +217.1R; trade every day. Session filter (`--grid session`, entry cutoffs 10:30-16:55 NY): overlap-only (entries by 12:00 NY) +57.4R (2023-26) / +181.3R (2018-22) vs all-day +102.4R / +159.8R, with 2025 -23.7R; entries by 10:30 have the best R/trade in both periods (+0.157 / +0.317) but half the trades. Keep the 16:55 cutoff. News filter (`--grid news`, `configs/news_us_high_impact.csv`: CPI/NFP from BLS release archives, FOMC from federalreserve.gov, 2018-2026): skipping CPI+NFP+FOMC days costs ~22R in both periods; skipping FOMC only ~neutral (+105.7R / +158.8R); keep trading news days. Volatility filter (`--grid vol`, ranks vs previous 100 days): skipping the quietest 20% of days by prior-day ATR(14)/price keeps R/trade (+0.188 / +0.153), lowers max DD in both periods (28.3R / 27.0R vs 36.3R / 35.3R) and leaves no losing year, but total 2018-22 R falls to +121.1R (fewer trades); KZ-range 20% similar (+110.1R / +129.8R); 33% cuts too many trades. Smoother, not more profitable. v2 = next-candle confirm, break-candle stop: 805 trades, WR 21.9%, +68.3R, PF 1.11, max DD 30R, losing streak 18; 2018-22 check +0.062R. Exit sweep (`--grid exit`, 5 stop sizes x 4 trails on v2 entries): **2x ATR(14, M5) stop, fixed 4R** +0.136R (2023-26, 753 trades, max DD 36R) / +0.153R (2018-22), 8 of 9 years positive, pooled t ~3.0 after ~30 variants. Breakeven and ATR trailing cut the 4R winners and lowered 2023-26 results in every stop size. FVG retest-and-confirm entry (`--grid fvg`): only 57% of confirmed breaks give an entry; best +0.106R (2023-26) but +0.022R (2018-22), less steady by year; no stop choice beats next-close + 2x ATR. FVG limit entries (near edge / 50%): fill more often (616-717 trades) but win less; best near edge + 2x ATR +0.092R / +0.072R, 50% with candle stops negative. Paper-trade before trusting |

## Next

Pick a new hypothesis (do not stack knobs on rejected XAU / majors HTF lines).
