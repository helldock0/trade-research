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
| `nykz_breakout` | NAS100 M5: 07:00-10:00 NY range, body-close break, enter next close, stop beyond break bar +2pt, 4R, hold to SL/TP. 2023-01 → 2026-10 with real bid/ask: 800 trades, WR 19.6%, ~-0.044R (-34.8R, max DD 75.8R). Zero-spread ~+0.010R. Longs ~0R, shorts ~-0.093R. Variant sweep (`scripts/compare_nykz_breakout.py`, 5 break rules x 2 stop rules, 2023-26 plus a 2018-22 check): next-candle confirmation +0.085R (2023-26, max DD 30R) / +0.062R (2018-22); close >=0.1% beyond + prior-candle stop +0.053R / +0.189R; body-size filter alone no help. Best pooled t ~2.5 after 10 tries: promising, not proven | v1 negative. **v2 = next-candle confirm (config default)**: 805 trades, WR 21.9%, +68.3R, PF 1.11, max DD 30R, losing streak 18; 2018-22 check +0.062R. Exit sweep (`--grid exit`, 5 stop sizes x 4 trails on v2 entries): **2x ATR(14, M5) stop, fixed 4R** +0.136R (2023-26, 753 trades, max DD 36R) / +0.153R (2018-22), 8 of 9 years positive, pooled t ~3.0 after ~30 variants. Breakeven and ATR trailing cut the 4R winners and lowered 2023-26 results in every stop size. FVG retest-and-confirm entry (`--grid fvg`): only 57% of confirmed breaks give an entry; best +0.106R (2023-26) but +0.022R (2018-22), less steady by year; no stop choice beats next-close + 2x ATR. FVG limit entries (near edge / 50%): fill more often (616-717 trades) but win less; best near edge + 2x ATR +0.092R / +0.072R, 50% with candle stops negative. Paper-trade before trusting |

## Next

Pick a new hypothesis (do not stack knobs on rejected XAU / majors HTF lines).
