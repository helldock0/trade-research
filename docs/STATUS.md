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
| `nykz_breakout` | NAS100 M5: 07:00-10:00 NY range, body-close break, enter next close, stop beyond break bar +2pt, 4R, hold to SL/TP. 2023-01 → 2026-10 with real bid/ask: 800 trades, WR 19.6%, ~-0.044R (-34.8R, max DD 75.8R). Zero-spread ~+0.010R. Longs ~0R, shorts ~-0.093R | v1 negative after spread |

## Next

Pick a new hypothesis (do not stack knobs on rejected XAU / majors HTF lines).
