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

## Active (PDF proxies — untested on live parquet)

| Strategy | Source PDF | Notes |
|----------|------------|-------|
| `ebp` | Omar Agag EBP | H4 engulfing + fib 25/50/75 limit/market; XAU+majors |
| `nowick` | Bard FX Nowick | M15 no-wick + 9-bar retest; trend=EMA50; structure SL |
| `fib_swing` | Fib Retracement Swing | H4 EMA tap + fractal 0.71 align + reversal |
| `judas` | ICT Judas Asia | M15 Asia sweep + MSS; no trendline liquidity; `use_signal_tp` |

## Parked (not coded — discretionary / incomplete / needs M1–M5)

| Strategy PDF | Why parked |
|--------------|------------|
| Matt’s Wicks | Checklist only; no complete entry/exit algorithm |
| Waqar Asim Scalping | Discretionary S/D + inducement; vague SL; needs M1 |
| Ali Khan DRT | Range/SMT detection too soft without invented rules |
| Doyle Exchange S/D | Objective OB/impulse undefined; needs 5m |
| Trader Mike Failed 2s | Needs 1H/15m/1m + FVG/MSB stack |
| Quarterly Theory + SMT | Dual-asset SSMT; defer until SMT feature module |

Do not invent hard rules for parked PDFs unless supplied explicitly.

## Next

Run `ebp` / `nowick` / `fib_swing` / `judas` on real `data/parquet` (or synthetic smoke), then FTMO + walk-forward. Do not stack knobs on rejected HTF/XAU lines.
