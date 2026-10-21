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
| `ny_range_fade` | NY 08:12-09:12 range fade, TP opposite edge; M1 2018-2026, 12 symbols (7 majors, XAU, NAS100/SPX500/US30/US2000): ~-0.16 RU/trade net over 26k trades (t -15.7), gross ~-0.04 RU; all 84 symbol x variant cells negative | NO-GO |

## Notes: `ny_range_fade`

- User-supplied rules: break above range high -> sell (TP range low); break below range low -> buy (TP range high). No stop in the rules.
- Runner: `python scripts/run_ny_range_fade.py` (own M1 simulator; P&L in range units, RU = P&L / range width).
- Gross edge ~0: FX/gold hit TP ~35-42% and otherwise bleed to the 16:00 NY flat; indices hit TP ~63% but carry 15-24 RU tail losses after the 09:30 open.
- Costs ~0.15 RU/trade on FX (14-25 pip ranges), ~0.07-0.09 on indices/gold. Stops (0.5x/1x range) cut tails but stay negative (best pooled -0.11 RU).
- M1 parquet for the 12 symbols now cached in `data/parquet` (Dukascopy, 2018-01-01 -> 2026-10-21).

## Active

None. All current lines are NO-GO.

## Next

Pick a new hypothesis (do not stack knobs on rejected XAU / majors HTF lines).
