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
| `rolling_box` | Rolling M15 box breakout, Tokyo open to 01:00 IST, 35 markets x 2,592 setups; best pooled OOS ~-0.17R, gross edge ~0 ([notes](rolling_box.md)) | NO-GO |
| `rolling_box_ema` | Same boxes, one trade/day in the EMA direction (M15 50/200, H1 200) from Tokyo/London/NY; EMA worsens results vs no EMA, best pooled OOS ~-0.21R ([notes](rolling_box.md#variant-one-trade-a-day-in-the-ema-direction-rolling_box_ema)) | NO-GO |

## Active

None. All current lines are NO-GO.

## Next

Pick a new hypothesis (do not stack knobs on rejected XAU / majors HTF lines).
Open leads from `rolling_box` (not validated): USDJPY session-hold breakout with a
box-middle stop (+0.21R gross every year, needs ~half the modelled costs; still the
only market positive after costs in `rolling_box_ema`, at +0.02R OOS) and NAS100
around the NY cash open. EMA trend filters did not help intraday box breakouts.
