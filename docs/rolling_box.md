# Rolling 15-minute box breakout (`rolling_box`)

**Verdict: NO-GO.** No setup makes money after costs, on any market, in either period.

## Idea

Every M15 candle from the Tokyo open (00:00 UTC / 05:30 IST) to the NY end
(19:30 UTC / 01:00 IST) is a box: 78 boxes per weekday. Trade the first candle
close beyond the box high (long) or low (short). Always flat at 01:00 IST.

## Grid (2,592 setups per market)

| Dimension | Values |
|-----------|--------|
| Confirmation candle | M1, M5, M15 close beyond the box |
| Breakout valid for | 15, 30, 45, 60, 90, 120 min after the box closes |
| Stop | opposite side, box middle, 1x ATR, 2x ATR (M15 ATR14 from entry) |
| Exit | 1R, 1.5R, 2R, 3R target; close after 15 or 60 min; hold to 01:00 IST; trail 1R after +1R; trail 1x ATR |
| Concurrency | one trade at a time per market, or every box independent |
| Cost filter | none, or skip when the stop is < 3x the round-trip cost |

Universe: 28 FX pairs, XAUUSD, US30, NAS100, SPX500, GER40, UK100, JPN225
(35 markets, 90,720 market/setup combinations). Dukascopy M1 bid, 2018-01 to
2026-10. Ranked on 2018-22, checked on 2023-26. Costs from `configs/costs.yaml`;
"half costs" halves spread, slippage and commission (roughly a raw-spread account).

## Results (all markets pooled, R per trade)

- 0 of 2,592 setups are positive after costs in 2018-22, in 2023-26, or at half
  costs in 2023-26.
- Before costs there is no edge: median -0.01R, best +0.017R (11% of setups > 0).
  Median after costs: -0.22R.
- Best 2018-22 setup (M15 close, 15 min, 2x ATR stop, hold to 01:00 IST, every
  box, stop >= 3x cost): -0.17R in 2018-22, -0.17R in 2023-26, +0.01R before costs.
- The user's default (M15 close, 60 min, opposite-side stop, 1R target, one per
  market): -0.29R / -0.31R, -0.01R before costs, 46% win rate.
- Every IST hour and every year 2018-2026 is negative for both setups.

What the settings change (median across setups):

- Window and confirmation candle barely matter (within 0.002R of each other).
- Stops: 2x ATR -0.21R, opposite side -0.25R, 1x ATR -0.32R, box middle -0.34R.
  Tighter stops just pay costs more often.
- The cost filter halves the loss (-0.21R vs -0.42R) by skipping tiny boxes.
- Hold to 01:00 IST is the only exit with a positive median before costs (+0.004R).

## By market

No market's best 2018-22 setup is positive in 2023-26 at full costs, and no setup
is positive in both periods on any market. Indices and gold lose least (costs are
a smaller share of the box). At half costs in 2023-26 only four markets are
positive: USDJPY +0.07R, XAUUSD +0.05R, GBPJPY +0.01R, NAS100 +0.01R.

Trading only the market-hours that were positive in 2018-22 (40+ trades) does
not carry over: median +0.03R in 2018-22 becomes -0.05R in 2023-26; 7% of
setups stay positive.

## Leads (not validated; found by searching many combinations)

- **USDJPY**, M1 close, 15 min, box-middle stop, hold to 01:00 IST, one at a time,
  stop >= 3x cost: +0.21R before costs, positive in every year 2018-2026 and on
  both longs and shorts; -0.05R after costs in both periods, +0.07R at half costs.
  A trend-following profile (77% of trades stopped, the rest run to session end).
- **NAS100 around the NY cash open** (boxes 17:30-19:30 IST) is positive in both
  periods for the best pooled setup. It is one of 665 market-hour cells, so part
  of this is likely chance.

## Reproduce

```bash
python scripts/download_data.py --full --intervals M1 --workers 8
python scripts/run_rolling_box.py --workers 4   # ~6 min; writes reports/rolling_box/
```

## Variant: one trade a day in the EMA direction (`rolling_box_ema`)

**Verdict: NO-GO.** The EMA filter does not help; it makes results slightly worse.

Same boxes, but each market takes only one trade per day: the first breakout after
the start time whose direction agrees with the EMA (box close above the EMA allows
only buys, below only sells; counter-trend breakouts are ignored). The EMA reads only
bars closed by the box close. `none` takes the first breakout in either direction and
is the control. Config: `configs/strategy_rolling_box_ema.yaml`.

| Dimension | Values |
|-----------|--------|
| Direction rule | none (control), M15 EMA 50, M15 EMA 200, H1 EMA 200 |
| Day starts | 05:30 IST (Tokyo), 12:30 IST (London), 18:30 IST (New York) |
| Confirmation candle | M1, M15 |
| Breakout valid for | 15, 60, 120 min |
| Stop | opposite side, box middle, 2x ATR |
| Exit | 1R, 2R, 3R target; hold to 01:00 IST; trail 1R after +1R |
| Cost filter | none, or skip when the stop is < 3x the round-trip cost |

2,160 setups per market, 75,600 market/setup combinations.

Results (all markets pooled, R per trade):

- 0 of 2,160 setups are positive after costs in 2018-22, in 2023-26, or at half
  costs in 2023-26. 32% are positive before costs, but by little (best +0.04R).
- Each EMA setup against the identical no-EMA setup (median change):
  M15 EMA 50 -0.0005R (47% of setups improved), M15 EMA 200 -0.007R (31%),
  H1 EMA 200 -0.011R (14%). The change is negative before costs too, so
  with-trend breakouts are slightly worse than either-direction ones.
- Best 2018-22 setup has no EMA: from 12:30 IST, M15 close, 120 min, opposite-side
  stop, hold to 01:00 IST, stop >= 3x cost. -0.16R in 2018-22, -0.21R in 2023-26,
  +0.04R before costs.
- The user's rules (from 05:30 IST, M15 close, 60 min, opposite-side stop, 1R):
  -0.38R with no EMA, -0.41R with the H1 EMA 200 (45% / 44% win rate).
- Start time: New York (18:30 IST) loses least (median -0.23R), Tokyo most (-0.26R).
- In-sample winners do not persist: of the 647 market/setup combinations positive in
  2018-22, only 20% are still positive in 2023-26.
- USDJPY again stands out before costs (90% of its setups positive, median +0.05R).
  Its best setup (M15 EMA 200, from 05:30 IST, M15 close, box-middle stop, hold to
  01:00 IST, stop >= 3x cost) is +0.08R in 2018-22 and +0.02R in 2023-26 after costs,
  +0.31R before costs; only 3 of its 2,160 setups are positive in both periods.

Reproduce: `python scripts/run_rolling_box.py --config strategy_rolling_box_ema.yaml --out reports/rolling_box_ema --workers 4`
