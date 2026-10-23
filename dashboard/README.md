# v3 dashboard

Source of the claude.ai dashboard for the NAS100 NY kill-zone breakout (rule v3):
https://claude.ai/artifact/CDaM6oCD234XTVAjJxnPze

- `index.html`: the page. It runs inside claude.ai's dashboard viewer (`dash.*` helpers), not as a standalone file.
- `data/`: the datasets it reads, with Blueberry NAS100.p costs (spread at least 1.26 points, swap long -$58.5 / short +$7.6 per lot a night).

| File | Made by |
|---|---|
| `nas100_kz_v3_trades.csv` | v3 trades 2023-01 to 2026-10, `configs/strategy_nykz_breakout_blueberry.yaml`, $1,000 per R (`swap_r` = swap in R) |
| `nas100_kz_v3_trades_zero_spread.csv` | the same rules with no spread or swap |
| `v3_ftmo_pass_rates.csv` | `python scripts/prop_sim_nykz_breakout.py --config strategy_nykz_breakout_blueberry.yaml --risks 0.25,0.5,0.75,1,1.5,2` |
| `blueberry_10k_plan.csv` | `python scripts/instant_sim_nykz_breakout.py --config strategy_nykz_breakout_blueberry.yaml --balance 10000 --targets-usd 100,350 --risks 0.25,0.5` |
| `blueberry_addons.csv` | the same with `--min-days 7`, `--min-days 0`, `--min-active-days 3` and `--max-trade-loss-pct 2` |
