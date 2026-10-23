"""First payout on an instant-funded account for a NY kill-zone breakout config.

Defaults are Blueberry Funded Instant Elite (accounts bought from 12 Mar 2026,
help.blueberryfunded.com): 10% trailing max drawdown that locks at the
starting balance, 1.5% max loss per trade idea, payout after >= $100 realised
profit, 5 active days (each closing >= 0.5% profit) and a 14-day cycle.
An account is started at every trade from --start and replayed on the real
trade sequence; rates count accounts that finished before the data ran out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import ROOT as PROJECT_ROOT, ensure_dirs, load_strategy_config
from src.data.dukascopy_ingest import load_parquet, parquet_path
from src.prop.instant import OPEN, PAYOUT, TRADE_CAP, TRAIL, first_payout
from src.strategy.nykz_breakout import merge_bid_ask, run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Instant-funded first payout for NY kill-zone breakout")
    parser.add_argument("--config", default="strategy_nykz_breakout.yaml")
    parser.add_argument("--start", default="2018-03-01")
    parser.add_argument("--balance", type=float, default=2500.0)
    parser.add_argument("--targets-usd", default="100,125", help="realised profit needed (125 = $100 after an 80% split)")
    parser.add_argument("--risks", default="0.25,0.3,0.4,0.5,0.75,1,1.25,1.5", help="% of the starting balance per trade")
    parser.add_argument("--max-dd-pct", type=float, default=10.0)
    parser.add_argument("--max-trade-loss-pct", type=float, default=1.5)
    parser.add_argument("--min-active-days", type=int, default=5)
    parser.add_argument("--active-day-pct", type=float, default=0.5)
    parser.add_argument("--min-days", type=int, default=14)
    parser.add_argument("--skip-news", default="", help="e.g. FOMC or CPI,NFP,FOMC (uses configs/news_us_high_impact.csv)")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    bars = merge_bid_ask(
        load_parquet(parquet_path(cfg["symbol"], cfg["timeframe"], cache, side="bid")),
        load_parquet(parquet_path(cfg["symbol"], cfg["timeframe"], cache, side="ask")),
    )
    sig = {**cfg["signal"], "trade_from": args.start}
    if args.skip_news:
        sig |= {"news_calendar": "configs/news_us_high_impact.csv", "skip_news_events": args.skip_news.split(",")}
    trades, _ = run(bars, {**cfg, "signal": sig})
    r = np.array([t.r_multiple for t in trades])
    mae = np.array([t.mae_r for t in trades])
    mfe = np.array([t.mfe_r for t in trades])
    epoch = pd.Timestamp("2000-01-01")
    start_day = np.array([(pd.Timestamp(t.day) - epoch).days for t in trades])
    exit_day = np.array([(pd.Timestamp(t.exit_time.tz_convert("America/New_York").date()) - epoch).days for t in trades])

    rows = []
    for target_usd in (float(x) for x in args.targets_usd.split(",")):
        rules = {
            "max_dd_pct": args.max_dd_pct, "max_trade_loss_pct": args.max_trade_loss_pct,
            "target_pct": target_usd / args.balance * 100, "min_active_days": args.min_active_days,
            "active_day_pct": args.active_day_pct, "min_days": args.min_days,
        }
        for risk in (float(x) for x in args.risks.split(",")):
            outs = [first_payout(r[i:], mae[i:], mfe[i:], start_day[i:], exit_day[i:], risk, rules) for i in range(len(r))]
            done = [(i, res, i + k) for i, (res, k) in enumerate(outs) if res != OPEN]
            paid = [(i, j) for i, res, j in done if res == PAYOUT]
            days = np.array([exit_day[j] - start_day[i] for i, j in paid])
            n = len(done)
            rows.append({
                "target_usd": target_usd, "risk_pct": risk, "risk_usd": risk / 100 * args.balance, "accounts": n,
                "payout": len(paid) / n, "trailing_dd": sum(res == TRAIL for _, res, _ in done) / n,
                "trade_loss_cap": sum(res == TRADE_CAP for _, res, _ in done) / n,
                "median_days": float(np.median(days)) if len(days) else None,
                "p75_days": float(np.percentile(days, 75)) if len(days) else None,
            })

    tag = f"_skip_{args.skip_news.replace(',', '_')}" if args.skip_news else ""
    out = PROJECT_ROOT / "reports" / f"instant_sim_{cfg['strategy']}{tag}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"${args.balance:,.0f} account, {len(trades)} trades {args.start} onward" + (f", skipping {args.skip_news} days" if args.skip_news else ""))
    print(f"{'target':>7} {'risk':>6} {'$/trade':>8} | {'payout':>7} {'trail DD':>8} {'1.5% cap':>8} | {'days med':>8} {'p75':>5}")
    for x in rows:
        print(f"${x['target_usd']:>6.0f} {x['risk_pct']:5.2f}% {x['risk_usd']:8.2f} | {x['payout']:7.0%} {x['trailing_dd']:8.0%} {x['trade_loss_cap']:8.0%} | "
              f"{x['median_days'] or 0:8.0f} {x['p75_days'] or 0:5.0f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
