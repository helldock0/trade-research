"""Backtest the NY kill-zone breakout on NAS100 M5 with real bid/ask spread.

Standalone (not in the registry): the shared engine fills FX-style mid prices
with a fixed spread, while this strategy needs ask/bid bars and gap fills.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ROOT as PROJECT_ROOT, ensure_dirs, load_strategy_config
from src.data.dukascopy_ingest import load_parquet, parquet_path
from src.strategy.nykz_breakout import merge_bid_ask, run, trades_frame, without_spread


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="NY kill-zone breakout backtest (NAS100)")
    parser.add_argument("--config", default="strategy_nykz_breakout.yaml")
    parser.add_argument("--start", default=None, help="override data.start (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="override data.end (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    sid = cfg["strategy"]
    sym, tf = cfg["symbol"], cfg["timeframe"]
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    bars = merge_bid_ask(
        load_parquet(parquet_path(sym, tf, cache, side="bid")),
        load_parquet(parquet_path(sym, tf, cache, side="ask")),
    )

    start = args.start or cfg["data"]["start"]
    end = args.end or cfg["data"].get("end")
    # A year of earlier bars warms up the volatility filters; trading starts at `start`.
    bars = bars.loc[pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=365) :]
    if end:
        bars = bars.loc[: pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    cfg = {**cfg, "signal": {**cfg["signal"], "trade_from": start}}

    trades, summary = run(bars, cfg)
    _, baseline = run(without_spread(bars), cfg)

    trades_rel = f"reports/backtest_trades_{sid}.csv"
    out = {
        "strategy": sid,
        "config": args.config,
        "symbol": sym,
        "data_from": start,
        "data_to": str(bars.index[-1]),
        "metrics": summary,
        "zero_spread": {
            k: baseline[k]
            for k in ("n_trades", "win_rate", "expectancy_r", "total_r", "profit_factor", "max_dd_r")
            if k in baseline
        },
        "trades_path": trades_rel,
    }
    report = PROJECT_ROOT / "reports" / f"backtest_summary_{sid}.json"
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if trades:
        trades_frame(trades).to_csv(PROJECT_ROOT / trades_rel, index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
