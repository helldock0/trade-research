"""Compare NY kill-zone breakout variants: clean-break rules x stop rules.

Each variant runs on the config's test period and on an earlier check period
(2018-2022 by default), with the real bid/ask spread.
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
from src.strategy.nykz_breakout import merge_bid_ask, run

BREAK_RULES = [
    ("any close beyond", {}),
    ("close >= 30 pts beyond", {"min_break_points": 30.0}),
    ("close >= 0.1% beyond", {"min_break_pct": 0.1}),
    ("next candle confirms", {"confirm_close": True}),
    ("body >= 1.5x range avg", {"min_body_vs_range": 1.5}),
]
V1_RULES = {
    "min_break_points": 0.0,
    "min_break_pct": 0.0,
    "min_body_vs_range": 0.0,
    "confirm_close": False,
    "stop_includes_prior_bar": False,
}
STOP_RULES = [
    ("break candle", {}),
    ("break + prior candle", {"stop_includes_prior_bar": True}),
]


def _slice(bars: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    out = bars.loc[pd.Timestamp(start, tz="UTC") :]
    if end:
        out = out.loc[: pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare NY kill-zone breakout variants")
    parser.add_argument("--config", default="strategy_nykz_breakout.yaml")
    parser.add_argument("--check-start", default="2018-01-01")
    parser.add_argument("--check-end", default="2022-12-31")
    parser.add_argument("--day", default=None, help="also show each variant's trade on this NY date")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    sym, tf = cfg["symbol"], cfg["timeframe"]
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    bars = merge_bid_ask(
        load_parquet(parquet_path(sym, tf, cache, side="bid")),
        load_parquet(parquet_path(sym, tf, cache, side="ask")),
    )
    periods = {
        "test": _slice(bars, cfg["data"]["start"], cfg["data"].get("end")),
        "check": _slice(bars, args.check_start, args.check_end),
    }

    rows = []
    for break_name, break_cfg in BREAK_RULES:
        for stop_name, stop_cfg in STOP_RULES:
            variant = {**cfg, "signal": {**cfg["signal"], **V1_RULES, **break_cfg, **stop_cfg}}
            row: dict = {"break_rule": break_name, "stop_rule": stop_name}
            for period, period_bars in periods.items():
                trades, s = run(period_bars, variant)
                row[period] = {
                    k: s.get(k, 0.0)
                    for k in ("n_trades", "win_rate", "expectancy_r", "total_r", "max_dd_r", "by_year")
                }
                if args.day and period == "test":
                    row["day"] = next(
                        ({"side": t.side, "entry": t.entry, "stop": t.stop, "r": t.r_multiple}
                         for t in trades if t.day == args.day),
                        None,
                    )
            rows.append(row)

    out = PROJECT_ROOT / "reports" / f"compare_{cfg['strategy']}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{'break rule':24} {'stop':22} {'trades':>6} {'win%':>6} {'total R':>8} {'R/trade':>8} {'maxDD R':>8} | "
          f"{'2018-22 R':>9} {'R/trade':>8}" + (f" | {args.day}" if args.day else ""))
    for r in rows:
        t, c = r["test"], r["check"]
        line = (f"{r['break_rule']:24} {r['stop_rule']:22} {t['n_trades']:6d} {t['win_rate']:6.1%} "
                f"{t['total_r']:+8.1f} {t['expectancy_r']:+8.3f} {t['max_dd_r']:8.1f} | "
                f"{c['total_r']:+9.1f} {c['expectancy_r']:+8.3f}")
        if args.day:
            d = r.get("day")
            line += " | " + (f"{'long' if d['side'] > 0 else 'short'} {d['entry']:.1f} stop {d['stop']:.1f} {d['r']:+.1f}R"
                             if d else "no trade")
        print(line)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
