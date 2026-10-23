"""Run a NY kill-zone breakout config (v3 by default) on other markets.

Each symbol needs M5 bid and ask parquet files. The break-candle buffer
(used only to skip entries that close beyond the break candle) is scaled to
the same share of price as the config's buffer on its own symbol. FX pairs
also pay a round-turn commission (--fx-commission-pct of notional, ~$7 per
standard lot), since raw ECN spreads come with one.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from src.strategy.nykz_breakout import merge_bid_ask, run, summarize, without_spread

MARKETS = "NAS100,US30,US500,GER40,UK100,JP225,EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="NY kill-zone breakout on other markets")
    parser.add_argument("--config", default="strategy_nykz_breakout.yaml")
    parser.add_argument("--symbols", default=MARKETS)
    parser.add_argument("--check-start", default="2018-01-01")
    parser.add_argument("--check-end", default="2022-12-31")
    parser.add_argument("--fx-commission-pct", type=float, default=0.007, help="round turn, % of notional")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    start = cfg["data"]["start"]
    load = lambda sym: merge_bid_ask(
        load_parquet(parquet_path(sym, cfg["timeframe"], cache, side="bid")),
        load_parquet(parquet_path(sym, cfg["timeframe"], cache, side="ask")),
    )
    base_bars = load(cfg["symbol"])
    buffer_frac = float(cfg["signal"]["stop_buffer_points"]) / float(base_bars["close"].loc[start:].median())

    rows = []
    for sym in args.symbols.split(","):
        try:
            bars = base_bars if sym == cfg["symbol"] else load(sym)
        except FileNotFoundError as exc:
            print(f"[skip] {sym}: {exc}")
            continue
        buffer = float(cfg["signal"]["stop_buffer_points"]) if sym == cfg["symbol"] else buffer_frac * float(bars["close"].loc[start:].median())
        sig = {**cfg["signal"], "stop_buffer_points": buffer}
        periods = {
            "test": (bars.loc[pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=365) :], start),
            "check": (bars.loc[pd.Timestamp(args.check_start, tz="UTC") : pd.Timestamp(args.check_end, tz="UTC") + pd.Timedelta(days=1)], args.check_start),
        }
        row = {"symbol": sym}
        for name, (b, trade_from) in periods.items():
            c = {**cfg, "symbol": sym, "signal": {**sig, "trade_from": trade_from}}
            trades, s = run(b, c)
            _, zero = run(without_spread(b), c)
            if sym.isalpha() and len(sym) == 6:  # FX pair: pay the commission in R
                frac = args.fx_commission_pct / 100
                trades = [dataclasses.replace(t, r_multiple=t.r_multiple - frac * t.entry / t.risk_points) for t in trades]
                s = {**s, **summarize(trades)}
            row[name] = {k: s.get(k, 0.0) for k in ("n_trades", "win_rate", "expectancy_r", "total_r", "max_dd_r", "by_year")}
            row[name]["zero_spread_r_per_trade"] = zero.get("expectancy_r", 0.0)
            row[name]["spread_share_of_stop"] = float(np.mean([t.spread_points / t.risk_points for t in trades])) if trades else 0.0
        rows.append(row)
        t, c = row["test"], row["check"]
        losing = sum(v["total_r"] < 0 for v in {**c["by_year"], **t["by_year"]}.values())
        print(f"{sym:7} test n {t['n_trades']:4d} WR {t['win_rate']:5.1%} {t['total_r']:+7.1f}R {t['expectancy_r']:+.3f}R/t DD {t['max_dd_r']:5.1f} | "
              f"check {c['total_r']:+7.1f}R {c['expectancy_r']:+.3f}R/t DD {c['max_dd_r']:5.1f} | zero-spread {t['zero_spread_r_per_trade']:+.3f}/{c['zero_spread_r_per_trade']:+.3f} | "
              f"spread {t['spread_share_of_stop']:.1%} of stop | losing years {losing}", flush=True)

    out = PROJECT_ROOT / "reports" / f"markets_{cfg['strategy']}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
