"""Sweep the rolling 15-minute box breakout over the full parameter grid.

Example:
    python scripts/run_rolling_box.py --workers 4
    python scripts/run_rolling_box.py --symbols EURUSD,XAUUSD,US30
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

from src.config import ROOT as PROJECT_ROOT
from src.config import ensure_dirs, load_costs_config, load_strategy_config
from src.research.box_sweep import add_metrics, config_table, portfolio_summary, run_sweep

SHOW_COLS = [
    "confirm_tf",
    "window",
    "stop",
    "exit",
    "concurrency",
    "min_risk_cost",
    "n",
    "exp_r_is",
    "exp_r_oos",
    "exp_r_gross",
    "win_rate",
    "profit_factor",
]


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip().upper() for x in value.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rolling 15-minute box breakout sweep")
    parser.add_argument("--config", default="strategy_rolling_box.yaml")
    parser.add_argument("--symbols", default=None, help="Comma-separated subset of the universe")
    parser.add_argument("--workers", type=int, default=4, help="Parallel symbol processes")
    parser.add_argument("--out", default="reports/rolling_box")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--min-trades-is",
        type=int,
        default=300,
        help="Minimum in-sample trades for a config to be ranked",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    costs = load_costs_config()
    symbols = _parse_csv(args.symbols) or list(cfg["universe"])
    cache_dir = PROJECT_ROOT / cfg["data"]["cache_dir"]
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = config_table(cfg)
    print(f"[sweep] {len(symbols)} symbols x {len(configs)} configs", flush=True)
    summary, by_hour, by_year, missing = run_sweep(symbols, cfg, costs, cache_dir, args.workers)

    summary = add_metrics(summary).merge(configs, on="cfg_id")
    portfolio = portfolio_summary(summary).merge(configs, on="cfg_id")
    configs.to_csv(out_dir / "configs.csv", index=False)
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    portfolio.to_csv(out_dir / "portfolio.csv", index=False)
    by_hour.to_parquet(out_dir / "by_hour.parquet", index=False)
    by_year.to_parquet(out_dir / "by_year.parquet", index=False)

    ranked = portfolio.loc[portfolio["n_is"] >= args.min_trades_is].sort_values("exp_r_is", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nTop configs, all symbols pooled (ranked in-sample, OOS alongside):")
        print(ranked[SHOW_COLS].head(args.top).round(3).to_string(index=False))

        per_symbol = summary.loc[summary["n_is"] >= args.min_trades_is]
        best = per_symbol.sort_values("exp_r_is", ascending=False).groupby("symbol").head(1)
        print("\nBest in-sample config per symbol:")
        print(best[["symbol"] + SHOW_COLS].round(3).to_string(index=False))

    meta = {
        "config": args.config,
        "symbols": sorted(set(summary["symbol"])),
        "missing": missing,
        "n_configs": len(configs),
        "oos_start": cfg["validation"]["oos_start"],
        "outputs": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
