"""Backtest the NY 08:12-09:12 range fade (base config + sensitivity variants) on M1 bars."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ROOT as PROJECT_ROOT, ensure_dirs, load_costs_config, load_strategy_config
from src.data.dukascopy_ingest import load_parquet, parquet_path
from src.strategy.ny_range_fade import run_fade, summarize

TABLE_COLS = [
    "n_trades",
    "win_rate",
    "tp_rate",
    "expectancy_ru",
    "gross_expectancy_ru",
    "t_stat",
    "profit_factor",
    "avg_win_ru",
    "avg_loss_ru",
    "worst_ru",
    "max_dd_ru",
    "expectancy_pips",
]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    keep = pd.Series(True, index=df.index)
    if start:
        keep &= ts >= pd.Timestamp(start, tz="UTC")
    if end:
        keep &= ts < pd.Timestamp(end, tz="UTC")
    return df.loc[keep]


def _table(rows: dict[str, dict]) -> str:
    frame = pd.DataFrame(rows).T
    cols = [c for c in TABLE_COLS if c in frame.columns]
    return frame[cols].to_string(float_format=lambda v: f"{v:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="NY 08:12-09:12 range fade backtest (M1)")
    parser.add_argument("--config", default="strategy_ny_range_fade.yaml")
    parser.add_argument("--symbols", default=None, help="comma-separated override of universe")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", default="reports/ny_range_fade")
    args = parser.parse_args(argv)

    ensure_dirs()
    base = load_strategy_config(args.config)
    costs = load_costs_config()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else base["universe"]
    start = args.start or base["data"]["start"]
    end = args.end or base["data"].get("end")
    cache = PROJECT_ROOT / base["data"]["cache_dir"]

    bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = parquet_path(sym, base.get("timeframe", "M1"), cache)
        if not path.exists():
            print(f"[skip] {sym}: missing {path}")
            continue
        bars[sym] = _filter_dates(load_parquet(path), start, end)
    if not bars:
        raise SystemExit("No M1 data found; run scripts/download_data.py --intervals M1 first")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {"base": {}, **(base.get("variants") or {})}
    summary: dict = {"symbols": list(bars), "start": start, "end": end, "variants": {}}
    pooled_rows: dict[str, dict] = {}

    for name, override in variants.items():
        cfg = deep_merge({k: v for k, v in base.items() if k != "variants"}, override)
        tf, counts = run_fade(bars, cfg, costs)
        tf.to_csv(out_dir / f"trades_{name}.csv", index=False)
        pooled = summarize(tf)
        pooled_rows[name] = pooled
        per_symbol = {s: summarize(g) for s, g in tf.groupby("symbol")} if not tf.empty else {}
        per_year = (
            {str(y): summarize(g) for y, g in tf.groupby(pd.to_datetime(tf["day"]).dt.year)}
            if not tf.empty
            else {}
        )
        summary["variants"][name] = {
            "override": override,
            "pooled": pooled,
            "per_symbol": per_symbol,
            "per_year": per_year,
            "counts": counts,
        }
        if name == "base":
            print("\n== base: per symbol ==")
            print(_table(per_symbol))
            print("\n== base: per year (all symbols) ==")
            print(_table(per_year))

    print("\n== pooled, all symbols: base vs variants ==")
    print(_table(pooled_rows))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
