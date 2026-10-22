"""Turn-of-month bar backtest: `tom` vs a mid-month `placebo` (+ US500 cross-check).

    python scripts/run_tom_check.py          # signal check first (caches the FRED calendar)
    python scripts/run_tom_backtest.py
    python scripts/run_tom_backtest.py --variants tom,placebo --no-ftmo

Bars: data/parquet/{NAS100,US500}_M15.parquet (cloud: scripts/bars_from_data_branch.py --pairs NAS100,US500).
Writes reports/tom_compare.json and reports/tom_compare.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, ensure_dirs, load_costs_config, load_ftmo_config, load_strategy_config
from src.data.dukascopy_ingest import load_symbol_bars
from src.research.macro_compare import compare_variants, render_markdown, variants_universe, write_compare_report

TITLE = "Turn of the month: variant comparison"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Turn-of-month backtest vs placebo")
    parser.add_argument("--config", default="strategy_tom.yaml")
    parser.add_argument("--variants", default=None, help="comma list (default: all in the config)")
    parser.add_argument("--no-ftmo", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    names = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    bars = {s: load_symbol_bars(s, cfg.get("timeframe", "M15")) for s in variants_universe(cfg, names)}
    result = compare_variants(cfg, bars, load_costs_config(), None if args.no_ftmo else load_ftmo_config(), names=names)
    paths = write_compare_report(result, PROJECT_ROOT / "reports", stem="tom_compare", title=TITLE)
    print(render_markdown(result, title=TITLE))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
