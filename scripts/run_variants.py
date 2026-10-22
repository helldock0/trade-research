"""Run a strategy config's `variants:` through backtest, walk-forward, gates and the FTMO path.

    python scripts/run_variants.py --config strategy_imom.yaml --stem imom_compare
    python scripts/run_variants.py --config strategy_tom.yaml --stem tom_compare --variants tom,placebo

Uses each strategy's registered signal function. Writes reports/<stem>.json and reports/<stem>.md.
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare a config's variants")
    parser.add_argument("--config", required=True, help="strategy yaml under configs/")
    parser.add_argument("--stem", default=None, help="report name (default: <strategy>_compare)")
    parser.add_argument("--variants", default=None, help="comma list (default: all in the config)")
    parser.add_argument("--no-ftmo", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    names = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    stem = args.stem or f"{cfg['strategy']}_compare"
    title = f"{cfg['strategy']}: variant comparison"
    bars = {s: load_symbol_bars(s, cfg.get("timeframe", "M15")) for s in variants_universe(cfg, names)}
    result = compare_variants(cfg, bars, load_costs_config(), None if args.no_ftmo else load_ftmo_config(), names=names)
    paths = write_compare_report(result, PROJECT_ROOT / "reports", stem=stem, title=title)
    print(render_markdown(result, title=title))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
