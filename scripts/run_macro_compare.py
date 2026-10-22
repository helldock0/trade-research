"""Compare macro_swing variants (fixed target vs trailing stop vs correlation-gated basket).

    python scripts/run_macro_compare.py                      # every variant in the config
    python scripts/run_macro_compare.py --variants trail,basket
    python scripts/run_macro_compare.py --no-ftmo

Variants live under `variants:` in configs/strategy_macro_swing.yaml.
Writes reports/macro_compare.json and reports/macro_compare.md (or macro_compare_<tag>.*).
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ROOT as PROJECT_ROOT,
    ensure_dirs,
    load_costs_config,
    load_ftmo_config,
    load_strategy_config,
)
from src.data.dukascopy_ingest import load_symbol_bars
from src.data.macro import load_macro_panel
from src.features.news import load_news_calendar
from src.research.macro_compare import compare_variants, render_markdown, variants_universe, write_compare_report
from src.strategy.macro_swing import generate_signals


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare macro_swing trade-management variants")
    parser.add_argument("--config", default="strategy_macro_swing.yaml")
    parser.add_argument("--variants", default=None, help="comma list (default: all in the config)")
    parser.add_argument("--no-ftmo", action="store_true")
    parser.add_argument("--tag", default=None, help="report name suffix")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    costs = load_costs_config()
    ftmo = None if args.no_ftmo else load_ftmo_config()
    names = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None

    panel = load_macro_panel(cfg["macro"])
    if not panel:
        print("No cached macro series; run scripts/download_macro.py first.", file=sys.stderr)
        sys.exit(1)
    news_cfg = cfg.get("news", {}) or {}
    news = load_news_calendar(news_cfg.get("calendar_csv")) if news_cfg.get("enabled", True) else None
    # load the panel and calendar once; every variant and fold reuses them
    gen = partial(generate_signals, macro_panel=panel, news_df=news)

    bars = {s: load_symbol_bars(s, cfg.get("timeframe", "M15")) for s in variants_universe(cfg, names)}
    result = compare_variants(cfg, bars, costs, ftmo, names=names, signal_fn=gen)
    paths = write_compare_report(result, PROJECT_ROOT / "reports", tag=args.tag)
    print(render_markdown(result))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
