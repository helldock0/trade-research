"""Intraday-momentum signal check on M15 bars (pre-registered rule in configs/strategy_imom.yaml).

    python scripts/bars_from_data_branch.py --pairs US500,NAS100,US30   # cloud sessions
    python scripts/run_imom_check.py

Writes reports/imom_check.json and reports/imom_check.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, load_costs_config, load_strategy_config
from src.data.dukascopy_ingest import load_symbol_bars
from src.data.macro import fetch_series, save_series, series_path
from src.research.imom_check import render_markdown, run_imom_check, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Intraday momentum signal check (bars, net of costs)")
    parser.add_argument("--config", default="strategy_imom.yaml")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    cal = cfg["signal"].get("calendar_series")
    if cal and not series_path(cal).exists():
        save_series(fetch_series({"source": "fred", "id": cal}, cal), cal)
    symbols = [cfg["universe"][0]] + list(cfg["validation"].get("cross_checks", []))
    bars = {}
    for s in symbols:
        try:
            bars[s.upper()] = load_symbol_bars(s, cfg.get("timeframe", "M15"))
        except FileNotFoundError:
            print(f"no bars for {s}; skipped", file=sys.stderr)
    res = run_imom_check(bars, cfg, load_costs_config())
    paths = write_report(res, PROJECT_ROOT / "reports")
    print(render_markdown(res))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
