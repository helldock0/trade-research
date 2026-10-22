"""Buy-the-dip signal check on FRED daily index closes (pre-registered rule in configs/strategy_dip.yaml).

    python scripts/run_dip_check.py
    python scripts/run_dip_check.py --no-download

Writes reports/dip_check.json and reports/dip_check.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, load_costs_config, load_strategy_config
from src.data.macro import fetch_series, load_series, save_series, series_path
from src.research.dip_check import render_markdown, run_dip_check, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Buy-the-dip signal check (FRED daily closes)")
    parser.add_argument("--config", default="strategy_dip.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    for name in cfg["validation"]["series"]:
        if not args.no_download and not series_path(name).exists():
            save_series(fetch_series({"source": "fred", "id": name}, name), name)
    series = {n: load_series(n) for n in cfg["validation"]["series"]}
    series = {n: s for n, s in series.items() if not s.empty}
    if not series:
        print("No cached series; run without --no-download.", file=sys.stderr)
        sys.exit(1)
    res = run_dip_check(series, cfg, load_costs_config())
    paths = write_report(res, cfg, PROJECT_ROOT / "reports")
    print(render_markdown(res, cfg))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
