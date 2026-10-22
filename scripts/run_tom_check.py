"""Turn-of-month signal check on FRED daily index closes (pre-registered rule in the config).

    python scripts/run_tom_check.py                  # fetch missing series, run, write reports
    python scripts/run_tom_check.py --no-download

Writes reports/tom_check.json and reports/tom_check.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, load_strategy_config
from src.data.macro import DEFAULT_MACRO_DIR, fetch_series, load_series, save_series, series_path
from src.research.tom_check import render_markdown, run_tom_check, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Turn-of-month signal check (daily closes, no bars)")
    parser.add_argument("--config", default="strategy_tom.yaml")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download cached series")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    names = list(cfg["validation"]["series"])
    for name in names:
        if args.no_download or (series_path(name).exists() and not args.force):
            continue
        save_series(fetch_series({"source": "fred", "id": name}, name), name)
    series = {n: load_series(n) for n in names}
    series = {n: s for n, s in series.items() if not s.empty}
    if not series:
        print(f"No cached series in {DEFAULT_MACRO_DIR}; run without --no-download.", file=sys.stderr)
        sys.exit(1)
    res = run_tom_check(series, cfg)
    paths = write_report(res, PROJECT_ROOT / "reports")
    print(render_markdown(res))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
