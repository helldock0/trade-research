"""Relative-value signal check on FRED daily rates (pre-registered rule in configs/strategy_rv.yaml).

    python scripts/run_rv_check.py
    python scripts/run_rv_check.py --no-download

Crosses are built from their FRED USD legs (cached as data/macro/fx_*.csv).
Writes reports/rv_check.json and reports/rv_check.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, load_strategy_config
from src.research.macro_ic import fetch_fx, load_fx
from src.research.rv_check import render_markdown, run_rv_check, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Relative-value signal check (FRED daily)")
    parser.add_argument("--config", default="strategy_rv.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    pairs = list(cfg["validation"]["pairs"])
    if not args.no_download:
        errors = {k: v for k, v in fetch_fx(pairs).items() if "error" in v}
        if errors:
            print(json.dumps(errors, indent=2), file=sys.stderr)
    res = run_rv_check(load_fx(pairs), cfg)
    paths = write_report(res, cfg, PROJECT_ROOT / "reports")
    print(render_markdown(res, cfg))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
