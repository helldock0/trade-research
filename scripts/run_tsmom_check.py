"""Time-series momentum signal check on daily history (pre-registered rule in configs/strategy_tsmom.yaml).

    python scripts/run_tsmom_check.py
    python scripts/run_tsmom_check.py --no-download

Daily histories: FRED FX (fx_*), FRED indices, or YH_* names from Yahoo (cached in data/macro/).
Writes reports/tsmom_check.json and reports/tsmom_check.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, load_strategy_config
from src.data.macro import fetch_series, load_series, save_series, series_path
from src.research.tsmom_check import render_markdown, run_tsmom_check, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Time-series momentum signal check (daily history)")
    parser.add_argument("--config", default="strategy_tsmom.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    val = cfg["validation"]
    for name in dict.fromkeys(val["history"].values()):
        if args.no_download or series_path(name).exists():
            continue
        if name in (val.get("yahoo") or {}):
            spec = {"source": "yahoo", "symbol": val["yahoo"][name]}
        else:
            spec = {"source": "fred", "id": (val.get("fred_fx") or {}).get(name, name)}
        save_series(fetch_series(spec, name), name)
    history = {name: load_series(name) for name in dict.fromkeys(val["history"].values())}
    res = run_tsmom_check(history, cfg)
    paths = write_report(res, cfg, PROJECT_ROOT / "reports")
    print(render_markdown(res, cfg))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
