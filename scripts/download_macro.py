"""Fetch the macro series named in a strategy config into data/macro/.

    python scripts/download_macro.py                 # fetch missing series
    python scripts/download_macro.py --force         # refresh everything
    python scripts/download_macro.py --show          # print latest currency scores
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_strategy_config
from src.data.macro import download_macro, load_macro_panel
from src.features.macro_score import build_currency_scores, latest_table, pair_bias


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download macro series for the macro swing strategy")
    parser.add_argument("--config", default="strategy_macro_swing.yaml")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--force", action="store_true", help="re-download cached series")
    parser.add_argument("--show", action="store_true", help="print the latest scores and pair bias")
    parser.add_argument("--no-download", action="store_true", help="only show cached data")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    macro_cfg = cfg["macro"]
    if not args.no_download:
        status = download_macro(macro_cfg, force=args.force, start=args.start)
        print(json.dumps(status, indent=2))
    if args.show or args.no_download:
        panel = load_macro_panel(macro_cfg)
        res = build_currency_scores(panel, macro_cfg)
        tab = latest_table(res)
        if tab.empty:
            print("No scores (no cached series).")
            return
        with_pd = tab.round(2)
        print(with_pd.to_string())
        bias = pair_bias(res["scores"], cfg["universe"]).dropna(how="all")
        if not bias.empty:
            print("\nLatest pair bias:")
            print(bias.iloc[-1].round(2).to_string())


if __name__ == "__main__":
    main()
