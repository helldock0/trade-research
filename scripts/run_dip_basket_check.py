"""Per-index inclusion check for the buy-the-dip basket (rule in configs/strategy_dip_basket.yaml).

    python scripts/run_dip_basket_check.py
    python scripts/run_dip_basket_check.py --no-download

Daily histories: FRED ids, or YH_* names fetched from Yahoo (cached in data/macro/).
Writes reports/dip_basket_check.json and reports/dip_basket_check.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT as PROJECT_ROOT, load_costs_config, load_strategy_config
from src.data.macro import fetch_series, load_series, save_series, series_path
from src.research.dip_check import render_basket_markdown, run_basket_check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Buy-the-dip basket: per-index inclusion check")
    parser.add_argument("--config", default="strategy_dip_basket.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    val = cfg["validation"]
    yahoo = val.get("yahoo") or {}
    for name in dict.fromkeys(val["history"].values()):
        if args.no_download or series_path(name).exists():
            continue
        spec = {"source": "yahoo", "symbol": yahoo[name]} if name in yahoo else {"source": "fred", "id": name}
        save_series(fetch_series(spec, name), name)
    history = {name: load_series(name) for name in dict.fromkeys(val["history"].values())}
    res = run_basket_check(history, cfg, load_costs_config())
    out = PROJECT_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dip_basket_check.json").write_text(json.dumps(res, indent=2, default=lambda o: None), encoding="utf-8")
    md = render_basket_markdown(res, cfg)
    (out / "dip_basket_check.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Report: {out / 'dip_basket_check.md'}")


if __name__ == "__main__":
    main()
