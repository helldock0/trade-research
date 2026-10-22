"""Independent re-check of gj-strat's USDJPY gotobi and vol-box edges (configs/jpy_recheck.yaml).

    python -I scripts/run_jpy_recheck.py
    python -I scripts/run_jpy_recheck.py --datastore /path/to/gj-strat/datastore --log /path/to/trades.csv

Reads the raw Dukascopy minute files from gj-strat's datastore with this repo's own decoder
(no gj-strat code is imported or run). Writes reports/jpy_recheck.{json,md} and the trade lists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ROOT as PROJECT_ROOT
from src.config import load_strategy_config
from src.research.jpy_recheck import (
    load_datastore,
    load_rates,
    render_markdown,
    run_recheck,
    unseen_market,
    write_report,
)

LOG = "results/patterns_usdjpy/trades_P1_fix-short_to_0600UTC_gotobi.csv"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="USDJPY gotobi / vol-box re-check")
    parser.add_argument("--config", default="jpy_recheck.yaml")
    parser.add_argument("--datastore", default=None, help="folder of {SYMBOL}*.tar day files (default: config)")
    parser.add_argument("--log", default=None, help="gj-strat's gotobi trade log (default: next to the datastore)")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    store = Path(args.datastore or cfg["datastore"])
    log_path = Path(args.log) if args.log else store.parent / LOG
    m = load_datastore(store, cfg["symbol"], cfg["start"], cfg["end"])
    log = pd.read_csv(log_path) if log_path.exists() else None
    res = run_recheck(m, cfg, load_rates(PROJECT_ROOT / "data" / "macro"), log)
    del m
    if res.get("robustness") is not None:
        res["robustness"]["unseen"] = {
            mk: unseen_market(load_datastore(store, mk, cfg["start"], cfg["end"]), cfg, float(cfg["pip"]))
            for mk in cfg["robustness"].get("unseen_markets") or []
        }
    paths = write_report(res, PROJECT_ROOT / "reports")
    print(render_markdown(res))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
