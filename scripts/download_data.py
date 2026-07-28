from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/...` without editable install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_strategy_config
from src.data.bi5_download import download_via_bi5, generate_synthetic_ohlc
from src.data.dukascopy_ingest import parquet_path, save_parquet
from src.features.news import write_empty_news_template
from src.config import ROOT as PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Dukascopy FX data for CSSM")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic OHLC instead of downloading (offline CI)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="If set with --synthetic, span this many years ending now",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    write_empty_news_template()
    cfg = load_strategy_config()
    symbols = cfg["universe"]
    start_s = args.start or cfg["data"]["start"]
    end_s = args.end or cfg["data"].get("end")
    start = datetime.fromisoformat(start_s).replace(tzinfo=timezone.utc)
    end = (
        datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc)
        if end_s
        else datetime.now(timezone.utc)
    )

    if args.synthetic:
        if args.years:
            end = datetime.now(timezone.utc)
            start = end.replace(year=end.year - args.years)
        cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
        for sym in symbols:
            for interval, rule in (("M15", "15min"), ("H1", "1h")):
                path = parquet_path(sym, interval, cache)
                if path.exists() and not args.force:
                    print(f"[skip] {sym}_{interval}")
                    continue
                df = generate_synthetic_ohlc(sym, start, end, rule=rule)
                save_parquet(df, path)
                print(f"[synthetic] {sym}_{interval}: {len(df)} bars")
        return

    # 1) Try dukascopy-python OHLC (fast). 2) bi5 ticks. 3) tell user about --synthetic.
    try:
        from src.data.dukascopy_ingest import download_universe

        download_universe(symbols, start.date().isoformat(), end.date().isoformat(), force=args.force)
        return
    except Exception as exc:
        print(f"[warn] dukascopy-python OHLC failed: {exc}")

    try:
        download_via_bi5(symbols, start, end, force=args.force)
    except Exception as exc:
        print(f"[error] bi5 download failed: {exc}")
        print("Tip: retry, or use --synthetic --years 8 for pipeline testing.")
        raise


if __name__ == "__main__":
    main()
