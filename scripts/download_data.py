from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/...` without editable install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# dukascopy-python is extremely chatty at INFO (per-candle timestamps)
logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)

from src.config import ensure_dirs, load_strategy_config
from src.data.bi5_download import download_via_bi5, generate_synthetic_ohlc
from src.data.dukascopy_ingest import (
    ALL_TIMEFRAMES,
    FULL_FX_UNIVERSE,
    FULL_UNIVERSE,
    download_universe,
    parquet_path,
    save_parquet,
)
from src.features.news import write_empty_news_template
from src.config import ROOT as PROJECT_ROOT


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [x.strip().upper() for x in value.replace(";", ",").split(",") if x.strip()]
    return items or None


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
    parser.add_argument(
        "--full-fx",
        action="store_true",
        help="Download majors + minor crosses + XAUUSD (ignores strategy universe)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download --full-fx plus index CFDs (US30, NAS100, SPX500, GER40, UK100, JPN225)",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (overrides --full-fx / strategy universe)",
    )
    parser.add_argument(
        "--intervals",
        default=None,
        help=(
            "Comma-separated intervals. Default: M15 (strategy). "
            f"Use 'all' for {','.join(ALL_TIMEFRAMES)}"
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Seconds to sleep between Dukascopy year chunks (rate limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel symbol downloads (network-bound; 4-8 is reasonable)",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    write_empty_news_template()
    cfg = load_strategy_config()

    symbols = _parse_csv(args.symbols)
    if symbols is None:
        if args.full:
            symbols = list(FULL_UNIVERSE)
        elif args.full_fx:
            symbols = list(FULL_FX_UNIVERSE)
        else:
            symbols = list(cfg["universe"])

    if args.intervals is None:
        intervals = ["M15"]
    elif args.intervals.strip().lower() == "all":
        intervals = list(ALL_TIMEFRAMES)
    else:
        intervals = _parse_csv(args.intervals) or ["M15"]

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
        download_universe(
            symbols,
            start.date().isoformat(),
            end.date().isoformat(),
            intervals=intervals,
            force=args.force,
            sleep_s=args.sleep,
            continue_on_error=True,
            workers=args.workers,
        )
        return
    except Exception as exc:
        print(f"[warn] dukascopy-python OHLC failed: {exc}")

    # bi5 fallback only covers the original major subset + M15/H1
    try:
        download_via_bi5(symbols, start, end, force=args.force)
    except Exception as exc:
        print(f"[error] bi5 download failed: {exc}")
        print("Tip: retry, or use --synthetic --years 8 for pipeline testing.")
        raise


if __name__ == "__main__":
    main()
