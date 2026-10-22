"""Fetch the economic calendar (FXStreet) and optional FRED series into data/.

Examples:
    python scripts/download_calendar.py                      # 2018-01-01 .. next week, merge into cache
    python scripts/download_calendar.py --start 2024-01-01   # refresh a slice
    python scripts/download_calendar.py --gasoline           # also fetch FRED GASREGW
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.calendar import (
    DEFAULT_CALENDAR_PATH,
    fetch_fxstreet_calendar,
    load_event_calendar,
    merge_calendars,
    save_calendar,
)
from src.features.nowcast import fetch_fred_series


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download economic calendar + macro series")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument(
        "--end",
        default=(datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%d"),
    )
    parser.add_argument("--out", default=str(DEFAULT_CALENDAR_PATH))
    parser.add_argument("--no-merge", action="store_true", help="overwrite instead of merging into the cache")
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between monthly requests")
    parser.add_argument("--gasoline", action="store_true", help="also fetch FRED GASREGW for the CPI nowcast")
    parser.add_argument("--fred", default="", help="comma-separated extra FRED series ids to cache")
    parser.add_argument("--force", action="store_true", help="re-download cached FRED series")
    args = parser.parse_args(argv)

    out = Path(args.out)
    print(f"Fetching FXStreet calendar {args.start} .. {args.end}")
    new = fetch_fxstreet_calendar(args.start, args.end, sleep_s=args.sleep)
    print(f"  fetched {len(new)} rows")
    if not args.no_merge:
        existing = load_event_calendar(out)
        if not existing.empty:
            merged = merge_calendars(existing, new)
            print(f"  merged with {len(existing)} cached rows -> {len(merged)}")
            new = merged
    path = save_calendar(new, out)
    print(f"  saved {path}")

    series = [s.strip() for s in args.fred.split(",") if s.strip()]
    if args.gasoline and "GASREGW" not in series:
        series.append("GASREGW")
    for sid in series:
        s = fetch_fred_series(sid, force=args.force)
        print(f"FRED {sid}: {len(s)} obs, last {s.index[-1].date() if len(s) else 'n/a'}")


if __name__ == "__main__":
    main()
