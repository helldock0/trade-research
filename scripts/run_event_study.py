"""Run the pre / post / fade event study for the event_drift config.

    python scripts/run_event_study.py
    python scripts/run_event_study.py --signal-mode nowcast_gap --tag cpi_nowcast
    python scripts/run_event_study.py --variants pre,post
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ROOT as PROJECT_ROOT,
    ensure_dirs,
    load_costs_config,
    load_ftmo_config,
    load_strategy_config,
)
from src.data.dukascopy_ingest import load_symbol_bars
from src.research.event_study import (
    forecast_quality,
    render_forecast_markdown,
    render_markdown,
    run_event_study,
    write_event_study_report,
)
from src.strategy.event_drift import VARIANTS, load_calendar_for, prepare_events


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Event study: pre-release drift, surprise, fade")
    parser.add_argument("--config", default="strategy_event_drift.yaml")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--signal-mode", default=None, help="override signal.signal_mode")
    parser.add_argument("--min-abs-z", type=float, default=None, help="override signal.min_abs_z")
    parser.add_argument("--tag", default=None, help="report name suffix (default: signal mode)")
    parser.add_argument("--no-ftmo", action="store_true")
    parser.add_argument(
        "--calendar-only",
        action="store_true",
        help="skip price bars; report whether the signal predicts the realized surprise",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    strategy = load_strategy_config(args.config)
    if args.signal_mode:
        strategy["signal"]["signal_mode"] = args.signal_mode
    if args.min_abs_z is not None:
        strategy["signal"]["min_abs_z"] = args.min_abs_z
    costs = load_costs_config()
    ftmo = None if args.no_ftmo else load_ftmo_config()
    variants = tuple(v.strip().lower() for v in args.variants.split(",") if v.strip())
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        print(f"Unknown variants: {bad}. Known: {VARIANTS}", file=sys.stderr)
        sys.exit(1)

    tag = args.tag or str(strategy["signal"].get("signal_mode", "surprise_momentum"))
    if args.calendar_only:
        events = prepare_events(load_calendar_for(strategy), strategy)
        fq = forecast_quality(events)
        md = render_forecast_markdown(fq, title=f"Forecast quality: {tag}")
        out = PROJECT_ROOT / "reports" / f"forecast_quality_{tag}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(md)
        print(f"Report: {out}")
        return

    bars = {s: load_symbol_bars(s, strategy.get("timeframe", "M15")) for s in strategy["universe"]}
    result = run_event_study(bars, strategy, costs, ftmo, variants=variants)
    paths = write_event_study_report(result, PROJECT_ROOT / "reports", tag=tag)
    print(render_markdown(result, title=f"Event study: {tag}"))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
