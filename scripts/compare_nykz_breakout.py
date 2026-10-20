"""Compare NY kill-zone breakout variants, one grid at a time.

--grid break: clean-break rules x stop placement, each from plain v1 rules.
--grid exit:  stop size x trailing, on the config's entry rule.
--grid fvg:   next-close vs FVG retest-and-confirm and FVG limit entries, with v2 and ATR stops.
--grid rr:    the config's rules with different take-profit targets.
--grid weekday: the config's rules, skipping one weekday at a time.
--grid session: the config's rules with earlier entry cutoffs (New York time).

Each variant runs on the config's test period and on an earlier check period
(2018-2022 by default), with the real bid/ask spread.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ROOT as PROJECT_ROOT, ensure_dirs, load_strategy_config
from src.data.dukascopy_ingest import load_parquet, parquet_path
from src.strategy.nykz_breakout import merge_bid_ask, run

BREAK_RULES = [
    ("any close beyond", {}),
    ("close >= 30 pts beyond", {"min_break_points": 30.0}),
    ("close >= 0.1% beyond", {"min_break_pct": 0.1}),
    ("next candle confirms", {"confirm_close": True}),
    ("body >= 1.5x range avg", {"min_body_vs_range": 1.5}),
]
V1_RULES = {
    "min_break_points": 0.0,
    "min_break_pct": 0.0,
    "min_body_vs_range": 0.0,
    "confirm_close": False,
    "stop_includes_prior_bar": False,
}
STOP_RULES = [
    ("break candle", {}),
    ("break + prior candle", {"stop_includes_prior_bar": True}),
]
PLAIN_ENTRY = {"entry_mode": "next_close", "stop_ref": "break"}
FIXED_EXITS = {
    "atr_stop_mult": 0.0,
    "min_stop_atr": 0.0,
    "breakeven_at_r": 0.0,
    "trail_atr_mult": 0.0,
    "trail_start_r": 1.0,
    "take_profit_r": 4.0,
}
STOP_SIZES = [
    ("break candle", {}),
    ("1.5x ATR", {"atr_stop_mult": 1.5}),
    ("2x ATR", {"atr_stop_mult": 2.0}),
    ("3x ATR", {"atr_stop_mult": 3.0}),
    ("candle, min 1.5x ATR", {"min_stop_atr": 1.5}),
]
TRAILS = [
    ("fixed 4R", {}),
    ("breakeven at 1R", {"breakeven_at_r": 1.0}),
    ("trail 2x ATR, 4R target", {"trail_atr_mult": 2.0}),
    ("trail 2x ATR, no target", {"trail_atr_mult": 2.0, "take_profit_r": 0.0}),
]
FVG_VARIANTS = [
    ("next close (v2)", "break candle", {}),
    ("next close (v2)", "2x ATR", {"atr_stop_mult": 2.0}),
    ("FVG retest", "break candle", {"entry_mode": "fvg_retest"}),
    ("FVG retest", "FVG candle 1", {"entry_mode": "fvg_retest", "stop_ref": "fvg"}),
    ("FVG retest", "2x ATR", {"entry_mode": "fvg_retest", "atr_stop_mult": 2.0}),
] + [
    (f"FVG limit, {level_name}", stop_name, {"entry_mode": "fvg_limit", "fvg_limit_level": level, **stop_cfg})
    for level_name, level in (("near edge", 0.0), ("50%", 0.5))
    for stop_name, stop_cfg in (
        ("break candle", {}),
        ("FVG candle 1", {"stop_ref": "fvg"}),
        ("2x ATR", {"atr_stop_mult": 2.0}),
    )
]


WEEKDAY_VARIANTS = [("config rules", "every day", {})] + [
    ("config rules", f"skip {d}", {"skip_weekdays": [d]}) for d in ("Mon", "Tue", "Wed", "Thu", "Fri")
]
SESSION_VARIANTS = [
    ("config rules", f"entries until {hhmm}" + (" (overlap)" if hhmm == "12:00" else ""), {"session": {"entry_cutoff": hhmm}})
    for hhmm in ("10:30", "11:00", "11:30", "12:00", "13:00", "14:00", "15:00", "16:55")
]
RR_VARIANTS = [("config rules", f"{r:g}R target", {"take_profit_r": r}) for r in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)]


def _product(rules_a: list, rules_b: list) -> list:
    return [(a, b, {**cfg_a, **cfg_b}) for a, cfg_a in rules_a for b, cfg_b in rules_b]


GRIDS = {
    # name: (settings every variant starts from, variants, column labels)
    "break": ({**PLAIN_ENTRY, **V1_RULES, **FIXED_EXITS}, _product(BREAK_RULES, STOP_RULES), ("break rule", "stop")),
    "exit": ({**PLAIN_ENTRY, **FIXED_EXITS}, _product(STOP_SIZES, TRAILS), ("stop", "trail")),
    "fvg": ({**PLAIN_ENTRY, **FIXED_EXITS}, FVG_VARIANTS, ("entry", "stop")),
    "rr": ({}, RR_VARIANTS, ("rules", "target")),
    "weekday": ({}, WEEKDAY_VARIANTS, ("rules", "days")),
    "session": ({}, SESSION_VARIANTS, ("rules", "entry window")),
}


def _slice(bars: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    out = bars.loc[pd.Timestamp(start, tz="UTC") :]
    if end:
        out = out.loc[: pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare NY kill-zone breakout variants")
    parser.add_argument("--config", default="strategy_nykz_breakout.yaml")
    parser.add_argument("--check-start", default="2018-01-01")
    parser.add_argument("--check-end", default="2022-12-31")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="break")
    parser.add_argument("--day", default=None, help="also show each variant's trade on this NY date")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    sym, tf = cfg["symbol"], cfg["timeframe"]
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    bars = merge_bid_ask(
        load_parquet(parquet_path(sym, tf, cache, side="bid")),
        load_parquet(parquet_path(sym, tf, cache, side="ask")),
    )
    periods = {
        "test": _slice(bars, cfg["data"]["start"], cfg["data"].get("end")),
        "check": _slice(bars, args.check_start, args.check_end),
    }

    base, variants, (label_a, label_b) = GRIDS[args.grid]
    rows = []
    for name_a, name_b, extra in variants:
        extra = dict(extra)
        session = {**cfg["session"], **extra.pop("session", {})}
        variant = {**cfg, "session": session, "signal": {**cfg["signal"], **base, **extra}}
        row: dict = {"a": name_a, "b": name_b}
        for period, period_bars in periods.items():
            trades, s = run(period_bars, variant)
            row[period] = {
                k: s.get(k, 0.0)
                for k in ("n_trades", "win_rate", "expectancy_r", "total_r", "max_dd_r", "by_year")
            }
            if args.day and period == "test":
                row["day"] = next(
                    ({"side": t.side, "entry": t.entry, "stop": t.stop, "r": t.r_multiple}
                     for t in trades if t.day == args.day),
                    None,
                )
        rows.append(row)

    out = PROJECT_ROOT / "reports" / f"compare_{cfg['strategy']}_{args.grid}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"{label_a:24} {label_b:24} {'trades':>6} {'win%':>6} {'total R':>8} {'R/trade':>8} {'maxDD R':>8} | "
          f"{'2018-22 R':>9} {'R/trade':>8}" + (f" | {args.day}" if args.day else ""))
    for r in rows:
        t, c = r["test"], r["check"]
        line = (f"{r['a']:24} {r['b']:24} {t['n_trades']:6d} {t['win_rate']:6.1%} "
                f"{t['total_r']:+8.1f} {t['expectancy_r']:+8.3f} {t['max_dd_r']:8.1f} | "
                f"{c['total_r']:+9.1f} {c['expectancy_r']:+8.3f}")
        if args.day:
            d = r.get("day")
            line += " | " + (f"{'long' if d['side'] > 0 else 'short'} {d['entry']:.1f} stop {d['stop']:.1f} {d['r']:+.1f}R"
                             if d else "no trade")
        print(line)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
