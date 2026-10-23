"""NY kill-zone breakout v3 with the two side edges (USDJPY gotobi, NAS100 dip).

Prints each line and each book (v3 alone, + gotobi, + gotobi + dip) on the
2018-22 check and 2023-26 test periods, then replays every book on the
instant-funded account (Blueberry Instant Elite defaults, as in
instant_sim_nykz_breakout.py) and the FTMO 2-step challenge. The lines run
side by side (each one trade at a time); books are replayed in exit order.
Trade lists go to reports/ for the dashboard.

Needs NAS100 and USDJPY M5 bid + ask, plus hourly bid bars from mid-2016 to
warm up the daily signals:
    python scripts/download_data.py --symbols NAS100,USDJPY --intervals H1 --start 2016-06-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import ROOT as PROJECT_ROOT, ensure_dirs, load_ftmo_config, load_strategy_config
from src.data.dukascopy_ingest import load_parquet, parquet_path
from src.prop.challenge import OPEN as CH_OPEN, PASS, challenge_outcomes
from src.prop.instant import OPEN, PAYOUT, TRADE_CAP, TRAIL, first_payout
from src.strategy.nykz_breakout import merge_bid_ask, run, trades_frame
from src.strategy.side_edges import cash_sessions, dip_trades, gotobi_trades, ny_close_daily

PERIODS = {"2018-22": ("2018-01-01", "2022-12-31"), "2023-26": ("2023-01-01", "2026-12-31")}


def _line_stats(trades: list) -> dict:
    r = np.array([t.r_multiple for t in sorted(trades, key=lambda t: t.exit_time)])
    if not len(r):
        return {"n_trades": 0}
    equity = np.concatenate([[0.0], np.cumsum(r)])
    years = len({t.day[:4] for t in trades})
    return {
        "n_trades": len(r), "per_year": len(r) / max(years, 1), "win_rate": float((r > 0).mean()),
        "r_per_trade": float(r.mean()), "t_stat": float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 1 else 0.0,
        "total_r": float(r.sum()), "max_dd_r": float((np.maximum.accumulate(equity) - equity).max()),
        "profit_factor": float(r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else float("inf"),
    }


def _instant(trades: list, risk: float, rules: dict) -> dict:
    trades = sorted(trades, key=lambda t: t.exit_time)
    r, mae, mfe = (np.array([getattr(t, k) for t in trades]) for k in ("r_multiple", "mae_r", "mfe_r"))
    epoch = pd.Timestamp("2000-01-01")
    start_day = np.array([(pd.Timestamp(t.entry_time.tz_convert("America/New_York").date()) - epoch).days for t in trades])
    exit_day = np.array([(pd.Timestamp(t.exit_time.tz_convert("America/New_York").date()) - epoch).days for t in trades])
    outs = [first_payout(r[i:], mae[i:], mfe[i:], start_day[i:], exit_day[i:], risk, rules) for i in range(len(r))]
    done = [(i, res, i + k) for i, (res, k) in enumerate(outs) if res != OPEN]
    days = np.array([exit_day[j] - start_day[i] for i, res, j in done if res == PAYOUT])
    n = len(done)
    return {
        "accounts": n, "reached": sum(res == PAYOUT for _, res, _ in done) / n,
        "lost_trailing_dd": sum(res == TRAIL for _, res, _ in done) / n,
        "lost_trade_cap": sum(res == TRADE_CAP for _, res, _ in done) / n,
        "median_days": float(np.median(days)) if len(days) else None,
        "p75_days": float(np.percentile(days, 75)) if len(days) else None,
    }


def _ftmo(trades: list, risk: float, ftmo: dict) -> float:
    trades = sorted(trades, key=lambda t: t.entry_time)
    out = challenge_outcomes(np.array([t.r_multiple for t in trades]), np.array([t.mae_r for t in trades]), risk, ftmo)
    done = [o for o in out if o[0] != CH_OPEN]
    return sum(o[0] == PASS for o in done) / len(done)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="v3 + USDJPY gotobi + NAS100 dip")
    parser.add_argument("--config", default="strategy_nykz_breakout.yaml")
    parser.add_argument("--start", default="2018-03-01", help="first trade (v3 needs earlier bars to warm up)")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=0.25, help="% of the starting balance per trade, every line")
    parser.add_argument("--targets-usd", default="100,350", help="first payout, fee back")
    parser.add_argument("--split", type=float, default=0.8, help="trader's profit share, for income per month")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    m5 = lambda sym: merge_bid_ask(
        load_parquet(parquet_path(sym, "M5", cache, side="bid")), load_parquet(parquet_path(sym, "M5", cache, side="ask"))
    )
    nas, usdjpy = m5("NAS100"), m5("USDJPY")
    # daily signals need ~1 year before the first trade: hourly bars from mid-2016
    lines = {
        "v3": run(nas, {**cfg, "signal": {**cfg["signal"], "trade_from": args.start}})[0],
        "gotobi": gotobi_trades(usdjpy, ny_close_daily(load_parquet(parquet_path("USDJPY", "H1", cache))), {"trade_from": args.start}),
        "dip": dip_trades(nas, cash_sessions(load_parquet(parquet_path("NAS100", "H1", cache))), {"trade_from": args.start}),
    }
    books = {"v3 alone": ["v3"], "v3 + gotobi": ["v3", "gotobi"], "v3 + gotobi + dip": ["v3", "gotobi", "dip"]}
    book_trades = {b: [t for name in names for t in lines[name]] for b, names in books.items()}

    months = len(pd.period_range(args.start[:7], max(t.exit_time for t in lines["v3"]).strftime("%Y-%m"), freq="M"))
    ftmo = load_ftmo_config()
    results: dict = {"lines": {}, "books": {}}
    for name, trades in {**lines, **book_trades}.items():
        key = "lines" if name in lines else "books"
        row = {p: _line_stats([t for t in trades if lo <= t.day <= hi]) for p, (lo, hi) in PERIODS.items()}
        row["all"] = _line_stats(trades)
        row["usd_per_month"] = row["all"]["total_r"] * args.risk / 100 * args.balance / months * args.split
        if key == "books":
            row["instant"] = {}
            for target in (float(x) for x in args.targets_usd.split(",")):
                rules = {"max_dd_pct": 10.0, "max_trade_loss_pct": 1.5, "target_pct": target / args.balance * 100,
                         "min_active_days": 5, "active_day_pct": 0.5, "min_days": 14}
                row["instant"][f"${target:.0f}"] = _instant(trades, args.risk, rules)
            row["ftmo_pass"] = {f"{x}%": _ftmo(trades, x, ftmo) for x in (0.5, 1.0)}
        results[key][name] = row
    monthly = pd.DataFrame({n: pd.Series([t.r_multiple for t in ts], index=[t.day[:7] for t in ts]).groupby(level=0).sum() for n, ts in lines.items()}).fillna(0.0)
    results["monthly_corr_with_v3"] = {n: float(monthly["v3"].corr(monthly[n])) for n in ("gotobi", "dip")}

    out = PROJECT_ROOT / "reports" / "combo_nykz_breakout.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, trades in lines.items():
        trades_frame(trades).to_csv(PROJECT_ROOT / "reports" / f"combo_{name}_trades.csv", index=False)

    for key in ("lines", "books"):
        print(f"\n{key}")
        for name, row in results[key].items():
            for p in (*PERIODS, "all"):
                x = row[p]
                if not x["n_trades"]:
                    continue
                print(f"  {name:18} {p:7} n {x['n_trades']:4d} ({x['per_year']:5.1f}/yr) WR {x['win_rate']:4.0%} {x['r_per_trade']:+.3f}R/t (t {x['t_stat']:4.2f}) "
                      f"total {x['total_r']:+7.1f}R DD {x['max_dd_r']:5.1f}R PF {x['profit_factor']:.2f}")
            print(f"  {'':18} ${row['usd_per_month']:.0f}/month after the split at {args.risk}% of ${args.balance:,.0f}")
    print(f"\nmonthly R correlation with v3: " + ", ".join(f"{k} {v:+.2f}" for k, v in results["monthly_corr_with_v3"].items()))
    print(f"\n${args.balance:,.0f} instant account at {args.risk}% risk (accounts started at every trade from {args.start}):")
    for name, row in results["books"].items():
        for target, x in row["instant"].items():
            print(f"  {name:18} {target:>5}: reached {x['reached']:4.0%}  trailing DD {x['lost_trailing_dd']:4.0%}  1.5% cap {x['lost_trade_cap']:4.0%}  "
                  f"days median {x['median_days'] or 0:4.0f} p75 {x['p75_days'] or 0:4.0f}  ({x['accounts']} accounts)")
        print(f"  {'':18} FTMO 2-step pass: " + "  ".join(f"{k} {v:.0%}" for k, v in row["ftmo_pass"].items()))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
