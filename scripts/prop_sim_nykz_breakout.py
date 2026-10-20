"""FTMO-style pass rates for NY kill-zone breakout configs at several risk levels.

A 2-step challenge (configs/ftmo_2step.yaml) is started at every trade from
--start onward and replayed on the real trade sequence. Results count only
challenges that finished (passed or failed) before the data ran out.
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
from src.prop.challenge import FAIL, OPEN, PASS, challenge_outcomes
from src.strategy.nykz_breakout import merge_bid_ask, run

CONFIGS = {"v3": "strategy_nykz_breakout.yaml", "v4 prop": "strategy_nykz_breakout_prop.yaml"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTMO pass rates for NY kill-zone breakout")
    parser.add_argument("--start", default="2018-03-01", help="first trade day (earlier bars warm up filters)")
    parser.add_argument("--risks", default="0.25,0.5,0.75,1,1.5,2", help="risk per trade, % of balance")
    args = parser.parse_args(argv)
    risks = [float(x) for x in args.risks.split(",")]

    ensure_dirs()
    ftmo = load_ftmo_config()
    rows = []
    for name, cfg_name in CONFIGS.items():
        cfg = load_strategy_config(cfg_name)
        cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
        bars = merge_bid_ask(
            load_parquet(parquet_path(cfg["symbol"], cfg["timeframe"], cache, side="bid")),
            load_parquet(parquet_path(cfg["symbol"], cfg["timeframe"], cache, side="ask")),
        )
        trades, _ = run(bars, {**cfg, "signal": {**cfg["signal"], "trade_from": args.start}})
        r = np.array([t.r_multiple for t in trades])
        mae = np.array([t.mae_r for t in trades])
        start_day = pd.to_datetime([t.day for t in trades])
        end_day = pd.to_datetime([t.exit_time.tz_convert("America/New_York").date() for t in trades])

        for risk in risks:
            out = challenge_outcomes(r, mae, risk, ftmo)
            done = [o for o in out if o[0] != OPEN]
            passed = [o for o in done if o[0] == PASS]
            days = np.array([(end_day[k] - start_day[i]).days for _, _, i, k in passed])
            by_era = {}
            for era, lo, hi in (("2018-22", "2018", "2022-12-31"), ("2023-26", "2023", "2027")):
                era_done = [o for o in done if lo <= str(start_day[o[2]].date()) <= hi]
                by_era[era] = sum(o[0] == PASS for o in era_done) / len(era_done) if era_done else None
            rows.append({
                "version": name,
                "risk_pct": risk,
                "challenges": len(done),
                "pass_rate": len(passed) / len(done) if done else 0.0,
                "fail_phase1": sum(o[1] == "phase1" for o in done) / len(done) if done else 0.0,
                "fail_phase2": sum(o[1] == "phase2" for o in done) / len(done) if done else 0.0,
                "pass_rate_by_start": by_era,
                "median_days_to_pass": float(np.median(days)) if len(days) else None,
                "p75_days_to_pass": float(np.percentile(days, 75)) if len(days) else None,
                "unfinished": len(out) - len(done),
            })

    out_path = PROJECT_ROOT / "reports" / "prop_sim_nykz_breakout.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{'version':8} {'risk':>5} {'n':>5} {'pass':>6} {'fail P1':>8} {'fail P2':>8} | {'pass 18-22':>10} {'pass 23-26':>10} | {'days med':>8} {'p75':>5}")
    for x in rows:
        era = x["pass_rate_by_start"]
        fmt = lambda v: f"{v:10.0%}" if v is not None else f"{'-':>10}"
        print(f"{x['version']:8} {x['risk_pct']:4.2f}% {x['challenges']:5d} {x['pass_rate']:6.0%} {x['fail_phase1']:8.0%} {x['fail_phase2']:8.0%} | "
              f"{fmt(era['2018-22'])} {fmt(era['2023-26'])} | {x['median_days_to_pass'] or 0:8.0f} {x['p75_days_to_pass'] or 0:5.0f}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
