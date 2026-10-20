"""NY kill-zone breakout on several markets, one trade at a time across all of them.

Compares each market alone, the combined one-at-a-time portfolio (first
market wins ties), and every market traded independently at once, on the
test and 2018-22 check periods, plus FTMO 2-step pass rates over 2018-26.
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
from src.prop.challenge import OPEN, PASS, challenge_outcomes
from src.strategy.nykz_breakout import merge_bid_ask, run, run_portfolio


def _stats(trades: list, months: pd.Index) -> dict:
    """Trades in exit order: totals, max drawdown, average month and losing years."""
    trades = sorted(trades, key=lambda t: t.exit_time)
    r = np.array([t.r_multiple for t in trades])
    equity = np.concatenate([[0.0], np.cumsum(r)])
    df = pd.DataFrame({"m": [t.day[:7] for t in trades], "y": [t.day[:4] for t in trades], "r": r})
    by_month = df.groupby("m").r.sum().reindex(months, fill_value=0.0)
    return {
        "n_trades": len(r),
        "win_rate": float((r > 0).mean()),
        "total_r": float(r.sum()),
        "r_per_trade": float(r.mean()),
        "max_dd_r": float((np.maximum.accumulate(equity) - equity).max()),
        "avg_r_per_month": float(by_month.mean()),
        "losing_years": int((df.groupby("y").r.sum() < 0).sum()),
    }


def _ftmo(trades: list, risk: float, ftmo: dict) -> float:
    trades = sorted(trades, key=lambda t: t.entry_time)
    out = challenge_outcomes(np.array([t.r_multiple for t in trades]), np.array([t.mae_r for t in trades]), risk, ftmo)
    done = [o for o in out if o[0] != OPEN]
    return sum(o[0] == PASS for o in done) / len(done)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-trade-at-a-time NY kill-zone breakout portfolio")
    parser.add_argument("--config", default="strategy_nykz_breakout.yaml")
    parser.add_argument("--symbols", default="NAS100,US500", help="priority order for same-time signals")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_strategy_config(args.config)
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    start = cfg["data"]["start"]
    bars = {
        sym: merge_bid_ask(
            load_parquet(parquet_path(sym, cfg["timeframe"], cache, side="bid")),
            load_parquet(parquet_path(sym, cfg["timeframe"], cache, side="ask")),
        )
        for sym in args.symbols.split(",")
    }
    # The break-candle buffer keeps the same share of price on every market.
    base = cfg["symbol"]
    frac = float(cfg["signal"]["stop_buffer_points"]) / float(bars[base]["close"].loc[start:].median())
    buffers = {s: float(cfg["signal"]["stop_buffer_points"]) if s == base else frac * float(b["close"].loc[start:].median()) for s, b in bars.items()}

    def window(first: str, last: str | None, trade_from: str) -> dict:
        lo = pd.Timestamp(first, tz="UTC")
        hi = pd.Timestamp(last, tz="UTC") + pd.Timedelta(days=1) if last else None
        return {
            s: (b.loc[lo:hi], {**cfg, "symbol": s, "signal": {**cfg["signal"], "stop_buffer_points": buffers[s], "trade_from": trade_from}})
            for s, b in bars.items()
        }

    periods = {
        "2023-26": (window(str(pd.Timestamp(start) - pd.Timedelta(days=365))[:10], None, start), pd.period_range(start[:7], "2026-09", freq="M")),
        "2018-22": (window("2018-01-01", "2022-12-31", "2018-01-01"), pd.period_range("2018-01", "2022-12", freq="M")),
        "all": (window("2018-01-01", None, "2018-03-01"), None),
    }
    ftmo = load_ftmo_config()
    results: dict = {}
    for period, (markets, months) in periods.items():
        alone = {s: run(b, c)[0] for s, (b, c) in markets.items()}
        combined, skipped = run_portfolio(markets)
        variants = {**{f"{s} only": t for s, t in alone.items()},
                    "one at a time": [t for _, t in combined],
                    "all at once": [t for ts in alone.values() for t in ts]}
        if months is None:  # full history: FTMO pass rates (one trade at a time only)
            results["ftmo_pass_rate"] = {
                name: {f"{risk}%": _ftmo(t, risk, ftmo) for risk in (0.5, 1.0)}
                for name, t in variants.items() if name != "all at once"
            }
            continue
        months = months.astype(str)
        results[period] = {name: _stats(t, months) for name, t in variants.items()}
        results[period]["one at a time"]["by_market"] = {s: sum(1 for m, _ in combined if m == s) for s in markets}
        results[period]["one at a time"]["skipped_setups"] = skipped

    out = PROJECT_ROOT / "reports" / "portfolio_nykz_breakout.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for period in ("2023-26", "2018-22"):
        print(f"\n{period}")
        for name, x in results[period].items():
            extra = f"  trades by market {x['by_market']}" if "by_market" in x else ""
            print(f"  {name:15} n {x['n_trades']:4d} WR {x['win_rate']:5.1%} total {x['total_r']:+7.1f}R  {x['r_per_trade']:+.3f}R/t  "
                  f"DD {x['max_dd_r']:5.1f}R  {x['avg_r_per_month']:+.2f}R/month  losing years {x['losing_years']}{extra}")
    print("\nFTMO 2-step pass rate (challenges started at every trade, 2018-26):")
    for name, x in results["ftmo_pass_rate"].items():
        print(f"  {name:15} " + "  ".join(f"{k} risk: {v:.0%}" for k, v in x.items()))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
