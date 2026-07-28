from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.config import ROOT as PROJECT_ROOT, ensure_dirs, load_costs_config, load_strategy_config
from src.data.dukascopy_ingest import load_symbol_bars
from src.strategy.cssm import generate_signals


def load_universe_bars(symbols: list[str], interval: str = "M15"):
    return {s: load_symbol_bars(s, interval) for s in symbols}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CSSM backtest")
    parser.add_argument("--report", default="reports/backtest_summary.json")
    args = parser.parse_args(argv)

    ensure_dirs()
    strategy = load_strategy_config()
    costs = load_costs_config()
    symbols = strategy["universe"]

    bars = load_universe_bars(symbols, "M15")
    signals = generate_signals(bars, strategy, costs)
    result = run_backtest(signals, bars, strategy, costs)

    out = {
        "n_signals": len(signals),
        "metrics": result.metrics,
        "trades_path": "reports/backtest_trades.csv",
    }
    report = PROJECT_ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")

    tf = result.trades_frame()
    if not tf.empty:
        tf.to_csv(PROJECT_ROOT / "reports" / "backtest_trades.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
