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
from src.research.promote import evaluate_promotion, write_promotion_report
from src.research.walk_forward import walk_forward
from src.strategy.cssm import generate_signals
from src.mt5_gate import maybe_write_mt5_stub


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward + Monte Carlo + promote")
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--test-months", type=int, default=6)
    args = parser.parse_args(argv)

    ensure_dirs()
    strategy = load_strategy_config()
    costs = load_costs_config()
    symbols = strategy["universe"]
    bars = {s: load_symbol_bars(s, "M15") for s in symbols}

    folds = walk_forward(
        bars,
        strategy,
        costs,
        train_months=args.train_months,
        test_months=args.test_months,
    )

    signals = generate_signals(bars, strategy, costs)
    full = run_backtest(signals, bars, strategy, costs)

    verdict = evaluate_promotion(folds, full.trades)
    path = write_promotion_report(verdict)
    maybe_write_mt5_stub(verdict["verdict"] == "GO")

    summary = {
        "verdict": verdict["verdict"],
        "checks": verdict["checks"],
        "oos": {k: v for k, v in verdict["oos"].items() if k != "folds"},
        "monte_carlo": verdict["monte_carlo"],
        "report": str(path),
    }
    (PROJECT_ROOT / "reports" / "walkforward_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
