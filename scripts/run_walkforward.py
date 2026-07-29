from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.config import (
    ROOT as PROJECT_ROOT,
    ensure_dirs,
    load_costs_config,
    load_strategy_config,
    resolve_strategy_config_name,
)
from src.data.dukascopy_ingest import load_symbol_bars
from src.research.promote import evaluate_promotion, write_promotion_report
from src.research.walk_forward import walk_forward
from src.strategy.registry import resolve_signal_fn, strategy_id
from src.mt5_gate import maybe_write_mt5_stub


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward + Monte Carlo + promote")
    parser.add_argument(
        "--strategy",
        default=None,
        help="cssm | orb | csmo | htf_pullback | d1_mom | fail_fade | vwap_fade | msnr | donchian_atr | xau_htf_pullback | htf_adx_pullback",
    )
    parser.add_argument("--config", default=None, help="strategy yaml filename under configs/")
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--test-months", type=int, default=6)
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg_name = resolve_strategy_config_name(args.strategy, args.config)
    strategy = load_strategy_config(cfg_name)
    sid = strategy_id(strategy)
    costs = load_costs_config()
    symbols = strategy["universe"]
    bars = {s: load_symbol_bars(s, "M15") for s in symbols}
    gen = resolve_signal_fn(strategy)

    folds = walk_forward(
        bars,
        strategy,
        costs,
        train_months=args.train_months,
        test_months=args.test_months,
        signal_fn=gen,
    )

    signals = gen(bars, strategy, costs)
    full = run_backtest(signals, bars, strategy, costs)

    verdict = evaluate_promotion(folds, full.trades)
    path = write_promotion_report(
        verdict,
        path=PROJECT_ROOT / "reports" / f"promotion_verdict_{sid}.json",
    )
    maybe_write_mt5_stub(verdict["verdict"] == "GO", strategy_name=sid)

    summary = {
        "strategy": sid,
        "config": cfg_name,
        "verdict": verdict["verdict"],
        "checks": verdict["checks"],
        "oos": {k: v for k, v in verdict["oos"].items() if k != "folds"},
        "monte_carlo": verdict["monte_carlo"],
        "report": str(path),
    }
    (PROJECT_ROOT / "reports" / f"walkforward_summary_{sid}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
