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
from src.strategy.registry import resolve_signal_fn, strategy_id


def load_universe_bars(symbols: list[str], interval: str = "M15"):
    return {s: load_symbol_bars(s, interval) for s in symbols}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument(
        "--strategy",
        default=None,
        help="cssm | orb | csmo | htf_pullback | d1_mom | fail_fade | vwap_fade | msnr | donchian_atr | xau_htf_pullback | htf_adx_pullback",
    )
    parser.add_argument("--config", default=None, help="strategy yaml filename under configs/")
    parser.add_argument("--report", default=None, help="output summary json path")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg_name = resolve_strategy_config_name(args.strategy, args.config)
    strategy = load_strategy_config(cfg_name)
    sid = strategy_id(strategy)
    costs = load_costs_config()
    symbols = strategy["universe"]
    gen = resolve_signal_fn(strategy)

    bars = load_universe_bars(symbols, "M15")
    signals = gen(bars, strategy, costs)
    result = run_backtest(signals, bars, strategy, costs)

    report_rel = args.report or f"reports/backtest_summary_{sid}.json"
    trades_rel = f"reports/backtest_trades_{sid}.csv"
    out = {
        "strategy": sid,
        "config": cfg_name,
        "n_signals": len(signals),
        "metrics": result.metrics,
        "trades_path": trades_rel,
    }
    report = PROJECT_ROOT / report_rel
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")

    tf = result.trades_frame()
    if not tf.empty:
        tf.to_csv(PROJECT_ROOT / trades_rel, index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
