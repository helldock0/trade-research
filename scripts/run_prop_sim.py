from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.config import (
    ROOT as PROJECT_ROOT,
    ensure_dirs,
    load_costs_config,
    load_ftmo_config,
    load_strategy_config,
    resolve_strategy_config_name,
)
from src.data.dukascopy_ingest import load_symbol_bars
from src.prop.ftmo_simulator import simulate_ftmo_path
from src.strategy.registry import resolve_signal_fn, strategy_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTMO 2-Step path simulation")
    parser.add_argument(
        "--strategy",
        default=None,
        help="cssm | orb | csmo | htf_pullback | d1_mom | fail_fade | vwap_fade | msnr | donchian_atr | xau_htf_pullback | htf_adx_pullback",
    )
    parser.add_argument("--config", default=None, help="strategy yaml filename under configs/")
    args = parser.parse_args(argv)
    ensure_dirs()

    cfg_name = resolve_strategy_config_name(args.strategy, args.config)
    strategy = load_strategy_config(cfg_name)
    sid = strategy_id(strategy)
    costs = load_costs_config()
    ftmo = load_ftmo_config()
    bars = {s: load_symbol_bars(s, "M15") for s in strategy["universe"]}
    gen = resolve_signal_fn(strategy)
    signals = gen(bars, strategy, costs)
    result = run_backtest(signals, bars, strategy, costs)
    sim = simulate_ftmo_path(result.trades, ftmo)

    out = {
        "strategy": sid,
        "config": cfg_name,
        "overall_pass": sim.overall_pass,
        "notes": sim.notes,
        "phase1": asdict(sim.phase1),
        "phase2": asdict(sim.phase2) if sim.phase2 else None,
        "n_trades": len(result.trades),
        "backtest_metrics": result.metrics,
    }
    path = PROJECT_ROOT / "reports" / f"ftmo_sim_{sid}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
