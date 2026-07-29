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
from src.strategy.registry import list_strategies, resolve_signal_fn, strategy_id


def parse_strategies_arg(raw: str | None) -> list[str]:
    """Parse comma-separated strategy ids; default = all known strategies."""
    known = list_strategies()
    if raw is None or not str(raw).strip():
        return known
    requested = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    if not requested:
        return known
    unknown = [s for s in requested if s not in known]
    if unknown:
        raise ValueError(
            f"Unknown strateg{'y' if len(unknown) == 1 else 'ies'}: {', '.join(unknown)}. "
            f"Known: {', '.join(known)}"
        )
    return requested


def run_batch(strategies: list[str]) -> dict:
    """Run backtest + FTMO sim for each strategy; write per-strategy and batch reports."""
    ensure_dirs()
    costs = load_costs_config()
    ftmo = load_ftmo_config()
    results: list[dict] = []

    for sid_req in strategies:
        row: dict = {
            "strategy": sid_req,
            "config": None,
            "n_signals": 0,
            "n_trades": 0,
            "metrics": None,
            "overall_pass": None,
            "notes": [],
            "phase1_breach": None,
            "phase2_breach": None,
            "error": None,
        }
        try:
            cfg_name = resolve_strategy_config_name(sid_req, None)
            strategy = load_strategy_config(cfg_name)
            sid = strategy_id(strategy)
            row["strategy"] = sid
            row["config"] = cfg_name

            symbols = strategy["universe"]
            gen = resolve_signal_fn(strategy)
            bars = {s: load_symbol_bars(s, "M15") for s in symbols}
            signals = gen(bars, strategy, costs)
            result = run_backtest(signals, bars, strategy, costs)
            sim = simulate_ftmo_path(result.trades, ftmo)

            report_rel = f"reports/backtest_summary_{sid}.json"
            trades_rel = f"reports/backtest_trades_{sid}.csv"
            bt_out = {
                "strategy": sid,
                "config": cfg_name,
                "n_signals": len(signals),
                "metrics": result.metrics,
                "trades_path": trades_rel,
            }
            report = PROJECT_ROOT / report_rel
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(bt_out, indent=2), encoding="utf-8")

            tf = result.trades_frame()
            if not tf.empty:
                tf.to_csv(PROJECT_ROOT / trades_rel, index=False)

            ftmo_out = {
                "strategy": sid,
                "config": cfg_name,
                "overall_pass": sim.overall_pass,
                "notes": sim.notes,
                "phase1": asdict(sim.phase1),
                "phase2": asdict(sim.phase2) if sim.phase2 else None,
                "n_trades": len(result.trades),
                "backtest_metrics": result.metrics,
            }
            ftmo_path = PROJECT_ROOT / "reports" / f"ftmo_sim_{sid}.json"
            ftmo_path.write_text(json.dumps(ftmo_out, indent=2, default=str), encoding="utf-8")

            row.update(
                {
                    "n_signals": len(signals),
                    "n_trades": len(result.trades),
                    "metrics": result.metrics,
                    "overall_pass": sim.overall_pass,
                    "notes": sim.notes,
                    "phase1_breach": sim.phase1.breach,
                    "phase2_breach": sim.phase2.breach if sim.phase2 else None,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 — batch continues on per-strategy failure
            row["error"] = str(exc)

        results.append(row)

    batch = {"strategies": strategies, "results": results}
    batch_path = PROJECT_ROOT / "reports" / "batch_summary.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch, indent=2, default=str), encoding="utf-8")
    return batch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch backtest + FTMO simulation for all strategies")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated subset of strategies (default: all). "
        f"Known: {', '.join(list_strategies())}",
    )
    args = parser.parse_args(argv)

    try:
        strategies = parse_strategies_arg(args.strategies)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    batch = run_batch(strategies)
    print(json.dumps(batch, indent=2, default=str))


if __name__ == "__main__":
    main()
