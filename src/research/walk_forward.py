from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.backtest.engine import run_backtest
from src.strategy.cssm import Signal, generate_signals
from dataclasses import dataclass


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_trades: int
    test_profitable: bool


def _as_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "timestamp" not in x.columns:
        x = x.rename_axis("timestamp").reset_index()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    cols = ["timestamp", "open", "high", "low", "close"]
    if "volume" in x.columns:
        cols.append("volume")
    else:
        x["volume"] = 0.0
        cols.append("volume")
    return x[cols].sort_values("timestamp").reset_index(drop=True)


def _filter_bars(
    bars_by_symbol: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out = {}
    for s, df in bars_by_symbol.items():
        x = _as_frame(df)
        out[s] = x[(x["timestamp"] >= start) & (x["timestamp"] < end)].reset_index(drop=True)
    return out


def _filter_signals(signals: list[Signal], start: pd.Timestamp, end: pd.Timestamp) -> list[Signal]:
    return [s for s in signals if start <= s.timestamp < end]


def walk_forward(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    train_months: int = 18,
    test_months: int = 6,
    min_folds: int = 4,
    initial_balance: float = 100000.0,
    signal_fn: Callable | None = None,
) -> list[FoldResult]:
    """Rolling 18m/6m walk-forward. Parameters are frozen (no in-fold optimization)."""
    any_df = _as_frame(next(iter(bars_by_symbol.values())))
    start = any_df["timestamp"].min().normalize()
    end = any_df["timestamp"].max().normalize() + pd.Timedelta(days=1)

    gen = signal_fn or generate_signals
    all_signals = gen(bars_by_symbol, strategy_cfg, costs_cfg)

    folds: list[FoldResult] = []
    fold = 0
    cursor = start
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break

        warm = pd.DateOffset(days=90)
        train_bars_w = _filter_bars(bars_by_symbol, train_start - warm, train_end)
        test_bars_w = _filter_bars(bars_by_symbol, test_start - warm, test_end)

        train_sigs = _filter_signals(all_signals, train_start, train_end)
        test_sigs = _filter_signals(all_signals, test_start, test_end)

        train_res = run_backtest(train_sigs, train_bars_w, strategy_cfg, costs_cfg, initial_balance)
        test_res = run_backtest(test_sigs, test_bars_w, strategy_cfg, costs_cfg, initial_balance)

        folds.append(
            FoldResult(
                fold=fold,
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                train_metrics=train_res.metrics,
                test_metrics=test_res.metrics,
                test_trades=test_res.metrics.get("n_trades", 0),
                test_profitable=test_res.metrics.get("net_pnl", 0) > 0,
            )
        )
        fold += 1
        cursor = cursor + pd.DateOffset(months=test_months)

    return folds


def aggregate_oos(folds: list[FoldResult]) -> dict[str, Any]:
    if not folds:
        return {
            "n_folds": 0,
            "profitable_folds": 0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_dd_pct": 0.0,
        }
    exp = [f.test_metrics.get("expectancy_r", 0.0) for f in folds]
    pfs = [
        f.test_metrics.get("profit_factor", 0.0)
        for f in folds
        if f.test_metrics.get("n_trades", 0) > 0
    ]
    dds = [f.test_metrics.get("max_dd_pct", 0.0) for f in folds]
    return {
        "n_folds": len(folds),
        "profitable_folds": sum(1 for f in folds if f.test_profitable),
        "expectancy_r": float(sum(exp) / len(exp)),
        "profit_factor": float(sum(pfs) / len(pfs)) if pfs else 0.0,
        "max_dd_pct": float(max(dds) if dds else 0.0),
        "folds": [
            {
                "fold": f.fold,
                "test_start": str(pd.Timestamp(f.test_start).date()),
                "test_end": str(pd.Timestamp(f.test_end).date()),
                "test_metrics": f.test_metrics,
                "profitable": f.test_profitable,
            }
            for f in folds
        ],
    }
