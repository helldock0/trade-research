"""Donchian + ATR unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_donchian_atr_config_and_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("donchian_atr") == "strategy_donchian_atr.yaml"
    assert default_config_name("donchian") == "strategy_donchian_atr.yaml"
    cfg = load_strategy_config("strategy_donchian_atr.yaml")
    assert strategy_id(cfg) == "donchian_atr"
    assert cfg["signal"]["donchian_period"] == 20
    assert cfg["signal"]["atr_stop_mult"] == 1.5
    assert cfg["signal"]["take_profit_r"] == 2.0
    assert cfg["signal"]["exit_mode"] == "swing"
    assert cfg["signal"]["htf_rule"] == "1h"
    assert resolve_signal_fn(cfg).__module__.endswith("donchian_atr")


def test_donchian_channel_no_lookahead_shift():
    from src.features.donchian import donchian_channel

    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    high = pd.Series([1.0, 1.1, 1.2, 1.15, 1.3, 1.25, 1.4, 1.35, 1.5, 1.45], index=idx)
    low = high - 0.05
    upper, lower = donchian_channel(high, low, period=3)
    # First usable upper at i=3 uses highs[0:3] max = 1.2 (shifted)
    assert np.isnan(upper.iloc[2])
    assert abs(float(upper.iloc[3]) - 1.2) < 1e-12
    # At i=6, prior 3 highs are i=3,4,5 -> 1.15, 1.3, 1.25 -> max 1.3
    assert abs(float(upper.iloc[6]) - 1.3) < 1e-12
    # Current bar high must not leak into same-bar channel
    assert float(upper.iloc[6]) < float(high.iloc[6])


def test_donchian_break_side_long_short():
    from src.features.donchian import donchian_break_side

    assert donchian_break_side(1.21, 1.20, 1.10) == 1
    assert donchian_break_side(1.09, 1.20, 1.10) == -1
    assert donchian_break_side(1.15, 1.20, 1.10) == 0
    assert donchian_break_side(1.20, 1.20, 1.10) == 0  # need strict break


def test_donchian_generate_signals_long_break():
    """Synthetic H1 uptrend break in London emits a long signal."""
    from src.config import load_strategy_config
    from src.strategy.donchian_atr import generate_signals

    # Build M15 covering enough H1 bars; force a clear upside break after flat range
    start = pd.Timestamp("2024-01-02", tz="UTC")  # Tuesday
    n = 80 * 4  # ~80 H1 bars of M15
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = np.full(len(idx), 1.1000, dtype=float)
    # After warmup, spike above prior 20-high on a London H1 (Thu 08:00)
    break_h1 = start + pd.Timedelta(days=2, hours=8)
    high = np.full(len(idx), 1.1010, dtype=float)
    low = np.full(len(idx), 1.0990, dtype=float)
    for i, ts in enumerate(idx):
        if ts < break_h1:
            close[i] = 1.1000
            high[i] = 1.1010
            low[i] = 1.0990
        elif ts < break_h1 + pd.Timedelta(hours=1):
            close[i] = 1.1050
            high[i] = 1.1055
            low[i] = 1.1030
        else:
            close[i] = 1.1040
            high[i] = 1.1045
            low[i] = 1.1035

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        }
    )

    cfg = load_strategy_config("strategy_donchian_atr.yaml")
    cfg["news"]["enabled"] = False
    cfg["signal"]["min_htf_bars"] = 25
    cfg["signal"]["donchian_period"] = 20
    cfg["universe"] = ["EURUSD"]
    costs = {
        "pip_size": {"EURUSD": 0.0001},
        "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7},
        "pairs": {},
    }
    sigs = generate_signals({"EURUSD": df}, cfg, costs)
    longs = [s for s in sigs if s.side == 1]
    assert len(longs) >= 1
    s = longs[0]
    assert s.stop < s.entry
    assert s.take_profit > s.entry
    assert "donchian_atr" in s.reason
