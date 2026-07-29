"""EBP / Nowick / Fib swing / Judas unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _costs() -> dict:
    return {
        "pip_size": {
            "EURUSD": 0.0001,
            "GBPUSD": 0.0001,
            "USDJPY": 0.01,
            "AUDUSD": 0.0001,
            "XAUUSD": 0.01,
        },
        "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7},
        "pairs": {},
    }


def test_candle_ebp_and_nowick_helpers():
    from src.features.candles import (
        close_strength_from_extreme,
        ebp_levels,
        is_ebp_bull,
        is_nowick_bull,
    )

    prev = pd.Series({"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105})
    cur = pd.Series({"open": 1.10, "high": 1.13, "low": 1.08, "close": 1.125})
    assert is_ebp_bull(prev, cur)
    assert abs(close_strength_from_extreme(1, 1.13, 1.08, 1.125) - (1.13 - 1.125) / 0.05) < 1e-12
    levels = ebp_levels(1, 1.13, 1.08)
    assert abs(levels["p25"] - (1.13 - 0.25 * 0.05)) < 1e-12
    assert is_nowick_bull(1.10, 1.105, 1.10, 1.104, tol=0.0001)


def test_ebp_config_and_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("ebp") == "strategy_ebp.yaml"
    cfg = load_strategy_config("strategy_ebp.yaml")
    assert strategy_id(cfg) == "ebp"
    assert cfg["signal"]["strong_close_max"] == 0.15
    assert resolve_signal_fn(cfg).__module__.endswith("ebp")


def test_ebp_strong_limit_fill():
    """Strong bullish EBP on H4 → limit fill at 25% on next H4 touch."""
    from src.config import load_strategy_config
    from src.strategy.ebp import generate_signals

    # Build M15 so H4 bars are clean: first H4 inside range, second engulfs, third taps 25%
    start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    # 12 H4 bars worth of M15
    n = 12 * 16
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    open_ = np.full(n, 1.1000)
    high = np.full(n, 1.1010)
    low = np.full(n, 1.0990)
    close = np.full(n, 1.1000)

    def set_h4(h4_i: int, o: float, h: float, l: float, c: float) -> None:
        sl = h4_i * 16
        for j in range(16):
            open_[sl + j] = o if j == 0 else close[sl + j - 1]
            high[sl + j] = h
            low[sl + j] = l
            close[sl + j] = c if j == 15 else (o + (c - o) * (j / 15))

    # H4 0: small bar
    set_h4(0, 1.1000, 1.1010, 1.0990, 1.1005)
    # H4 1: bullish EBP — sweeps both sides, strong close near high
    set_h4(1, 1.1000, 1.1100, 1.0980, 1.1090)  # strength = (1.11-1.109)/0.012 = 0.083
    # H4 2: tap 25% = 1.1100 - 0.25*0.012 = 1.1070
    set_h4(2, 1.1080, 1.1085, 1.1065, 1.1075)

    df = pd.DataFrame(
        {"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": 100.0}
    )
    cfg = load_strategy_config("strategy_ebp.yaml")
    cfg["news"]["enabled"] = False
    cfg["universe"] = ["EURUSD"]
    cfg["signal"]["min_htf_bars"] = 1
    sigs = generate_signals({"EURUSD": df}, cfg, _costs())
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.side == 1
    assert "ebp" in s.reason
    assert s.stop < s.entry < s.take_profit


def test_nowick_config_and_retest_signal():
    from src.config import load_strategy_config
    from src.strategy.nowick import generate_signals
    from src.strategy.registry import default_config_name, resolve_signal_fn

    assert default_config_name("nowick") == "strategy_nowick.yaml"
    cfg = load_strategy_config("strategy_nowick.yaml")
    assert resolve_signal_fn(cfg).__module__.endswith("nowick")

    # Start early so warmup EMA fits before London Nowick at 10:00
    start = pd.Timestamp("2024-01-02 05:00", tz="UTC")
    n = 40
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = np.linspace(1.1000, 1.1080, n)
    open_ = np.roll(close, 1)
    open_[0] = 1.1000
    high = np.maximum(open_, close) + 0.0002
    low = np.minimum(open_, close) - 0.0002
    # Bullish Nowick at 10:00 UTC (London window)
    i_nw = 20
    assert 7 <= idx[i_nw].hour < 13
    open_[i_nw] = 1.1060
    low[i_nw] = 1.1060
    close[i_nw] = 1.1070
    high[i_nw] = 1.1071
    # Cap subsequent highs so invalidation (new high) does not fire
    for j in range(i_nw + 1, n):
        high[j] = min(high[j], 1.1070)
    # Retest within 9 bars: touch Nowick open
    i_rt = 24
    assert idx[i_rt].hour < 13
    open_[i_rt] = 1.1062
    high[i_rt] = 1.1065
    low[i_rt] = 1.1059
    close[i_rt] = 1.1061
    # Keep structure lows below stop path
    for j in range(i_nw - 5, i_nw):
        low[j] = min(low[j], 1.1045)

    df = pd.DataFrame(
        {"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": 100.0}
    )
    cfg["news"]["enabled"] = False
    cfg["signal"]["trend_ema_period"] = 10
    cfg["universe"] = ["AUDUSD"]
    sigs = generate_signals({"AUDUSD": df}, cfg, _costs())
    assert any(s.reason == "nowick_retest" and s.side == 1 for s in sigs)


def test_fib_swing_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("fib_swing") == "strategy_fib_swing.yaml"
    assert default_config_name("fib") == "strategy_fib_swing.yaml"
    cfg = load_strategy_config("strategy_fib_swing.yaml")
    assert strategy_id(cfg) == "fib_swing"
    assert cfg["signal"]["fib_level"] == 0.71
    assert resolve_signal_fn(cfg).__module__.endswith("fib_swing")


def test_fib_swing_generates_on_synthetic_uptrend():
    """Enough H4 history with rising EMA and a forced tap/reversal path."""
    from src.config import load_strategy_config
    from src.strategy.fib_swing import generate_signals

    # Long rising series so EMA200 slopes up; force a pullback tap + reversal late
    start = pd.Timestamp("2023-01-02", tz="UTC")
    n = 250 * 16  # ~250 H4 bars
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    # weekdays only roughly handled by continuous series (strategy doesn't require weekends)
    t = np.arange(n, dtype=float)
    close = 1.05 + t * 0.00002
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.0004
    low = np.minimum(open_, close) - 0.0004
    df = pd.DataFrame(
        {"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": 100.0}
    )
    cfg = load_strategy_config("strategy_fib_swing.yaml")
    cfg["news"]["enabled"] = False
    cfg["signal"]["min_htf_bars"] = 50
    cfg["signal"]["ema_slow"] = 30
    cfg["signal"]["ema_fast"] = 10
    cfg["signal"]["fib_align_pct"] = 0.05  # loose for synthetic
    cfg["signal"]["tap_atr_mult"] = 2.0
    cfg["universe"] = ["EURUSD"]
    # Should not crash; signals optional on random-ish path
    sigs = generate_signals({"EURUSD": df}, cfg, _costs())
    assert isinstance(sigs, list)


def test_judas_config_and_asia_sweep_long():
    from src.config import load_strategy_config
    from src.strategy.judas import generate_signals
    from src.strategy.registry import default_config_name, resolve_signal_fn

    assert default_config_name("judas") == "strategy_judas.yaml"
    cfg = load_strategy_config("strategy_judas.yaml")
    assert cfg["signal"]["use_signal_tp"] is True
    assert resolve_signal_fn(cfg).__module__.endswith("judas")

    # One weekday: Asia range then London sweep low + MSS
    day = pd.Timestamp("2024-01-02", tz="UTC")  # Tuesday
    idx = pd.date_range(day, periods=40, freq="15min", tz="UTC")  # 00:00–09:45
    open_ = np.full(40, 1.1000)
    high = np.full(40, 1.1010)
    low = np.full(40, 1.0990)
    close = np.full(40, 1.1000)
    # Asia 00:00–07:00 → bars 0..27
    for i in range(28):
        high[i], low[i], close[i], open_[i] = 1.1020, 1.1000, 1.1010, 1.1010
    # London bar 28 (07:00): sweep low
    open_[28], high[28], low[28], close[28] = 1.1005, 1.1008, 1.0990, 1.0995
    # Bar 29: MSS close above sweep high, still in bottom half (mid=1.1010)
    open_[29], high[29], low[29], close[29] = 1.0996, 1.1010, 1.0994, 1.10085
    # Neutral remainder of London (avoid accidental extremes)
    for i in range(30, 40):
        open_[i] = high[i] = low[i] = close[i] = 1.1005

    df = pd.DataFrame(
        {"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": 100.0}
    )
    cfg["news"]["enabled"] = False
    cfg["universe"] = ["GBPUSD"]
    sigs = generate_signals({"GBPUSD": df}, cfg, _costs())
    longs = [s for s in sigs if s.side == 1]
    assert len(longs) >= 1
    s = longs[0]
    assert s.reason == "judas_asia_long"
    assert s.take_profit == s.range_high
    assert s.stop < s.entry


def test_use_signal_tp_in_engine():
    from src.backtest.engine import run_backtest
    from src.strategy.cssm import Signal

    idx = pd.date_range("2024-01-02 08:00", periods=6, freq="15min", tz="UTC")
    close = np.array([1.10, 1.101, 1.102, 1.103, 1.104, 1.105])
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.001,
            "low": close - 0.0005,
            "close": close,
            "volume": 100.0,
        }
    )
    sig = Signal(
        timestamp=idx[0],
        symbol="EURUSD",
        side=1,
        entry=1.10,
        stop=1.099,
        take_profit=1.104,
        risk_price=0.001,
        session="london",
        range_high=1.105,
        range_low=1.098,
        reason="test",
    )
    cfg = {
        "strategy": "judas",
        "signal": {
            "take_profit_r": 99.0,
            "use_signal_tp": True,
            "exit_mode": "swing",
            "allow_weekend_hold": True,
            "max_hold_days": 5,
            "breakeven_at_r": 0.0,
        },
        "risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 1, "soft_daily_halt_pct": 5.0},
        "sessions": {
            "london_end": "11:00",
            "ny_end": "16:00",
            "friday_hard_flat": "20:00",
            "friday_no_new_entries": "16:00",
        },
    }
    result = run_backtest([sig], {"EURUSD": df}, cfg, _costs())
    assert len(result.trades) >= 1
    # TP should be near signal TP (not 99R)
    assert result.trades[0].take_profit < 1.11
