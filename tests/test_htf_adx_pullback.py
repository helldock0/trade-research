"""HTF ADX pullback registry / filter / signal tests."""

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
        "pip_size": {"EURUSD": 0.0001},
        "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7},
        "pairs": {},
    }


def _base_cfg(**signal_overrides):
    from src.config import load_strategy_config

    cfg = load_strategy_config("strategy_htf_adx_pullback.yaml")
    cfg["news"]["enabled"] = False
    cfg["universe"] = ["EURUSD"]
    cfg["signal"]["min_htf_bars"] = 15
    cfg["signal"]["htf_ema_period"] = 20
    cfg["signal"]["entry_ema_period"] = 8
    cfg["signal"]["adx_period"] = 5
    cfg["signal"]["atr_period"] = 8
    cfg["signal"]["enable_ny_session"] = False
    cfg["signal"].update(signal_overrides)
    return cfg


def _trending_with_london_pullback(
    *,
    pullback_depth: float,
    reclaim: bool = True,
) -> pd.DataFrame:
    """Strong uptrend then London pullback (and optional reclaim) on last weekday."""
    # ~40 business days of M15 → enough H4 bars for short EMA/ADX
    days = pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")
    rows = []
    price = 1.0500
    signal_day = days[-1]
    for d in days:
        for h in range(0, 24):
            for m in (0, 15, 30, 45):
                ts = d + pd.Timedelta(hours=h, minutes=m)
                # Steady grind higher outside signal pattern
                if d < signal_day:
                    price += 0.00008
                    c = price
                    o, hi, lo = c - 0.00002, c + 0.0001, c - 0.0001
                elif h < 7:
                    price += 0.00005
                    c = price
                    o, hi, lo = c - 0.00002, c + 0.0001, c - 0.0001
                elif h == 7 and m == 0:
                    # Deep (or shallow) pullback bar through EMA
                    c = price - pullback_depth
                    o = price
                    hi = price + 0.00005
                    lo = c - 0.00005
                elif h == 7 and m == 15 and reclaim:
                    # Reclaim above EMA
                    c = price + 0.0003
                    o = price - pullback_depth * 0.5
                    hi = c + 0.00005
                    lo = o - 0.00005
                    price = c
                else:
                    price += 0.00003
                    c = price
                    o, hi, lo = c - 0.00002, c + 0.00008, c - 0.00008
                rows.append(
                    {
                        "timestamp": ts,
                        "open": o,
                        "high": hi,
                        "low": lo,
                        "close": c,
                        "volume": 100.0,
                    }
                )
    return pd.DataFrame(rows)


def test_htf_adx_pullback_config_and_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("htf_adx_pullback") == "strategy_htf_adx_pullback.yaml"
    assert default_config_name("htf_adx") == "strategy_htf_adx_pullback.yaml"
    assert default_config_name("adx_pullback") == "strategy_htf_adx_pullback.yaml"
    cfg = load_strategy_config("strategy_htf_adx_pullback.yaml")
    assert strategy_id(cfg) == "htf_adx_pullback"
    assert cfg["universe"] == ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    assert cfg["signal"]["htf_ema_period"] == 200
    assert cfg["signal"]["entry_ema_period"] == 21
    assert cfg["signal"]["adx_min"] == 25.0
    assert cfg["signal"]["min_pullback_atr"] == 0.25
    assert cfg["signal"]["exit_mode"] == "swing"
    assert cfg["signal"]["take_profit_r"] == 2.0
    assert resolve_signal_fn(cfg).__module__.endswith("htf_adx_pullback")


def test_htf_adx_on_ltf_no_lookahead_shift():
    from src.features.trend import htf_adx_on_ltf

    idx = pd.date_range("2020-01-01", periods=4000, freq="15min", tz="UTC")
    close = np.linspace(1.0, 1.2, len(idx))
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "volume": 1.0,
        }
    )
    adx = htf_adx_on_ltf(df, htf_rule="4h", adx_period=14)
    assert pd.isna(adx.iloc[0])
    # After warmup ADX should be finite on a trending series
    assert adx.iloc[-500:].notna().mean() > 0.9
    assert float(adx.iloc[-1]) > 0


def test_adx_below_threshold_blocks_signal():
    from src.strategy.htf_adx_pullback import generate_signals

    df = _trending_with_london_pullback(pullback_depth=0.004)
    cfg = _base_cfg(adx_min=200.0, min_pullback_atr=0.1)  # impossible ADX floor
    sigs = generate_signals({"EURUSD": df}, cfg, _costs())
    assert sigs == []


def test_shallow_pullback_blocks_signal():
    from src.strategy.htf_adx_pullback import generate_signals

    # Tiny dip — fails min_pullback_atr with high threshold
    df = _trending_with_london_pullback(pullback_depth=0.00005)
    cfg = _base_cfg(adx_min=0.0, min_pullback_atr=5.0)
    sigs = generate_signals({"EURUSD": df}, cfg, _costs())
    assert sigs == []


def test_deep_pullback_reclaim_emits_long():
    from src.strategy.htf_adx_pullback import generate_signals

    df = _trending_with_london_pullback(pullback_depth=0.004)
    cfg = _base_cfg(adx_min=0.0, min_pullback_atr=0.15)
    sigs = generate_signals({"EURUSD": df}, cfg, _costs())
    longs = [s for s in sigs if s.side == 1]
    assert len(longs) >= 1
    s = longs[0]
    assert s.stop < s.entry
    assert s.take_profit > s.entry
    assert "htf_adx_pullback" in s.reason
