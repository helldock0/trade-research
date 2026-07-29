"""XAUUSD HTF pullback registry / config tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_xau_htf_pullback_config_and_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("xau_htf_pullback") == "strategy_xau_htf_pullback.yaml"
    assert default_config_name("xau") == "strategy_xau_htf_pullback.yaml"
    cfg = load_strategy_config("strategy_xau_htf_pullback.yaml")
    assert strategy_id(cfg) == "xau_htf_pullback"
    assert cfg["universe"] == ["XAUUSD"]
    assert cfg["signal"]["htf_ema_period"] == 200
    assert cfg["signal"]["entry_ema_period"] == 21
    assert cfg["signal"]["exit_mode"] == "swing"
    assert cfg["signal"]["take_profit_r"] == 2.0
    assert resolve_signal_fn(cfg).__module__.endswith("xau_htf_pullback")
