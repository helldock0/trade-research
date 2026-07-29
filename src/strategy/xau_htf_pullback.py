"""XAUUSD-only HTF pullback — thin wrapper over htf_pullback signal logic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategy.htf_pullback import generate_signals as _htf_generate_signals


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list:
    """Delegate to HTF pullback; universe is expected to be XAUUSD-only via config."""
    return _htf_generate_signals(bars_by_symbol, strategy_cfg, costs_cfg, news_df=news_df)
