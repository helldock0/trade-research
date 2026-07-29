"""Donchian channel helpers — prior-bar channel only (no lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.bars import atr


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Prior completed N-bar Donchian high/low (shifted by 1).

    At bar i, upper/lower use highs/lows from i-period .. i-1 only.
    """
    if period < 1:
        raise ValueError("donchian period must be >= 1")
    upper = high.rolling(period, min_periods=period).max().shift(1)
    lower = low.rolling(period, min_periods=period).min().shift(1)
    return upper, lower


def donchian_break_side(
    close: float,
    upper: float,
    lower: float,
) -> int:
    """Return +1 long break, -1 short break, 0 otherwise."""
    if not (np.isfinite(close) and np.isfinite(upper) and np.isfinite(lower)):
        return 0
    if close > upper:
        return 1
    if close < lower:
        return -1
    return 0


def h1_donchian_frame(
    h1: pd.DataFrame,
    period: int = 20,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Annotate H1 OHLC with prior Donchian channel, ATR, and break side."""
    out = h1.copy()
    upper, lower = donchian_channel(out["high"], out["low"], period=period)
    out["don_upper"] = upper
    out["don_lower"] = lower
    out["atr"] = atr(out["high"], out["low"], out["close"], period=atr_period)
    sides = [
        donchian_break_side(float(c), float(u) if np.isfinite(u) else np.nan, float(l) if np.isfinite(l) else np.nan)
        for c, u, l in zip(out["close"], out["don_upper"], out["don_lower"])
    ]
    out["break_side"] = sides
    return out
