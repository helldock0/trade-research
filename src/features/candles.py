"""Candle-pattern helpers for PDF strategy proxies (no lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_nowick_bull(open_: float, high: float, low: float, close: float, tol: float) -> bool:
    """Bullish Nowick: no lower wick (open ≈ low), close > open."""
    if close <= open_:
        return False
    return (open_ - low) <= tol and (high - max(open_, close)) >= 0


def is_nowick_bear(open_: float, high: float, low: float, close: float, tol: float) -> bool:
    """Bearish Nowick: no upper wick (open ≈ high), close < open."""
    if close >= open_:
        return False
    return (high - open_) <= tol and (min(open_, close) - low) >= 0


def is_ebp_bull(prev: pd.Series, cur: pd.Series) -> bool:
    """Bullish EBP: sweeps prior H and L, closes above prior body high."""
    if float(cur["low"]) > float(prev["low"]) or float(cur["high"]) < float(prev["high"]):
        return False
    prior_body_high = max(float(prev["open"]), float(prev["close"]))
    return float(cur["close"]) > prior_body_high


def is_ebp_bear(prev: pd.Series, cur: pd.Series) -> bool:
    """Bearish EBP: sweeps prior H and L, closes below prior body low."""
    if float(cur["low"]) > float(prev["low"]) or float(cur["high"]) < float(prev["high"]):
        return False
    prior_body_low = min(float(prev["open"]), float(prev["close"]))
    return float(cur["close"]) < prior_body_low


def close_strength_from_extreme(side: int, high: float, low: float, close: float) -> float:
    """Fraction of range from the extreme in trade direction (0 = strong at extreme)."""
    rng = high - low
    if rng <= 0:
        return 1.0
    if side > 0:
        return (high - close) / rng
    return (close - low) / rng


def ebp_levels(side: int, high: float, low: float) -> dict[str, float]:
    """Fib levels on EBP bar: 0 = extreme in trade direction, 1 = origin."""
    rng = high - low
    if side > 0:
        # Long: 0=high, 1=low (retracement down into bar)
        return {
            "origin": low,
            "p15": high - 0.15 * rng,
            "p25": high - 0.25 * rng,
            "p50": high - 0.50 * rng,
            "p75": high - 0.75 * rng,
            "extreme": high,
        }
    return {
        "origin": high,
        "p15": low + 0.15 * rng,
        "p25": low + 0.25 * rng,
        "p50": low + 0.50 * rng,
        "p75": low + 0.75 * rng,
        "extreme": low,
    }


def williams_fractal_highs(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Fractal highs confirmed `right` bars later; NaN until confirmed."""
    n = len(high)
    out = pd.Series(np.nan, index=high.index, dtype=float)
    for i in range(left, n - right):
        window = high.iloc[i - left : i + right + 1]
        if float(high.iloc[i]) >= float(window.max()) and (window == high.iloc[i]).sum() == 1:
            # Confirm on bar i+right
            out.iloc[i + right] = float(high.iloc[i])
    return out


def williams_fractal_lows(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    n = len(low)
    out = pd.Series(np.nan, index=low.index, dtype=float)
    for i in range(left, n - right):
        window = low.iloc[i - left : i + right + 1]
        if float(low.iloc[i]) <= float(window.min()) and (window == low.iloc[i]).sum() == 1:
            out.iloc[i + right] = float(low.iloc[i])
    return out


def ema_slope_up(ema: pd.Series, lookback: int) -> pd.Series:
    """True when EMA is higher than `lookback` bars ago."""
    prev = ema.shift(lookback)
    return (ema > prev) & ema.notna() & prev.notna()


def ema_slope_down(ema: pd.Series, lookback: int) -> pd.Series:
    prev = ema.shift(lookback)
    return (ema < prev) & ema.notna() & prev.notna()


def is_bullish_reversal(prev: pd.Series, cur: pd.Series) -> bool:
    """Engulfing body or pin bar (lower wick >= body, close in upper half)."""
    o, h, l, c = float(cur["open"]), float(cur["high"]), float(cur["low"]), float(cur["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    engulf = c > o and c >= max(po, pc) and o <= min(po, pc)
    pin = (o - l) >= body and (h - c) <= 0.5 * rng and c >= l + 0.5 * rng
    return bool(engulf or pin)


def is_bearish_reversal(prev: pd.Series, cur: pd.Series) -> bool:
    o, h, l, c = float(cur["open"]), float(cur["high"]), float(cur["low"]), float(cur["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    engulf = c < o and c <= min(po, pc) and o >= max(po, pc)
    pin = (h - o) >= body and (c - l) <= 0.5 * rng and c <= h - 0.5 * rng
    return bool(engulf or pin)


def htf_signal_m15_ts(htf_start: pd.Timestamp, htf_delta: pd.Timedelta) -> pd.Timestamp:
    """Last M15 timestamp inside a completed HTF bar."""
    return pd.Timestamp(htf_start) + htf_delta - pd.Timedelta(minutes=15)
