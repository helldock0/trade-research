"""Higher-timeframe trend helpers (no lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index, resample_ohlc, true_range


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def wilder_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder ADX; NaN until enough bars."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = true_range(high, low, close)

    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_w
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / di_sum).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def htf_bias_on_ltf(
    m15: pd.DataFrame,
    htf_rule: str = "4h",
    htf_ema_period: int = 200,
) -> pd.Series:
    """Map completed HTF bias (+1/-1/0) onto M15 index without lookahead.

    HTF bar at timestamp T covers [T, T+rule). Its OHLC is only known at T+rule,
    so we shift(1) before forward-filling onto M15.
    """
    x = ensure_utc_index(m15)
    htf = resample_ohlc(x.reset_index(), htf_rule)
    if htf.empty:
        return pd.Series(0, index=x.index, dtype=int)
    htf["ema"] = ema(htf["close"], htf_ema_period)
    bias = pd.Series(0, index=htf.index, dtype=int)
    bias = bias.mask(htf["close"] > htf["ema"], 1)
    bias = bias.mask(htf["close"] < htf["ema"], -1)
    # Only prior completed HTF bar
    bias_shifted = bias.shift(1)
    return bias_shifted.reindex(x.index, method="ffill").fillna(0).astype(int)


def htf_adx_on_ltf(
    m15: pd.DataFrame,
    htf_rule: str = "4h",
    adx_period: int = 14,
) -> pd.Series:
    """Map completed HTF Wilder ADX onto M15 index without lookahead.

    Same shift(1) + ffill contract as htf_bias_on_ltf.
    """
    x = ensure_utc_index(m15)
    htf = resample_ohlc(x.reset_index(), htf_rule)
    if htf.empty:
        return pd.Series(np.nan, index=x.index, dtype=float)
    adx = wilder_adx(htf["high"], htf["low"], htf["close"], period=adx_period)
    adx_shifted = adx.shift(1)
    return adx_shifted.reindex(x.index, method="ffill")


def m15_indicators(
    m15: pd.DataFrame,
    entry_ema_period: int = 21,
    atr_period: int = 14,
) -> pd.DataFrame:
    x = ensure_utc_index(m15)
    out = x.copy()
    out["ema_entry"] = ema(out["close"], entry_ema_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], period=atr_period)
    return out
