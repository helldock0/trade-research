"""Fibonacci retracement swing — H4 EMA tap + 0.71 fib + reversal candle."""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index, resample_ohlc
from src.features.candles import (
    ema_slope_down,
    ema_slope_up,
    htf_signal_m15_ts,
    is_bearish_reversal,
    is_bullish_reversal,
    williams_fractal_highs,
    williams_fractal_lows,
)
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.features.trend import ema
from src.strategy.cssm import Signal


def _pip_size(symbol: str, costs_cfg: dict) -> float:
    return float(costs_cfg.get("pip_size", {}).get(symbol, 0.0001 if "JPY" not in symbol else 0.01))


def _session_name_for_ts(
    ts: pd.Timestamp,
    windows: list[tuple[str, time, time]],
) -> str:
    for sname, w_start, w_end in windows:
        sm = session_mask(pd.DatetimeIndex([ts]), w_start, w_end)
        if bool(sm.iloc[0]):
            return sname
    return "swing"


def _map_to_m15(m15_index: pd.DatetimeIndex, htf_ts: pd.Timestamp, htf_delta: pd.Timedelta) -> pd.Timestamp | None:
    sig_ts = htf_signal_m15_ts(htf_ts, htf_delta)
    if sig_ts in m15_index:
        return pd.Timestamp(sig_ts)
    end = htf_ts + htf_delta
    pos = m15_index.searchsorted(end, side="left") - 1
    if pos < 0:
        return None
    sig_ts = pd.Timestamp(m15_index[pos])
    return None if sig_ts < htf_ts else sig_ts


def _last_confirmed_fractal(series: pd.Series, up_to: int) -> tuple[int, float] | None:
    """Most recent non-NaN fractal value at or before index `up_to`."""
    for j in range(up_to, -1, -1):
        v = series.iloc[j]
        if np.isfinite(v):
            return j, float(v)
    return None


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """H4 fib swing proxy.

    Conventions (YAML):
    - Trend = 200 EMA slope over slope_lookback bars (not flat)
    - Tap = low/high within tap_atr_mult * ATR of 50 or 200 EMA
    - Fib for longs: from last fractal high (0) to last prior fractal low (1);
      require |fib_0.71 - tapped_ema| / price <= fib_align_pct
    - Entry: bullish/bearish reversal closes back beyond EMA and 0.71
    - SL = extreme of tap bar (min 10 pips); TP = take_profit_r
    """
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    news_frame = news_df if news_df is not None else pd.DataFrame()
    news_on = bool(news_cfg.get("enabled", True))
    blackout_min = int(news_cfg.get("blackout_minutes", 60))

    htf_rule = str(sig_cfg.get("htf_rule", "4h"))
    ema_slow_n = int(sig_cfg.get("ema_slow", 200))
    ema_fast_n = int(sig_cfg.get("ema_fast", 50))
    slope_lb = int(sig_cfg.get("slope_lookback", 5))
    fib_level = float(sig_cfg.get("fib_level", 0.71))
    align_pct = float(sig_cfg.get("fib_align_pct", 0.0025))
    tap_atr_mult = float(sig_cfg.get("tap_atr_mult", 0.35))
    atr_period = int(sig_cfg.get("atr_period", 14))
    min_stop_pips = float(sig_cfg.get("min_stop_pips", 10.0))
    tp_r = float(sig_cfg.get("take_profit_r", 2.0))
    min_htf = int(sig_cfg.get("min_htf_bars", 220))
    allow_weekend_hold = bool(sig_cfg.get("allow_weekend_hold", True))
    htf_delta = pd.Timedelta(htf_rule)
    friday_cut = sessions.get("friday_no_new_entries")

    session_windows: list[tuple[str, time, time]] = [
        ("london", sessions["london_start"], sessions["london_end"]),
        ("ny", sessions["ny_start"], sessions["ny_end"]),
    ]

    signals: list[Signal] = []

    for sym, df in bars_by_symbol.items():
        m15 = ensure_utc_index(df)
        m15_index = m15.index
        htf = resample_ohlc(m15.reset_index(), htf_rule)
        if len(htf) < min_htf + 10:
            continue
        pip = _pip_size(sym, costs_cfg)
        min_stop = min_stop_pips * pip

        htf = htf.copy()
        htf["ema200"] = ema(htf["close"], ema_slow_n)
        htf["ema50"] = ema(htf["close"], ema_fast_n)
        tr = pd.concat(
            [
                (htf["high"] - htf["low"]).abs(),
                (htf["high"] - htf["close"].shift(1)).abs(),
                (htf["low"] - htf["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        htf["atr"] = tr.rolling(atr_period, min_periods=atr_period).mean()
        htf["slope_up"] = ema_slope_up(htf["ema200"], slope_lb)
        htf["slope_down"] = ema_slope_down(htf["ema200"], slope_lb)
        # Fractal values stamped on confirmation bar; also keep pivot index via search
        frac_hi = williams_fractal_highs(htf["high"])
        frac_lo = williams_fractal_lows(htf["low"])

        # Track last tap state
        tap: dict[str, Any] | None = None

        for i in range(max(min_htf, ema_slow_n + slope_lb + 5), len(htf)):
            row = htf.iloc[i]
            prev = htf.iloc[i - 1]
            htf_ts = pd.Timestamp(htf.index[i])
            e200 = float(row["ema200"])
            e50 = float(row["ema50"])
            atr_v = float(row["atr"])
            if not np.isfinite(e200) or not np.isfinite(e50) or not np.isfinite(atr_v) or atr_v <= 0:
                tap = None
                continue

            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
            tap_tol = tap_atr_mult * atr_v

            # Detect EMA tap in trend
            if bool(row["slope_up"]):
                for ema_v, which in ((e50, "ema50"), (e200, "ema200")):
                    if lo <= ema_v + tap_tol and hi >= ema_v - tap_tol:
                        tap = {
                            "side": 1,
                            "ema": ema_v,
                            "which": which,
                            "tap_i": i,
                            "tap_low": lo,
                            "tap_high": hi,
                        }
                        break
            elif bool(row["slope_down"]):
                for ema_v, which in ((e50, "ema50"), (e200, "ema200")):
                    if hi >= ema_v - tap_tol and lo <= ema_v + tap_tol:
                        tap = {
                            "side": -1,
                            "ema": ema_v,
                            "which": which,
                            "tap_i": i,
                            "tap_low": lo,
                            "tap_high": hi,
                        }
                        break
            else:
                tap = None
                continue

            if tap is None or int(tap["tap_i"]) > i:
                continue
            # Entry only on bars after tap (same bar OK if reversal)
            side = int(tap["side"])
            ema_ref = float(tap["ema"])

            # Fractals known as of prior confirmation (use series up to i)
            if side > 0:
                fh = _last_confirmed_fractal(frac_hi, i)
                fl = _last_confirmed_fractal(frac_lo, i)
                if fh is None or fl is None:
                    continue
                # Prefer fractal high after fractal low (rally then pullback)
                swing_high = fh[1]
                swing_low = fl[1]
                if swing_high <= swing_low:
                    continue
                fib71 = swing_high - fib_level * (swing_high - swing_low)
                # Invalidate if wick beyond fib origin (swing high)
                if hi > swing_high:
                    tap = None
                    continue
                if abs(fib71 - ema_ref) / max(abs(cl), 1e-12) > align_pct:
                    continue
                if not is_bullish_reversal(prev, row):
                    continue
                if not (cl >= ema_ref and cl >= fib71):
                    continue
                stop = min(float(tap["tap_low"]), lo) - 0.0
                if cl - stop < min_stop:
                    stop = cl - min_stop
            else:
                fh = _last_confirmed_fractal(frac_hi, i)
                fl = _last_confirmed_fractal(frac_lo, i)
                if fh is None or fl is None:
                    continue
                swing_high = fh[1]
                swing_low = fl[1]
                if swing_high <= swing_low:
                    continue
                fib71 = swing_low + fib_level * (swing_high - swing_low)
                if lo < swing_low:
                    tap = None
                    continue
                if abs(fib71 - ema_ref) / max(abs(cl), 1e-12) > align_pct:
                    continue
                if not is_bearish_reversal(prev, row):
                    continue
                if not (cl <= ema_ref and cl <= fib71):
                    continue
                stop = max(float(tap["tap_high"]), hi)
                if stop - cl < min_stop:
                    stop = cl + min_stop

            sig_ts = _map_to_m15(m15_index, htf_ts, htf_delta)
            if sig_ts is None:
                continue
            if (
                not allow_weekend_hold
                and friday_cut is not None
                and is_friday(sig_ts)
                and sig_ts.time() >= friday_cut
            ):
                continue
            if news_on and is_news_blackout(sig_ts, sym, news_frame, blackout_min):
                continue

            entry = cl
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            tp = entry + tp_r * risk if side > 0 else entry - tp_r * risk
            signals.append(
                Signal(
                    timestamp=sig_ts,
                    symbol=sym,
                    side=side,
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    risk_price=risk,
                    session=_session_name_for_ts(sig_ts, session_windows),
                    range_high=swing_high,
                    range_low=swing_low,
                    reason=f"fib_swing_{tap['which']}",
                )
            )
            tap = None

    signals.sort(key=lambda s: s.timestamp)
    return signals
