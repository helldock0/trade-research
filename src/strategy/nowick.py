"""Bard FX Nowick (no-wick) compensation play — M15 proxy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index
from src.features.candles import is_nowick_bear, is_nowick_bull
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.features.trend import ema
from src.strategy.cssm import Signal


def _pip_size(symbol: str, costs_cfg: dict) -> float:
    return float(costs_cfg.get("pip_size", {}).get(symbol, 0.0001 if "JPY" not in symbol else 0.01))


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Nowick in trend → limit retest within N bars → 1R target.

    Conventions (documented in YAML):
    - Trend = close vs EMA(trend_ema_period)
    - Nowick tolerance = nowick_wick_tol_pips
    - Structure SL = min/max low/high of structure_lookback bars before Nowick ± buffer
    - Session window = london_start..london_end (PDF: ~07:00–13:00 UTC)
    """
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    news_frame = news_df if news_df is not None else pd.DataFrame()
    news_on = bool(news_cfg.get("enabled", True))
    blackout_min = int(news_cfg.get("blackout_minutes", 60))

    trend_ema_n = int(sig_cfg.get("trend_ema_period", 50))
    retest_bars = int(sig_cfg.get("retest_max_bars", 9))
    structure_lb = int(sig_cfg.get("structure_lookback", 5))
    stop_buffer_pips = float(sig_cfg.get("stop_buffer_pips", 3.0))
    wick_tol_pips = float(sig_cfg.get("nowick_wick_tol_pips", 0.2))
    tp_r = float(sig_cfg.get("take_profit_r", 1.0))
    allow_weekend_hold = bool(sig_cfg.get("allow_weekend_hold", False))
    friday_cut = sessions.get("friday_no_new_entries")
    win_start, win_end = sessions["london_start"], sessions["london_end"]

    signals: list[Signal] = []

    for sym, df in bars_by_symbol.items():
        bars = ensure_utc_index(df)
        if len(bars) < trend_ema_n + structure_lb + retest_bars + 5:
            continue
        pip = _pip_size(sym, costs_cfg)
        tol = wick_tol_pips * pip
        buffer = stop_buffer_pips * pip
        bars = bars.copy()
        bars["ema"] = ema(bars["close"], trend_ema_n)

        pending: dict[str, Any] | None = None

        for i in range(trend_ema_n + structure_lb, len(bars)):
            ts = pd.Timestamp(bars.index[i])
            row = bars.iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            ema_v = float(row["ema"])
            if not np.isfinite(ema_v):
                continue

            in_session = bool(session_mask(pd.DatetimeIndex([ts]), win_start, win_end).iloc[0])

            if pending is not None:
                age = i - int(pending["setup_i"])
                if age > retest_bars:
                    pending = None
                else:
                    side = int(pending["side"])
                    limit = float(pending["limit"])
                    stop = float(pending["stop"])
                    # Invalidate if new extreme beyond Nowick structure before retest
                    if side > 0 and h > float(pending["nowick_high"]):
                        pending = None
                    elif side < 0 and l < float(pending["nowick_low"]):
                        pending = None
                    elif (side > 0 and l <= stop) or (side < 0 and h >= stop):
                        pending = None
                    else:
                        touched = (side > 0 and l <= limit) or (side < 0 and h >= limit)
                        if touched and in_session:
                            if not (
                                not allow_weekend_hold
                                and friday_cut is not None
                                and is_friday(ts)
                                and ts.time() >= friday_cut
                            ) and not (
                                news_on and is_news_blackout(ts, sym, news_frame, blackout_min)
                            ):
                                entry = limit
                                risk = abs(entry - stop)
                                if risk > 0:
                                    tp = entry + tp_r * risk if side > 0 else entry - tp_r * risk
                                    signals.append(
                                        Signal(
                                            timestamp=ts,
                                            symbol=sym,
                                            side=side,
                                            entry=entry,
                                            stop=stop,
                                            take_profit=tp,
                                            risk_price=risk,
                                            session="london",
                                            range_high=float(pending["nowick_high"]),
                                            range_low=float(pending["nowick_low"]),
                                            reason="nowick_retest",
                                        )
                                    )
                            pending = None
                            continue

            if pending is not None or not in_session:
                continue
            if (
                not allow_weekend_hold
                and friday_cut is not None
                and is_friday(ts)
                and ts.time() >= friday_cut
            ):
                continue
            if news_on and is_news_blackout(ts, sym, news_frame, blackout_min):
                continue

            uptrend = c > ema_v
            downtrend = c < ema_v
            bull = uptrend and is_nowick_bull(o, h, l, c, tol)
            bear = downtrend and is_nowick_bear(o, h, l, c, tol)
            if not bull and not bear:
                continue

            hist = bars.iloc[i - structure_lb : i]
            if bull:
                side = 1
                struct = float(hist["low"].min())
                stop = struct - buffer
                # Limit = Nowick body (open/close zone mid → open for bullish Nowick)
                limit = o
            else:
                side = -1
                struct = float(hist["high"].max())
                stop = struct + buffer
                limit = o

            if abs(c - stop) <= 0:
                continue
            pending = {
                "setup_i": i,
                "side": side,
                "limit": limit,
                "stop": stop,
                "nowick_high": h,
                "nowick_low": l,
            }

    signals.sort(key=lambda s: s.timestamp)
    return signals
