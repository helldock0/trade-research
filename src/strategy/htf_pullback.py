"""HTF trend + M15 pullback — H4 EMA bias, M15 EMA reclaim entries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.features.trend import htf_bias_on_ltf, m15_indicators
from src.strategy.cssm import Signal


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Generate HTF-bias M15 pullback signals with no lookahead."""
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))

    htf_rule = str(sig_cfg.get("htf_rule", "4h"))
    htf_ema = int(sig_cfg.get("htf_ema_period", 200))
    entry_ema = int(sig_cfg.get("entry_ema_period", 21))
    atr_period = int(sig_cfg.get("atr_period", 14))
    atr_mult = float(sig_cfg.get("atr_stop_mult", 1.5))
    tp_r = float(sig_cfg.get("take_profit_r", 2.0))
    min_htf = int(sig_cfg.get("min_htf_bars", 220))
    enable_ny = bool(sig_cfg.get("enable_ny_session", True))
    allow_weekend_hold = bool(sig_cfg.get("allow_weekend_hold", False))

    prepared: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        x = m15_indicators(df, entry_ema_period=entry_ema, atr_period=atr_period)
        bias = htf_bias_on_ltf(x, htf_rule=htf_rule, htf_ema_period=htf_ema)
        # Warmup: no bias until ~min_htf completed H4 bars (~16 M15 bars each)
        warmup = min_htf * 16
        if len(bias) > warmup:
            bias = bias.copy()
            bias.iloc[:warmup] = 0
        x["bias"] = bias
        prepared[sym] = x

    any_sym = next(iter(prepared))
    all_days = sorted({ts.normalize() for ts in prepared[any_sym].index if ts.dayofweek < 5})

    signals: list[Signal] = []
    traded_day: set[tuple[str, pd.Timestamp]] = set()

    windows: list[tuple[str, Any, Any]] = [
        ("london", sessions["london_start"], sessions["london_end"]),
    ]
    if enable_ny:
        windows.append(("ny", sessions["ny_start"], sessions["ny_end"]))

    for day in all_days:
        day = pd.Timestamp(day)
        if day.tzinfo is None:
            day = day.tz_localize("UTC")

        for sym, bars in prepared.items():
            day_key = (sym, day.normalize())
            if day_key in traded_day:
                continue
            day_bars = bars.loc[day : day + pd.Timedelta(hours=23, minutes=59)]
            if len(day_bars) < 3:
                continue

            for session_name, w_start, w_end in windows:
                if day_key in traded_day:
                    break
                win_mask = session_mask(day_bars.index, w_start, w_end)
                window = day_bars.loc[win_mask]
                if len(window) < 2:
                    continue

                times = list(window.index)
                for i in range(1, len(times)):
                    ts = times[i]
                    prev_ts = times[i - 1]
                    if (
                        not allow_weekend_hold
                        and "friday_no_new_entries" in sessions
                        and is_friday(ts)
                        and ts.time() >= sessions["friday_no_new_entries"]
                    ):
                        break
                    if news_cfg.get("enabled", True) and is_news_blackout(
                        ts,
                        sym,
                        news_df if news_df is not None else pd.DataFrame(),
                        int(news_cfg.get("blackout_minutes", 60)),
                    ):
                        continue

                    bar = window.loc[ts]
                    prev = window.loc[prev_ts]
                    bias = int(bar["bias"])
                    if bias == 0:
                        continue
                    ema_v = float(bar["ema_entry"])
                    atr_v = float(bar["atr"])
                    if not np.isfinite(ema_v) or not np.isfinite(atr_v) or atr_v <= 0:
                        continue

                    close = float(bar["close"])
                    prev_close = float(prev["close"])
                    prev_ema = float(prev["ema_entry"])
                    if not np.isfinite(prev_ema):
                        continue

                    # Pullback reclaim: prior bar on/through EMA, current close back with trend
                    long_setup = (
                        bias > 0
                        and prev_close <= prev_ema
                        and close > ema_v
                    )
                    short_setup = (
                        bias < 0
                        and prev_close >= prev_ema
                        and close < ema_v
                    )
                    if not (long_setup or short_setup):
                        continue

                    side = 1 if long_setup else -1
                    entry = close
                    stop_dist = atr_mult * atr_v
                    if side > 0:
                        stop = entry - stop_dist
                        risk = entry - stop
                        tp = entry + tp_r * risk
                    else:
                        stop = entry + stop_dist
                        risk = stop - entry
                        tp = entry - tp_r * risk
                    if risk <= 0:
                        continue

                    signals.append(
                        Signal(
                            timestamp=ts,
                            symbol=sym,
                            side=side,
                            entry=entry,
                            stop=stop,
                            take_profit=tp,
                            risk_price=risk,
                            session=session_name,
                            range_high=float(bar["high"]),
                            range_low=float(bar["low"]),
                            reason=f"{session_name}_htf_pullback",
                        )
                    )
                    traded_day.add(day_key)
                    break

    signals.sort(key=lambda s: s.timestamp)
    return signals
