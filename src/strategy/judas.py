"""ICT Judas Swing (Asia H/L) — M15 proxy of the 5m model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import asian_range_for_day, is_friday, session_mask
from src.strategy.cssm import Signal


def _pip_size(symbol: str, costs_cfg: dict) -> float:
    return float(costs_cfg.get("pip_size", {}).get(symbol, 0.0001 if "JPY" not in symbol else 0.01))


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Asia range sweep + MSS during early London.

    Proxy conventions (YAML / STATUS):
    - Asia = asian_start..asian_end UTC
    - Entry window = london_start..london_end (PDF first ~2.5h London)
    - Sweep = London wick beyond Asia H/L
    - MSS long = close above high of the sweep bar (and above Asia mid filter)
    - MSS short = close below low of the sweep bar
    - Half-range filter: longs only if entry < Asia mid; shorts if entry > mid
    - Invalidate if price extends >50% of Asia width outside the range before entry
    - Trendline liquidity omitted (discretionary)
    - TP = opposite Asia side via use_signal_tp
    """
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    news_frame = news_df if news_df is not None else pd.DataFrame()
    news_on = bool(news_cfg.get("enabled", True))
    blackout_min = int(news_cfg.get("blackout_minutes", 60))

    extension_frac = float(sig_cfg.get("invalidate_extension", 0.5))
    stop_buffer_pips = float(sig_cfg.get("stop_buffer_pips", 1.0))
    min_range_pips = float(sig_cfg.get("min_range_pips", 5.0))
    max_mss_bars = int(sig_cfg.get("max_mss_bars", 8))
    allow_weekend_hold = bool(sig_cfg.get("allow_weekend_hold", False))
    friday_cut = sessions.get("friday_no_new_entries")
    # take_profit_r kept for engine fallback; use_signal_tp preferred
    tp_r = float(sig_cfg.get("take_profit_r", 1.0))

    signals: list[Signal] = []

    for sym, df in bars_by_symbol.items():
        bars = ensure_utc_index(df)
        pip = _pip_size(sym, costs_cfg)
        buffer = stop_buffer_pips * pip
        all_days = sorted({ts.normalize() for ts in bars.index if ts.dayofweek < 5})

        for day in all_days:
            day = pd.Timestamp(day)
            if day.tzinfo is None:
                day = day.tz_localize("UTC")
            day_bars = bars.loc[day : day + pd.Timedelta(hours=23, minutes=59)]
            if day_bars.empty:
                continue
            ar = asian_range_for_day(day_bars, sessions["asian_start"], sessions["asian_end"])
            if ar is None:
                continue
            asia_hi, asia_lo = ar
            width = asia_hi - asia_lo
            if width <= 0 or width / pip < min_range_pips:
                continue
            mid = 0.5 * (asia_hi + asia_lo)
            ext_up = asia_hi + extension_frac * width
            ext_dn = asia_lo - extension_frac * width
            # Float-safe bound: invalidate only clearly beyond the 50% extension
            eps = max(pip * 0.05, 1e-12)

            win_mask = session_mask(day_bars.index, sessions["london_start"], sessions["london_end"])
            window = day_bars.loc[win_mask]
            if len(window) < 2:
                continue

            sweep_side: int | None = None
            sweep_i: int | None = None
            sweep_extreme: float | None = None
            times = list(window.index)

            for i, ts in enumerate(times):
                row = window.loc[ts]
                hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])

                # Hard invalidation of the day
                if hi > ext_up + eps or lo < ext_dn - eps:
                    break

                if (
                    not allow_weekend_hold
                    and friday_cut is not None
                    and is_friday(ts)
                    and ts.time() >= friday_cut
                ):
                    break
                if news_on and is_news_blackout(ts, sym, news_frame, blackout_min):
                    continue

                if sweep_side is None:
                    if lo < asia_lo:
                        sweep_side = 1  # swept lows → look for long MSS
                        sweep_i = i
                        sweep_extreme = lo
                    elif hi > asia_hi:
                        sweep_side = -1
                        sweep_i = i
                        sweep_extreme = hi
                    continue

                assert sweep_i is not None and sweep_extreme is not None
                if i - sweep_i > max_mss_bars:
                    break

                sweep_bar = window.loc[times[sweep_i]]
                if sweep_side > 0:
                    # Update swing low
                    sweep_extreme = min(sweep_extreme, lo)
                    mss = cl > float(sweep_bar["high"])
                    if not mss:
                        continue
                    if cl >= mid:  # must be bottom half
                        continue
                    entry = cl
                    stop = sweep_extreme - buffer
                    tp = asia_hi
                    risk = entry - stop
                    if risk <= 0 or tp <= entry:
                        continue
                    # Store intended R for engine fallback
                    _ = tp_r
                    signals.append(
                        Signal(
                            timestamp=ts,
                            symbol=sym,
                            side=1,
                            entry=entry,
                            stop=stop,
                            take_profit=tp,
                            risk_price=risk,
                            session="london",
                            range_high=asia_hi,
                            range_low=asia_lo,
                            reason="judas_asia_long",
                        )
                    )
                    break
                else:
                    sweep_extreme = max(sweep_extreme, hi)
                    mss = cl < float(sweep_bar["low"])
                    if not mss:
                        continue
                    if cl <= mid:
                        continue
                    entry = cl
                    stop = sweep_extreme + buffer
                    tp = asia_lo
                    risk = stop - entry
                    if risk <= 0 or tp >= entry:
                        continue
                    signals.append(
                        Signal(
                            timestamp=ts,
                            symbol=sym,
                            side=-1,
                            entry=entry,
                            stop=stop,
                            take_profit=tp,
                            risk_price=risk,
                            session="london",
                            range_high=asia_hi,
                            range_low=asia_lo,
                            reason="judas_asia_short",
                        )
                    )
                    break

    signals.sort(key=lambda s: s.timestamp)
    return signals
