"""Donchian + ATR — H1 channel breakout with ATR stops."""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index, resample_ohlc
from src.features.donchian import h1_donchian_frame
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.strategy.cssm import Signal


def _h1_signal_m15_ts(h1_start: pd.Timestamp, htf_delta: pd.Timedelta) -> pd.Timestamp:
    """Last M15 timestamp inside a completed H1 bar (no lookahead past bar end)."""
    return pd.Timestamp(h1_start) + htf_delta - pd.Timedelta(minutes=15)


def _session_name_for_ts(
    ts: pd.Timestamp,
    windows: list[tuple[str, time, time]],
) -> str:
    for sname, w_start, w_end in windows:
        sm = session_mask(pd.DatetimeIndex([ts]), w_start, w_end)
        if bool(sm.iloc[0]):
            return sname
    return ""


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Generate H1 Donchian breakout signals mapped onto M15 timestamps."""
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    news_frame = news_df if news_df is not None else pd.DataFrame()
    news_on = bool(news_cfg.get("enabled", True))
    blackout_min = int(news_cfg.get("blackout_minutes", 60))

    htf_rule = str(sig_cfg.get("htf_rule", "1h"))
    period = int(sig_cfg.get("donchian_period", 20))
    atr_period = int(sig_cfg.get("atr_period", 14))
    atr_mult = float(sig_cfg.get("atr_stop_mult", 1.5))
    tp_r = float(sig_cfg.get("take_profit_r", 2.0))
    min_htf = int(sig_cfg.get("min_htf_bars", 40))
    enable_ny = bool(sig_cfg.get("enable_ny_session", True))
    allow_weekend_hold = bool(sig_cfg.get("allow_weekend_hold", False))
    htf_delta = pd.Timedelta(htf_rule)

    session_windows: list[tuple[str, time, time]] = [
        ("london", sessions["london_start"], sessions["london_end"]),
    ]
    if enable_ny:
        session_windows.append(("ny", sessions["ny_start"], sessions["ny_end"]))

    friday_cut = sessions.get("friday_no_new_entries")
    signals: list[Signal] = []

    for sym, df in bars_by_symbol.items():
        m15 = ensure_utc_index(df)
        m15_index = m15.index
        h1 = resample_ohlc(m15.reset_index(), htf_rule)
        if len(h1) < min_htf + period + 2:
            continue
        h1x = h1_donchian_frame(h1, period=period, atr_period=atr_period)

        for i in range(len(h1x)):
            if i < min_htf:
                continue
            row = h1x.iloc[i]
            side = int(row["break_side"])
            if side == 0:
                continue
            atr_v = float(row["atr"])
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue

            h1_ts = pd.Timestamp(h1x.index[i])
            # Session filter on H1 bar timestamp (label=left start)
            session_name = _session_name_for_ts(h1_ts, session_windows)
            if not session_name:
                continue

            sig_ts = _h1_signal_m15_ts(h1_ts, htf_delta)
            if sig_ts not in m15_index:
                # Fallback: nearest M15 at or before H1 end
                end = h1_ts + htf_delta
                pos = m15_index.searchsorted(end, side="left") - 1
                if pos < 0:
                    continue
                sig_ts = pd.Timestamp(m15_index[pos])
                if sig_ts < h1_ts:
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

            entry = float(row["close"])
            stop_dist = atr_mult * atr_v
            if side > 0:
                stop = entry - stop_dist
                risk = entry - stop
                tp = entry + tp_r * risk
                z_hi = float(row["don_upper"])
                z_lo = float(row["don_lower"])
            else:
                stop = entry + stop_dist
                risk = stop - entry
                tp = entry - tp_r * risk
                z_hi = float(row["don_upper"])
                z_lo = float(row["don_lower"])
            if risk <= 0:
                continue

            signals.append(
                Signal(
                    timestamp=sig_ts,
                    symbol=sym,
                    side=side,
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    risk_price=risk,
                    session=session_name,
                    range_high=z_hi,
                    range_low=z_lo,
                    reason=f"{session_name}_donchian_atr",
                )
            )

    signals.sort(key=lambda s: s.timestamp)
    return signals
