"""Session VWAP stretch fade — mean-revert M15 stretches back to session VWAP."""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.features.trend import wilder_adx
from src.strategy.cssm import Signal

# Re-export for tests / callers that import from this module
__all__ = ["generate_signals", "session_vwap", "wilder_adx"]


def session_vwap(tp: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP over the provided session slice (caller resets)."""
    vol = volume.astype(float).copy()
    vol = vol.where(vol > 0, 1.0)
    pv = (tp * vol).cumsum()
    vv = vol.cumsum()
    return pv / vv


def _trade_session_mask(
    index: pd.DatetimeIndex,
    sessions: dict[str, time],
    enable_ny: bool,
) -> pd.Series:
    london = session_mask(index, sessions["london_start"], sessions["london_end"])
    if not enable_ny:
        return london
    ny = session_mask(index, sessions["ny_start"], sessions["ny_end"])
    return london | ny


def _vwap_window_for_day(
    day_bars: pd.DataFrame,
    sessions: dict[str, time],
    reset_key: str,
) -> pd.DataFrame:
    """Bars from VWAP reset through end of calendar day."""
    if reset_key == "overnight_anchor":
        # Session that started previous evening: from prior day anchor to day end.
        # For same-day generation we only have day_bars; use overnight_anchor if present
        # else fall back to london. Caller should pass extended bars when using overnight.
        start = sessions.get("overnight_anchor", sessions["london_start"])
    else:
        start = sessions["london_start"]
    end = time(23, 59)
    mask = session_mask(day_bars.index, start, end)
    return day_bars.loc[mask]


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    if news_df is None:
        news_df = pd.DataFrame()

    atr_period = int(sig_cfg.get("atr_period", 14))
    adx_period = int(sig_cfg.get("adx_period", 14))
    stretch_atr = float(sig_cfg.get("stretch_atr", 1.5))
    atr_stop_mult = float(sig_cfg.get("atr_stop_mult", 1.0))
    min_tp_r = float(sig_cfg.get("take_profit_r", 0.8))
    adx_max = float(sig_cfg.get("adx_max", 25.0))
    enable_ny = bool(sig_cfg.get("enable_ny_session", True))
    reset_key = str(sig_cfg.get("vwap_reset", "london_start")).lower()

    prepared: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        x = ensure_utc_index(df)
        if "volume" not in x.columns:
            x["volume"] = 1.0
        x["atr"] = atr(x["high"], x["low"], x["close"], period=atr_period)
        x["adx"] = wilder_adx(x["high"], x["low"], x["close"], period=adx_period)
        x["tp"] = (x["high"] + x["low"] + x["close"]) / 3.0
        prepared[sym] = x

    any_sym = next(iter(prepared))
    all_days = sorted({ts.normalize() for ts in prepared[any_sym].index if ts.dayofweek < 5})

    signals: list[Signal] = []

    for day in all_days:
        day = pd.Timestamp(day)
        if day.tzinfo is None:
            day = day.tz_localize("UTC")

        for sym, bars in prepared.items():
            if reset_key == "overnight_anchor":
                prev = day - pd.Timedelta(days=1)
                while prev.dayofweek >= 5:
                    prev -= pd.Timedelta(days=1)
                anchor = sessions["overnight_anchor"]
                start_ts = pd.Timestamp(
                    year=prev.year,
                    month=prev.month,
                    day=prev.day,
                    hour=anchor.hour,
                    minute=anchor.minute,
                    tz="UTC",
                )
                end_ts = day + pd.Timedelta(hours=23, minutes=59)
                sess_bars = bars.loc[start_ts:end_ts]
            else:
                day_bars = bars.loc[day : day + pd.Timedelta(hours=23, minutes=59)]
                sess_bars = _vwap_window_for_day(day_bars, sessions, reset_key)

            if len(sess_bars) < max(atr_period, adx_period) // 2 + 2:
                continue

            vwap = session_vwap(sess_bars["tp"], sess_bars["volume"])
            trade_mask = _trade_session_mask(sess_bars.index, sessions, enable_ny)
            candidates = sess_bars.loc[trade_mask]
            if candidates.empty:
                continue

            emitted = False
            for ts, row in candidates.iterrows():
                if emitted:
                    break
                if (
                    "friday_no_new_entries" in sessions
                    and is_friday(ts)
                    and ts.time() >= sessions["friday_no_new_entries"]
                ):
                    continue
                if news_cfg.get("enabled", True) and is_news_blackout(
                    ts, sym, news_df, int(news_cfg.get("blackout_minutes", 60))
                ):
                    continue

                atr_v = float(row["atr"])
                adx_v = float(row["adx"])
                close = float(row["close"])
                vwap_v = float(vwap.loc[ts])
                if not np.isfinite(atr_v) or atr_v <= 0:
                    continue
                if not np.isfinite(adx_v) or adx_v >= adx_max:
                    continue
                if not np.isfinite(vwap_v):
                    continue

                stretch = (close - vwap_v) / atr_v
                if stretch <= -stretch_atr:
                    side = 1  # below VWAP -> long fade
                elif stretch >= stretch_atr:
                    side = -1
                else:
                    continue

                risk = atr_stop_mult * atr_v
                if risk <= 0:
                    continue
                entry = close
                if side > 0:
                    stop = entry - risk
                    tp = vwap_v
                    reward = tp - entry
                else:
                    stop = entry + risk
                    tp = vwap_v
                    reward = entry - tp
                if reward <= 0:
                    continue
                if reward / risk < min_tp_r:
                    continue

                session_name = "london"
                if enable_ny and session_mask(
                    pd.DatetimeIndex([ts]), sessions["ny_start"], sessions["ny_end"]
                ).iloc[0]:
                    session_name = "ny"

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
                        range_high=max(entry, vwap_v),
                        range_low=min(entry, vwap_v),
                        reason="vwap_fade",
                    )
                )
                emitted = True

    signals.sort(key=lambda s: s.timestamp)
    return signals
