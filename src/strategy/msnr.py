"""MSNR — H4 swing zones with M15 rejection entries."""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index, resample_ohlc
from src.features.momentum import parse_session_times
from src.features.msnr import ZoneBook
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.features.trend import htf_bias_on_ltf, m15_indicators
from src.strategy.cssm import Signal


def _session_name_series(
    index: pd.DatetimeIndex,
    windows: list[tuple[str, time, time]],
) -> pd.Series:
    """Map each bar to session name or empty string."""
    out = pd.Series("", index=index, dtype=object)
    for sname, w_start, w_end in windows:
        m = session_mask(index, w_start, w_end)
        out = out.where(~m, sname)
    return out


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Generate MSNR signals with no lookahead."""
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))

    htf_rule = str(sig_cfg.get("htf_rule", "4h"))
    htf_ema = int(sig_cfg.get("htf_ema_period", 200))
    pivot_left = int(sig_cfg.get("pivot_left", 2))
    pivot_right = int(sig_cfg.get("pivot_right", 2))
    max_touches = int(sig_cfg.get("max_touches", 2))
    atr_period = int(sig_cfg.get("atr_period", 14))
    atr_buf = float(sig_cfg.get("atr_stop_buffer_mult", 0.25))
    tp_r = float(sig_cfg.get("take_profit_r", 2.0))
    min_htf = int(sig_cfg.get("min_htf_bars", 220))
    enable_ny = bool(sig_cfg.get("enable_ny_session", True))
    allow_weekend_hold = bool(sig_cfg.get("allow_weekend_hold", False))
    news_on = bool(news_cfg.get("enabled", True))
    blackout_min = int(news_cfg.get("blackout_minutes", 60))
    news_frame = news_df if news_df is not None else pd.DataFrame()

    session_windows: list[tuple[str, time, time]] = [
        ("london", sessions["london_start"], sessions["london_end"]),
    ]
    if enable_ny:
        session_windows.append(("ny", sessions["ny_start"], sessions["ny_end"]))

    friday_cut = sessions.get("friday_no_new_entries")
    signals: list[Signal] = []
    traded_day: set[tuple[str, pd.Timestamp]] = set()

    for sym, df in bars_by_symbol.items():
        bars = m15_indicators(df, entry_ema_period=21, atr_period=atr_period)
        bias = htf_bias_on_ltf(bars, htf_rule=htf_rule, htf_ema_period=htf_ema)
        warmup = min_htf * 16
        if len(bias) > warmup:
            bias = bias.copy()
            bias.iloc[:warmup] = 0
        bars["bias"] = bias

        htf = resample_ohlc(ensure_utc_index(df).reset_index(), htf_rule)
        book = ZoneBook.from_htf(
            htf,
            left=pivot_left,
            right=pivot_right,
            htf_rule=htf_rule,
            max_touches=max_touches,
        )
        sess = _session_name_series(bars.index, session_windows)

        lows = bars["low"].to_numpy(dtype=float)
        highs = bars["high"].to_numpy(dtype=float)
        closes = bars["close"].to_numpy(dtype=float)
        biases = bars["bias"].to_numpy(dtype=int)
        atrs = bars["atr"].to_numpy(dtype=float)
        sess_arr = sess.to_numpy()
        times = bars.index

        for i in range(len(bars)):
            ts = times[i]
            lo = lows[i]
            hi = highs[i]
            close = closes[i]
            b = int(biases[i])
            atr_v = float(atrs[i])
            session_name = str(sess_arr[i])

            live = book.active(ts)
            candidates = [z for z in live if z.intersects(lo, hi)]

            day_key = (sym, pd.Timestamp(ts).normalize())
            can_trade = (
                bool(session_name)
                and day_key not in traded_day
                and b != 0
                and np.isfinite(atr_v)
                and atr_v > 0
            )
            if can_trade:
                if (
                    not allow_weekend_hold
                    and friday_cut is not None
                    and is_friday(ts)
                    and ts.time() >= friday_cut
                ):
                    can_trade = False
                elif news_on and is_news_blackout(ts, sym, news_frame, blackout_min):
                    can_trade = False

            if can_trade:
                for z in candidates:
                    long_ok = b > 0 and z.side > 0 and z.rejection_long(close)
                    short_ok = b < 0 and z.side < 0 and z.rejection_short(close)
                    if not (long_ok or short_ok):
                        continue
                    side = 1 if long_ok else -1
                    entry = close
                    buf = atr_buf * atr_v
                    if side > 0:
                        stop = z.z_low - buf
                        risk = entry - stop
                        tp = entry + tp_r * risk
                    else:
                        stop = z.z_high + buf
                        risk = stop - entry
                        tp = entry - tp_r * risk
                    if risk <= 0:
                        continue
                    kind = "retest" if z.flipped else ("support" if side > 0 else "res")
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
                            range_high=z.z_high,
                            range_low=z.z_low,
                            reason=f"{session_name}_msnr_{kind}",
                        )
                    )
                    traded_day.add(day_key)
                    break

            book.note_m15_bar(ts, lo, hi)
            # Drop dead zones periodically to keep touch scans cheap
            if i % 2048 == 0 and len(book.zones) > 64:
                book.zones = [z for z in book.zones if z.active]

    signals.sort(key=lambda s: s.timestamp)
    return signals
