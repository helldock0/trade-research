"""Omar Agag EBP (Engulfing Bar Play) — H4 swing proxy."""

from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd

from src.data.bars import ensure_utc_index, resample_ohlc
from src.features.candles import (
    close_strength_from_extreme,
    ebp_levels,
    htf_signal_m15_ts,
    is_ebp_bear,
    is_ebp_bull,
)
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.strategy.cssm import Signal


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
    if sig_ts < htf_ts:
        return None
    return sig_ts


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """H4 EBP → limit/market entry on subsequent bars (no lookahead)."""
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    news_frame = news_df if news_df is not None else pd.DataFrame()
    news_on = bool(news_cfg.get("enabled", True))
    blackout_min = int(news_cfg.get("blackout_minutes", 60))

    htf_rule = str(sig_cfg.get("htf_rule", "4h"))
    strong_max = float(sig_cfg.get("strong_close_max", 0.15))
    market_min = float(sig_cfg.get("market_entry_min_retracement", 0.50))
    tp_r = float(sig_cfg.get("take_profit_r", 2.0))
    max_wait = int(sig_cfg.get("max_wait_htf_bars", 12))
    min_htf = int(sig_cfg.get("min_htf_bars", 40))
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
        if len(htf) < min_htf + 3:
            continue

        pending: dict[str, Any] | None = None

        for i in range(1, len(htf)):
            row = htf.iloc[i]
            prev = htf.iloc[i - 1]
            htf_ts = pd.Timestamp(htf.index[i])

            # Fill or invalidate pending limit
            if pending is not None:
                age = i - int(pending["setup_i"])
                if age > max_wait:
                    pending = None
                else:
                    side = int(pending["side"])
                    limit = float(pending["limit"])
                    stop = float(pending["stop"])
                    hi, lo = float(row["high"]), float(row["low"])
                    invalidated = (side > 0 and lo <= stop) or (side < 0 and hi >= stop)
                    touched = (side > 0 and lo <= limit) or (side < 0 and hi >= limit)
                    if invalidated:
                        pending = None
                    elif touched:
                        sig_ts = _map_to_m15(m15_index, htf_ts, htf_delta)
                        if sig_ts is not None:
                            if not (
                                not allow_weekend_hold
                                and friday_cut is not None
                                and is_friday(sig_ts)
                                and sig_ts.time() >= friday_cut
                            ) and not (
                                news_on and is_news_blackout(sig_ts, sym, news_frame, blackout_min)
                            ):
                                entry = limit
                                risk = abs(entry - stop)
                                if risk > 0:
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
                                            range_high=float(pending["high"]),
                                            range_low=float(pending["low"]),
                                            reason=str(pending["reason"]),
                                        )
                                    )
                        pending = None
                        continue

            if pending is not None:
                continue

            if is_ebp_bull(prev, row):
                side = 1
            elif is_ebp_bear(prev, row):
                side = -1
            else:
                continue

            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
            if hi <= lo:
                continue
            strength = close_strength_from_extreme(side, hi, lo, cl)
            levels = ebp_levels(side, hi, lo)

            if strength <= strong_max:
                limit, stop, reason = levels["p25"], levels["p75"], "ebp_strong_25"
            elif strength >= market_min:
                # Very indecisive: market entry on this bar close
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
                stop = levels["origin"]
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
                        range_high=hi,
                        range_low=lo,
                        reason="ebp_market_50",
                    )
                )
                continue
            else:
                limit, stop, reason = levels["p50"], levels["origin"], "ebp_indecisive_50"

            pending = {
                "setup_i": i,
                "side": side,
                "limit": limit,
                "stop": stop,
                "high": hi,
                "low": lo,
                "reason": reason,
            }

    signals.sort(key=lambda s: s.timestamp)
    return signals
