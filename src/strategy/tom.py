"""Turn of the month (TOM): long a US index CFD around the month boundary.

Window: the last `days_before` trading days of a month plus the first
`days_after` of the next. Entry at the cash close of the trading day before
the window, exit at the cash close of its last day. A protective ATR stop sits
under the entry; there is no target. `placebo_start_day: k` moves the same
window to trading days k .. k+days_before+days_after-1 of each month, which
measures the index's plain drift for comparison.

Trading days come from the US exchange calendar (dates of a FRED index series,
`calendar_series`), so CFD sessions on US holidays do not shift the window.
Sessions are the 24h ending at the cash close in exchange time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.data.macro import load_series
from src.research.tom_check import tom_mask
from src.strategy.cssm import Signal


def cash_sessions(m15: pd.DataFrame, exchange_tz: str = "America/New_York", cash_close: str = "16:00") -> pd.DataFrame:
    """Daily OHLC per exchange session (prior cash close -> this cash close) plus the last bar's timestamp."""
    x = ensure_utc_index(m15)
    hh, mm = (int(v) for v in cash_close.split(":"))
    local = x.index.tz_convert(exchange_tz)
    # shift so the cash close lands on midnight: every bar before it belongs to that day's session
    shift = pd.Timedelta(hours=24 - hh, minutes=-mm)
    day = (local + shift).normalize().tz_localize(None)
    g = x.assign(_day=day, _ts=x.index).groupby("_day")
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), last_ts=("_ts", "last"))
    out.index = pd.DatetimeIndex(out.index, name="session")
    return out


def window_mask(days: pd.DatetimeIndex, days_before: int, days_after: int, placebo_start_day: int | None = None) -> pd.Series:
    if placebo_start_day is None:
        return tom_mask(days, days_before, days_after)
    month = days.to_period("M")
    pos = pd.Series(np.arange(len(days)), index=days).groupby(month.values).cumcount() + 1
    k = int(placebo_start_day)
    return ((pos >= k) & (pos < k + days_before + days_after)).rename("tom")


def windows(days: pd.DatetimeIndex, mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(entry day, exit day): the day before each run of masked days, and the run's last day.

    Runs touching either end of `days` are dropped: a partial month at the edge
    of the calendar mislabels its first / last trading days.
    """
    m = mask.reindex(days).fillna(False).to_numpy(dtype=bool)
    out = []
    i = 0
    while i < len(days):
        if m[i]:
            j = i
            while j + 1 < len(days) and m[j + 1]:
                j += 1
            if i > 0 and j < len(days) - 1:
                out.append((days[i - 1], days[j]))
            i = j + 1
        else:
            i += 1
    return out


def trading_calendar(cfg: dict[str, Any], sessions: pd.DataFrame) -> pd.DatetimeIndex:
    name = (cfg.get("signal") or {}).get("calendar_series", "NASDAQ100")
    cal = load_series(name) if name else pd.Series(dtype=float)
    if cal.empty:
        return pd.DatetimeIndex(sessions.index[sessions.index.dayofweek < 5])
    days = pd.DatetimeIndex(cal.index.tz_localize(None).normalize())
    lo, hi = sessions.index.min(), sessions.index.max()
    return days[(days >= lo) & (days <= hi)]


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    calendar: pd.DatetimeIndex | None = None,
) -> list[Signal]:
    sig = strategy_cfg["signal"]
    sess_cfg = strategy_cfg.get("sessions", {}) or {}
    tz = str(sess_cfg.get("exchange_tz", "America/New_York"))
    close_t = str(sess_cfg.get("cash_close", "16:00"))
    before, after = int(sig.get("days_before", 1)), int(sig.get("days_after", 3))
    placebo = sig.get("placebo_start_day")
    stop_mult = float(sig.get("stop_atr_mult", 2.0))
    n_atr = int(sig.get("atr_period", 20))

    signals: list[Signal] = []
    for sym, df in bars_by_symbol.items():
        if df is None or len(df) == 0:
            continue
        sessions = cash_sessions(df, tz, close_t)
        days = calendar if calendar is not None else trading_calendar(strategy_cfg, sessions)
        days = pd.DatetimeIndex([d for d in days if d in sessions.index])
        if len(days) < n_atr + 2:
            continue
        daily = sessions.loc[days]
        # ATR through the previous session: known before the entry close
        a = atr(daily["high"], daily["low"], daily["close"], period=n_atr).shift(1)
        for d_in, d_out in windows(days, window_mask(days, before, after, placebo)):
            risk = stop_mult * float(a.get(d_in, np.nan))
            if not np.isfinite(risk) or risk <= 0:
                continue
            row_in, row_out = daily.loc[d_in], daily.loc[d_out]
            entry = float(row_in["close"])
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(row_in["last_ts"]),
                    symbol=sym.upper(),
                    side=1,
                    entry=entry,
                    stop=entry - risk,
                    take_profit=float("nan"),
                    risk_price=risk,
                    session="tom",
                    range_high=entry + risk,
                    range_low=entry - risk,
                    reason=f"tom|{'placebo' + str(placebo) if placebo else 'window'}|{d_in.date()}->{d_out.date()}",
                    exit_time=pd.Timestamp(row_out["last_ts"]),
                )
            )
    signals.sort(key=lambda s: s.timestamp)
    return signals
