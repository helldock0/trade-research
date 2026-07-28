from __future__ import annotations

from datetime import time

import pandas as pd


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def session_mask(index: pd.DatetimeIndex, start: time, end: time) -> pd.Series:
    t = index.time
    if start <= end:
        return pd.Series([(start <= x < end) for x in t], index=index)
    # wraps midnight
    return pd.Series([(x >= start or x < end) for x in t], index=index)


def asian_range_for_day(
    day_bars: pd.DataFrame,
    asian_start: time,
    asian_end: time,
) -> tuple[float, float] | None:
    mask = session_mask(day_bars.index, asian_start, asian_end)
    window = day_bars.loc[mask]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min())


def london_ny_range(
    day_bars: pd.DataFrame,
    start: time,
    end: time,
) -> tuple[float, float] | None:
    mask = session_mask(day_bars.index, start, end)
    window = day_bars.loc[mask]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min())


def is_friday(ts: pd.Timestamp) -> bool:
    return int(ts.dayofweek) == 4


def past_friday_cutoff(ts: pd.Timestamp, cutoff: time) -> bool:
    return is_friday(ts) and ts.timetz().replace(tzinfo=None) >= cutoff
