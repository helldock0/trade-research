"""Time-series momentum (trend following) across FX, gold and index CFDs.

On the last trading day of each month, the position in each market is the sign
of its return over the past `lookback_days` daily closes. A trade runs until a
month end where the sign changes; then it is closed and the opposite trade
opened at the same close. Daily bars roll at `sessions.close_hour_utc`; the
data's final month (possibly partial) takes no decision, so a trade still open
then is closed by the engine at the end of the data. A catastrophe stop sits
`stop_atr_mult` daily ATRs from the entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, daily_ohlc
from src.strategy.cssm import Signal


def month_end_positions(close: pd.Series, lookback: int) -> pd.Series:
    """Sign of the lookback return on each month's last trading day (final month dropped); 0 before the lookback fills."""
    c = pd.Series(close, dtype=float).dropna()
    past = np.log(c) - np.log(c).shift(int(lookback))
    ends = pd.DatetimeIndex(pd.Series(c.index, index=c.index).groupby(c.index.to_period("M")).max().to_numpy())[:-1]
    return np.sign(past.reindex(ends)).fillna(0).astype(int)


def position_runs(pos: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp | None, int]]:
    """(entry date, exit date or None if still open, side) for each run of the same nonzero position."""
    runs, side, start = [], 0, None
    for d, p in pos.items():
        if p != side:
            if side != 0:
                runs.append((start, d, side))
            side, start = int(p), d
    if side != 0:
        runs.append((start, None, side))
    return runs


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
) -> list[Signal]:
    sig = strategy_cfg["signal"]
    hour = int((strategy_cfg.get("sessions") or {}).get("close_hour_utc", 21))
    lookback = int(sig.get("lookback_days", 252))
    stop_mult = float(sig.get("stop_atr_mult", 5.0))
    n_atr = int(sig.get("atr_period", 20))
    universe = [s.upper() for s in (strategy_cfg.get("universe") or bars_by_symbol)]
    signals: list[Signal] = []
    for sym in universe:
        df = bars_by_symbol.get(sym)
        if df is None or len(df) == 0:
            continue
        d = daily_ohlc(df, hour)
        a = atr(d["high"], d["low"], d["close"], period=n_atr).shift(1)
        for entry_day, exit_day, side in position_runs(month_end_positions(d["close"], lookback)):
            risk = stop_mult * float(a.get(entry_day, np.nan))
            if not np.isfinite(risk) or risk <= 0:
                continue
            entry = float(d.at[entry_day, "close"])
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(d.at[entry_day, "last_ts"]),
                    symbol=sym,
                    side=side,
                    entry=entry,
                    stop=entry - side * risk,
                    take_profit=float("nan"),
                    risk_price=risk,
                    session="tsmom",
                    range_high=entry + risk,
                    range_low=entry - risk,
                    reason=f"tsmom|{'long' if side > 0 else 'short'}|{entry_day.date()}->{exit_day.date() if exit_day is not None else 'open'}",
                    exit_time=pd.Timestamp(d.at[exit_day, "last_ts"]) if exit_day is not None else None,
                )
            )
    signals.sort(key=lambda s: s.timestamp)
    return signals
