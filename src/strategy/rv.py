"""Relative value: band mean reversion on crosses of linked economies (AUDNZD, EURGBP).

When the daily close is `entry_z` standard deviations below (above) its
`band_period`-day mean, go long (short) at the close; exit at the first close
back across the mean or after `max_days` trading days. One position per pair.
`band_trades` works on any daily close series, so the FRED signal check and the
bar strategy trade exactly the same rules. Daily bars roll at
`sessions.close_hour_utc`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, daily_ohlc
from src.strategy.cssm import Signal


def band_trades(close: pd.Series, sig: dict[str, Any]) -> pd.DataFrame:
    """Entry / exit dates and side of the rule on a daily close series (decisions at each close)."""
    c = pd.Series(close, dtype=float).dropna()
    n = int(sig.get("band_period", 20))
    mu = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    z = ((c - mu) / sd.where(sd > 0)).to_numpy()
    cv, mv = c.to_numpy(), mu.to_numpy()
    entry_z, max_days = float(sig.get("entry_z", 2.0)), int(sig.get("max_days", 20))
    rows = []
    i, m = 0, len(c)
    while i < m:
        side = 1 if z[i] <= -entry_z else (-1 if z[i] >= entry_z else 0)
        if side == 0:
            i += 1
            continue
        j = i + 1
        while j < m and not ((side > 0 and cv[j] >= mv[j]) or (side < 0 and cv[j] <= mv[j])) and (j - i) < max_days:
            j += 1
        if j >= m:
            break  # still open when the data ends
        rows.append((c.index[i], c.index[j], side, j - i, cv[i], cv[j]))
        i = j + 1
    return pd.DataFrame(rows, columns=["entry_date", "exit_date", "side", "days", "entry_px", "exit_px"])


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
) -> list[Signal]:
    sig = strategy_cfg["signal"]
    hour = int((strategy_cfg.get("sessions") or {}).get("close_hour_utc", 21))
    stop_mult = float(sig.get("stop_atr_mult", 3.0))
    n_atr = int(sig.get("atr_period", 20))
    signals: list[Signal] = []
    for sym in [s.upper() for s in (strategy_cfg.get("universe") or bars_by_symbol)]:
        df = bars_by_symbol.get(sym)
        if df is None or len(df) == 0:
            continue
        d = daily_ohlc(df, hour)
        a = atr(d["high"], d["low"], d["close"], period=n_atr).shift(1)
        for _, t in band_trades(d["close"], sig).iterrows():
            risk = stop_mult * float(a.get(t["entry_date"], np.nan))
            if not np.isfinite(risk) or risk <= 0:
                continue
            side, entry = int(t["side"]), float(t["entry_px"])
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(d.at[t["entry_date"], "last_ts"]),
                    symbol=sym,
                    side=side,
                    entry=entry,
                    stop=entry - side * risk,
                    take_profit=float("nan"),
                    risk_price=risk,
                    session="rv",
                    range_high=entry + risk,
                    range_low=entry - risk,
                    reason=f"rv|{'long' if side > 0 else 'short'}|{t['entry_date'].date()}->{t['exit_date'].date()}",
                    exit_time=pd.Timestamp(d.at[t["exit_date"], "last_ts"]),
                )
            )
    signals.sort(key=lambda s: s.timestamp)
    return signals
