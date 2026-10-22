"""Correlation gate for running several positions at once (point-in-time).

A new trade is refused when it repeats the bet of a position already open:
signed correlation = corr(daily returns of the two symbols) * side_new * side_open.
Long EURUSD with long GBPUSD is positive (same short-USD bet); long EURUSD with
long USDJPY is also positive (both short USD). Daily closes are taken at the
21:00 UTC rollover, and the correlation used on day d ends with day d-1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def daily_closes(bars: dict[str, pd.DataFrame], close_hour_utc: int = 21) -> pd.DataFrame:
    """Weekday closes at the bar before `close_hour_utc` (UTC-indexed bars)."""
    cols = {}
    for sym, df in bars.items():
        if df.empty:
            continue
        c = df["close"].resample("24h", offset=f"{int(close_hour_utc)}h", label="right").last()
        c.index = c.index.normalize()
        cols[sym] = c
    out = pd.DataFrame(cols)
    return out[out.index.dayofweek < 5].dropna(how="all")


class CorrelationGate:
    def __init__(self, bars: dict[str, pd.DataFrame], max_signed_corr: float, window_days: int = 60):
        self.max_signed_corr = float(max_signed_corr)
        rets = np.log(daily_closes(bars)).diff()
        minp = max(10, int(window_days) // 2)
        syms = list(rets.columns)
        self._corr: dict[tuple[str, str], pd.Series] = {}
        for i, a in enumerate(syms):
            for b in syms[i + 1 :]:
                c = rets[a].rolling(int(window_days), min_periods=minp).corr(rets[b]).shift(1)
                self._corr[(a, b)] = self._corr[(b, a)] = c

    @classmethod
    def from_config(cls, risk_cfg: dict[str, Any], bars: dict[str, pd.DataFrame]) -> CorrelationGate | None:
        cap = risk_cfg.get("max_signed_corr")
        if cap is None:
            return None
        return cls(bars, float(cap), int(risk_cfg.get("corr_window_days", 60)))

    def corr(self, a: str, b: str, ts: pd.Timestamp) -> float:
        s = self._corr.get((a, b))
        if s is None or s.empty:
            return float("nan")
        v = s.asof(ts.normalize())
        return float(v) if v is not None else float("nan")

    def allows(self, symbol: str, side: int, open_positions: list[dict[str, Any]], ts: pd.Timestamp) -> bool:
        """False when any open position's signed correlation exceeds the cap (or is unknown)."""
        for p in open_positions:
            c = self.corr(symbol, p["symbol"], ts)
            if math.isnan(c) or c * side * int(p["side"]) > self.max_signed_corr:
                return False
        return True
