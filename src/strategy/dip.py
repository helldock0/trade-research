"""Buy the dip in an uptrend: RSI(2) < entry level with the close above its trend average.

Long at the cash close; exit at the first close above the exit average or after
`max_days` trading days; one position per index (no new entry until the
previous trade has closed). Each index can use its own exchange time zone,
cash close and trading calendar (`sessions.per_symbol`). `dip_trades` works on
any daily close series, so the signal check on daily history and the bar
strategy trade exactly the same rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr
from src.strategy.cssm import Signal
from src.strategy.tom import cash_sessions, trading_calendar


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Wilder RSI."""
    d = pd.Series(close, dtype=float).diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return 100.0 - 100.0 / (1.0 + up / dn)


def dip_trades(close: pd.Series, sig: dict[str, Any]) -> pd.DataFrame:
    """Entry / exit dates of the rule on a daily close series (decisions at each close)."""
    c = pd.Series(close, dtype=float).dropna()
    trend = c.rolling(int(sig.get("trend_sma", 200))).mean()
    exit_avg = c.rolling(int(sig.get("exit_sma", 5))).mean()
    r = rsi(c, int(sig.get("rsi_period", 2)))
    entry_ok = ((c > trend) & (r < float(sig.get("rsi_entry", 10)))).to_numpy()
    above_exit = (c > exit_avg).to_numpy()
    max_days = int(sig.get("max_days", 10))
    rows = []
    i, n = 0, len(c)
    while i < n:
        if not entry_ok[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not above_exit[j] and (j - i) < max_days:
            j += 1
        if j >= n:
            break  # still open when the data ends
        rows.append((c.index[i], c.index[j], j - i, c.iloc[i], c.iloc[j]))
        i = j + 1
    return pd.DataFrame(rows, columns=["entry_date", "exit_date", "days", "entry_px", "exit_px"])


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    calendar: pd.DatetimeIndex | None = None,
) -> list[Signal]:
    sig = strategy_cfg["signal"]
    sess_cfg = strategy_cfg.get("sessions", {}) or {}
    per_symbol = {str(k).upper(): v or {} for k, v in (sess_cfg.get("per_symbol") or {}).items()}
    # only the universe is traded; other bars (FX pairs for P&L conversion) are left alone
    universe = {s.upper() for s in (strategy_cfg.get("universe") or bars_by_symbol)}
    stop_mult = float(sig.get("stop_atr_mult", 2.5))
    n_atr = int(sig.get("atr_period", 20))
    signals: list[Signal] = []
    for sym, df in bars_by_symbol.items():
        if df is None or len(df) == 0 or sym.upper() not in universe:
            continue
        ps = per_symbol.get(sym.upper(), {})
        tz = str(ps.get("exchange_tz", sess_cfg.get("exchange_tz", "America/New_York")))
        close_t = str(ps.get("cash_close", sess_cfg.get("cash_close", "16:00")))
        sessions = cash_sessions(df, tz, close_t)
        cal_name = ps.get("calendar_series", sig.get("calendar_series", "NASDAQ100"))
        days = calendar if calendar is not None else trading_calendar({"signal": {"calendar_series": cal_name}}, sessions)
        daily = sessions.reindex(pd.DatetimeIndex(days)).dropna(subset=["close"])
        # ATR through the previous session: known before the entry close
        a = atr(daily["high"], daily["low"], daily["close"], period=n_atr).shift(1)
        for _, t in dip_trades(daily["close"], sig).iterrows():
            risk = stop_mult * float(a.get(t["entry_date"], np.nan))
            if not np.isfinite(risk) or risk <= 0:
                continue
            entry = float(t["entry_px"])
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(daily.at[t["entry_date"], "last_ts"]),
                    symbol=sym.upper(),
                    side=1,
                    entry=entry,
                    stop=entry - risk,
                    take_profit=float("nan"),
                    risk_price=risk,
                    session="dip",
                    range_high=entry + risk,
                    range_low=entry - risk,
                    reason=f"dip|{t['entry_date'].date()}->{t['exit_date'].date()}",
                    exit_time=pd.Timestamp(daily.at[t["exit_date"], "last_ts"]),
                )
            )
    signals.sort(key=lambda s: s.timestamp)
    return signals
