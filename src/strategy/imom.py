"""Market intraday momentum (imom): trade the last half hour of the US cash session.

At `entry_time` (exchange time, 15:30) go with the sign of the day's return so
far (mode `day`: previous cash close -> entry), or of the first half hour
(mode `first`: previous close -> 10:00, Gao et al. 2018), and exit at
`exit_time` (16:00). Mode `long` is always long over the same half hour, the
placebo. The protective stop is `stop_range_mult` times the mean high-low range
of that half hour over the prior `range_lookback` days. Trading days come from
the US exchange calendar; days without a full cash close are skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index
from src.strategy.cssm import Signal
from src.strategy.tom import trading_calendar

MODES = ("day", "first", "long")


def _minus_15(hhmm: str) -> str:
    return (pd.Timestamp(f"2000-01-01 {hhmm}") - pd.Timedelta(minutes=15)).strftime("%H:%M")


def day_table(
    m15: pd.DataFrame,
    exchange_tz: str = "America/New_York",
    entry_time: str = "15:30",
    exit_time: str = "16:00",
    first_time: str = "10:00",
) -> pd.DataFrame:
    """Per exchange date: closes of the M15 bars ending at the first / entry / exit times,
    their timestamps, and the high-low range from entry to exit."""
    x = ensure_utc_index(m15)
    local = x.index.tz_convert(exchange_tz)
    hm = pd.Index(local.strftime("%H:%M"))
    date = pd.DatetimeIndex(local.tz_localize(None).normalize())
    cols: dict[str, pd.Series] = {}

    def at(hhmm: str, field: str) -> pd.Series:
        sel = np.asarray(hm == _minus_15(hhmm))
        vals = x.index[sel] if field == "ts" else x[field].to_numpy()[sel]
        s = pd.Series(vals, index=date[sel])
        return s[~s.index.duplicated(keep="last")]

    cols["first"] = at(first_time, "close")
    cols["entry"] = at(entry_time, "close")
    cols["entry_ts"] = at(entry_time, "ts")
    cols["exit"] = at(exit_time, "close")
    cols["exit_ts"] = at(exit_time, "ts")
    # range of the bars from entry to exit (bar starts in [entry, exit))
    t_in, t_out = pd.Timestamp(f"2000-01-01 {entry_time}").time(), pd.Timestamp(f"2000-01-01 {exit_time}").time()
    tod = pd.Index(local.time)
    win = (tod >= t_in) & (tod < t_out)
    w = x.loc[np.asarray(win), ["high", "low"]].copy()
    w["d"] = date[np.asarray(win)]
    g = w.groupby("d")
    cols["range"] = g["high"].max() - g["low"].min()
    out = pd.DataFrame(cols)
    out.index.name = "date"
    return out


def signal_frame(m15: pd.DataFrame, cfg: dict[str, Any], calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """day_table restricted to trading days with a full close, plus the signal, side and stop distance."""
    sig = cfg.get("signal", {}) or {}
    tz = str((cfg.get("sessions") or {}).get("exchange_tz", "America/New_York"))
    mode = str(sig.get("mode", "day")).lower()
    if mode not in MODES:
        raise ValueError(f"signal.mode must be one of {MODES}")
    t = day_table(m15, tz, str(sig.get("entry_time", "15:30")), str(sig.get("exit_time", "16:00")))
    days = calendar if calendar is not None else trading_calendar(cfg, t)
    t = t.reindex(pd.DatetimeIndex(days))
    t["prev_close"] = t["exit"].shift(1)                     # previous trading day's cash close
    t = t.dropna(subset=["entry", "exit", "prev_close"])
    ref = t["first"] if mode == "first" else t["entry"]
    t["signal"] = np.log(ref / t["prev_close"])
    t["side"] = 1 if mode == "long" else np.sign(t["signal"]).astype(int)
    if bool(sig.get("strong_only", False)) and mode != "long":
        look = int(sig.get("strong_lookback", 60))
        med = t["signal"].abs().rolling(look, min_periods=look // 2).median().shift(1)
        t.loc[~(t["signal"].abs() >= med), "side"] = 0
    look_r = int(sig.get("range_lookback", 20))
    t["stop_dist"] = float(sig.get("stop_range_mult", 1.0)) * t["range"].rolling(look_r, min_periods=look_r // 2).mean().shift(1)
    t["last_ret"] = np.log(t["exit"] / t["entry"])
    return t


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    calendar: pd.DatetimeIndex | None = None,
) -> list[Signal]:
    mode = str(strategy_cfg["signal"].get("mode", "day"))
    signals: list[Signal] = []
    for sym, df in bars_by_symbol.items():
        if df is None or len(df) == 0:
            continue
        t = signal_frame(df, strategy_cfg, calendar)
        t = t[(t["side"] != 0) & (t["stop_dist"] > 0)]
        for d, r in t.iterrows():
            side, entry, risk = int(r["side"]), float(r["entry"]), float(r["stop_dist"])
            signals.append(
                Signal(
                    timestamp=pd.Timestamp(r["entry_ts"]),
                    symbol=sym.upper(),
                    side=side,
                    entry=entry,
                    stop=entry - side * risk,
                    take_profit=float("nan"),
                    risk_price=risk,
                    session="imom",
                    range_high=entry + risk,
                    range_low=entry - risk,
                    reason=f"imom|{mode}|{pd.Timestamp(d).date()}|sig={float(r['signal']):+.4f}",
                    exit_time=pd.Timestamp(r["exit_ts"]),
                )
            )
    signals.sort(key=lambda s: s.timestamp)
    return signals
