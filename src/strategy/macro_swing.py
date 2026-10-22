"""Macro swing — fundamentals pick the pair and direction, technicals time it.

Daily decision at `sessions.signal_time` (UTC). Pair bias comes from
`src.features.macro_score` (rates repricing, carry, risk, commodities,
positioning). A pair is tradable only when |bias| >= min_abs_bias, in the bias
direction. The D1 trigger (`none` | `pullback` | `breakout`) decides the day.
Stops are ATR-based (optionally trailing, see `signal.trailing_stop_r`); the
position is closed when the bias flips (scheduled exit), at take-profit when
one is set, or at max_hold_days. With risk.max_open_positions > 1 the top
candidates of each day are offered and the engine's correlation gate decides
which of them may join the open positions (a basket).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.data.macro import load_macro_panel
from src.features.macro_score import build_currency_scores, pair_bias
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import parse_hhmm
from src.strategy.cssm import Signal
from src.strategy.d1_mom import build_eod_frame

TRIGGERS = ("none", "pullback", "breakout")


def trigger_side(row: pd.Series, prev: pd.Series | None, mode: str, want: int) -> bool:
    """Does today's EOD row confirm a trade in direction `want` under `mode`?"""
    mode = (mode or "none").lower()
    if mode == "none":
        return True
    if mode == "pullback":
        ema = row.get("ema")
        if ema is None or math.isnan(ema):
            return False
        if want > 0:
            return row["close"] > ema and row["low"] <= ema and row["close"] > row["open"]
        return row["close"] < ema and row["high"] >= ema and row["close"] < row["open"]
    if mode == "breakout":
        hi, lo = row.get("don_hi"), row.get("don_lo")
        if hi is None or lo is None or math.isnan(hi) or math.isnan(lo):
            return False
        return row["close"] > hi if want > 0 else row["close"] < lo
    raise ValueError(f"Unknown trigger mode: {mode}")


def first_bias_exit(bias: pd.Series, after: pd.Timestamp, side: int, exit_abs: float) -> pd.Timestamp | None:
    """First decision day after `after` where the bias no longer supports `side`."""
    fut = bias[bias.index > after]
    for d, b in fut.items():
        if pd.isna(b):
            continue
        if (side > 0 and b <= exit_abs) or (side < 0 and b >= -exit_abs):
            return d
    return None


def prepare_eod(m15: pd.DataFrame, signal_hhmm: str, sig: dict[str, Any]) -> pd.DataFrame:
    d1 = build_eod_frame(m15, signal_hhmm)
    d1["atr"] = atr(d1["high"], d1["low"], d1["close"], period=int(sig.get("atr_period", 20))).shift(1)
    ema_n = int(sig.get("ema_period", 20))
    d1["ema"] = d1["close"].ewm(span=ema_n, adjust=False, min_periods=ema_n).mean().shift(1)
    don = int(sig.get("donchian_period", 20))
    d1["don_hi"] = d1["high"].rolling(don, min_periods=don).max().shift(1)
    d1["don_lo"] = d1["low"].rolling(don, min_periods=don).min().shift(1)
    return d1


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
    macro_panel: dict[str, pd.Series] | None = None,
) -> list[Signal]:
    sig = strategy_cfg["signal"]
    macro_cfg = strategy_cfg.get("macro", {}) or {}
    news_cfg = strategy_cfg.get("news", {}) or {}
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    if news_df is None:
        news_df = pd.DataFrame()

    panel = macro_panel if macro_panel is not None else load_macro_panel(macro_cfg)
    if not panel:
        return []
    symbols = [s.upper() for s in bars_by_symbol]
    currencies = sorted({s[:3] for s in symbols} | {s[3:6] for s in symbols})
    scored = build_currency_scores(panel, macro_cfg, currencies=currencies)
    bias = pair_bias(scored["scores"], symbols)
    if bias.empty:
        return []

    signal_hhmm = str(strategy_cfg.get("sessions", {}).get("signal_time", "21:00"))
    signal_t = parse_hhmm(signal_hhmm)
    min_bias = float(sig.get("min_abs_bias", 0.5))
    exit_abs = float(sig.get("exit_abs_bias", 0.0))
    exit_on_flip = bool(sig.get("exit_on_bias_flip", True))
    mode = str(sig.get("trigger", "none")).lower()
    if mode not in TRIGGERS:
        raise ValueError(f"signal.trigger must be one of {TRIGGERS}")
    stop_mult = float(sig.get("stop_atr_mult", 2.0))
    tp_r = float(sig.get("take_profit_r") or 0.0)
    cooldown = int(sig.get("reentry_cooldown_days", 0) or 0)
    # one candidate per day for a single slot; up to max_open_positions for a basket
    per_day = max(1, int((strategy_cfg.get("risk") or {}).get("max_open_positions", 1) or 1))

    m15 = {s.upper(): ensure_utc_index(df) for s, df in bars_by_symbol.items()}
    eod = {s: prepare_eod(df, signal_hhmm, sig) for s, df in m15.items()}
    days = sorted(set().union(*[set(d.index) for d in eod.values()]))
    bias_daily = bias.reindex(pd.DatetimeIndex(days)).ffill() if days else bias

    last_signal_day: dict[str, pd.Timestamp] = {}
    signals: list[Signal] = []
    for day in days:
        cands: list[tuple[float, str, int, pd.Series]] = []
        for sym in symbols:
            if sym not in bias_daily.columns or day not in eod[sym].index:
                continue
            b = bias_daily.at[day, sym]
            if pd.isna(b) or abs(b) < min_bias:
                continue
            row = eod[sym].loc[day]
            a = row["atr"]
            if pd.isna(a) or float(a) <= 0:
                continue
            want = 1 if b > 0 else -1
            if not trigger_side(row, None, mode, want):
                continue
            last = last_signal_day.get(sym)
            if cooldown > 0 and last is not None and (day - last).days < cooldown:
                continue
            cands.append((abs(float(b)), sym, want, row))
        if not cands:
            continue
        # strongest bias first; the engine fills them in order while slots and the correlation gate allow
        for _, sym, side, row in sorted(cands, key=lambda c: -c[0])[:per_day]:
            entry_ts = pd.Timestamp(
                year=day.year, month=day.month, day=day.day, hour=signal_t.hour, minute=signal_t.minute, tz="UTC"
            )
            bars = m15[sym]
            if entry_ts not in bars.index:
                continue
            if news_cfg.get("enabled", True) and is_news_blackout(
                entry_ts, sym, news_df, int(news_cfg.get("blackout_minutes", 60))
            ):
                continue
            entry = float(bars.loc[entry_ts]["close"])
            risk = stop_mult * float(row["atr"])
            stop = entry - side * risk
            tp = entry + side * tp_r * risk if tp_r > 0 else float("nan")
            exit_time = None
            if exit_on_flip:
                d_exit = first_bias_exit(bias_daily[sym], day, side, exit_abs)
                if d_exit is not None:
                    exit_time = pd.Timestamp(
                        year=d_exit.year, month=d_exit.month, day=d_exit.day,
                        hour=signal_t.hour, minute=signal_t.minute, tz="UTC",
                    )
            b = float(bias_daily.at[day, sym])
            last_signal_day[sym] = day
            signals.append(
                Signal(
                    timestamp=entry_ts,
                    symbol=sym,
                    side=side,
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    risk_price=risk,
                    session="macro",
                    range_high=entry + risk,
                    range_low=entry - risk,
                    reason=f"macro_swing|{mode}|bias={b:+.2f}",
                    exit_time=exit_time,
                )
            )
    signals.sort(key=lambda s: s.timestamp)
    return signals
