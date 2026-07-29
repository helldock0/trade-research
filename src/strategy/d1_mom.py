"""D1 Cross-Sectional Momentum — Menkhoff-style 20d rank with swing holds."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.features.momentum import pick_cross_section_candidate
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import parse_hhmm
from src.strategy.cssm import Signal


def build_eod_frame(m15: pd.DataFrame, signal_time_hhmm: str = "21:00") -> pd.DataFrame:
    """One row per weekday using the signal-time M15 bar as the daily OHLC proxy.

    Avoids calendar resample lookahead past the decision timestamp.
    """
    x = ensure_utc_index(m15)
    t = parse_hhmm(signal_time_hhmm)
    mask = (x.index.hour == t.hour) & (x.index.minute == t.minute) & (x.index.dayofweek < 5)
    eod = x.loc[mask, ["open", "high", "low", "close", "volume"]].copy()
    # Index by calendar date (normalize) for joins across symbols
    eod.index = eod.index.normalize()
    eod = eod[~eod.index.duplicated(keep="last")].sort_index()
    return eod


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Emit at most one signal per weekday at signal_time (default 21:00 UTC)."""
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    if news_df is None:
        news_df = pd.DataFrame()

    formation = int(sig_cfg.get("formation_days", 20))
    history_n = int(sig_cfg.get("history_days", 60))
    winner_pct = float(sig_cfg.get("winner_percentile", 0.80))
    loser_pct = float(sig_cfg.get("loser_percentile", 0.20))
    atr_period = int(sig_cfg.get("atr_period", 20))
    stop_mult = float(sig_cfg.get("stop_atr_mult", 1.5))
    tp_r = float(sig_cfg.get("take_profit_r", 2.0))
    signal_hhmm = str(strategy_cfg.get("sessions", {}).get("signal_time", "21:00"))
    signal_t = parse_hhmm(signal_hhmm)

    m15: dict[str, pd.DataFrame] = {
        s: ensure_utc_index(df) for s, df in bars_by_symbol.items()
    }
    daily: dict[str, pd.DataFrame] = {}
    for s, df in m15.items():
        d1 = build_eod_frame(df, signal_hhmm)
        d1["ret_nd"] = d1["close"].pct_change(formation)
        # ATR from EOD series; shift 1 so stop uses prior completed day only
        d1["atr"] = atr(d1["high"], d1["low"], d1["close"], period=atr_period).shift(1)
        daily[s] = d1

    any_s = next(iter(daily))
    days = list(daily[any_s].index)

    ret_history: dict[str, list[float]] = {s: [] for s in daily}
    signals: list[Signal] = []
    min_hist = max(20, history_n // 3)

    for day in days:
        rets: dict[str, float] = {}
        atrs: dict[str, float] = {}
        for s, d1 in daily.items():
            if day not in d1.index:
                continue
            row = d1.loc[day]
            r = row["ret_nd"]
            a = row["atr"]
            if pd.isna(r) or pd.isna(a) or float(a) <= 0:
                continue
            rets[s] = float(r)
            atrs[s] = float(a)

        hist_snap = {s: list(v) for s, v in ret_history.items()}
        pick = pick_cross_section_candidate(
            rets, hist_snap, winner_pct, loser_pct, min_history=min_hist
        )

        for s, r in rets.items():
            ret_history[s].append(r)
            if len(ret_history[s]) > history_n:
                ret_history[s] = ret_history[s][-history_n:]

        if pick is None:
            continue

        sym, side, _ = pick
        entry_ts = pd.Timestamp(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=signal_t.hour,
            minute=signal_t.minute,
            tz="UTC",
        )
        bars = m15[sym]
        if entry_ts not in bars.index:
            continue

        if news_cfg.get("enabled", True) and is_news_blackout(
            entry_ts, sym, news_df, int(news_cfg.get("blackout_minutes", 60))
        ):
            continue

        entry = float(bars.loc[entry_ts]["close"])
        risk = stop_mult * atrs[sym]
        if risk <= 0:
            continue
        if side > 0:
            stop = entry - risk
            tp = entry + tp_r * risk
        else:
            stop = entry + risk
            tp = entry - tp_r * risk

        signals.append(
            Signal(
                timestamp=entry_ts,
                symbol=sym,
                side=side,
                entry=entry,
                stop=stop,
                take_profit=tp,
                risk_price=risk,
                session="d1",
                range_high=entry + risk,
                range_low=entry - risk,
                reason="d1_mom_20d",
            )
        )

    signals.sort(key=lambda s: s.timestamp)
    return signals
