"""CSMO — Cross-Sectional Momentum Open (no Asian range break)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.features.momentum import overnight_return, parse_session_times, pick_cross_section_candidate
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday
from src.strategy.cssm import Signal


def _daily_atr_series(bars: pd.DataFrame, period: int) -> pd.Series:
    x = ensure_utc_index(bars)
    daily = x.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    daily = daily.dropna()
    a = atr(daily["high"], daily["low"], daily["close"], period=period)
    a = a.shift(1)
    return a.reindex(x.index, method="ffill")


# Back-compat alias
pick_csmo_candidate = pick_cross_section_candidate


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """London-open CSMO signals; one candidate per day max."""
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    if news_df is None:
        news_df = pd.DataFrame()

    lookback = int(sig_cfg.get("overnight_lookback_days", 60))
    winner_pct = float(sig_cfg.get("winner_percentile", 0.80))
    loser_pct = float(sig_cfg.get("loser_percentile", 0.20))
    atr_period = int(sig_cfg.get("atr_period", 20))
    stop_atr_mult = float(sig_cfg.get("stop_atr_mult", 0.75))
    tp_r = float(sig_cfg.get("take_profit_r", 1.5))

    norm: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        x = ensure_utc_index(df)
        x["atr_d"] = _daily_atr_series(x.reset_index(), atr_period)
        norm[sym] = x

    any_sym = next(iter(norm))
    all_days = sorted({ts.normalize() for ts in norm[any_sym].index if ts.dayofweek < 5})

    overnight_history: dict[str, list[float]] = {s: [] for s in norm}
    signals: list[Signal] = []

    london_open = sessions["london_start"]

    for day in all_days:
        day = pd.Timestamp(day)
        if day.tzinfo is None:
            day = day.tz_localize("UTC")

        # Entry bar: 07:00 UTC M15
        entry_ts = pd.Timestamp(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=london_open.hour,
            minute=london_open.minute,
            tz="UTC",
        )
        if is_friday(entry_ts) and entry_ts.time() >= sessions["friday_no_new_entries"]:
            # Still update history below for consistency
            pass

        overnight_rets: dict[str, float] = {}
        for sym, bars in norm.items():
            r = overnight_return(bars, day, sessions["overnight_anchor"], london_open)
            if r is None:
                continue
            overnight_rets[sym] = r

        # History for percentile uses prior days only (append after decision)
        hist_snapshot = {s: list(v) for s, v in overnight_history.items()}

        pick = pick_csmo_candidate(
            overnight_rets,
            hist_snapshot,
            winner_pct=winner_pct,
            loser_pct=loser_pct,
            min_history=max(20, lookback // 3),
        )

        # Update rolling overnight history (cap lookback)
        for sym, r in overnight_rets.items():
            overnight_history[sym].append(r)
            if len(overnight_history[sym]) > lookback:
                overnight_history[sym] = overnight_history[sym][-lookback:]

        if pick is None:
            continue

        sym, side, _ov = pick
        bars = norm[sym]
        if entry_ts not in bars.index:
            # nearest at-or-before
            hist = bars.loc[:entry_ts]
            if hist.empty:
                continue
            entry_ts = hist.index[-1]
            if entry_ts.normalize() != day.normalize():
                continue

        if is_friday(entry_ts) and entry_ts.time() >= sessions["friday_no_new_entries"]:
            continue

        if news_cfg.get("enabled", True) and is_news_blackout(
            entry_ts, sym, news_df, int(news_cfg.get("blackout_minutes", 60))
        ):
            continue

        bar = bars.loc[entry_ts]
        entry = float(bar["close"])
        atr_v = float(bar["atr_d"]) if pd.notna(bar["atr_d"]) else np.nan
        if not np.isfinite(atr_v) or atr_v <= 0:
            continue

        risk = stop_atr_mult * atr_v
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
                session="london",
                range_high=entry + risk,
                range_low=entry - risk,
                reason="csmo_london_open",
            )
        )

    signals.sort(key=lambda s: s.timestamp)
    return signals
