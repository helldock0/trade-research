"""Failed-break fade — fade London false breaks of the Asian range."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.features.momentum import parse_session_times
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask
from src.strategy.cssm import Signal


def _pip_size(symbol: str, costs_cfg: dict) -> float:
    return float(costs_cfg.get("pip_size", {}).get(symbol, 0.0001 if "JPY" not in symbol else 0.01))


def _daily_atr_series(bars: pd.DataFrame, period: int) -> pd.Series:
    x = ensure_utc_index(bars)
    daily = x.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    daily = daily.dropna()
    a = atr(daily["high"], daily["low"], daily["close"], period=period)
    a = a.shift(1)
    return a.reindex(x.index, method="ffill")


def _range_pct_thresholds(
    range_frac_history: pd.Series,
    lookback: int,
    q_low: float,
    q_high: float,
) -> tuple[float, float] | None:
    hist = range_frac_history.dropna().iloc[-lookback:]
    if len(hist) < max(20, lookback // 4):
        return None
    return float(hist.quantile(q_low)), float(hist.quantile(q_high))


def detect_fail_fade(
    closes: list[float],
    rh: float,
    rl: float,
    break_buf: float = 0.0,
) -> tuple[int, int] | None:
    """Return (break_index, fail_index) for first break then reclaim, or None.

    Side is implied: break above rh -> fade short; break below rl -> fade long.
    Indices are into the closes list.
    """
    break_i: int | None = None
    break_up: bool | None = None
    for i, c in enumerate(closes):
        if break_i is None:
            if c > rh + break_buf:
                break_i, break_up = i, True
            elif c < rl - break_buf:
                break_i, break_up = i, False
            continue
        # Fail = close back inside range (strictly inside)
        if rl < c < rh:
            return break_i, i
    return None


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))
    if news_df is None:
        news_df = pd.DataFrame()

    norm: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        x = ensure_utc_index(df)
        x["atr_d"] = _daily_atr_series(x.reset_index(), int(sig_cfg["atr_period"]))
        norm[sym] = x

    any_sym = next(iter(norm))
    all_days = sorted({ts.normalize() for ts in norm[any_sym].index if ts.dayofweek < 5})

    range_hist: dict[str, list[tuple[pd.Timestamp, float]]] = {s: [] for s in norm}
    signals: list[Signal] = []

    for day in all_days:
        day = pd.Timestamp(day)
        if day.tzinfo is None:
            day = day.tz_localize("UTC")

        for sym, bars in norm.items():
            day_bars = bars.loc[day : day + pd.Timedelta(hours=23, minutes=59)]
            if day_bars.empty:
                continue
            asian_mask = session_mask(day_bars.index, sessions["asian_start"], sessions["asian_end"])
            asian = day_bars.loc[asian_mask]
            if asian.empty:
                continue

            hi, lo = float(asian["high"].max()), float(asian["low"].min())
            width = hi - lo
            pip = _pip_size(sym, costs_cfg)
            width_pips = width / pip if pip > 0 else np.nan
            min_pips = float(sig_cfg.get("min_range_pips", 0.0))
            max_pips = float(sig_cfg.get("max_range_pips", 1e9))
            if not np.isfinite(width_pips) or width_pips < min_pips or width_pips > max_pips:
                continue

            atr_v = float(asian["atr_d"].iloc[-1]) if not np.isnan(asian["atr_d"].iloc[-1]) else np.nan
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            frac = width / atr_v
            range_hist[sym].append((day, frac))
            hist_series = pd.Series(
                [v for _, v in range_hist[sym]], index=[d for d, _ in range_hist[sym]]
            )
            thr = _range_pct_thresholds(
                hist_series.iloc[:-1],
                int(sig_cfg["atr_lookback_days"]),
                float(sig_cfg["range_pct_low"]),
                float(sig_cfg["range_pct_high"]),
            )
            if thr is None or not (thr[0] <= frac <= thr[1]):
                continue

            win_mask = session_mask(
                day_bars.index, sessions["london_start"], sessions["london_end"]
            )
            window = day_bars.loc[win_mask]
            if len(window) < 2:
                continue

            times = list(window.index)
            closes = [float(window.loc[t]["close"]) for t in times]
            break_buf = float(sig_cfg.get("breakout_buffer_pips", 0.0)) * pip
            detected = detect_fail_fade(closes, hi, lo, break_buf=break_buf)
            if detected is None:
                continue
            _break_i, fail_i = detected
            ts = times[fail_i]
            if (
                "friday_no_new_entries" in sessions
                and is_friday(ts)
                and ts.time() >= sessions["friday_no_new_entries"]
            ):
                continue
            if news_cfg.get("enabled", True) and is_news_blackout(
                ts, sym, news_df, int(news_cfg.get("blackout_minutes", 60))
            ):
                continue

            fail_close = closes[fail_i]
            # Infer side from which extreme was broken first
            break_close = closes[_break_i]
            buffer = float(sig_cfg["stop_buffer_pips"]) * pip
            tp_r = float(sig_cfg["take_profit_r"])

            if break_close > hi + break_buf:
                side = -1  # upside fail -> short
                entry = fail_close
                stop = hi + buffer
                risk = stop - entry
                if risk <= 0:
                    continue
                tp = entry - tp_r * risk
            else:
                side = 1  # downside fail -> long
                entry = fail_close
                stop = lo - buffer
                risk = entry - stop
                if risk <= 0:
                    continue
                tp = entry + tp_r * risk

            signals.append(
                Signal(
                    timestamp=ts,
                    symbol=sym,
                    side=side,
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    risk_price=risk,
                    session="london",
                    range_high=hi,
                    range_low=lo,
                    reason="london_fail_fade",
                )
            )

    signals.sort(key=lambda s: s.timestamp)
    return signals
