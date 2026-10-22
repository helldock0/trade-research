from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.features.momentum import (
    overnight_return,
    pair_trade_direction,
    parse_session_times,
    top_tercile_mask,
    usd_signed_return,
    cross_sectional_ranks,
)
from src.features.news import is_news_blackout, load_news_calendar
from src.features.sessions import is_friday, session_mask


@dataclass
class Signal:
    timestamp: pd.Timestamp
    symbol: str
    side: int  # +1 long / -1 short
    entry: float
    stop: float
    take_profit: float
    risk_price: float
    session: str
    range_high: float
    range_low: float
    reason: str
    # Optional scheduled exit (event studies); engine closes at first bar >= exit_time
    exit_time: pd.Timestamp | None = None


def _pip_size(symbol: str, costs_cfg: dict) -> float:
    return float(costs_cfg.get("pip_size", {}).get(symbol, 0.0001 if "JPY" not in symbol else 0.01))


def _daily_atr_series(bars: pd.DataFrame, period: int) -> pd.Series:
    """Compute ATR on daily bars derived from M15."""
    x = ensure_utc_index(bars)
    daily = x.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    daily = daily.dropna()
    a = atr(daily["high"], daily["low"], daily["close"], period=period)
    # forward-fill onto M15 index using prior completed day only (shift 1)
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


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    news_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Generate CSSM signals with no lookahead (decisions on bar close)."""
    sessions = parse_session_times(strategy_cfg)
    sig_cfg = strategy_cfg["signal"]
    news_cfg = strategy_cfg.get("news", {})
    if news_df is None and news_cfg.get("enabled", True):
        news_df = load_news_calendar(news_cfg.get("calendar_csv"))

    # Normalize bars
    norm: dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_symbol.items():
        x = ensure_utc_index(df)
        x["atr_d"] = _daily_atr_series(x.reset_index(), sig_cfg["atr_period"])
        norm[sym] = x

    # Common calendar days from first symbol
    any_sym = next(iter(norm))
    all_days = sorted({ts.normalize() for ts in norm[any_sym].index if ts.dayofweek < 5})

    # Rolling range fraction history per symbol
    range_hist: dict[str, list[tuple[pd.Timestamp, float]]] = {s: [] for s in norm}

    signals: list[Signal] = []
    open_day_trade: set[tuple[str, pd.Timestamp]] = set()  # one setup attempt tracking

    for day in all_days:
        day = pd.Timestamp(day)
        if day.tzinfo is None:
            day = day.tz_localize("UTC")

        # --- overnight cross-section at London open ---
        usd_rets: dict[str, float] = {}
        pair_rets: dict[str, float] = {}
        for sym, bars in norm.items():
            r = overnight_return(
                bars, day, sessions["overnight_anchor"], sessions["london_start"]
            )
            if r is None:
                continue
            pair_rets[sym] = r
            usd_rets[sym] = usd_signed_return(sym, r)

        if len(usd_rets) < 3:
            continue

        ranks = cross_sectional_ranks(usd_rets)
        usd_side = top_tercile_mask(ranks)

        # Asian ranges + regime gate
        day_meta: dict[str, dict] = {}
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
            atr_v = float(asian["atr_d"].iloc[-1]) if not np.isnan(asian["atr_d"].iloc[-1]) else np.nan
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            frac = width / atr_v
            range_hist[sym].append((day, frac))
            hist_series = pd.Series([v for _, v in range_hist[sym]], index=[d for d, _ in range_hist[sym]])
            # thresholds from history excluding today
            hist_ex = hist_series.iloc[:-1]
            thr = _range_pct_thresholds(
                hist_ex,
                sig_cfg["atr_lookback_days"],
                sig_cfg["range_pct_low"],
                sig_cfg["range_pct_high"],
            )
            regime_ok = False
            if thr is not None:
                regime_ok = thr[0] <= frac <= thr[1]
            day_meta[sym] = {
                "asian_high": hi,
                "asian_low": lo,
                "width": width,
                "frac": frac,
                "regime_ok": regime_ok,
                "atr": atr_v,
                "day_bars": day_bars,
            }

        # London session signals
        _scan_session(
            day=day,
            session_name="london",
            window_start=sessions["london_start"],
            window_end=sessions["london_end"],
            day_meta=day_meta,
            usd_side=usd_side,
            sessions=sessions,
            sig_cfg=sig_cfg,
            costs_cfg=costs_cfg,
            news_df=news_df if news_df is not None else pd.DataFrame(),
            news_cfg=news_cfg,
            signals=signals,
            open_day_trade=open_day_trade,
            use_asian_range=True,
        )

        # NY session — consolidation from london_end to ny_start
        if sig_cfg.get("enable_ny_session", True):
            for sym, meta in day_meta.items():
                day_bars = meta["day_bars"]
                consol_mask = session_mask(day_bars.index, sessions["london_end"], sessions["ny_start"])
                consol = day_bars.loc[consol_mask]
                if consol.empty:
                    meta["ny_high"] = meta["asian_high"]
                    meta["ny_low"] = meta["asian_low"]
                else:
                    meta["ny_high"] = float(consol["high"].max())
                    meta["ny_low"] = float(consol["low"].min())
                    width = meta["ny_high"] - meta["ny_low"]
                    atr_v = meta["atr"]
                    frac = width / atr_v if atr_v else np.nan
                    hist_series = pd.Series(
                        [v for _, v in range_hist[sym]], index=[d for d, _ in range_hist[sym]]
                    )
                    thr = _range_pct_thresholds(
                        hist_series.iloc[:-1],
                        sig_cfg["atr_lookback_days"],
                        sig_cfg["range_pct_low"],
                        sig_cfg["range_pct_high"],
                    )
                    meta["ny_regime_ok"] = bool(thr and thr[0] <= frac <= thr[1]) if thr else False
                if "ny_regime_ok" not in meta:
                    meta["ny_regime_ok"] = meta["regime_ok"]

            _scan_session(
                day=day,
                session_name="ny",
                window_start=sessions["ny_start"],
                window_end=sessions["ny_end"],
                day_meta=day_meta,
                usd_side=usd_side,
                sessions=sessions,
                sig_cfg=sig_cfg,
                costs_cfg=costs_cfg,
                news_df=news_df if news_df is not None else pd.DataFrame(),
                news_cfg=news_cfg,
                signals=signals,
                open_day_trade=open_day_trade,
                use_asian_range=False,
            )

    signals.sort(key=lambda s: s.timestamp)
    return signals


def _scan_session(
    day: pd.Timestamp,
    session_name: str,
    window_start: time,
    window_end: time,
    day_meta: dict,
    usd_side: dict[str, int],
    sessions: dict,
    sig_cfg: dict,
    costs_cfg: dict,
    news_df: pd.DataFrame,
    news_cfg: dict,
    signals: list[Signal],
    open_day_trade: set,
    use_asian_range: bool,
) -> None:
    for sym, meta in day_meta.items():
        side_usd = usd_side.get(sym, 0)
        direction = pair_trade_direction(sym, side_usd)
        if direction == 0:
            continue

        if use_asian_range:
            if not meta.get("regime_ok"):
                continue
            rh, rl = meta["asian_high"], meta["asian_low"]
        else:
            if not meta.get("ny_regime_ok", False):
                continue
            rh, rl = meta.get("ny_high", meta["asian_high"]), meta.get("ny_low", meta["asian_low"])

        day_bars = meta["day_bars"]
        win_mask = session_mask(day_bars.index, window_start, window_end)
        window = day_bars.loc[win_mask]
        if window.empty:
            continue

        pip = _pip_size(sym, costs_cfg)
        buffer = float(sig_cfg["stop_buffer_pips"]) * pip
        tp_r = float(sig_cfg["take_profit_r"])
        require_confirm = bool(sig_cfg.get("require_confirm_bar", False))

        times = list(window.index)
        for i, ts in enumerate(times):
            if is_friday(ts) and ts.time() >= sessions["friday_no_new_entries"]:
                break
            key = (sym, day.normalize(), session_name)
            if key in open_day_trade:
                break

            bar = window.loc[ts]
            if news_cfg.get("enabled", True) and is_news_blackout(
                ts, sym, news_df, int(news_cfg.get("blackout_minutes", 60))
            ):
                continue

            close = float(bar["close"])
            break_long = direction > 0 and close > rh
            break_short = direction < 0 and close < rl
            if not (break_long or break_short):
                continue

            # v1.2: confirmation on the next M15 close still beyond the range
            entry_ts = ts
            entry_px = close
            if require_confirm:
                if i + 1 >= len(times):
                    continue
                conf_ts = times[i + 1]
                if is_friday(conf_ts) and conf_ts.time() >= sessions["friday_no_new_entries"]:
                    continue
                if news_cfg.get("enabled", True) and is_news_blackout(
                    conf_ts, sym, news_df, int(news_cfg.get("blackout_minutes", 60))
                ):
                    continue
                conf_close = float(window.loc[conf_ts]["close"])
                if break_long and not (conf_close > rh and conf_close > close):
                    continue
                if break_short and not (conf_close < rl and conf_close < close):
                    continue
                entry_ts = conf_ts
                entry_px = conf_close

            if direction > 0:
                entry = entry_px
                stop = rl - buffer
                risk = entry - stop
                if risk <= 0:
                    continue
                tp = entry + tp_r * risk
            else:
                entry = entry_px
                stop = rh + buffer
                risk = stop - entry
                if risk <= 0:
                    continue
                tp = entry - tp_r * risk

            reason = f"{session_name}_break_mom"
            if require_confirm:
                reason = f"{session_name}_break_confirm"

            signals.append(
                Signal(
                    timestamp=entry_ts,
                    symbol=sym,
                    side=direction,
                    entry=entry,
                    stop=stop,
                    take_profit=tp,
                    risk_price=risk,
                    session=session_name,
                    range_high=rh,
                    range_low=rl,
                    reason=reason,
                )
            )
            open_day_trade.add(key)
            break
