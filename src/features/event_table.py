"""Build the per-event trade table used by both the strategy and the raw study.

Bar convention (matches the engine): a bar labeled L spans [L, L+bar_len) and
its close is observed at L+bar_len. The engine fills a signal at the close of
the bar whose label equals the signal timestamp, and a scheduled exit fills at
the close of the first bar whose label >= exit_time.

Per (event, pair) row:
    entry_label / entry_px          entry bar for pre & post variants
    pre_exit_time / pre_label       scheduled exit before the print (clean flag)
    post_exit_time / post_label     scheduled exit after the print
    ref_label / react_label         pre-print reference close, first post-print close
    fade_entry_label / fade_exit_*  fade-the-reaction leg
    *_ret (price units), *_r (R units net of round-trip cost), *_atr (ATR units)
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.backtest.costs import pair_cost, pip_size
from src.data.bars import ensure_utc_index
from src.strategy.cssm import _daily_atr_series

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

EVENT_TABLE_COLUMNS = [
    "timestamp", "currency", "country", "event", "preliminary", "symbol", "polarity", "signal_z", "side",
    "actual", "forecast", "surprise_z",
    "entry_label", "entry_px", "atr", "stop_dist", "cost_px",
    "pre_exit_time", "pre_label", "pre_px", "pre_clean", "pre_ret", "pre_r", "pre_atr",
    "post_exit_time", "post_label", "post_px", "post_ret", "post_r", "post_atr",
    "ref_label", "react_label", "reaction", "reaction_atr", "pred_move_atr",
    "fade_side", "fade_entry_label", "fade_entry_px", "fade_exit_time", "fade_exit_label",
    "fade_exit_px", "fade_ret", "fade_r", "fade_atr",
]


def bar_minutes(timeframe: str | None) -> int:
    tf = str(timeframe or "M15").upper()
    if tf not in _TF_MINUTES:
        raise ValueError(f"Unsupported timeframe for event table: {timeframe}")
    return _TF_MINUTES[tf]


def label_at_or_before(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    pos = int(index.searchsorted(ts, side="right")) - 1
    return index[pos] if pos >= 0 else None


def label_at_or_after(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    pos = int(index.searchsorted(ts, side="left"))
    return index[pos] if pos < len(index) else None


def round_trip_cost_px(symbol: str, costs_cfg: dict[str, Any]) -> float:
    c = pair_cost(symbol, costs_cfg)
    return 2.0 * (c["half_spread_pips"] + c["slippage_pips"]) * pip_size(symbol, costs_cfg)


def trade_side(currency: str, symbol: str, polarity: int, signal_z: float) -> int:
    """Side for `symbol` given a pre-release signal on `currency`."""
    if signal_z == 0 or math.isnan(signal_z):
        return 0
    base_sign = 1 if symbol.upper()[:3] == currency.upper() else -1
    return int(polarity) * (1 if signal_z > 0 else -1) * base_sign


def _px(closes: pd.Series, label: pd.Timestamp | None) -> float:
    if label is None:
        return math.nan
    return float(closes.loc[label])


def _sign(x: float) -> int:
    if x is None or math.isnan(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def build_event_table(
    events: pd.DataFrame,
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
) -> pd.DataFrame:
    sig = strategy_cfg.get("signal", {})
    ev_cfg = strategy_cfg.get("events", {})
    bar_len = pd.Timedelta(minutes=bar_minutes(strategy_cfg.get("timeframe", "M15")))
    entry_off = pd.Timedelta(hours=float(sig.get("entry_hours_before", 48)))
    pre_margin = pd.Timedelta(minutes=int(sig.get("pre_exit_margin_min", 30)))
    post_hold = pd.Timedelta(minutes=int(sig.get("post_hold_min", 60)))
    fade_entry = pd.Timedelta(minutes=int(sig.get("fade_entry_min", 15)))
    fade_hold = pd.Timedelta(minutes=int(sig.get("fade_hold_min", 240)))
    fade_req = bool(sig.get("fade_require_signal", True))
    stop_mult = float(sig.get("stop_atr_mult", 1.5))
    atr_period = int(sig.get("atr_period", 20))
    per_event = int(sig.get("pairs_per_event", 1))
    pairs_map = {
        str(k).upper(): [str(p).upper() for p in (v or [])]
        for k, v in (ev_cfg.get("currency_pairs") or {}).items()
    }

    bars = {s.upper(): ensure_utc_index(df) for s, df in bars_by_symbol.items()}
    atrs = {s: _daily_atr_series(df, atr_period) for s, df in bars.items()}
    costs = {s: round_trip_cost_px(s, costs_cfg) for s in bars}

    rows: list[dict[str, Any]] = []
    if events is None or events.empty:
        return pd.DataFrame(columns=EVENT_TABLE_COLUMNS)

    for ev in events.itertuples(index=False):
        z = float(getattr(ev, "signal_z", math.nan))
        if math.isnan(z) or z == 0:
            continue
        cur = str(ev.currency).upper()
        pol = int(getattr(ev, "polarity", 1))
        T = pd.Timestamp(ev.timestamp)
        T = T.tz_localize("UTC") if T.tzinfo is None else T.tz_convert("UTC")
        for sym in [p for p in pairs_map.get(cur, []) if p in bars][:per_event]:
            idx = bars[sym].index
            closes = bars[sym]["close"]
            side = trade_side(cur, sym, pol, z)
            if side == 0:
                continue

            entry_label = label_at_or_before(idx, T - entry_off)
            if entry_label is None:
                continue
            atr_v = atrs[sym].get(entry_label, math.nan)
            atr_v = float(atr_v) if atr_v is not None else math.nan
            if math.isnan(atr_v) or atr_v <= 0:
                continue
            stop_dist = stop_mult * atr_v
            entry_px = _px(closes, entry_label)
            cost = costs[sym]

            pre_exit_time = T - pre_margin - bar_len
            pre_label = label_at_or_after(idx, pre_exit_time)
            pre_clean = (
                pre_label is not None
                and pre_label > entry_label
                and (pre_label + bar_len) <= (T - pre_margin)
            )
            pre_px = _px(closes, pre_label) if pre_clean else math.nan
            pre_ret = side * (pre_px - entry_px) if pre_clean else math.nan

            post_exit_time = T + post_hold
            post_label = label_at_or_after(idx, post_exit_time)
            post_px = _px(closes, post_label)
            post_ret = side * (post_px - entry_px) if post_label is not None else math.nan

            ref_label = label_at_or_before(idx, T - 2 * bar_len)
            react_label = label_at_or_after(idx, T)
            reaction = (
                _px(closes, react_label) - _px(closes, ref_label)
                if (ref_label is not None and react_label is not None)
                else math.nan
            )
            rs = _sign(reaction)
            fade_side = 0
            if rs != 0 and ((not fade_req) or rs == side):
                fade_side = -rs
            fade_entry_label = label_at_or_after(idx, T + fade_entry)
            fade_exit_time = T + fade_hold
            fade_exit_label = label_at_or_after(idx, fade_exit_time)
            fade_ok = (
                fade_side != 0
                and fade_entry_label is not None
                and fade_exit_label is not None
                and fade_exit_label > fade_entry_label
            )
            fade_entry_px = _px(closes, fade_entry_label) if fade_entry_label is not None else math.nan
            fade_exit_px = _px(closes, fade_exit_label) if fade_exit_label is not None else math.nan
            fade_ret = fade_side * (fade_exit_px - fade_entry_px) if fade_ok else math.nan

            rows.append(
                {
                    "timestamp": T,
                    "currency": cur,
                    "country": str(getattr(ev, "country", "") or ""),
                    "event": str(ev.event),
                    "preliminary": bool(getattr(ev, "preliminary", False)),
                    "symbol": sym,
                    "polarity": pol,
                    "signal_z": z,
                    "side": side,
                    "actual": float(getattr(ev, "actual", math.nan)),
                    "forecast": float(getattr(ev, "forecast", math.nan)),
                    "surprise_z": float(getattr(ev, "surprise_z", math.nan)),
                    "entry_label": entry_label,
                    "entry_px": entry_px,
                    "atr": atr_v,
                    "stop_dist": stop_dist,
                    "cost_px": cost,
                    "pre_exit_time": pre_exit_time,
                    "pre_label": pre_label if pre_clean else pd.NaT,
                    "pre_px": pre_px,
                    "pre_clean": bool(pre_clean),
                    "pre_ret": pre_ret,
                    "pre_r": (pre_ret - cost) / stop_dist if pre_clean else math.nan,
                    "pre_atr": pre_ret / atr_v if pre_clean else math.nan,
                    "post_exit_time": post_exit_time,
                    "post_label": post_label if post_label is not None else pd.NaT,
                    "post_px": post_px,
                    "post_ret": post_ret,
                    "post_r": (post_ret - cost) / stop_dist if post_label is not None else math.nan,
                    "post_atr": post_ret / atr_v if post_label is not None else math.nan,
                    "ref_label": ref_label if ref_label is not None else pd.NaT,
                    "react_label": react_label if react_label is not None else pd.NaT,
                    "reaction": reaction,
                    "reaction_atr": reaction / atr_v if not math.isnan(reaction) else math.nan,
                    "pred_move_atr": side * reaction / atr_v if not math.isnan(reaction) else math.nan,
                    "fade_side": fade_side,
                    "fade_entry_label": fade_entry_label if fade_ok else pd.NaT,
                    "fade_entry_px": fade_entry_px if fade_ok else math.nan,
                    "fade_exit_time": fade_exit_time,
                    "fade_exit_label": fade_exit_label if fade_ok else pd.NaT,
                    "fade_exit_px": fade_exit_px if fade_ok else math.nan,
                    "fade_ret": fade_ret,
                    "fade_r": (fade_ret - cost) / stop_dist if fade_ok else math.nan,
                    "fade_atr": fade_ret / atr_v if fade_ok else math.nan,
                }
            )

    if not rows:
        return pd.DataFrame(columns=EVENT_TABLE_COLUMNS)
    out = pd.DataFrame(rows)[EVENT_TABLE_COLUMNS]
    return out.sort_values(["timestamp", "symbol", "event"]).reset_index(drop=True)
