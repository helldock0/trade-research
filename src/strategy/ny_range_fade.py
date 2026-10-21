"""NY range fade — fade the first break of the 08:12-09:12 New York range.

User-supplied rules:
  - Range: high/low of the M1 bars from 08:12 to 09:12 New York time.
  - Break above the range high -> sell, TP at the range low.
  - Break below the range low  -> buy,  TP at the range high.

The rules give no stop and no time exit, so both are config knobs:
  - stop_mode: none (literal) | range_mult (stop = broken edge +/- k * range width)
               | atr (stop = entry +/- k * ATR(atr_period) on atr_timeframe bars)
  - entry_end / flat_time: last NY time for new entries / force-flat NY time

Self-contained M1 simulator: the shared engine recomputes TP as an R multiple and
uses fixed-UTC session exits, which does not fit a fixed-price TP on NY local time.
P&L is reported in pips and range units (RU = P&L / range width).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import (
    entry_fill_price,
    exit_fill_price,
    pair_cost,
    pip_size,
    pip_value_per_lot,
)
from src.data.bars import atr, ensure_utc_index, resample_ohlc
from src.features.sessions import parse_hhmm


@dataclass
class FadeTrade:
    symbol: str
    day: pd.Timestamp  # NY session date
    side: int
    entry_time: pd.Timestamp  # bar-open timestamps (UTC)
    exit_time: pd.Timestamp
    entry_price: float  # fills after spread/slippage
    exit_price: float
    range_high: float
    range_low: float
    range_pips: float
    take_profit: float
    stop: float  # nan when stop_mode == none
    stop_ru: float  # entry-to-stop distance in range widths (nan when no stop)
    exit_reason: str  # tp | stop | time
    gross_ru: float
    pnl_pips: float
    pnl_ru: float
    mae_ru: float  # max adverse excursion vs entry, in range widths


def _minute_of_day(hhmm: str) -> int:
    t = parse_hhmm(hhmm)
    return t.hour * 60 + t.minute


def _session_minutes(strategy_cfg: dict[str, Any]) -> dict[str, int]:
    s = strategy_cfg["sessions"]
    out = {k: _minute_of_day(s[k]) for k in ("range_start", "range_end", "entry_end", "flat_time")}
    if not out["range_start"] < out["range_end"] <= out["entry_end"] <= out["flat_time"]:
        raise ValueError(f"Expected range_start < range_end <= entry_end <= flat_time, got {s}")
    return out


def completed_atr(bars: pd.DataFrame, rule: str, period: int) -> pd.Series:
    """ATR(period) on `rule` bars, as known at each M1 bar's open (completed bars only)."""
    htf = resample_ohlc(bars, rule)
    a = atr(htf["high"], htf["low"], htf["close"], period=period)
    # A bar labelled T (closed=left) is only complete at T + rule
    a.index = a.index + pd.tseries.frequencies.to_offset(rule)
    return a.reindex(ensure_utc_index(bars).index, method="ffill")


def simulate_symbol(
    bars: pd.DataFrame,
    symbol: str,
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
) -> tuple[list[FadeTrade], dict[str, int]]:
    """Simulate the fade on one symbol's M1 bars. Returns (trades, day counters)."""
    sig = strategy_cfg["signal"]
    mins_cfg = _session_minutes(strategy_cfg)
    rs, re_, entry_end, flat = (
        mins_cfg["range_start"],
        mins_cfg["range_end"],
        mins_cfg["entry_end"],
        mins_cfg["flat_time"],
    )
    entry_mode = str(sig.get("entry_mode", "close"))
    if entry_mode not in ("close", "touch"):
        raise ValueError(f"entry_mode must be close|touch, got {entry_mode}")
    stop_mode = str(sig.get("stop_mode", "none"))
    if stop_mode not in ("none", "range_mult", "atr"):
        raise ValueError(f"stop_mode must be none|range_mult|atr, got {stop_mode}")
    stop_mult = float(sig.get("stop_range_mult", 1.0))
    atr_mult = float(sig.get("stop_atr_mult", 1.0))
    has_stop = stop_mode != "none"
    max_trades = int(sig.get("max_trades_per_day", 1))
    min_range_bars = int(sig.get("min_range_bars", 0))
    min_range_pips = float(sig.get("min_range_pips", 0.0) or 0.0)
    max_range_pips = float(sig.get("max_range_pips") or np.inf)

    pip = pip_size(symbol, costs_cfg)
    comm_usd = pair_cost(symbol, costs_cfg)["commission_per_lot_usd"]

    x = ensure_utc_index(bars)
    if stop_mode == "atr":
        x["atr"] = completed_atr(x, str(sig.get("atr_timeframe", "15min")), int(sig.get("atr_period", 14)))
    local = x.index.tz_convert(strategy_cfg["sessions"]["timezone"])
    mins = np.asarray(local.hour * 60 + local.minute)
    keep = (mins >= rs) & (mins < flat) & np.asarray(local.dayofweek < 5)
    x, local, mins = x.loc[keep], local[keep], mins[keep]
    utc_idx = x.index
    o = x["open"].to_numpy(float)
    h = x["high"].to_numpy(float)
    lo = x["low"].to_numpy(float)
    c = x["close"].to_numpy(float)
    atr_v = x["atr"].to_numpy(float) if stop_mode == "atr" else None

    day_key = local.normalize()
    day_ns = day_key.asi8
    starts = np.r_[0, np.flatnonzero(np.diff(day_ns)) + 1]
    ends = np.r_[starts[1:], len(day_ns)]

    trades: list[FadeTrade] = []
    counts = {
        "days": 0,
        "days_with_range": 0,
        "skipped_ambiguous": 0,
        "skipped_beyond_stop": 0,
        "skipped_no_atr": 0,
    }

    for s, e in zip(starts, ends):
        counts["days"] += 1
        m = mins[s:e]
        r_lo, r_hi = np.searchsorted(m, rs), np.searchsorted(m, re_)
        if r_hi - r_lo == 0 or r_hi - r_lo < min_range_bars:
            continue
        rh = float(h[s + r_lo : s + r_hi].max())
        rl = float(lo[s + r_lo : s + r_hi].min())
        width = rh - rl
        width_pips = width / pip
        if width <= 0 or width_pips < min_range_pips or width_pips > max_range_pips:
            continue
        counts["days_with_range"] += 1

        n = e - s
        scan_end = int(np.searchsorted(m, entry_end))
        i = int(r_hi)
        taken = 0
        while taken < max_trades and i < scan_end:
            if entry_mode == "close":
                up = c[s + i : s + scan_end] > rh
                dn = c[s + i : s + scan_end] < rl
            else:
                up = h[s + i : s + scan_end] > rh
                dn = lo[s + i : s + scan_end] < rl
            hits = np.flatnonzero(up | dn)
            if hits.size == 0:
                break
            k = i + int(hits[0])
            if up[hits[0]] and dn[hits[0]]:
                # One M1 bar spans the whole range: fill order unknowable
                counts["skipped_ambiguous"] += 1
                break
            side = -1 if up[hits[0]] else 1
            g = s + k
            if entry_mode == "close":
                entry_mid = c[g]
                first_exit = k + 1
            else:
                # Resting limit at the broken edge (better fill if the bar gapped through)
                entry_mid = max(rh, o[g]) if side < 0 else min(rl, o[g])
                first_exit = k
            tp = rl if side < 0 else rh
            if stop_mode == "range_mult":
                stop = rh + stop_mult * width if side < 0 else rl - stop_mult * width
            elif stop_mode == "atr":
                if not np.isfinite(atr_v[g]) or atr_v[g] <= 0:
                    counts["skipped_no_atr"] += 1
                    break
                stop = entry_mid - side * atr_mult * atr_v[g]
            else:
                stop = np.nan
            if has_stop and (entry_mid >= stop if side < 0 else entry_mid <= stop):
                counts["skipped_beyond_stop"] += 1
                break

            ex_h, ex_l, ex_o = h[s + first_exit : e], lo[s + first_exit : e], o[s + first_exit : e]
            if side < 0:
                tp_hit = ex_l <= tp
                sl_hit = ex_h >= stop if has_stop else np.zeros(len(ex_h), bool)
            else:
                tp_hit = ex_h >= tp
                sl_hit = ex_l <= stop if has_stop else np.zeros(len(ex_h), bool)
            if entry_mode == "touch" and len(tp_hit):
                tp_hit[0] = False  # entry bar: only adverse moves are provably after the fill
            f_tp = int(np.argmax(tp_hit)) if tp_hit.any() else len(tp_hit)
            f_sl = int(np.argmax(sl_hit)) if sl_hit.any() else len(sl_hit)

            if f_sl < len(sl_hit) and f_sl <= f_tp:
                # Same-bar TP+SL resolves to the stop (conservative); gaps fill at the open
                rel = f_sl
                exit_mid = max(stop, ex_o[rel]) if side < 0 else min(stop, ex_o[rel])
                if entry_mode == "touch" and rel == 0:
                    exit_mid = stop
                reason = "stop"
            elif f_tp < len(tp_hit):
                rel, exit_mid, reason = f_tp, tp, "tp"
            else:
                rel = len(tp_hit) - 1
                exit_mid = c[e - 1] if rel >= 0 else entry_mid
                reason = "time"
            x_rel = first_exit + rel if rel >= 0 else k

            path_h = h[s + first_exit : s + x_rel + 1]
            path_l = lo[s + first_exit : s + x_rel + 1]
            if side < 0:
                adverse = (path_h.max() - entry_mid) if path_h.size else 0.0
            else:
                adverse = (entry_mid - path_l.min()) if path_l.size else 0.0
            if reason == "stop":
                adverse = max(adverse, abs(stop - entry_mid))

            e_fill = entry_fill_price(side, entry_mid, symbol, costs_cfg)
            x_fill = exit_fill_price(side, exit_mid, symbol, costs_cfg)
            comm_pips = comm_usd / pip_value_per_lot(symbol, entry_mid)
            pnl_pips = side * (x_fill - e_fill) / pip - comm_pips

            trades.append(
                FadeTrade(
                    symbol=symbol,
                    day=day_key[s].tz_localize(None),
                    side=side,
                    entry_time=utc_idx[g],
                    exit_time=utc_idx[s + x_rel],
                    entry_price=e_fill,
                    exit_price=x_fill,
                    range_high=rh,
                    range_low=rl,
                    range_pips=width_pips,
                    take_profit=tp,
                    stop=stop,
                    stop_ru=abs(stop - entry_mid) / width,
                    exit_reason=reason,
                    gross_ru=side * (exit_mid - entry_mid) / width,
                    pnl_pips=pnl_pips,
                    pnl_ru=pnl_pips * pip / width,
                    mae_ru=max(adverse, 0.0) / width,
                )
            )
            taken += 1
            i = x_rel + 1
            if i >= n:
                break

    return trades, counts


def run_fade(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Simulate every symbol; returns (trades frame sorted by entry, per-symbol counters)."""
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for sym, bars in bars_by_symbol.items():
        trades, cnt = simulate_symbol(bars, sym, strategy_cfg, costs_cfg)
        rows.extend(asdict(t) for t in trades)
        counts[sym] = cnt
    tf = pd.DataFrame(rows)
    if not tf.empty:
        tf = tf.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    return tf, counts


def summarize(tf: pd.DataFrame) -> dict[str, Any]:
    """Expectancy/PF/drawdown in range units (equal 1-RU weight per trade)."""
    if tf.empty:
        return {"n_trades": 0}
    ru = tf["pnl_ru"].to_numpy(float)
    wins, losses = ru[ru > 0].sum(), -ru[ru < 0].sum()
    cum = np.cumsum(ru)
    dd = cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]
    std = ru.std(ddof=1) if len(ru) > 1 else np.nan
    reasons = tf["exit_reason"].value_counts(normalize=True)
    return {
        "n_trades": int(len(tf)),
        "win_rate": float((ru > 0).mean()),
        "tp_rate": float(reasons.get("tp", 0.0)),
        "stop_rate": float(reasons.get("stop", 0.0)),
        "time_exit_rate": float(reasons.get("time", 0.0)),
        "expectancy_ru": float(ru.mean()),
        "gross_expectancy_ru": float(tf["gross_ru"].mean()),
        "t_stat": float(ru.mean() / (std / np.sqrt(len(ru)))) if std and std > 0 else float("nan"),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "avg_win_ru": float(ru[ru > 0].mean()) if (ru > 0).any() else 0.0,
        "avg_loss_ru": float(ru[ru < 0].mean()) if (ru < 0).any() else 0.0,
        "worst_ru": float(ru.min()),
        "total_ru": float(cum[-1]),
        "max_dd_ru": float(-dd.min()),
        "expectancy_pips": float(tf["pnl_pips"].mean()),
        "median_range_pips": float(tf["range_pips"].median()),
        "median_stop_ru": float(tf["stop_ru"].median()),
        "p95_mae_ru": float(tf["mae_ru"].quantile(0.95)),
    }
