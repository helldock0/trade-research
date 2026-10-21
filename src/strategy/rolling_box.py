"""Rolling 15-minute box breakout (Tokyo open -> NY end).

Every M15 candle between ``sessions.start`` and ``sessions.end`` (UTC) is a box.
The first ``confirm_tf``-minute candle that closes beyond the box high (long) or
low (short) within ``window`` minutes of the box close is the entry, filled at
that candle's close. Trades are simulated on M1 bars and are always flat at
``sessions.end``. All times are integer minutes since the Unix epoch (UTC).

Stop specs:  ``opposite`` (other side of box), ``mid`` (box midpoint),
             ``atr:K`` (K x M15 ATR from entry).
Exit specs:  ``tp:R`` (fixed target), ``time:N`` (close N minutes after entry),
             ``session`` (hold to session end), ``trail_r:A:D`` (once +A R,
             trail D R behind the best price), ``trail_atr:K`` (trail K x ATR
             behind the best price from entry).
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from src.data.bars import atr as atr_series
from src.data.bars import ensure_utc_index, resample_ohlc

try:
    from numba import njit
except ImportError:  # pragma: no cover - pure-Python fallback (slow)

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


STOP_OPPOSITE, STOP_MID, STOP_ATR = 0, 1, 2
EXIT_TP, EXIT_TIME, EXIT_SESSION, EXIT_TRAIL_R, EXIT_TRAIL_ATR = 0, 1, 2, 3, 4
REASON_STOP, REASON_TP, REASON_TIME, REASON_SESSION, REASON_TRAIL, REASON_NONE = range(6)
REASON_NAMES = ("stop", "tp", "time", "session", "trail_stop", "none")


def parse_stop(spec: str) -> tuple[int, float]:
    parts = spec.strip().lower().split(":")
    if parts[0] == "opposite":
        return STOP_OPPOSITE, 0.0
    if parts[0] == "mid":
        return STOP_MID, 0.0
    if parts[0] == "atr":
        return STOP_ATR, float(parts[1]) if len(parts) > 1 else 1.0
    raise ValueError(f"Unknown stop spec: {spec}")


def parse_exit(spec: str) -> tuple[int, float, float]:
    parts = spec.strip().lower().split(":")
    kind = parts[0]
    if kind == "tp":
        return EXIT_TP, float(parts[1]), 0.0
    if kind == "time":
        return EXIT_TIME, float(parts[1]), 0.0
    if kind == "session":
        return EXIT_SESSION, 0.0, 0.0
    if kind == "trail_r":
        activate = float(parts[1]) if len(parts) > 1 else 1.0
        distance = float(parts[2]) if len(parts) > 2 else activate
        return EXIT_TRAIL_R, activate, distance
    if kind == "trail_atr":
        return EXIT_TRAIL_ATR, float(parts[1]) if len(parts) > 1 else 1.0, 0.0
    raise ValueError(f"Unknown exit spec: {spec}")


def to_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    return (pd.DatetimeIndex(index).as_unit("s").asi8 // 60).astype(np.int64)


def _hhmm_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def m1_arrays(m1: pd.DataFrame) -> dict[str, np.ndarray]:
    x = ensure_utc_index(m1)
    x = x[~x.index.duplicated(keep="last")]
    return {
        "t": to_minutes(x.index),
        "open": x["open"].to_numpy(np.float64),
        "high": x["high"].to_numpy(np.float64),
        "low": x["low"].to_numpy(np.float64),
        "close": x["close"].to_numpy(np.float64),
    }


def build_boxes(
    m1: pd.DataFrame,
    session_start: time,
    session_end: time,
    box_minutes: int = 15,
    atr_period: int = 14,
) -> pd.DataFrame:
    """One row per box candle that closes inside the weekday session window."""
    start_min, end_min = _hhmm_minutes(session_start), _hhmm_minutes(session_end)
    if start_min >= end_min:
        raise ValueError("Session window must not wrap midnight (start < end in UTC)")
    bars = resample_ohlc(m1, f"{box_minutes}min")
    # ATR on the continuous box-timeframe series, inclusive of the box bar
    # (known at box close).
    bars["atr"] = atr_series(bars["high"], bars["low"], bars["close"], period=atr_period)
    idx = bars.index
    tod = idx.hour * 60 + idx.minute
    keep = (idx.dayofweek < 5) & (tod >= start_min) & (tod + box_minutes <= end_min)
    b = bars.loc[keep]
    start = to_minutes(b.index)
    day0 = start - (b.index.hour * 60 + b.index.minute).to_numpy()
    return pd.DataFrame(
        {
            "box_start": start,
            "box_end": start + box_minutes,
            "high": b["high"].to_numpy(np.float64),
            "low": b["low"].to_numpy(np.float64),
            "atr": b["atr"].to_numpy(np.float64),
            "sess_end": day0 + end_min,
            "hour_utc": b.index.hour.to_numpy(),
            "year": b.index.year.to_numpy(),
        }
    )


def confirm_candles(m1: pd.DataFrame, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """(candle start minutes, candle close) for the confirmation timeframe."""
    bars = ensure_utc_index(m1) if minutes == 1 else resample_ohlc(m1, f"{minutes}min")
    bars = bars[~bars.index.duplicated(keep="last")]
    return to_minutes(bars.index), bars["close"].to_numpy(np.float64)


@njit(cache=True)
def first_breakouts(box_end, box_hi, box_lo, sess_end, c_start, c_close, c_minutes, max_window):
    """First confirmation close beyond each box within ``max_window`` minutes.

    Returns (side, entry_t, entry_px); side 0 means no breakout. The entry
    candle must close strictly before the session end.
    """
    n = box_end.shape[0]
    nc = c_start.shape[0]
    side = np.zeros(n, np.int8)
    entry_t = np.full(n, -1, np.int64)
    entry_px = np.full(n, np.nan)
    j = 0
    for b in range(n):
        while j < nc and c_start[j] < box_end[b]:
            j += 1
        if box_hi[b] <= box_lo[b]:
            continue
        k = j
        while k < nc:
            cend = c_start[k] + c_minutes
            if cend > box_end[b] + max_window or cend >= sess_end[b]:
                break
            if c_close[k] > box_hi[b]:
                side[b] = 1
            elif c_close[k] < box_lo[b]:
                side[b] = -1
            if side[b] != 0:
                entry_t[b] = cend
                entry_px[b] = c_close[k]
                break
            k += 1
    return side, entry_t, entry_px


@njit(cache=True)
def simulate_trades(
    m1_t,
    m1_o,
    m1_h,
    m1_l,
    m1_c,
    entry_idx,
    entry_t,
    side,
    entry_px,
    box_hi,
    box_lo,
    box_atr,
    sess_end,
    stop_types,
    stop_params,
    exit_types,
    exit_p1,
    exit_p2,
):
    """Simulate every (stop, exit) variant for every entry on M1 bars.

    Prices are mirrored for shorts so one long-only loop handles both sides.
    Within a bar the stop is checked before the target (conservative). Trailing
    stops update after the bar closes and apply from the next bar.

    Returns pnl[n,S,E] (price move in trade direction, no costs),
    exit_t[n,S,E] (minute the exit bar closes), reason[n,S,E] and the initial
    risk[n,S] (entry-to-stop distance; NaN when the stop is invalid).
    """
    n = entry_idx.shape[0]
    n_stops = stop_types.shape[0]
    n_exits = exit_types.shape[0]
    nb = m1_t.shape[0]
    pnl = np.full((n, n_stops, n_exits), np.nan)
    exit_t = np.full((n, n_stops, n_exits), -1, np.int64)
    reason = np.full((n, n_stops, n_exits), REASON_NONE, np.int8)
    risk = np.full((n, n_stops), np.nan)
    for i in range(n):
        sd = side[i]
        if sd == 0:
            continue
        ent = entry_px[i] * sd
        for s in range(n_stops):
            st = stop_types[s]
            if st == STOP_OPPOSITE:
                stop_px = box_lo[i] if sd > 0 else box_hi[i]
            elif st == STOP_MID:
                stop_px = 0.5 * (box_hi[i] + box_lo[i])
            else:
                stop_px = entry_px[i] - sd * stop_params[s] * box_atr[i]
            stop0 = stop_px * sd
            r0 = ent - stop0
            if not (r0 > 0.0):
                continue
            risk[i, s] = r0
            for e in range(n_exits):
                et = exit_types[e]
                tp = ent + exit_p1[e] * r0 if et == EXIT_TP else np.inf
                stop = stop0
                best = ent
                moved = False
                k = entry_idx[i]
                while k < nb and m1_t[k] < sess_end[i]:
                    if sd > 0:
                        o, h, lo, c = m1_o[k], m1_h[k], m1_l[k], m1_c[k]
                    else:
                        o, h, lo, c = -m1_o[k], -m1_l[k], -m1_h[k], -m1_c[k]
                    px = np.nan
                    rs = -1
                    if o <= stop:
                        px = o
                        rs = REASON_TRAIL if moved else REASON_STOP
                    elif lo <= stop:
                        px = stop
                        rs = REASON_TRAIL if moved else REASON_STOP
                    elif h >= tp:
                        px = tp
                        rs = REASON_TP
                    elif et == EXIT_TIME and m1_t[k] + 1 >= entry_t[i] + exit_p1[e]:
                        px = c
                        rs = REASON_TIME
                    elif k + 1 >= nb or m1_t[k + 1] >= sess_end[i]:
                        px = c
                        rs = REASON_SESSION
                    if rs >= 0:
                        pnl[i, s, e] = px - ent
                        exit_t[i, s, e] = m1_t[k] + 1
                        reason[i, s, e] = rs
                        break
                    if h > best:
                        best = h
                    if et == EXIT_TRAIL_R:
                        if best - ent >= exit_p1[e] * r0:
                            ns = best - exit_p2[e] * r0
                            if ns > stop:
                                stop = ns
                                moved = True
                    elif et == EXIT_TRAIL_ATR:
                        ns = best - exit_p1[e] * box_atr[i]
                        if ns > stop:
                            stop = ns
                            moved = True
                    k += 1
    return pnl, exit_t, reason, risk


@njit(cache=True)
def one_at_a_time(entry_t, exit_t, mask):
    """Greedy filter: take a trade only when the previous one has closed.

    Inputs must be sorted by entry time.
    """
    keep = np.zeros(entry_t.shape[0], np.bool_)
    last_exit = np.int64(-1)
    for i in range(entry_t.shape[0]):
        if mask[i] and entry_t[i] >= last_exit:
            keep[i] = True
            last_exit = exit_t[i]
    return keep


@njit(cache=True)
def max_drawdown(r):
    """Peak-to-trough drop of the cumulative R curve (positive number)."""
    peak = 0.0
    cum = 0.0
    dd = 0.0
    for x in r:
        cum += x
        if cum > peak:
            peak = cum
        if peak - cum > dd:
            dd = peak - cum
    return dd


def breakout_candidates(
    m1: pd.DataFrame,
    m1a: dict[str, np.ndarray],
    boxes: pd.DataFrame,
    confirm_minutes: int,
    max_window: int,
) -> pd.DataFrame:
    """Boxes that broke out within ``max_window``, with entry details."""
    c_start, c_close = confirm_candles(m1, confirm_minutes)
    side, entry_t, entry_px = first_breakouts(
        boxes["box_end"].to_numpy(),
        boxes["high"].to_numpy(),
        boxes["low"].to_numpy(),
        boxes["sess_end"].to_numpy(),
        c_start,
        c_close,
        np.int64(confirm_minutes),
        np.int64(max_window),
    )
    cand = boxes.assign(side=side, entry_t=entry_t, entry_px=entry_px)
    cand = cand.loc[cand["side"] != 0].copy()
    cand["delay"] = cand["entry_t"] - cand["box_end"]
    cand["entry_idx"] = np.searchsorted(m1a["t"], cand["entry_t"].to_numpy(), side="left")
    cand = cand.sort_values(["entry_t", "box_start"], kind="stable").reset_index(drop=True)
    return cand


def simulate_candidates(
    m1a: dict[str, np.ndarray],
    cand: pd.DataFrame,
    stops: list[str],
    exits: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    st = [parse_stop(s) for s in stops]
    ex = [parse_exit(e) for e in exits]
    return simulate_trades(
        m1a["t"],
        m1a["open"],
        m1a["high"],
        m1a["low"],
        m1a["close"],
        cand["entry_idx"].to_numpy(np.int64),
        cand["entry_t"].to_numpy(np.int64),
        cand["side"].to_numpy(np.int8),
        cand["entry_px"].to_numpy(np.float64),
        cand["high"].to_numpy(np.float64),
        cand["low"].to_numpy(np.float64),
        cand["atr"].to_numpy(np.float64),
        cand["sess_end"].to_numpy(np.int64),
        np.array([s[0] for s in st], np.int64),
        np.array([s[1] for s in st], np.float64),
        np.array([e[0] for e in ex], np.int64),
        np.array([e[1] for e in ex], np.float64),
        np.array([e[2] for e in ex], np.float64),
    )


def net_r(pnl, risk, side_cost: float, round_trip_cost: float):
    """R multiple after costs; 1R = distance from the filled entry to the stop."""
    return (pnl - round_trip_cost) / (risk + side_cost)
