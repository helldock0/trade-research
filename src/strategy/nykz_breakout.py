"""NY kill-zone breakout — NAS100 M5 with real bid/ask fills.

Rules (session times are New York local, so US DST is handled):
- Range = bid high/low of bars opening in [range_start, range_end).
- The first later bar whose bid close is above the range high (long) or below
  the range low (short) is the break bar. One attempt per NY day.
- Entry at the close of the next bar: longs buy the ask, shorts sell the bid.
  That bar must close no later than entry_cutoff.
- Stop = break bar low - buffer (long) / high + buffer (short). The day is
  skipped when the entry bar's bid close is at or beyond the stop.
- Target = entry +/- take_profit_r * |entry - stop|, measured from the fill.
- Held until stop or target (overnight and over weekends). One trade at a
  time: a day that breaks while a trade is still open is skipped.
- Long exits trigger and fill on bid, short exits on ask. A bar that opens
  past a level fills at its open (gap); the stop wins when one bar spans both.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import ensure_utc_index
from src.features.sessions import parse_hhmm

OHLC = ["open", "high", "low", "close"]
BAR_MINUTES = 5
BAR = pd.Timedelta(minutes=BAR_MINUTES)


@dataclass
class Setup:
    day: str  # NY date
    side: int  # +1 long / -1 short
    range_high: float
    range_low: float
    break_time: pd.Timestamp  # break bar open (UTC)
    entry_time: pd.Timestamp  # entry bar close (UTC)
    entry_pos: int  # row of the entry bar in the bid/ask frame
    entry: float
    stop: float
    take_profit: float
    risk_points: float
    spread_points: float


@dataclass
class KZTrade:
    day: str
    side: int
    range_high: float
    range_low: float
    break_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    stop: float
    take_profit: float
    risk_points: float
    spread_points: float
    exit_time: pd.Timestamp  # open of the bar the exit happened in (UTC)
    exit_price: float
    exit_reason: str
    r_multiple: float


def merge_bid_ask(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """Bid OHLC as open/high/low/close plus ask_open/... on shared timestamps."""
    b = ensure_utc_index(bid)[OHLC]
    a = ensure_utc_index(ask)[OHLC].add_prefix("ask_")
    return b.join(a, how="inner")


def without_spread(bars: pd.DataFrame) -> pd.DataFrame:
    """Copy with ask == bid, for a zero-cost baseline."""
    out = bars.copy()
    for c in OHLC:
        out[f"ask_{c}"] = out[c]
    return out


def _minutes(hhmm: str) -> int:
    t = parse_hhmm(hhmm)
    return t.hour * 60 + t.minute


def find_setups(bars: pd.DataFrame, cfg: dict[str, Any]) -> list[Setup]:
    """One candidate trade per NY day (first break only), in time order."""
    sess = cfg["session"]
    sig = cfg["signal"]
    range_start = _minutes(sess["range_start"])
    range_end = _minutes(sess["range_end"])
    cutoff = _minutes(sess["entry_cutoff"])
    min_bars = int(sig.get("min_range_bars", 1))
    buffer = float(sig["stop_buffer_points"])
    tp_r = float(sig["take_profit_r"])

    local = bars.index.tz_convert(sess["timezone"])
    minute = np.asarray(local.hour * 60 + local.minute)
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    cl = bars["close"].to_numpy()
    ask_cl = bars["ask_close"].to_numpy()

    setups: list[Setup] = []
    for day, rows in pd.Series(np.arange(len(bars))).groupby(local.date):
        rows = rows.to_numpy()
        m = minute[rows]
        rng = rows[(m >= range_start) & (m < range_end)]
        if len(rng) < min_bars:
            continue
        rh, rl = float(hi[rng].max()), float(lo[rng].min())

        scan = rows[m >= range_end]
        brk = next((k for k, p in enumerate(scan) if cl[p] > rh or cl[p] < rl), None)
        if brk is None or brk + 1 >= len(scan):
            continue
        b, e = scan[brk], scan[brk + 1]
        if minute[e] + BAR_MINUTES > cutoff:
            continue

        side = 1 if cl[b] > rh else -1
        if side > 0:
            stop = float(lo[b]) - buffer
            if cl[e] <= stop:
                continue
            entry = float(ask_cl[e])
        else:
            stop = float(hi[b]) + buffer
            if cl[e] >= stop:
                continue
            entry = float(cl[e])
        risk = abs(entry - stop)

        setups.append(
            Setup(
                day=str(day),
                side=side,
                range_high=rh,
                range_low=rl,
                break_time=bars.index[b],
                entry_time=bars.index[e] + BAR,
                entry_pos=int(e),
                entry=entry,
                stop=stop,
                take_profit=entry + side * tp_r * risk,
                risk_points=risk,
                spread_points=float(ask_cl[e] - cl[e]),
            )
        )
    return setups


def _first_exit(
    side: int,
    stop: float,
    tp: float,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    start: int,
) -> tuple[int, float, str]:
    n = len(o)
    for j in range(start, n):
        if side > 0:
            if o[j] <= stop:
                return j, float(o[j]), "stop_gap"
            if o[j] >= tp:
                return j, float(o[j]), "tp_gap"
            if l[j] <= stop:
                return j, stop, "stop"
            if h[j] >= tp:
                return j, tp, "tp"
        else:
            if o[j] >= stop:
                return j, float(o[j]), "stop_gap"
            if o[j] <= tp:
                return j, float(o[j]), "tp_gap"
            if h[j] >= stop:
                return j, stop, "stop"
            if l[j] <= tp:
                return j, tp, "tp"
    return n - 1, float(c[n - 1]), "data_end"


def simulate(bars: pd.DataFrame, setups: list[Setup]) -> tuple[list[KZTrade], int]:
    """Fill setups one at a time. Returns (trades, setups skipped while busy)."""
    bid = [bars[c].to_numpy() for c in OHLC]
    ask = [bars[f"ask_{c}"].to_numpy() for c in OHLC]

    trades: list[KZTrade] = []
    busy_until: pd.Timestamp | None = None
    skipped = 0
    for s in setups:
        if busy_until is not None and s.entry_time < busy_until:
            skipped += 1
            continue
        px = bid if s.side > 0 else ask
        j, exit_px, reason = _first_exit(s.side, s.stop, s.take_profit, *px, start=s.entry_pos + 1)
        exit_time = bars.index[j]
        busy_until = exit_time + BAR

        fields = asdict(s)
        fields.pop("entry_pos")
        trades.append(
            KZTrade(
                **fields,
                exit_time=exit_time,
                exit_price=exit_px,
                exit_reason=reason,
                r_multiple=s.side * (exit_px - s.entry) / s.risk_points,
            )
        )
    return trades, skipped


def _stats(r: np.ndarray) -> dict[str, Any]:
    return {
        "n_trades": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "expectancy_r": float(r.mean()) if len(r) else 0.0,
        "total_r": float(r.sum()),
    }


def summarize(trades: list[KZTrade]) -> dict[str, Any]:
    r = np.array([t.r_multiple for t in trades], dtype=float)
    out = _stats(r)
    if not trades:
        return out

    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    equity = np.concatenate([[0.0], np.cumsum(r)])
    streak = longest = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        longest = max(longest, streak)

    sides = np.array([t.side for t in trades])
    years = np.array([t.day[:4] for t in trades])
    out.update(
        {
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "max_dd_r": float((np.maximum.accumulate(equity) - equity).max()),
            "max_losing_streak": int(longest),
            "avg_risk_points": float(np.mean([t.risk_points for t in trades])),
            "avg_spread_points": float(np.mean([t.spread_points for t in trades])),
            "exit_reasons": dict(Counter(t.exit_reason for t in trades)),
            "long": _stats(r[sides > 0]),
            "short": _stats(r[sides < 0]),
            "by_year": {y: _stats(r[years == y]) for y in sorted(set(years))},
        }
    )
    return out


def run(bars: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[KZTrade], dict[str, Any]]:
    setups = find_setups(bars, cfg)
    trades, skipped = simulate(bars, setups)
    summary = {"n_setups": len(setups), "skipped_open_trade": skipped, **summarize(trades)}
    return trades, summary


def trades_frame(trades: list[KZTrade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])
