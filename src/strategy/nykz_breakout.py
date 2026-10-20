"""NY kill-zone breakout — NAS100 M5 with real bid/ask fills.

Rules (session times are New York local, so US DST is handled):
- Range = bid high/low of bars opening in [range_start, range_end).
- The first later bar whose bid close is above the range high (long) or below
  the range low (short) is the break bar. One attempt per NY day.
  Optional "clean break" filters: the close must clear the level by
  min_break_points / min_break_pct of price, and the body must be at least
  min_body_vs_range x the average range-bar body. With confirm_close, the
  next bar must also close outside the range, else the scan waits for the
  next break.
- Entry at the close of the next bar: longs buy the ask, shorts sell the bid.
  That bar must close no later than entry_cutoff.
- Stop = break bar low - buffer (long) / high + buffer (short); with
  stop_includes_prior_bar, the bar before the break counts too. The day is
  skipped when the entry bar's bid close is at or beyond the stop.
  atr_stop_mult replaces that stop with entry -/+ mult x ATR (M5, at entry);
  min_stop_atr only widens a stop closer than that many ATRs.
- Target = entry +/- take_profit_r * |entry - stop|, measured from the fill;
  take_profit_r 0 means no target.
- Optional moving stop, updated after each bar closes: breakeven_at_r moves
  it to entry once the best price reaches that many R; trail_atr_mult trails
  it that many ATRs behind the best price once trail_start_r is reached.
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
    take_profit: float | None
    risk_points: float
    spread_points: float
    atr_points: float


@dataclass
class KZTrade:
    day: str
    side: int
    range_high: float
    range_low: float
    break_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    stop: float  # initial stop
    take_profit: float | None
    risk_points: float
    spread_points: float
    atr_points: float
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


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _break_side(o: float, c: float, rh: float, rl: float, need: float, body_floor: float) -> int:
    """+1 / -1 when a bar closes cleanly above / below the range, else 0."""
    if abs(c - o) < body_floor:
        return 0
    if c > rh + need:
        return 1
    if c < rl - need:
        return -1
    return 0


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
    min_pts = float(sig.get("min_break_points", 0.0))
    min_frac = float(sig.get("min_break_pct", 0.0)) / 100
    min_body = float(sig.get("min_body_vs_range", 0.0))
    confirm = bool(sig.get("confirm_close", False))
    stop_prior = bool(sig.get("stop_includes_prior_bar", False))
    atr_stop = float(sig.get("atr_stop_mult", 0.0))
    min_stop_atr = float(sig.get("min_stop_atr", 0.0))

    local = bars.index.tz_convert(sess["timezone"])
    minute = np.asarray(local.hour * 60 + local.minute)
    op = bars["open"].to_numpy()
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    cl = bars["close"].to_numpy()
    ask_cl = bars["ask_close"].to_numpy()
    atr = _atr(hi, lo, cl, int(sig.get("atr_period", 14)))

    setups: list[Setup] = []
    for day, rows in pd.Series(np.arange(len(bars))).groupby(local.date):
        rows = rows.to_numpy()
        m = minute[rows]
        rng = rows[(m >= range_start) & (m < range_end)]
        if len(rng) < min_bars:
            continue
        rh, rl = float(hi[rng].max()), float(lo[rng].min())
        body_floor = min_body * float(np.abs(cl[rng] - op[rng]).mean())

        scan = rows[m >= range_end]
        for k in range(len(scan) - 1):
            b, e = scan[k], scan[k + 1]
            side = _break_side(op[b], cl[b], rh, rl, max(min_pts, min_frac * cl[b]), body_floor)
            if side == 0:
                continue
            if confirm and not (cl[e] > rh if side > 0 else cl[e] < rl):
                continue  # failed confirmation: wait for the next break
            if minute[e] + BAR_MINUTES > cutoff:
                break

            ref = [b - 1, b] if stop_prior else [b]
            if side > 0:
                stop = float(lo[ref].min()) - buffer
                if cl[e] <= stop:
                    break
                entry = float(ask_cl[e])
            else:
                stop = float(hi[ref].max()) + buffer
                if cl[e] >= stop:
                    break
                entry = float(cl[e])
            a = float(atr[e])
            if atr_stop > 0:
                stop = entry - side * atr_stop * a
            elif abs(entry - stop) < min_stop_atr * a:
                stop = entry - side * min_stop_atr * a
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
                    take_profit=entry + side * tp_r * risk if tp_r > 0 else None,
                    risk_points=risk,
                    spread_points=float(ask_cl[e] - cl[e]),
                    atr_points=a,
                )
            )
            break
    return setups


def _first_exit(
    s: Setup,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    be_r: float = 0.0,
    trail_pts: float = 0.0,
    trail_start_r: float = 1.0,
) -> tuple[int, float, str]:
    """First bar to hit the stop or target. A moved stop applies from the next bar."""
    side, entry, tp = s.side, s.entry, s.take_profit
    stop, best = s.stop, entry
    for j in range(s.entry_pos + 1, len(o)):
        reason = "stop" if stop == s.stop else "breakeven" if stop == entry else "trail"
        if side > 0:
            if o[j] <= stop:
                return j, float(o[j]), reason + "_gap"
            if tp is not None and o[j] >= tp:
                return j, float(o[j]), "tp_gap"
            if l[j] <= stop:
                return j, stop, reason
            if tp is not None and h[j] >= tp:
                return j, tp, "tp"
            best = max(best, float(h[j]))
        else:
            if o[j] >= stop:
                return j, float(o[j]), reason + "_gap"
            if tp is not None and o[j] <= tp:
                return j, float(o[j]), "tp_gap"
            if h[j] >= stop:
                return j, stop, reason
            if tp is not None and l[j] <= tp:
                return j, tp, "tp"
            best = min(best, float(l[j]))
        gain_r = side * (best - entry) / s.risk_points
        if be_r and gain_r >= be_r:
            stop = max(stop, entry) if side > 0 else min(stop, entry)
        if trail_pts and gain_r >= trail_start_r:
            stop = max(stop, best - trail_pts) if side > 0 else min(stop, best + trail_pts)
    return len(o) - 1, float(c[-1]), "data_end"


def simulate(
    bars: pd.DataFrame, setups: list[Setup], sig: dict[str, Any] | None = None
) -> tuple[list[KZTrade], int]:
    """Fill setups one at a time. Returns (trades, setups skipped while busy)."""
    sig = sig or {}
    be_r = float(sig.get("breakeven_at_r", 0.0))
    trail_mult = float(sig.get("trail_atr_mult", 0.0))
    trail_start_r = float(sig.get("trail_start_r", 1.0))
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
        j, exit_px, reason = _first_exit(s, *px, be_r, trail_mult * s.atr_points, trail_start_r)
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
    trades, skipped = simulate(bars, setups, cfg["signal"])
    summary = {"n_setups": len(setups), "skipped_open_trade": skipped, **summarize(trades)}
    return trades, summary


def trades_frame(trades: list[KZTrade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])
