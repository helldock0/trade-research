from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from src.features.sessions import parse_hhmm


def usd_signed_return(pair: str, ret: float) -> float:
    """Express return as USD strength contribution.

    XXXUSD up => USD weaker => negative USD strength
    USDXXX up => USD stronger => positive USD strength
    """
    p = pair.upper()
    if p.startswith("USD"):
        return ret
    if p.endswith("USD"):
        return -ret
    return ret


def overnight_return(
    bars: pd.DataFrame,
    day: pd.Timestamp,
    anchor: time,
    session_open: time,
) -> float | None:
    """Return from prior day anchor close to session open close."""
    day = pd.Timestamp(day).tz_convert("UTC") if day.tzinfo else pd.Timestamp(day, tz="UTC")
    prev = day - pd.Timedelta(days=1)
    # skip weekends for prev lookback
    while prev.dayofweek >= 5:
        prev -= pd.Timedelta(days=1)

    def _bar_at(d: pd.Timestamp, t: time) -> pd.Series | None:
        ts = pd.Timestamp(
            year=d.year, month=d.month, day=d.day, hour=t.hour, minute=t.minute, tz="UTC"
        )
        # use last bar at or before ts
        hist = bars.loc[:ts]
        if hist.empty:
            return None
        return hist.iloc[-1]

    a = _bar_at(prev, anchor)
    b = _bar_at(day, session_open)
    if a is None or b is None:
        return None
    px0 = float(a["close"])
    px1 = float(b["close"])
    if px0 == 0:
        return None
    return (px1 / px0) - 1.0


def cross_sectional_ranks(usd_returns: dict[str, float]) -> dict[str, float]:
    """Rank 1..N (higher = stronger USD-aligned overnight move for that pair's trade direction).

    For trading we want continuation of overnight *pair* move, not only USD.
    Here we rank by raw overnight pair return magnitude directionally via signed pair return.
    """
    if not usd_returns:
        return {}
    items = sorted(usd_returns.items(), key=lambda kv: kv[1])
    n = len(items)
    ranks = {}
    for i, (sym, _) in enumerate(items):
        ranks[sym] = (i + 1) / n  # 0..1 percentile-ish
    return ranks


def top_tercile_mask(ranks: dict[str, float]) -> dict[str, int]:
    """Return +1 for top tercile (strong positive USD-signed overnight), -1 for bottom tercile.

    Trade direction for a pair:
    - top tercile USD strength: prefer trades that buy USD (short XXXUSD / long USDXXX)
    - bottom tercile: prefer sell USD
    """
    if not ranks:
        return {}
    vals = np.array(list(ranks.values()))
    hi = np.quantile(vals, 2 / 3)
    lo = np.quantile(vals, 1 / 3)
    out: dict[str, int] = {}
    for sym, r in ranks.items():
        if r >= hi:
            out[sym] = 1  # USD strong
        elif r <= lo:
            out[sym] = -1  # USD weak
        else:
            out[sym] = 0
    return out


def pair_trade_direction(pair: str, usd_side: int) -> int:
    """Map USD side (+1 strong USD) to pair buy(+1)/sell(-1)."""
    if usd_side == 0:
        return 0
    p = pair.upper()
    if p.endswith("USD") and not p.startswith("USD"):
        # EURUSD: strong USD => short pair
        return -usd_side
    if p.startswith("USD"):
        # USDJPY: strong USD => long pair
        return usd_side
    return 0


def trailing_percentile(series: pd.Series, lookback: int, q: float) -> pd.Series:
    return series.rolling(lookback, min_periods=max(20, lookback // 3)).quantile(q)


def pick_cross_section_candidate(
    rets: dict[str, float],
    history: dict[str, list[float]],
    winner_pct: float,
    loser_pct: float,
    min_history: int = 20,
) -> tuple[str, int, float] | None:
    """Pick (symbol, side, ret) from cross-section + own-history percentile gates.

    Rank by raw return: #1 long candidate, last short candidate.
    """
    if len(rets) < 2:
        return None

    ranked = sorted(rets.items(), key=lambda kv: kv[1], reverse=True)
    winner_sym, winner_ret = ranked[0]
    loser_sym, loser_ret = ranked[-1]

    candidates: list[tuple[str, int, float]] = []

    w_hist = history.get(winner_sym, [])
    if len(w_hist) >= min_history:
        thr = float(np.quantile(w_hist, winner_pct))
        if winner_ret >= thr:
            candidates.append((winner_sym, 1, winner_ret))

    l_hist = history.get(loser_sym, [])
    if len(l_hist) >= min_history:
        thr = float(np.quantile(l_hist, loser_pct))
        if loser_ret <= thr:
            candidates.append((loser_sym, -1, loser_ret))

    if not candidates:
        return None
    return max(candidates, key=lambda c: abs(c[2]))


def parse_session_times(cfg: dict) -> dict[str, time]:
    s = cfg["sessions"]
    out = {
        "asian_start": parse_hhmm(s["asian_start"]),
        "asian_end": parse_hhmm(s["asian_end"]),
        "london_start": parse_hhmm(s["london_start"]),
        "london_end": parse_hhmm(s["london_end"]),
        "ny_start": parse_hhmm(s["ny_start"]),
        "ny_end": parse_hhmm(s["ny_end"]),
        "overnight_anchor": parse_hhmm(s["overnight_anchor"]),
    }
    # Friday cutoffs optional when allow_weekend_hold strategies omit them
    if "friday_no_new_entries" in s and s["friday_no_new_entries"]:
        out["friday_no_new_entries"] = parse_hhmm(s["friday_no_new_entries"])
    if "friday_hard_flat" in s and s["friday_hard_flat"]:
        out["friday_hard_flat"] = parse_hhmm(s["friday_hard_flat"])
    return out
