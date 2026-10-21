"""First payout on an instant-funded account (Blueberry Funded Instant Elite style).

All amounts are % of the starting balance, with a fixed risk per trade (1R).
- Trailing max drawdown: the floor sits max_dd_pct below the highest equity
  seen, open-trade highs included, and stops rising at the starting balance.
- Any single trade losing more than max_trade_loss_pct is a hard breach.
- A payout needs realised profit >= target_pct, min_active_days days whose
  closed trades made >= active_day_pct, and min_days calendar days.
Inside a trade the worst order is assumed: a loser makes its best move before
its worst (raising the floor first), a winner its worst before its best.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PAYOUT, TRAIL, TRADE_CAP, OPEN = "payout", "trailing_dd", "trade_loss_cap", "open"


def first_payout(
    r: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    start_day: np.ndarray,
    exit_day: np.ndarray,
    risk_pct: float,
    rules: dict[str, Any],
) -> tuple[str, int]:
    """(payout | trailing_dd | trade_loss_cap | open, index of the deciding trade).

    start_day / exit_day are day numbers (e.g. days since an epoch) per trade.
    """
    max_dd = float(rules["max_dd_pct"])
    cap = float(rules["max_trade_loss_pct"])
    target = float(rules["target_pct"])
    min_active = int(rules["min_active_days"])
    active_pct = float(rules["active_day_pct"])
    min_days = int(rules["min_days"])

    equity = peak = 0.0
    floor = -max_dd
    day_pnl: dict[int, float] = {}
    active: set[int] = set()
    for k in range(len(r)):
        gain, low, high = r[k] * risk_pct, mae[k] * risk_pct, mfe[k] * risk_pct
        if -gain > cap + 1e-9:
            return TRADE_CAP, k
        if gain > 0 and equity - low < floor:  # a winner dips first
            return TRAIL, k
        peak = max(peak, equity + high)
        floor = max(floor, min(peak - max_dd, 0.0))
        if equity - low < floor:  # a loser runs up first, then falls
            return TRAIL, k
        equity += gain
        day_pnl[exit_day[k]] = day_pnl.get(exit_day[k], 0.0) + gain
        if day_pnl[exit_day[k]] >= active_pct:
            active.add(exit_day[k])
        if equity >= target and len(active) >= min_active and exit_day[k] - start_day[0] >= min_days:
            return PAYOUT, k
    return OPEN, len(r) - 1
