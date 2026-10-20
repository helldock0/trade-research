"""Prop-challenge pass rates from a trade list, starting a challenge at every trade.

Risk is a fixed % of the starting balance per trade (1R), as in FTMO-style
accounts. Each phase ends at the first trade that either breaches a limit or
reaches the target; a breach inside a trade (its worst open loss, mae_r)
counts before that trade's close. The daily loss check assumes at most one
trade per day, which holds for strategies that trade once a day.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PASS, FAIL, OPEN = "pass", "fail", "open"


def phase_outcome(
    r: np.ndarray, mae: np.ndarray, risk_pct: float, rules: dict[str, Any]
) -> tuple[str, int]:
    """(pass | fail | open, index of the deciding trade) for a phase starting at r[0]."""
    if len(r) == 0:
        return OPEN, -1
    gain = np.cumsum(r) * risk_pct  # closed P&L, % of the starting balance
    low = gain - r * risk_pct - mae * risk_pct  # lowest equity while each trade is open
    fail = (low < -float(rules["max_loss_pct"])) | (mae * risk_pct > float(rules["max_daily_loss_pct"]))
    ok = (gain >= float(rules["profit_target_pct"])) & (np.arange(1, len(r) + 1) >= int(rules["min_trading_days"]))
    i_fail = int(np.argmax(fail)) if fail.any() else len(r)
    i_pass = int(np.argmax(ok)) if ok.any() else len(r)
    if i_fail < len(r) and i_fail <= i_pass:
        return FAIL, i_fail
    if i_pass < len(r):
        return PASS, i_pass
    return OPEN, len(r) - 1


def challenge_outcomes(
    r: np.ndarray, mae: np.ndarray, risk_pct: float, ftmo_cfg: dict[str, Any]
) -> list[tuple[str, str, int, int]]:
    """For every start trade: (result, failed phase or "", start index, deciding index).

    result is pass (both phases), fail, or open (ran out of trades first).
    Phase 2 starts with the trade after the one that passed phase 1.
    """
    out = []
    for i in range(len(r)):
        res1, k1 = phase_outcome(r[i:], mae[i:], risk_pct, ftmo_cfg["phase1"])
        if res1 != PASS:
            out.append((res1, "phase1" if res1 == FAIL else "", i, i + k1))
            continue
        j = i + k1 + 1
        res2, k2 = phase_outcome(r[j:], mae[j:], risk_pct, ftmo_cfg["phase2"])
        out.append((res2, "phase2" if res2 == FAIL else "", i, j + k2 if k2 >= 0 else i + k1))
    return out
