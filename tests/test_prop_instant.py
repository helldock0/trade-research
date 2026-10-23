"""Instant-funded first-payout rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prop.instant import first_payout

RULES = {"max_dd_pct": 10.0, "max_trade_loss_pct": 1.5, "target_pct": 4.0,
         "min_active_days": 5, "active_day_pct": 0.5, "min_days": 14}


def _run(r, mae=None, mfe=None, days=None, risk=1.0, **rules):
    r = np.array(r, dtype=float)
    mae = np.array(mae if mae is not None else np.maximum(-r, 0) + 0.0)
    mfe = np.array(mfe if mfe is not None else np.maximum(r, 0))
    days = np.array(days if days is not None else np.arange(len(r)) * 3)
    return first_payout(r, mae, mfe, days, days, risk, {**RULES, **rules})


def test_payout_needs_five_active_days_and_fourteen_days():
    # Five 4R wins at 1% = +20%, one trade per 3 days: the 5th win closes on day 12, so the
    # payout waits for the next trade (day 15), even though it loses.
    assert _run([4, 4, 4, 4, 4, -1, 4]) == ("payout", 5)
    assert _run([4, 4, 4, 4, 4, -1, 4], min_days=0) == ("payout", 4)


def test_gap_loss_over_the_cap_is_a_hard_breach():
    assert _run([4, -1.6], risk=1.0) == ("trade_loss_cap", 1)
    assert _run([4, -1.6], risk=0.5)[0] == "open"


def test_open_profit_raises_the_floor_before_a_loss():
    # Each loser first runs +2R: the first one lifts the floor from -10% to -8%, so the 9th
    # straight loss (equity -8% -> -9%) breaches. Without the run-up 10 losses survive.
    r = [-1.0] * 10
    assert _run(r, mfe=[0.0] * 10)[0] == "open"
    assert _run(r, mfe=[2.0] * 10) == ("trailing_dd", 8)


def test_floor_locks_at_the_starting_balance():
    # +12% then losses. A floor still trailing would sit at +2% and breach on the 10th loss;
    # locked at 0% it breaches only on the 13th (equity 0% -> -1%). min_active_days 9: no payout.
    r = [4, 4, 4] + [-1] * 13
    assert _run(r, min_active_days=9) == ("trailing_dd", 15)
