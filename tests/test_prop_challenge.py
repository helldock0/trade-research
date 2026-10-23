"""Rolling prop-challenge pass/fail rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prop.challenge import challenge_outcomes, phase_outcome

RULES = {"profit_target_pct": 10.0, "max_loss_pct": 10.0, "max_daily_loss_pct": 5.0, "min_trading_days": 4}
FTMO = {"phase1": RULES, "phase2": {**RULES, "profit_target_pct": 5.0}}


def test_phase_passes_at_the_target_and_fails_at_max_loss():
    r = np.array([4.0, 4.0, 4.0, -1.0, 4.0])
    assert phase_outcome(r, np.zeros(5), 1.0, {**RULES, "min_trading_days": 1}) == ("pass", 2)  # 12%
    losses = np.full(12, -1.0)
    assert phase_outcome(losses, np.ones(12), 1.0, RULES) == ("fail", 10)  # 11th loss: -11% < -10%


def test_min_trading_days_delays_the_pass():
    r = np.array([6.0, 6.0, -1.0, 1.0, 1.0])
    # 12% after 2 trades, but the pass needs 4 trades: trade 4 ends at 12%.
    assert phase_outcome(r, np.zeros(5), 1.0, RULES) == ("pass", 3)


def test_open_loss_inside_a_winner_can_breach():
    r = np.array([-1.0] * 9 + [4.0])
    mae = np.array([1.0] * 9 + [0.9])
    # 9 losses at 1% = -9%; the winner dips 0.9% more = -9.9% (no breach) ...
    assert phase_outcome(r, mae, 1.0, RULES)[0] == "open"
    # ... but 1.2R deep would be -10.2%: fail on that trade even though it closes green.
    assert phase_outcome(r, np.array([1.0] * 9 + [1.2]), 1.0, RULES) == ("fail", 9)


def test_daily_loss_limit_from_one_gapped_trade():
    assert phase_outcome(np.array([-3.0]), np.array([3.0]), 2.0, RULES) == ("fail", 0)


def test_challenge_runs_phase2_after_phase1():
    r = np.array([4.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0])
    out = challenge_outcomes(r, np.zeros(len(r)), 1.0, FTMO)
    # From trade 0: phase 1 hits 10% at index 3; phase 2 (trades 4..) hits 5% at index 7 (4 trades).
    assert out[0] == ("pass", "", 0, 7)
    assert out[-1][0] == "open"
