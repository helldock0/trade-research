from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.engine import Trade


@dataclass
class PhaseResult:
    phase: str
    passed: bool
    breach: str | None
    ending_balance: float
    max_dd_pct: float
    worst_daily_loss_pct: float
    trading_days: int
    profit_pct: float
    trades_used: int


@dataclass
class FTMOSimResult:
    phase1: PhaseResult
    phase2: PhaseResult | None = None
    overall_pass: bool = False
    notes: list[str] = field(default_factory=list)


def _to_prague(ts: pd.Timestamp, tz_name: str = "Europe/Prague") -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ZoneInfo(tz_name))


def simulate_ftmo_path(
    trades: list[Trade],
    ftmo_cfg: dict[str, Any],
    start_balance: float | None = None,
) -> FTMOSimResult:
    """Replay closed trades against FTMO 2-Step style rules.

    Floating intrabar equity is approximated by using each trade's full loss
    as worst-case floating hit on the entry day (conservative for stops).
    """
    initial = float(start_balance or ftmo_cfg["account"]["initial_balance"])
    tz = ftmo_cfg.get("daily_reset", {}).get("timezone", "Europe/Prague")

    phase1 = _run_phase(
        trades=trades,
        phase_name="phase1",
        rules=ftmo_cfg["phase1"],
        initial=initial,
        tz=tz,
    )
    notes: list[str] = []
    phase2 = None
    overall = False

    if phase1.passed:
        # Continue with remaining trades after phase1 completion index
        remaining = trades[phase1.trades_used :]
        phase2 = _run_phase(
            trades=remaining,
            phase_name="phase2",
            rules=ftmo_cfg["phase2"],
            initial=initial,  # FTMO resets simulated account between phases typically
            tz=tz,
        )
        overall = phase2.passed
        if overall:
            notes.append("Passed FTMO 2-Step simulation on historical trade path.")
        else:
            notes.append(f"Phase2 failed: {phase2.breach}")
    else:
        notes.append(f"Phase1 failed: {phase1.breach}")

    return FTMOSimResult(phase1=phase1, phase2=phase2, overall_pass=overall, notes=notes)


def _run_phase(
    trades: list[Trade],
    phase_name: str,
    rules: dict[str, Any],
    initial: float,
    tz: str,
) -> PhaseResult:
    target = float(rules["profit_target_pct"])
    max_daily = float(rules["max_daily_loss_pct"])
    max_loss = float(rules["max_loss_pct"])
    min_days = int(rules["min_trading_days"])

    balance = initial
    peak_equity = initial
    worst_daily = 0.0
    max_dd = 0.0
    trading_days: set[str] = set()
    day_anchor_balance: dict[str, float] = {}
    day_worst_equity: dict[str, float] = {}

    used = 0
    breach = None
    passed = False

    for i, tr in enumerate(trades):
        used = i + 1
        day_key = _to_prague(tr.entry_time, tz).strftime("%Y-%m-%d")
        trading_days.add(day_key)

        if day_key not in day_anchor_balance:
            day_anchor_balance[day_key] = balance
            day_worst_equity[day_key] = balance

        # Conservative floating: if losing trade, assume MAE = full stop loss (=pnl if stop)
        float_equity = balance + min(0.0, tr.pnl_usd)
        day_worst_equity[day_key] = min(day_worst_equity[day_key], float_equity)

        daily_loss_pct = (
            (day_anchor_balance[day_key] - day_worst_equity[day_key]) / initial * 100.0
        )
        worst_daily = max(worst_daily, daily_loss_pct)
        if daily_loss_pct > max_daily:
            breach = f"max_daily_loss ({daily_loss_pct:.2f}% > {max_daily}%)"
            break

        balance += tr.pnl_usd
        equity = balance
        peak_equity = max(peak_equity, equity)
        dd_pct = (initial - equity) / initial * 100.0
        max_dd = max(max_dd, dd_pct)
        if dd_pct > max_loss:
            breach = f"max_loss ({dd_pct:.2f}% > {max_loss}%)"
            break

        profit_pct = (balance - initial) / initial * 100.0
        if profit_pct >= target and len(trading_days) >= min_days:
            passed = True
            break

    profit_pct = (balance - initial) / initial * 100.0
    if passed is False and breach is None:
        if profit_pct >= target and len(trading_days) < min_days:
            breach = f"min_trading_days ({len(trading_days)} < {min_days})"
        elif profit_pct < target:
            breach = f"profit_target_not_reached ({profit_pct:.2f}% < {target}%)"
        else:
            breach = "unknown"

    return PhaseResult(
        phase=phase_name,
        passed=passed,
        breach=breach,
        ending_balance=balance,
        max_dd_pct=max_dd,
        worst_daily_loss_pct=worst_daily,
        trading_days=len(trading_days),
        profit_pct=profit_pct,
        trades_used=used,
    )
