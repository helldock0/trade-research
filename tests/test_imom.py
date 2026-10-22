from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.research.imom_check import net_returns, round_trip_cost, run_imom_check
from src.strategy.imom import day_table, generate_signals, signal_frame

COSTS = {"default": {"half_spread_pips": 0.25, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}, "pip_size": {"US500": 1.0}, "contracts": {"US500": 1.0}}


def _cfg(**sig) -> dict:
    base = {
        "universe": ["US500"],
        "sessions": {"london_end": "11:00", "ny_end": "16:00", "exchange_tz": "America/New_York"},
        "signal": {"mode": "day", "entry_time": "15:30", "exit_time": "16:00", "strong_only": False, "strong_lookback": 10,
                   "range_lookback": 5, "stop_range_mult": 1.0, "take_profit_r": 0, "breakeven_at_r": 0.0, "exit_mode": "swing",
                   "allow_weekend_hold": True, "max_hold_days": 0},
        "risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 1, "soft_daily_halt_pct": 50.0},
        "validation": {"dev": ["2023-01-01", "2023-06-30"], "holdout": ["2023-07-01", None], "min_t": 2.0, "cross_checks": []},
    }
    base["signal"].update(sig)
    return base


def _bars(days: pd.DatetimeIndex, day_ret: np.ndarray, last_ret: np.ndarray, px0: float = 5000.0) -> pd.DataFrame:
    """M15 bars in New York time: flat until 15:30, the day's move lands in the 15:15 bar, the last half hour in two bars."""
    rows, px = [], px0
    for d, rd, rl in zip(days, day_ret, last_ret):
        for hh, mm in [(h, m) for h in range(9, 16) for m in (0, 15, 30, 45)]:
            ts = pd.Timestamp(d.date()).replace(hour=hh, minute=mm).tz_localize("America/New_York").tz_convert("UTC")
            start = px
            if (hh, mm) == (15, 15):
                px = px * math.exp(rd)
            elif (hh, mm) in ((15, 30), (15, 45)):
                px = px * math.exp(rl / 2)
            rows.append((ts, start, max(start, px) + 0.5, min(start, px) - 0.5, px))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"]).assign(volume=1.0)


def test_day_table_reads_new_york_clock_across_dst():
    days = pd.DatetimeIndex(["2024-03-08", "2024-03-11"])            # US DST starts 2024-03-10
    b = _bars(days, np.array([0.01, -0.01]), np.array([0.002, -0.002]))
    t = day_table(b)
    assert list(t.index) == list(days)
    assert t["entry_ts"].iloc[0] == pd.Timestamp("2024-03-08 20:15", tz="UTC")   # 15:15 EST
    assert t["entry_ts"].iloc[1] == pd.Timestamp("2024-03-11 19:15", tz="UTC")   # 15:15 EDT
    assert t["exit_ts"].iloc[1] == pd.Timestamp("2024-03-11 19:45", tz="UTC")
    assert math.log(t["exit"].iloc[0] / t["entry"].iloc[0]) == pytest.approx(0.002)


def test_signal_side_strong_filter_and_stop_use_only_past_days():
    days = pd.bdate_range("2023-01-02", periods=40)
    rng = np.random.default_rng(0)
    b = _bars(days, 0.01 * rng.standard_normal(40), 0.002 * rng.standard_normal(40))
    t = signal_frame(b, _cfg(), calendar=days)
    assert len(t) == 39                                               # the first day has no previous close
    assert (t["side"] == np.sign(t["signal"])).all()
    first_stop = t["stop_dist"].first_valid_index()
    assert t.index.get_loc(first_stop) >= 1                           # needs prior ranges, never today's
    strong = signal_frame(b, _cfg(strong_only=True), calendar=days)
    assert (strong["side"] == 0).sum() > 10 and set(strong["side"].unique()) <= {-1, 0, 1}
    assert (signal_frame(b, _cfg(mode="long"), calendar=days)["side"] == 1).all()
    with pytest.raises(ValueError):
        signal_frame(b, _cfg(mode="nope"), calendar=days)


def test_engine_trades_the_last_half_hour_and_check_detects_momentum():
    days = pd.bdate_range("2023-01-02", "2023-12-29")
    rng = np.random.default_rng(1)
    day_ret = 0.01 * rng.standard_normal(len(days))
    last = 0.0005 * np.sign(day_ret) + 0.001 * rng.standard_normal(len(days))   # momentum planted
    b = _bars(days, day_ret, last)
    cfg = _cfg()
    sigs = generate_signals({"US500": b}, cfg, COSTS, calendar=days)
    assert sigs and all((s.exit_time - s.timestamp) == pd.Timedelta(minutes=30) for s in sigs)
    res = run_backtest(sigs, {"US500": b}, cfg, COSTS)
    assert res.metrics["n_trades"] == len(sigs) and res.metrics["expectancy_r"] > 0
    assert {t.exit_reason for t in res.trades} <= {"time_exit", "stop"}
    nr = net_returns(b, "US500", cfg, COSTS, calendar=days)
    cost = round_trip_cost("US500", pd.Series([5000.0]), COSTS).iloc[0]
    assert cost == pytest.approx(0.5 / 5000.0)
    assert (nr["gross"] - nr["net"]).dropna().between(0.4 / 6000, 0.6 / 4000).all()
    chk = run_imom_check({"US500": b}, cfg, COSTS, calendar=days)
    assert chk["verdict"]["verdict"] == "PASS"
    # no momentum (sign-independent last half hour): fails
    flat = _bars(days, day_ret, 0.001 * rng.standard_normal(len(days)))
    assert run_imom_check({"US500": flat}, cfg, COSTS, calendar=days)["verdict"]["verdict"] == "NO-GO"
