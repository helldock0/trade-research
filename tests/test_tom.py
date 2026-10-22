from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.research.tom_check import full_months, render_markdown, run_tom_check, tom_mask, tom_stats, welch_t


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, tz="UTC")


def test_tom_mask_marks_last_day_and_first_three():
    idx = _bdays("2024-01-01", "2024-03-29").drop(pd.Timestamp("2024-02-19", tz="UTC"))  # a holiday
    m = tom_mask(idx, days_before=1, days_after=3)
    tom = [d.strftime("%m-%d") for d in idx[m.values]]
    assert tom == ["01-01", "01-02", "01-03", "01-31", "02-01", "02-02", "02-05", "02-29", "03-01", "03-04", "03-05", "03-29"]
    assert tom_mask(idx, 2, 1).sum() == 9


def test_tom_stats_recovers_a_planted_effect():
    idx = _bdays("2015-01-01", "2020-12-31")
    mask = tom_mask(idx).values
    rng = np.random.default_rng(0)
    r = 0.0001 * rng.standard_normal(len(idx)) + np.where(mask, 0.0010, 0.0)  # +10 bp on TOM days only
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    st = tom_stats(close)
    assert st["diff_bp"] == pytest.approx(10.0, abs=0.5)
    assert st["window_mean_bp"] == pytest.approx(40.0, abs=2.0)        # 4 TOM days a month
    assert st["window_hit"] == 1.0 and st["t"] > 20
    assert st["n_months"] == 70                                         # 72 months minus the two partial edge months
    assert 0.9 < st["share_of_return"] <= 1.05


def test_full_months_and_welch_t():
    idx = _bdays("2024-01-15", "2024-04-10")
    s = pd.Series(1.0, index=idx)
    kept = full_months(s)
    assert kept.index.min().month == 2 and kept.index.max().month == 3
    rng = np.random.default_rng(1)
    assert abs(welch_t(pd.Series(rng.standard_normal(500)), pd.Series(rng.standard_normal(5000)))) < 2.5
    assert welch_t(pd.Series([1.0]), pd.Series([1.0, 2.0])) == 0.0


def test_run_tom_check_verdict_follows_the_rule():
    idx = _bdays("2000-01-01", "2026-06-30")
    mask = tom_mask(idx).values
    rng = np.random.default_rng(2)
    noise = 0.01 * rng.standard_normal(len(idx))
    strong = pd.Series(100 * np.exp(np.cumsum(noise + np.where(mask, 0.003, 0.0))), index=idx)
    dead_after_2018 = pd.Series(100 * np.exp(np.cumsum(noise + np.where(mask & (idx.year < 2018), 0.003, np.where(mask, -0.002, 0.0)))), index=idx)
    cfg = {"signal": {"days_before": 1, "days_after": 3}, "validation": {"series": ["A", "B"], "holdout": [None, "2017-12-31"], "dev": ["2018-01-01", None], "min_t": 2.0}}
    ok = run_tom_check({"A": strong, "B": strong}, cfg)
    assert ok["verdict"]["verdict"] == "PASS" and ok["verdict"]["holdout_t"] >= 2.0
    assert {d["decade"] for d in ok["decades"]} == {2000, 2010, 2020}
    bad = run_tom_check({"A": strong, "B": dead_after_2018}, cfg)
    assert bad["verdict"]["verdict"] == "NO-GO" and bad["verdict"]["holdout_ok"] and not bad["verdict"]["dev_ok"]
    md = render_markdown(ok)
    assert "**PASS**" in md and "| A | holdout |" in md and "## By decade" in md
    assert math.isnan(run_tom_check({}, cfg)["verdict"]["holdout_t"])


# ----------------------------------------------------------------- strategy and index CFDs

from src.backtest.costs import contract_size, is_cross, position_size_lots
from src.backtest.engine import _pnl_usd, run_backtest
from src.strategy.tom import cash_sessions, generate_signals, window_mask, windows


def _index_m15(start: str, end: str, px0: float = 20000.0, drift: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="15min", tz="UTC")
    idx = idx[(idx.dayofweek < 5) & (idx.hour >= 7) & (idx.hour < 21)]
    k = np.arange(len(idx))
    close = px0 * (1 + drift * k + 0.002 * np.sin(k / 40))
    return pd.DataFrame({"timestamp": idx, "open": close, "high": close * 1.0005, "low": close * 0.9995, "close": close, "volume": 1.0})


def test_cash_sessions_end_at_the_new_york_close():
    s = cash_sessions(_index_m15("2024-07-01", "2024-07-12"))
    # EDT: 16:00 New York = 20:00 UTC, so the last bar of a session starts 19:45 UTC.
    # Friday-evening bars after the close form a "Saturday" session the trading calendar never uses.
    wk = s.loc["2024-07-01":"2024-07-11"]
    wk = wk[wk.index.dayofweek < 5]
    assert (wk["last_ts"].dt.hour == 19).all() and (wk["last_ts"].dt.minute == 45).all()
    assert (wk["last_ts"].dt.normalize().dt.tz_localize(None) == wk.index).all()
    w = cash_sessions(_index_m15("2024-01-08", "2024-01-13"))   # EST: 16:00 New York = 21:00 UTC; bars stop at 20:45
    w = w.loc["2024-01-08":"2024-01-12"]
    assert (w["last_ts"].dt.hour == 20).all() and (w["last_ts"].dt.minute == 45).all()


def test_windows_and_placebo_days():
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    tom = windows(days, window_mask(days, 1, 3))
    # Jan 31 + Feb 1, 2, 5; Feb 29 + Mar 1, 4, 5; the window starting Mar 29 is cut by the calendar end and dropped
    assert [(a.strftime("%m-%d"), b.strftime("%m-%d")) for a, b in tom] == [("01-30", "02-05"), ("02-28", "03-05")]
    mid = windows(days, window_mask(days, 1, 3, placebo_start_day=9))
    # trading days 9-12: entry at the close of day 8 (Jan 10, Feb 12, Mar 12), exit at day 12
    assert [(a.strftime("%m-%d"), b.strftime("%m-%d")) for a, b in mid] == [("01-10", "01-16"), ("02-12", "02-16"), ("03-12", "03-18")]


def test_tom_signals_and_index_trade_in_usd():
    bars = {"NAS100": _index_m15("2024-01-01", "2024-06-28", drift=2e-6)}
    cfg = {
        "signal": {"days_before": 1, "days_after": 3, "atr_period": 5, "stop_atr_mult": 2.0, "take_profit_r": 0, "exit_mode": "swing", "allow_weekend_hold": True, "breakeven_at_r": 0.0},
        "sessions": {"london_end": "11:00", "ny_end": "16:00", "exchange_tz": "America/New_York", "cash_close": "16:00"},
        "risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 1, "soft_daily_halt_pct": 50.0},
    }
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-06-28"))
    sigs = generate_signals(bars, cfg, {}, calendar=cal)
    assert len(sigs) == 5 and all(s.side == 1 and s.exit_time > s.timestamp for s in sigs)   # Feb..Jun boundaries
    assert sigs[0].reason.startswith("tom|window|2024-01-30->2024-02-05")
    costs = {"default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}, "pip_size": {"NAS100": 1.0}, "contracts": {"NAS100": 1.0}}
    res = run_backtest(sigs, bars, cfg, costs)
    assert len(res.trades) == 5 and all(t.exit_reason == "time_exit" for t in res.trades)
    t = res.trades[0]
    assert t.pnl_usd == pytest.approx((t.exit_price - t.entry_price) * t.lots * 1.0)
    # index CFDs: $1 per point per lot, never treated as an FX cross
    assert not is_cross("NAS100") and contract_size("NAS100", costs) == 1.0 and contract_size("EURUSD", costs) is None
    assert position_size_lots(100000, 0.5, 500.0, "NAS100", 20000.0, costs, None, 1.0) == pytest.approx(1.0)
    assert _pnl_usd(1, 20000.0, 20100.0, 2.0, "NAS100", None, 1.0) == pytest.approx(200.0)
