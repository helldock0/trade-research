from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.research.dip_check import run_dip_check, trade_table, window_stats
from src.strategy.dip import dip_trades, generate_signals, rsi

SIG = {"trend_sma": 20, "rsi_period": 2, "rsi_entry": 10, "exit_sma": 5, "max_days": 10, "atr_period": 5, "stop_atr_mult": 2.5}
def _dip_path(n_up: int = 40, slope: float = 1.0, drop: float = 1.5, after: list[float] | None = None) -> list[float]:
    """Steady uptrend, three small down closes (RSI(2) < 10 while still above the 20-day average), then `after`."""
    px = list(100 + slope * np.arange(n_up))
    for _ in range(3):
        px.append(px[-1] - drop)
    return px + list(after or [])


COSTS = {"default": {"half_spread_pips": 0.5, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}, "pip_size": {"NAS100": 1.0}}


def test_rsi_extremes_and_range():
    up = pd.Series(np.arange(1.0, 30.0))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    assert rsi(-up + 100).iloc[-1] == pytest.approx(0.0)
    r = rsi(pd.Series(100 + np.random.default_rng(0).standard_normal(200).cumsum()))
    assert r.dropna().between(0, 100).all()


def test_dip_trades_enter_on_dip_in_uptrend_and_exit_above_short_average():
    px = _dip_path(after=[134.5 + 0.8 * k for k in range(1, 18)])   # dip, then recovers
    days = pd.bdate_range("2024-01-01", periods=len(px))
    c = pd.Series(px, index=days)
    tr = dip_trades(c, SIG)
    assert len(tr) >= 1
    first = tr.iloc[0]
    assert first["entry_date"] == days[42]                        # the third down close
    sma5 = c.rolling(5).mean()
    exit_i = days.get_loc(first["exit_date"])
    assert c.iloc[exit_i] > sma5.iloc[exit_i]                     # first close above the 5-day average
    assert all(c.iloc[k] <= sma5.iloc[k] for k in range(days.get_loc(first["entry_date"]) + 1, exit_i))
    assert (tr["entry_date"].iloc[1:].values > tr["exit_date"].iloc[:-1].values).all()   # never overlapping
    # below the trend average nothing is bought
    assert dip_trades(pd.Series(np.linspace(200, 100, 60), index=days), SIG).empty


def test_max_days_and_open_trade_at_the_end():
    px = _dip_path(after=list(134.4 - 0.1 * np.arange(12)))      # dip, then drifts down: never above its 5-day average
    days = pd.bdate_range("2024-01-01", periods=len(px))
    c = pd.Series(px, index=days)
    tr = dip_trades(c, {**SIG, "max_days": 4})
    assert len(tr) >= 1 and (tr["days"] <= 4).all() and (tr["days"] == 4).any()
    assert dip_trades(c, {**SIG, "max_days": 100}).empty           # never exits before the data ends -> dropped


def _cfg(**val):
    v = {"series": ["A", "B"], "holdout": ["2000-01-01", "2009-12-31"], "dev": ["2010-01-01", None], "context": [None, "1999-12-31"],
         "min_t": 2.0, "financing_pct_per_year": 6.0, "cost_symbol": "NAS100"}
    v.update(val)
    return {"signal": SIG, "validation": v}


def _series(seed: int, revert: float) -> pd.Series:
    """Random walk with drift; `revert` > 0 makes a day after two down days bounce, < 0 makes it fall further."""
    days = pd.bdate_range("1995-01-01", "2014-12-31")
    rng = np.random.default_rng(seed)
    r = 0.0004 + 0.01 * rng.standard_normal(len(days))
    for k in range(2, len(r)):
        if r[k - 1] < 0 and r[k - 2] < 0:
            r[k] += revert
    return pd.Series(1000 * np.exp(np.cumsum(r)), index=days)


def test_trade_table_costs_and_financing():
    c = _series(1, 0.0)
    tr, up = trade_table(c, _cfg(), COSTS)
    assert len(tr) > 20 and len(up) > 1000
    expect = tr["gross"] - 1.0 / tr["entry_px"] - 0.06 * tr["cal_days"] / 365.0
    assert np.allclose(tr["net"], expect)
    tr_bp, _ = trade_table(c, _cfg(round_trip_cost_bp=2.0), COSTS)
    assert np.allclose(tr_bp["net"], tr_bp["gross"] - 2e-4 - 0.06 * tr_bp["cal_days"] / 365.0)
    w = window_stats(tr, up, "2000-01-01", "2009-12-31")
    assert w["n"] == ((tr["entry_date"] >= "2000-01-01") & (tr["entry_date"] <= "2009-12-31")).sum()


def test_run_dip_check_verdict():
    good = run_dip_check({"A": _series(2, 0.008), "B": _series(3, 0.008)}, _cfg(), COSTS)
    assert good["verdict"]["verdict"] == "PASS" and good["rows"][0]["holdout"]["t"] >= 2
    bad = run_dip_check({"A": _series(4, -0.008), "B": _series(5, -0.008)}, _cfg(), COSTS)
    assert bad["verdict"]["verdict"] == "NO-GO" and bad["rows"][0]["holdout"]["mean_excess_pct"] < 0
    assert {d["decade"] for d in good["decades"]} >= {2000, 2010}


def test_generate_signals_on_cash_close_bars():
    px = _dip_path(n_up=50, slope=0.4, drop=0.6, after=list(np.linspace(118.5, 125, 18)))
    days = pd.bdate_range("2024-01-02", periods=len(px))
    rows = []
    for d, p in zip(days, px):
        for hh, mm in ((15, 30), (15, 45)):
            ts = pd.Timestamp(d.date()).replace(hour=hh, minute=mm).tz_localize("America/New_York").tz_convert("UTC")
            rows.append((ts, p, p + 0.3, p - 0.3, p))
    bars = {"NAS100": pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"]).assign(volume=1.0)}
    cfg = {"signal": SIG, "sessions": {"exchange_tz": "America/New_York", "cash_close": "16:00"}}
    sigs = generate_signals(bars, cfg, COSTS, calendar=pd.DatetimeIndex(days))
    assert sigs and all(s.side == 1 and s.stop < s.entry and s.exit_time > s.timestamp for s in sigs)
    assert all(s.timestamp.tz_convert("America/New_York").strftime("%H:%M") == "15:45" for s in sigs)


def test_engine_charges_index_financing_over_calendar_time_only_where_configured():
    from src.backtest.engine import run_backtest
    from src.strategy.cssm import Signal

    idx = pd.date_range("2024-01-05 15:00", periods=4 * 24 * 4, freq="15min", tz="UTC")   # Fri -> Tue
    bars = {"NAS100": pd.DataFrame({"timestamp": idx, "open": 20000.0, "high": 20001.0, "low": 19999.0, "close": 20000.0, "volume": 1.0})}
    cfg = {"risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 1, "soft_daily_halt_pct": 50.0},
           "sessions": {"london_end": "11:00", "ny_end": "16:00"},
           "signal": {"exit_mode": "swing", "take_profit_r": 0, "breakeven_at_r": 0.0, "allow_weekend_hold": True}}
    free = {"default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}, "pip_size": {"NAS100": 1.0}, "contracts": {"NAS100": 1.0}}
    sig = Signal(idx[0], "NAS100", 1, 20000.0, 19500.0, float("nan"), 500.0, "dip", 20500.0, 19500.0, "t", exit_time=idx[0] + pd.Timedelta(days=3))
    t0 = run_backtest([sig], bars, cfg, free).trades[0]
    assert t0.pnl_usd == pytest.approx(0.0)                                  # flat price, no financing configured
    t1 = run_backtest([sig], bars, cfg, {**free, "financing_pct_per_year": {"NAS100": 6.0}}).trades[0]
    notional = 20000.0 * t1.lots * 1.0
    assert t1.pnl_usd == pytest.approx(-notional * 0.06 * 3 / 365, rel=1e-6)  # 3 calendar days, weekend included
