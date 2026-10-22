from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.research.tsmom_check import monthly_tsmom, render_markdown, run_tsmom_check


def _walk(drift_by_year: dict[int, float], start: str = "1990-01-01", end: str = "2024-12-31", vol: float = 0.01, seed: int = 0) -> pd.Series:
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    mu = np.array([drift_by_year.get(d.year, 0.0) for d in days])
    return pd.Series(100 * np.exp(np.cumsum(mu + vol * rng.standard_normal(len(days)))), index=days)


def test_monthly_tsmom_positions_returns_and_costs():
    s = _walk({}, start="2000-01-03", end="2003-12-31", seed=1)
    t = monthly_tsmom(s, lookback=252, vol_window=60, cost_bp=2.0, financing_pct=12.0)
    logc = np.log(s)
    m = t.index[5]
    i = s.index.get_loc(m)
    assert t.at[m, "pos"] == int(np.sign(logc.iloc[i] - logc.iloc[i - 252]))
    nxt = t.index[6]
    assert t.at[m, "ret"] == pytest.approx(logc.loc[nxt] - logc.loc[m])
    change = abs(t.at[m, "pos"] - t.at[t.index[4], "pos"])
    assert t.at[m, "net"] == pytest.approx(t.at[m, "pos"] * t.at[m, "ret"] - change * 1e-4 - 0.01)
    assert t.at[m, "scaled"] == pytest.approx(t.at[m, "net"] / t.at[m, "vol"])
    assert t.index.max() < pd.Timestamp("2003-12-01")          # the series' final month is dropped
    assert t.index.min() >= s.index[252]                        # needs a full lookback


def _cfg(n_markets: int = 3) -> dict:
    return {
        "signal": {"lookback_days": 252},
        "validation": {"history": {f"M{k}": f"S{k}" for k in range(n_markets)}, "holdout": ["1991-01-01", "2014-12-31"], "dev": ["2015-01-01", None],
                       "min_t": 2.0, "vol_window_days": 60, "round_trip_cost_bp": 2.0, "index_financing_pct_per_year": 6.0, "fx_financing_pct_per_year": 1.5},
    }


def test_check_finds_planted_trends_and_rejects_noise():
    rng = np.random.default_rng(3)
    trendy = {}
    for k in range(3):
        drifts = {y: float(rng.choice([-1, 1])) * 0.0015 for y in range(1990, 2025)}   # year-long trends
        trendy[f"S{k}"] = _walk(drifts, seed=10 + k)
    res = run_tsmom_check(trendy, _cfg())
    assert res["verdict"]["verdict"] == "PASS" and res["portfolio"]["holdout"]["t"] >= 2
    assert res["portfolio"]["holdout"]["avg_instruments"] == pytest.approx(3.0)
    noise = {f"S{k}": _walk({}, seed=20 + k) for k in range(3)}
    assert run_tsmom_check(noise, _cfg())["portfolio"]["holdout"]["t"] < 2.0
    md = render_markdown(res, _cfg())
    assert "**PASS**" in md and "## Per market" in md and "| M0 |" in md
    # markets whose month ends fall on different days still form one portfolio row per month
    shifted = {k: v.drop(v.index[v.index.is_month_end]) if k == "S1" else v for k, v in trendy.items()}
    res2 = run_tsmom_check(shifted, _cfg())
    n_months = res2["portfolio"]["holdout"]["n"]
    assert n_months == res["portfolio"]["holdout"]["n"] and n_months <= 12 * 24
    assert all(y["n"] <= 12 for y in res2["by_year_dev"])
    missing = run_tsmom_check({"S0": trendy["S0"]}, _cfg())
    assert any(r.get("missing") for r in missing["rows"]) and not math.isnan(missing["portfolio"]["holdout"]["mean"])


# ----------------------------------------------------------------- strategy on bars

from src.backtest.engine import run_backtest
from src.data.bars import daily_ohlc
from src.strategy.tsmom import generate_signals, month_end_positions, position_runs


def test_position_runs_and_month_end_positions():
    idx = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30", "2024-05-31"])
    runs = position_runs(pd.Series([0, 1, 1, -1, -1], index=idx))
    assert runs == [(idx[1], idx[3], 1), (idx[3], None, -1)]
    assert position_runs(pd.Series([1, 0, 1], index=idx[:3])) == [(idx[0], idx[1], 1), (idx[2], None, 1)]
    days = pd.bdate_range("2023-01-02", "2024-06-14")
    c = pd.Series(np.linspace(100, 150, len(days)), index=days)
    pos = month_end_positions(c, 100)
    assert pos.index.max() == pd.Timestamp("2024-05-31")              # June (the final, partial month) takes no decision
    assert (pos.iloc[:4] == 0).all() and (pos.iloc[6:] == 1).all()   # zero until 100 closes exist


def test_daily_ohlc_rolls_at_21_utc():
    idx = pd.date_range("2024-03-04 20:00", periods=8, freq="15min", tz="UTC")   # Mon 20:00 -> 21:45
    m15 = pd.DataFrame({"timestamp": idx, "open": np.arange(8.0), "high": np.arange(8.0) + 0.5, "low": np.arange(8.0) - 0.5, "close": np.arange(8.0), "volume": 1.0})
    d = daily_ohlc(m15)
    assert list(d.index) == [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")]
    assert d.at[pd.Timestamp("2024-03-04"), "close"] == 3.0 and d.at[pd.Timestamp("2024-03-04"), "last_ts"] == idx[3]   # 20:45 bar
    assert d.at[pd.Timestamp("2024-03-05"), "open"] == 4.0                                                               # 21:00 starts the next day


def test_generate_signals_reverses_at_the_same_close():
    idx = pd.date_range("2022-01-03", "2024-12-27", freq="15min", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    k = np.arange(len(idx))
    path = np.where(idx < pd.Timestamp("2023-06-01", tz="UTC"), 1.0 + 1.5e-6 * k, 1.0 + 1.5e-6 * np.searchsorted(idx, pd.Timestamp("2023-06-01", tz="UTC")) - 1.5e-6 * (k - np.searchsorted(idx, pd.Timestamp("2023-06-01", tz="UTC"))))
    close = path + 0.002 * np.sin(k / 50)
    bars = {"EURUSD": pd.DataFrame({"timestamp": idx, "open": close, "high": close + 0.0005, "low": close - 0.0005, "close": close, "volume": 1.0})}
    cfg = {"universe": ["EURUSD"], "sessions": {"london_end": "11:00", "ny_end": "16:00", "close_hour_utc": 21},
           "signal": {"lookback_days": 120, "atr_period": 10, "stop_atr_mult": 50.0, "take_profit_r": 0, "breakeven_at_r": 0.0, "exit_mode": "swing", "allow_weekend_hold": True, "max_hold_days": 0},
           "risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 1, "soft_daily_halt_pct": 50.0}}
    sigs = generate_signals(bars, cfg, {})
    sides = [s.side for s in sigs]
    assert sides[:2] == [1, -1] and sigs[0].exit_time == sigs[1].timestamp          # flip: exit and re-entry on one bar
    free = {"default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}}
    trades = run_backtest(sigs, bars, cfg, free).trades
    assert len(trades) == len(sigs) and trades[0].exit_time == trades[1].entry_time
    assert trades[0].r_multiple > 0 and trades[1].r_multiple > 0                    # caught both trends
