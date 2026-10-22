from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.research.rv_check import pair_trades, render_markdown, run_rv_check
from src.strategy.rv import band_trades, generate_signals

SIG = {"band_period": 20, "entry_z": 2.0, "max_days": 20, "atr_period": 10, "stop_atr_mult": 3.0}


def _ou(n: int, theta: float, seed: int, start: str = "1990-01-01", px0: float = 1.1, vol: float = 0.004) -> pd.Series:
    """Ornstein-Uhlenbeck around px0 in log space (theta > 0 pulls back; theta = 0 is a random walk)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = x[k - 1] - theta * x[k - 1] + vol * rng.standard_normal()
    return pd.Series(px0 * np.exp(x), index=pd.bdate_range(start, periods=n))


def test_band_trades_entries_exits_and_no_overlap():
    c = _ou(3000, 0.15, 1)
    tr = band_trades(c, SIG)
    assert len(tr) > 30 and set(tr["side"].unique()) <= {-1, 1}
    mu = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    for _, t in tr.head(10).iterrows():
        z = (c[t["entry_date"]] - mu[t["entry_date"]]) / sd[t["entry_date"]]
        assert (z <= -2.0) if t["side"] > 0 else (z >= 2.0)
        x = t["exit_date"]
        crossed = (c[x] >= mu[x]) if t["side"] > 0 else (c[x] <= mu[x])
        assert crossed or t["days"] == 20
    assert (tr["entry_date"].iloc[1:].values > tr["exit_date"].iloc[:-1].values).all()
    assert (tr["side"] * np.log(tr["exit_px"] / tr["entry_px"])).mean() > 0     # mean reversion pays


def test_pair_trades_costs_and_financing():
    tr = pair_trades(_ou(2000, 0.1, 2), SIG, cost_bp=3.0, financing_pct=1.5)
    assert np.allclose(tr["net"], tr["gross"] - 3e-4 - 0.015 * tr["cal_days"] / 365.0)


def _cfg():
    return {"signal": SIG, "validation": {"pairs": {"AAA": {"holdout": ["1991-01-01", "2014-12-31"], "round_trip_cost_bp": 3.0},
                                                    "BBB": {"holdout": ["1995-01-01", "2014-12-31"], "round_trip_cost_bp": 2.0}},
                                          "dev": ["2015-01-01", None], "min_t": 2.0, "financing_pct_per_year": 1.5}}


def test_check_passes_mean_reversion_and_rejects_trends():
    good = run_rv_check({"AAA": _ou(9000, 0.15, 3), "BBB": _ou(9000, 0.15, 4)}, _cfg())
    assert good["verdict"]["verdict"] == "PASS" and good["pooled"]["holdout"]["t"] >= 2
    trend = {k: pd.Series(1.1 * np.exp(np.cumsum(0.0008 + 0.004 * np.random.default_rng(s).standard_normal(9000))),
                          index=pd.bdate_range("1990-01-01", periods=9000)) for k, s in (("AAA", 5), ("BBB", 6))}
    bad = run_rv_check(trend, _cfg())
    assert bad["verdict"]["verdict"] == "NO-GO" and bad["pooled"]["holdout"]["mean_pct"] < 0
    md = render_markdown(good, _cfg())
    assert "**PASS**" in md and "| AAA |" in md
    assert any(r.get("missing") for r in run_rv_check({"AAA": _ou(3000, 0.15, 7)}, _cfg())["rows"])


def test_generate_signals_and_engine_on_bars():
    days = _ou(400, 0.15, 8, start="2023-01-02")
    rows = []
    for d, p in days.items():
        for mm in (30, 45):
            ts = pd.Timestamp(d.date()).replace(hour=20, minute=mm).tz_localize("UTC")
            rows.append((ts, p, p * 1.0005, p * 0.9995, p))
    bars = {"EURGBP": pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"]).assign(volume=1.0),
            "GBPUSD": pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"]).assign(volume=1.0)}
    cfg = {"universe": ["EURGBP"], "sessions": {"london_end": "11:00", "ny_end": "16:00", "close_hour_utc": 21},
           "signal": {**SIG, "take_profit_r": 0, "breakeven_at_r": 0.0, "exit_mode": "swing", "allow_weekend_hold": True, "max_hold_days": 0},
           "risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 2, "soft_daily_halt_pct": 50.0}}
    sigs = generate_signals(bars, cfg, {})
    assert sigs and {s.symbol for s in sigs} == {"EURGBP"}                       # GBPUSD is only for conversion
    assert all(s.exit_time > s.timestamp and (s.stop < s.entry) == (s.side > 0) for s in sigs)
    free = {"default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}}
    res = run_backtest(sigs, bars, cfg, free)
    assert res.metrics["n_trades"] == len(sigs)
