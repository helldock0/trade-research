from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_strategy_config
from src.strategy.vwap_fade import generate_signals, session_vwap, wilder_adx


def test_session_vwap_equal_volume():
    tp = pd.Series([1.0, 2.0, 3.0])
    vol = pd.Series([1.0, 1.0, 1.0])
    v = session_vwap(tp, vol)
    assert abs(float(v.iloc[0]) - 1.0) < 1e-9
    assert abs(float(v.iloc[1]) - 1.5) < 1e-9
    assert abs(float(v.iloc[2]) - 2.0) < 1e-9


def test_wilder_adx_finite_after_warmup():
    n = 80
    idx = pd.date_range("2024-01-02", periods=n, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(1.1, 1.12, n), index=idx)
    high = close + 0.0003
    low = close - 0.0003
    adx = wilder_adx(high, low, close, period=14)
    assert adx.iloc[40:].notna().all()


def test_vwap_fade_long_on_stretch_below():
    """Build a quiet session then dump price below VWAP in London -> long fade."""
    days = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    rows = []
    signal_day = days[-1]
    for d in days:
        for h in range(0, 24):
            for m in (0, 15, 30, 45):
                ts = d + pd.Timedelta(hours=h, minutes=m)
                base = 1.1000
                if d == signal_day and h >= 8 and h < 11:
                    # Stretch well below session mean accumulating from London open
                    c = 1.0920
                    o, hi, lo = c + 0.0001, c + 0.0002, c - 0.0002
                elif 7 <= h < 11:
                    c = base
                    o, hi, lo = c, c + 0.00015, c - 0.00015
                else:
                    c = base
                    o, hi, lo = c, c + 0.0001, c - 0.0001
                rows.append(
                    {
                        "timestamp": ts,
                        "open": o,
                        "high": hi,
                        "low": lo,
                        "close": c,
                        "volume": 100.0,
                    }
                )

    df = pd.DataFrame(rows)
    cfg = load_strategy_config("strategy_vwap_fade.yaml")
    cfg["news"]["enabled"] = False
    cfg["signal"]["stretch_atr"] = 1.0
    cfg["signal"]["adx_max"] = 100.0  # synthetic dump can elevate ADX
    cfg["signal"]["take_profit_r"] = 0.3
    cfg["universe"] = ["EURUSD"]
    costs = {
        "pip_size": {"EURUSD": 0.0001},
        "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7},
        "pairs": {},
    }
    sigs = generate_signals({"EURUSD": df}, cfg, costs)
    longs = [s for s in sigs if s.side == 1 and s.reason == "vwap_fade"]
    assert len(longs) >= 1
    assert longs[0].take_profit > longs[0].entry
