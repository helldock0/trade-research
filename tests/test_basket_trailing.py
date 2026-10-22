from __future__ import annotations

import math
from functools import partial

import numpy as np
import pandas as pd
import pytest

from src.backtest.correlation import CorrelationGate, daily_closes
from src.backtest.engine import run_backtest, trail_stop
from src.research.macro_compare import compare_variants, deep_merge, max_concurrent, render_markdown, variant_config
from src.strategy.cssm import Signal

FREE = {"default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}}


def _cfg(**over) -> dict:
    cfg = {
        "risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 1, "soft_daily_halt_pct": 50.0},
        "sessions": {"london_end": "11:00", "ny_end": "16:00"},
        "signal": {"exit_mode": "swing", "take_profit_r": 3.0, "breakeven_at_r": 0.0, "allow_weekend_hold": True},
    }
    return deep_merge(cfg, over)


def _bars(path: list[tuple[float, float, float]], start: str = "2024-01-01 00:00") -> pd.DataFrame:
    """M15 bars from (high, low, close) triples; open = previous close."""
    idx = pd.date_range(start, periods=len(path), freq="15min", tz="UTC")
    hi, lo, cl = (np.array(x, dtype=float) for x in zip(*path))
    op = np.r_[cl[0], cl[:-1]]
    return pd.DataFrame({"timestamp": idx, "open": op, "high": hi, "low": lo, "close": cl, "volume": 1.0})


def _sig(sym: str, ts: pd.Timestamp, side: int, entry: float, risk: float) -> Signal:
    return Signal(ts, sym, side, entry, entry - side * risk, entry + side * 3 * risk, risk, "macro", entry + risk, entry - risk, "t")


# ----------------------------------------------------------------- trailing stop


def test_trail_stop_ratchets_one_way():
    assert trail_stop(1, 1.09, 1.12, 0.01) == (pytest.approx(1.11), True)
    assert trail_stop(1, 1.115, 1.12, 0.01) == (1.115, False)        # never loosens
    assert trail_stop(-1, 1.11, 1.08, 0.01) == (pytest.approx(1.09), True)
    assert trail_stop(1, 1.09, 1.20, 0.0) == (1.09, False)


def test_trailing_stop_lets_winner_run_past_fixed_target_without_lookahead():
    path = [(1.1000, 1.1000, 1.1000)]                                   # signal bar, entry 1.1000, risk 0.0100
    path += [(1.1100 + 0.005 * i, 1.1000 + 0.005 * i, 1.1080 + 0.005 * i) for i in range(1, 50)]  # rally to ~1.35
    peak = path[-1][0]
    path += [(peak + 0.0100, peak - 0.0050, peak + 0.0050)]            # new high; low inside the old stop distance
    path += [(peak + 0.0060, peak - 0.0020, peak - 0.0010)]            # low crosses the new trail level
    bars = {"EURUSD": _bars(path)}
    sig = _sig("EURUSD", bars["EURUSD"]["timestamp"].iloc[0], 1, 1.1000, 0.0100)

    fixed = run_backtest([sig], bars, _cfg(), FREE)
    assert fixed.trades[0].exit_reason == "tp" and fixed.trades[0].r_multiple == pytest.approx(3.0, rel=1e-6)

    res = run_backtest([sig], bars, _cfg(signal={"take_profit_r": 0, "trailing_stop_r": 1.0}), FREE)
    t = res.trades[0]
    assert t.exit_reason == "trail_stop" and math.isnan(t.take_profit)
    # the bar that made the new high did not stop out on its own high; the next bar did
    assert t.exit_time == bars["EURUSD"]["timestamp"].iloc[-1]
    assert t.exit_price == pytest.approx(peak + 0.0100 - 0.0100)
    assert t.r_multiple > 20


def test_trailing_stop_short_side_and_stop_before_any_gain():
    path = [(1.1000, 1.1000, 1.1000), (1.1050, 1.0990, 1.1040), (1.1110, 1.1030, 1.1100)]
    bars = {"EURUSD": _bars(path)}
    sig = _sig("EURUSD", bars["EURUSD"]["timestamp"].iloc[0], -1, 1.1000, 0.0100)
    t = run_backtest([sig], bars, _cfg(signal={"take_profit_r": 0, "trailing_stop_r": 1.0}), FREE).trades[0]
    # best low 1.0990 moved the stop to 1.1090 before the move up hit it
    assert t.exit_reason == "trail_stop" and t.exit_price == pytest.approx(1.1090)
    assert t.r_multiple == pytest.approx(-0.9, rel=1e-6)


# ----------------------------------------------------------------- basket + correlation gate


def _daily_bars(rets: np.ndarray, start: str = "2023-01-02", px0: float = 1.1) -> pd.DataFrame:
    """One M15 bar per weekday at 20:45 UTC (the bar before the 21:00 close)."""
    days = pd.bdate_range(start, periods=len(rets), tz="UTC")
    idx = days + pd.Timedelta(hours=20, minutes=45)
    close = px0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"timestamp": idx, "open": close, "high": close * 1.0001, "low": close * 0.9999, "close": close, "volume": 1.0}).set_index("timestamp")


def test_daily_closes_and_gate_are_point_in_time():
    rng = np.random.default_rng(0)
    r = 0.005 * rng.standard_normal(120)
    a = _daily_bars(r)
    b = _daily_bars(-r)
    closes = daily_closes({"EURUSD": a, "USDJPY": b})
    assert (closes.index.dayofweek < 5).all() and len(closes) == 120
    assert closes["EURUSD"].iloc[5] == pytest.approx(a["close"].iloc[5])
    gate = CorrelationGate({"EURUSD": a, "USDJPY": b}, max_signed_corr=0.6, window_days=40)
    day = closes.index[80]
    rets = np.log(closes).diff()
    expect = rets["EURUSD"].iloc[41:80].corr(rets["USDJPY"].iloc[41:80])  # 40 returns ending the day before
    assert gate.corr("EURUSD", "USDJPY", day + pd.Timedelta(hours=21)) == pytest.approx(expect)
    assert gate.corr("EURUSD", "USDJPY", day) == pytest.approx(-1.0)
    # long EURUSD + long USDJPY is a hedge here (corr -1): allowed; long + short repeats the bet: refused
    open_long = [{"symbol": "EURUSD", "side": 1}]
    assert gate.allows("USDJPY", 1, open_long, day)
    assert not gate.allows("USDJPY", -1, open_long, day)
    assert not gate.allows("USDJPY", 1, open_long, closes.index[3])  # not enough history yet -> refuse
    assert CorrelationGate.from_config({}, {"EURUSD": a}) is None


def test_basket_holds_uncorrelated_positions_and_refuses_correlated_ones():
    rng = np.random.default_rng(1)
    n = 150
    common = 0.005 * rng.standard_normal(n)
    bars = {
        "EURUSD": _daily_bars(common),
        "GBPUSD": _daily_bars(common + 0.0005 * rng.standard_normal(n)),   # ~ same bet as EURUSD
        "AUDUSD": _daily_bars(0.005 * rng.standard_normal(n)),             # independent
    }
    ts = bars["EURUSD"].index[100]
    sigs = [_sig(s, ts, 1, float(bars[s].loc[ts, "close"]), 0.05) for s in ("EURUSD", "GBPUSD", "AUDUSD")]
    single = run_backtest(sigs, bars, _cfg(), FREE)
    assert len(single.trades) == 1 and single.trades[0].symbol == "EURUSD"
    no_gate = run_backtest(sigs, bars, _cfg(risk={"max_open_positions": 3}), FREE)
    assert sorted(t.symbol for t in no_gate.trades) == ["AUDUSD", "EURUSD", "GBPUSD"]
    assert max_concurrent(no_gate.trades) == 3
    gated = run_backtest(sigs, bars, _cfg(risk={"max_open_positions": 3, "max_signed_corr": 0.6, "corr_window_days": 40}), FREE)
    assert sorted(t.symbol for t in gated.trades) == ["AUDUSD", "EURUSD"]
    # a second signal on a symbol already open is ignored
    dup = run_backtest(sigs[:1] * 2, bars, _cfg(risk={"max_open_positions": 3}), FREE)
    assert len(dup.trades) == 1


# ----------------------------------------------------------------- variants and comparison


def test_variant_config_deep_merges_without_touching_base():
    base = {"signal": {"trigger": "pullback", "take_profit_r": 3.0}, "risk": {"max_open_positions": 1}, "variants": {"b": {"risk": {"max_open_positions": 3}}}}
    v = variant_config(base, "b")
    assert v["risk"]["max_open_positions"] == 3 and v["signal"]["trigger"] == "pullback" and "variants" not in v
    assert base["risk"]["max_open_positions"] == 1
    with pytest.raises(KeyError):
        variant_config(base, "nope")


def _macro_inputs():
    """Two pairs, a small macro panel with a trending EUR rate, M15 bars (no news, no cache)."""
    from src.config import load_strategy_config

    n = 400
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    t = np.arange(n)
    panel = {
        "rate_2y_USD": pd.Series(4.0 + 0.002 * rng.standard_normal(n).cumsum(), index=idx),
        "rate_2y_EUR": pd.Series(2.0 + 0.001 * t + 0.002 * rng.standard_normal(n).cumsum(), index=idx),
        "rate_2y_JPY": pd.Series(0.1 + 0.001 * rng.standard_normal(n).cumsum(), index=idx),
        "policy_rate_USD": pd.Series(5.25, index=idx),
        "policy_rate_EUR": pd.Series(4.0, index=idx),
        "policy_rate_JPY": pd.Series(0.1, index=idx),
    }
    cfg = load_strategy_config("strategy_macro_swing.yaml")
    cfg["universe"] = ["EURUSD", "USDJPY"]
    cfg["news"]["enabled"] = False
    cfg["macro"] = {
        "currencies": ["USD", "EUR", "JPY"],
        "min_components": 2,
        "lags": {"market_days": 1},
        "series": {
            "rate_2y": {c: {"source": "csv", "path": "x"} for c in ("USD", "EUR", "JPY")},
            "policy_rate": {c: {"source": "csv", "path": "x"} for c in ("USD", "EUR", "JPY")},
        },
        "components": {
            "rate_momentum": {"weight": 1.0, "input": "rate_2y", "transform": "diff", "lookback_days": 20, "standardize": "cross_section"},
            "carry": {"weight": 0.5, "input": "policy_rate", "transform": "level", "standardize": "cross_section"},
        },
    }
    cfg["signal"].update({"min_abs_bias": 0.3, "reentry_cooldown_days": 0, "max_hold_days": 0})
    bars = {}
    for sym, px in (("EURUSD", 1.1), ("USDJPY", 150.0)):
        m15 = pd.date_range("2023-06-01", "2024-01-31", freq="15min", tz="UTC")
        m15 = m15[m15.dayofweek < 5]
        k = np.arange(len(m15))
        close = px * (1 + 0.0004 * np.sin(k / 7) + 0.002 * np.sin(k / 900))
        bars[sym] = pd.DataFrame({"timestamp": m15, "open": close, "high": close * 1.0005, "low": close * 0.9995, "close": close, "volume": 1.0})
    return bars, panel, cfg


def test_macro_config_variants_and_compare_end_to_end():
    from src.config import load_strategy_config
    from src.strategy.macro_swing import generate_signals

    cfg = load_strategy_config("strategy_macro_swing.yaml")
    assert set(cfg["variants"]) >= {"fixed_tp", "trail", "basket", "hikes", "wide", "wide_hikes"}
    basket = variant_config(cfg, "basket")
    assert basket["risk"]["max_open_positions"] == 3 and basket["signal"]["take_profit_r"] == 0
    assert variant_config(cfg, "fixed_tp")["signal"]["take_profit_r"] == 3.0

    bars, panel, small = _macro_inputs()
    small["variants"] = {
        "fixed_tp": {"signal": {"trigger": "none"}},
        "basket": {
            "signal": {"trigger": "none", "take_profit_r": 0, "trailing_stop_r": 1.0},
            "risk": {"max_open_positions": 2, "max_signed_corr": 0.6},
        },
    }
    res = compare_variants(small, bars, FREE, None, signal_fn=partial(generate_signals, macro_panel=panel))
    rows = {r["variant"]: r for r in res["rows"]}
    assert set(rows) == {"fixed_tp", "basket"}
    assert rows["fixed_tp"]["n_signals"] > 0
    assert rows["basket"]["n_signals"] >= rows["fixed_tp"]["n_signals"]
    assert rows["fixed_tp"]["max_concurrent"] <= 1
    assert "tp" not in rows["basket"]["exit_reasons"]
    md = render_markdown(res)
    assert "| basket |" in md and "## Promotion gates" in md
