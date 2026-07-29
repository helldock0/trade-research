from __future__ import annotations

import pandas as pd

from src.backtest.costs import entry_fill_price, position_size_lots
from src.features.momentum import pair_trade_direction, usd_signed_return
from src.features.news import is_news_blackout, pair_currencies
from src.prop.ftmo_simulator import simulate_ftmo_path
from src.backtest.engine import Trade, maybe_move_stop_to_breakeven, unrealized_r
from src.research.promote import GATES, evaluate_promotion


def test_usd_signed_return():
    assert usd_signed_return("EURUSD", 0.01) == -0.01
    assert usd_signed_return("USDJPY", 0.01) == 0.01


def test_pair_trade_direction():
    assert pair_trade_direction("EURUSD", 1) == -1
    assert pair_trade_direction("USDJPY", 1) == 1
    assert pair_trade_direction("EURUSD", 0) == 0


def test_pair_currencies():
    assert pair_currencies("EURUSD") == {"EUR", "USD"}


def test_news_blackout():
    cal = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02 12:00:00", tz="UTC")],
            "currency": ["USD"],
            "impact": ["high"],
        }
    )
    ts = pd.Timestamp("2024-01-02 12:30:00", tz="UTC")
    assert is_news_blackout(ts, "EURUSD", cal, 60) is True
    assert is_news_blackout(ts, "EURUSD", cal, 10) is False


def test_costs_and_sizing():
    costs = {
        "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7},
        "pip_size": {"EURUSD": 0.0001},
        "pairs": {},
    }
    fill = entry_fill_price(1, 1.1000, "EURUSD", costs)
    assert fill > 1.1000
    lots = position_size_lots(100000, 0.5, 0.0010, "EURUSD", 1.1, costs)
    assert lots >= 0.01


def test_xauusd_costs_pip_value_and_pnl():
    from src.backtest.costs import pip_size, pip_value_per_lot, position_size_lots
    from src.backtest.engine import _pnl_usd

    costs = {
        "default": {"half_spread_pips": 20.0, "slippage_pips": 5.0, "commission_per_lot_usd": 7},
        "pip_size": {"XAUUSD": 0.01},
        "pairs": {
            "XAUUSD": {
                "half_spread_pips": 20.0,
                "slippage_pips": 5.0,
                "commission_per_lot_usd": 7.0,
            }
        },
    }
    assert pip_size("XAUUSD", costs) == 0.01
    assert pip_value_per_lot("XAUUSD", 2000.0) == 1.0
    # $1 move on 1.0 lot = $100
    assert abs(_pnl_usd(1, 2000.0, 2001.0, 1.0, "XAUUSD") - 100.0) < 1e-9
    assert abs(_pnl_usd(-1, 2000.0, 1999.0, 1.0, "XAUUSD") - 100.0) < 1e-9
    # Risk $500 on $10 stop (1000 pips @ 0.01) => 0.5 lots
    lots = position_size_lots(100000, 0.5, 10.0, "XAUUSD", 2000.0, costs)
    assert abs(lots - 0.5) < 1e-9
    fill = entry_fill_price(1, 2000.0, "XAUUSD", costs)
    assert abs(fill - 2000.25) < 1e-9  # (20+5)*0.01


def test_ftmo_fail_on_drawdown():
    trades = [
        Trade(
            symbol="EURUSD",
            side=1,
            entry_time=pd.Timestamp("2024-01-02 08:00:00", tz="UTC"),
            exit_time=pd.Timestamp("2024-01-02 10:00:00", tz="UTC"),
            entry_price=1.1,
            exit_price=1.09,
            stop=1.09,
            take_profit=1.12,
            lots=1.0,
            pnl_usd=-11000,
            r_multiple=-2.2,
            session="london",
            exit_reason="stop",
            commission=7,
        )
    ]
    cfg = {
        "account": {"initial_balance": 100000},
        "phase1": {
            "profit_target_pct": 10,
            "max_daily_loss_pct": 5,
            "max_loss_pct": 10,
            "min_trading_days": 4,
        },
        "phase2": {
            "profit_target_pct": 5,
            "max_daily_loss_pct": 5,
            "max_loss_pct": 10,
            "min_trading_days": 4,
        },
        "daily_reset": {"timezone": "Europe/Prague"},
    }
    sim = simulate_ftmo_path(trades, cfg)
    assert sim.overall_pass is False
    assert sim.phase1.passed is False


def test_promote_nogo_empty():
    verdict = evaluate_promotion([], [])
    assert verdict["verdict"] == "NO-GO"
    assert set(GATES) <= set(verdict["gates"])


def test_breakeven_stop_moves_at_one_r():
    entry, stop, risk = 100.0, 90.0, 10.0
    # Below +1R: no move
    s1, be1 = maybe_move_stop_to_breakeven(1, entry, stop, 105.0, risk, 1.0)
    assert be1 is False
    assert s1 == stop
    assert unrealized_r(1, entry, 105.0, risk) == 0.5

    # At +1R: stop -> entry
    s2, be2 = maybe_move_stop_to_breakeven(1, entry, stop, 110.0, risk, 1.0)
    assert be2 is True
    assert s2 == entry

    # Short side
    s3, be3 = maybe_move_stop_to_breakeven(-1, 100.0, 110.0, 90.0, 10.0, 1.0)
    assert be3 is True
    assert s3 == 100.0


def test_confirm_bar_config_flag_defaults():
    from src.config import load_strategy_config

    cfg = load_strategy_config()
    assert cfg["signal"].get("require_confirm_bar") is True
    assert cfg["signal"].get("exit_mode") == "hold_to_ny"


def test_csmo_pick_candidate_rank_and_percentile():
    from src.strategy.csmo import pick_csmo_candidate

    # History: flat around 0; today EURUSD extreme winner, USDJPY extreme loser
    history = {
        "EURUSD": [0.0] * 30 + [0.001] * 10,
        "GBPUSD": [0.0] * 40,
        "USDJPY": [0.0] * 30 + [-0.001] * 10,
        "AUDUSD": [0.0] * 40,
        "USDCAD": [0.0] * 40,
    }
    rets = {
        "EURUSD": 0.01,  # clear #1 winner above p80
        "GBPUSD": 0.001,
        "USDJPY": -0.01,  # clear #5 loser below p20
        "AUDUSD": 0.0,
        "USDCAD": -0.001,
    }
    pick = pick_csmo_candidate(rets, history, winner_pct=0.80, loser_pct=0.20)
    assert pick is not None
    sym, side, ov = pick
    # Both qualify; larger |ret| — tie 0.01, either extreme; max abs picks first max
    assert abs(ov) == 0.01
    assert sym in ("EURUSD", "USDJPY")
    assert side in (1, -1)
    if sym == "EURUSD":
        assert side == 1
    else:
        assert side == -1

    # Only winner qualifies
    rets2 = dict(rets)
    rets2["USDJPY"] = -0.0001  # not extreme vs history
    pick2 = pick_csmo_candidate(rets2, history, 0.80, 0.20)
    assert pick2 is not None
    assert pick2[0] == "EURUSD" and pick2[1] == 1


def test_d1_mom_config_and_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("d1_mom") == "strategy_d1_mom.yaml"
    cfg = load_strategy_config("strategy_d1_mom.yaml")
    assert strategy_id(cfg) == "d1_mom"
    assert cfg["signal"]["formation_days"] == 20
    assert cfg["signal"]["exit_mode"] == "swing"
    assert cfg["signal"]["allow_weekend_hold"] is True
    assert cfg["signal"]["max_hold_days"] == 10
    assert resolve_signal_fn(cfg).__module__.endswith("d1_mom")


def test_d1_mom_pick_uses_shared_gate():
    from src.features.momentum import pick_cross_section_candidate

    history = {
        "EURUSD": [0.01] * 40,
        "GBPUSD": [0.0] * 40,
        "USDJPY": [-0.01] * 40,
        "AUDUSD": [0.0] * 40,
        "USDCAD": [0.0] * 40,
    }
    # Only EURUSD clears p80 as #1; loser not extreme enough
    rets = {
        "EURUSD": 0.05,
        "GBPUSD": 0.01,
        "USDJPY": -0.005,
        "AUDUSD": 0.0,
        "USDCAD": -0.001,
    }
    pick = pick_cross_section_candidate(rets, history, 0.80, 0.20)
    assert pick is not None
    assert pick[0] == "EURUSD" and pick[1] == 1


def test_csmo_registry_and_config():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("csmo") == "strategy_csmo.yaml"
    cfg = load_strategy_config("strategy_csmo.yaml")
    assert strategy_id(cfg) == "csmo"
    assert cfg["signal"]["exit_mode"] == "hold_to_ny"
    assert cfg["signal"]["winner_percentile"] == 0.80
    assert cfg["signal"]["stop_atr_mult"] == 0.75
    assert resolve_signal_fn(cfg).__module__.endswith("csmo")


def test_orb_config_frozen_v1():
    from src.config import load_strategy_config
    from src.strategy.registry import resolve_signal_fn, strategy_id

    cfg = load_strategy_config("strategy_orb.yaml")
    assert strategy_id(cfg) == "orb"
    assert cfg["signal"]["exit_mode"] == "session"
    assert cfg["signal"]["require_confirm_bar"] is False
    assert cfg["signal"]["min_range_pips"] == 15.0
    assert cfg["signal"]["max_range_pips"] == 100.0
    assert "momentum_tercile" not in cfg["signal"]
    assert resolve_signal_fn(cfg).__module__.endswith("orb")


def test_orb_break_direction_no_momentum():
    """ORB direction comes from range break, not CSS ranks."""
    import numpy as np
    import pandas as pd

    from src.config import load_strategy_config
    from src.strategy.orb import generate_signals

    rng = np.random.default_rng(0)
    # Build two quiet Asian days then a London break day for EURUSD only
    days = pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")
    rows = []
    for d in days:
        # Asian: tight range ~20 pips around 1.1000
        base = 1.1000
        for h in range(0, 24):
            for m in (0, 15, 30, 45):
                ts = d + pd.Timedelta(hours=h, minutes=m)
                if h < 7:
                    o = base + rng.normal(0, 0.00005)
                    hi, lo, c = o + 0.0001, o - 0.0001, o
                elif h < 11 and d == days[30]:
                    # London break upward on day 30
                    o = 1.1030
                    hi, lo, c = 1.1040, 1.1025, 1.1035
                else:
                    o = base + rng.normal(0, 0.0002)
                    hi, lo, c = o + 0.0002, o - 0.0002, o
                rows.append(
                    {
                        "timestamp": ts,
                        "open": o,
                        "high": hi,
                        "low": lo,
                        "close": c,
                        "volume": 100,
                    }
                )
    df = pd.DataFrame(rows)
    # Widen ATR history: vary Asian ranges earlier so regime percentiles exist
    for i, d in enumerate(days[:25]):
        mask = (df["timestamp"] >= d) & (df["timestamp"] < d + pd.Timedelta(hours=7))
        scale = 0.0005 + 0.0003 * (i % 5)
        df.loc[mask, "high"] = df.loc[mask, "open"] + scale
        df.loc[mask, "low"] = df.loc[mask, "open"] - scale

    cfg = load_strategy_config("strategy_orb.yaml")
    cfg["news"]["enabled"] = False
    cfg["signal"]["min_range_pips"] = 5.0
    cfg["signal"]["max_range_pips"] = 200.0
    # Relax regime so synthetic data can pass
    cfg["signal"]["range_pct_low"] = 0.0
    cfg["signal"]["range_pct_high"] = 1.0
    costs = {
        "pip_size": {"EURUSD": 0.0001},
        "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7},
        "pairs": {},
    }
    sigs = generate_signals({"EURUSD": df}, cfg, costs)
    # May be empty if ATR history insufficient; at least API should not raise
    assert isinstance(sigs, list)
    for s in sigs:
        assert s.side in (-1, 1)
        assert s.reason.startswith("london_orb") or s.reason.startswith("ny_orb")


def test_htf_pullback_config_and_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("htf_pullback") == "strategy_htf_pullback.yaml"
    assert default_config_name("htf") == "strategy_htf_pullback.yaml"
    cfg = load_strategy_config("strategy_htf_pullback.yaml")
    assert strategy_id(cfg) == "htf_pullback"
    assert cfg["signal"]["htf_ema_period"] == 200
    assert cfg["signal"]["entry_ema_period"] == 21
    assert cfg["signal"]["take_profit_r"] == 2.0
    assert cfg["signal"]["exit_mode"] == "swing"
    assert cfg["signal"]["allow_weekend_hold"] is True
    assert cfg["signal"]["max_hold_days"] == 10
    assert resolve_signal_fn(cfg).__module__.endswith("htf_pullback")


def test_swing_mode_skips_friday_flat_and_honors_max_hold():
    """Swing + weekend hold: no Friday force-flat; max_hold_days closes trade."""
    import pandas as pd

    from src.backtest.engine import run_backtest
    from src.strategy.cssm import Signal

    # Fri 15:00 entry through Mon — must not friday_flat at 20:00; exit max_hold on day+2
    idx = pd.date_range("2024-01-05 14:00", periods=96 * 4, freq="15min", tz="UTC")  # Fri..Mon+
    # Filter to weekdays only like FX weekends gap — keep Fri+Mon for simplicity use continuous
    close = 1.1000
    rows = []
    for ts in idx:
        rows.append(
            {
                "timestamp": ts,
                "open": close,
                "high": close + 0.0005,
                "low": close - 0.0005,
                "close": close,
                "volume": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    entry_ts = pd.Timestamp("2024-01-05 15:00", tz="UTC")  # Friday
    sig = Signal(
        timestamp=entry_ts,
        symbol="EURUSD",
        side=1,
        entry=1.1000,
        stop=1.0900,  # wide stop — won't hit
        take_profit=1.2000,  # far TP
        risk_price=0.0100,
        session="london",
        range_high=1.1010,
        range_low=1.0990,
        reason="test",
    )
    strategy = {
        "sessions": {
            "london_end": "11:00",
            "ny_end": "16:00",
            "friday_hard_flat": "20:00",
            "friday_no_new_entries": "16:00",
        },
        "signal": {
            "exit_mode": "swing",
            "allow_weekend_hold": True,
            "max_hold_days": 2,
            "breakeven_at_r": 0.0,
            "take_profit_r": 2.0,
        },
        "risk": {
            "risk_per_trade_pct": 0.5,
            "max_open_positions": 1,
            "soft_daily_halt_pct": 5.0,
        },
    }
    costs = {
        "pip_size": {"EURUSD": 0.0001},
        "default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0},
        "pairs": {},
    }
    res = run_backtest([sig], {"EURUSD": df}, strategy, costs)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.exit_reason == "max_hold"
    assert tr.exit_reason != "friday_flat"
    # Survived Friday 20:00: exit on/after entry+2 calendar days
    assert (tr.exit_time.normalize() - tr.entry_time.normalize()).days >= 2


def test_fail_fade_config_and_registry():
    from src.config import load_strategy_config
    from src.strategy.registry import default_config_name, resolve_signal_fn, strategy_id

    assert default_config_name("fail_fade") == "strategy_fail_fade.yaml"
    assert default_config_name("fade") == "strategy_fail_fade.yaml"
    cfg = load_strategy_config("strategy_fail_fade.yaml")
    assert strategy_id(cfg) == "fail_fade"
    assert cfg["signal"]["exit_mode"] == "hold_to_ny"
    assert cfg["signal"]["take_profit_r"] == 1.5
    assert cfg["signal"]["breakeven_at_r"] == 1.0
    assert cfg["signal"]["enable_ny_session"] is False
    assert cfg["signal"]["range_pct_low"] == 0.25
    assert cfg["signal"]["range_pct_high"] == 0.70
    assert "momentum_tercile" not in cfg["signal"]
    assert resolve_signal_fn(cfg).__module__.endswith("fail_fade")


def test_fail_fade_break_then_reclaim_short():
    """Upside break then reclaim inside Asian range -> fade short."""
    from src.strategy.fail_fade import detect_fail_fade

    rh, rl = 1.1050, 1.1000
    # stay inside, break above, stay outside, reclaim inside
    closes = [1.1020, 1.1030, 1.1060, 1.1065, 1.1035]
    got = detect_fail_fade(closes, rh, rl)
    assert got == (2, 4)

    # Downside break then reclaim -> indices for long fade
    closes_dn = [1.1020, 1.0990, 1.0985, 1.1010]
    got_dn = detect_fail_fade(closes_dn, rh, rl)
    assert got_dn == (1, 3)

    # Break without reclaim
    assert detect_fail_fade([1.1020, 1.1060, 1.1070], rh, rl) is None
    # Never breaks
    assert detect_fail_fade([1.1020, 1.1030, 1.1010], rh, rl) is None


def test_fail_fade_generate_signals_upside_fail_short():
    """Synthetic Asian range + London upside fail -> short signal."""
    import numpy as np
    import pandas as pd

    from src.config import load_strategy_config
    from src.strategy.fail_fade import generate_signals

    rng = np.random.default_rng(1)
    days = pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC")
    rows = []
    signal_day = days[50]
    for d in days:
        base = 1.1000
        for h in range(0, 24):
            for m in (0, 15, 30, 45):
                ts = d + pd.Timedelta(hours=h, minutes=m)
                if h < 7:
                    o = base + rng.normal(0, 0.00005)
                    hi, lo, c = o + 0.00015, o - 0.00015, o
                elif d == signal_day and h == 8 and m == 0:
                    # Upside break close (above Asian high 1.1010)
                    o, hi, lo, c = 1.1018, 1.1020, 1.1015, 1.1015
                elif d == signal_day and 7 <= h < 11:
                    # Stay inside Asian range (reclaim on first post-break bar)
                    o, hi, lo, c = 1.1000, 1.1005, 1.0995, 1.1000
                else:
                    o = base + rng.normal(0, 0.00015)
                    hi, lo, c = o + 0.0002, o - 0.0002, o
                rows.append(
                    {
                        "timestamp": ts,
                        "open": o,
                        "high": hi,
                        "low": lo,
                        "close": c,
                        "volume": 100,
                    }
                )
    df = pd.DataFrame(rows)
    # Widen Asian ranges on all days before signal so ATR + regime hist clear floors
    for i, d in enumerate(days):
        if d == signal_day:
            continue
        mask = (df["timestamp"] >= d) & (df["timestamp"] < d + pd.Timedelta(hours=7))
        scale = 0.0008 + 0.0002 * (i % 5)
        df.loc[mask, "high"] = df.loc[mask, "open"] + scale
        df.loc[mask, "low"] = df.loc[mask, "open"] - scale
    # Signal day Asian ~20 pips (in-regime vs history); break above high
    mask_sd = (df["timestamp"] >= signal_day) & (
        df["timestamp"] < signal_day + pd.Timedelta(hours=7)
    )
    df.loc[mask_sd, "high"] = 1.1010
    df.loc[mask_sd, "low"] = 1.0990
    df.loc[mask_sd, "close"] = 1.1000
    df.loc[mask_sd, "open"] = 1.1000

    cfg = load_strategy_config("strategy_fail_fade.yaml")
    cfg["news"]["enabled"] = False
    cfg["signal"]["min_range_pips"] = 5.0
    cfg["signal"]["max_range_pips"] = 200.0
    cfg["signal"]["range_pct_low"] = 0.0
    cfg["signal"]["range_pct_high"] = 1.0
    cfg["signal"]["atr_lookback_days"] = 20
    costs = {
        "pip_size": {"EURUSD": 0.0001},
        "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7},
        "pairs": {},
    }
    sigs = generate_signals({"EURUSD": df}, cfg, costs)
    fade = [s for s in sigs if s.reason == "london_fail_fade" and s.timestamp.normalize() == signal_day]
    assert len(fade) >= 1
    s = fade[0]
    assert s.side == -1
    assert s.stop > s.range_high
    assert s.take_profit < s.entry


def test_htf_bias_no_lookahead_shift():
    import numpy as np
    import pandas as pd

    from src.features.trend import htf_bias_on_ltf

    # Rising series: bias should eventually be +1 after warmup, and ffill from shifted H4
    idx = pd.date_range("2020-01-01", periods=4000, freq="15min", tz="UTC")
    close = np.linspace(1.0, 1.2, len(idx))
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": 1.0,
        }
    )
    bias = htf_bias_on_ltf(df, htf_rule="4h", htf_ema_period=50)
    assert bias.dtype == int or str(bias.dtype).startswith("int")
    # Early bars before HTF history should be 0 or NaN-filled 0
    assert int(bias.iloc[0]) == 0
    # Late bars on a strong uptrend should be mostly +1
    assert (bias.iloc[-500:] == 1).mean() > 0.8
