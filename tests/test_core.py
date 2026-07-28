from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.backtest.costs import entry_fill_price, position_size_lots
from src.features.momentum import pair_trade_direction, usd_signed_return
from src.features.news import is_news_blackout, pair_currencies
from src.prop.ftmo_simulator import simulate_ftmo_path
from src.backtest.engine import Trade
from src.research.promote import GATES, evaluate_promotion
from src.research.walk_forward import FoldResult


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
