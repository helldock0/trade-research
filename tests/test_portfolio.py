from __future__ import annotations

import pandas as pd
import pytest

from src.research.portfolio import (
    calendar_folds,
    combine,
    ftmo_starts,
    leg_frame,
    metrics,
    monthly,
    render_markdown,
    run_portfolio,
    to_engine_trades,
)

FTMO = {"account": {"initial_balance": 100000}, "daily_reset": {"timezone": "Europe/Prague"},
        "phase1": {"profit_target_pct": 10.0, "max_daily_loss_pct": 5.0, "max_loss_pct": 10.0, "min_trading_days": 4},
        "phase2": {"profit_target_pct": 5.0, "max_daily_loss_pct": 5.0, "max_loss_pct": 10.0, "min_trading_days": 4}}


def _leg(leg: str, symbol: str, rows: list[tuple[str, str, int, float]]) -> pd.DataFrame:
    tr = pd.DataFrame(rows, columns=["entry_time", "exit_time", "side", "r"])
    return leg_frame(leg, symbol, tr)


def test_leg_frame_normalises_times_and_window():
    tr = pd.DataFrame({"entry_time": pd.to_datetime(["2017-12-29 14:00", "2018-01-02 14:00"], utc=True),
                       "exit_time": pd.to_datetime(["2018-01-02 15:00", "2018-01-03 15:00"], utc=True),
                       "side": [1, -1], "r_multiple": [1.0, -1.0]})
    out = leg_frame("x", "NAS100", tr, ["2018-01-01", "2018-12-31"])
    assert len(out) == 1 and out["entry_time"].dt.tz is None
    assert out.iloc[0]["r"] == -1.0 and out.iloc[0]["leg"] == "x"


def test_combine_and_same_side_gate():
    a = _leg("a", "NAS100", [("2024-01-02 10:00", "2024-01-04 10:00", 1, 1.0)])
    b = _leg("b", "NAS100", [("2024-01-03 10:00", "2024-01-03 12:00", 1, -1.0),     # same side, inside a: skipped
                             ("2024-01-03 11:00", "2024-01-03 13:00", -1, 2.0),     # other side: kept
                             ("2024-01-04 10:00", "2024-01-05 10:00", 1, 0.5)])     # a closed at this minute: kept
    c = _leg("c", "USDJPY", [("2024-01-03 10:00", "2024-01-03 16:00", 1, 0.3)])     # other symbol: kept
    assert len(combine([a, b, c])) == 5
    g = combine([a, b, c], skip_same_symbol_side=True)
    assert len(g) == 4 and -1.0 not in g["r"].tolist()


def test_metrics_drawdown_and_folds():
    tr = _leg("a", "X", [("2024-01-02", "2024-01-03", 1, 2.0), ("2024-01-04", "2024-01-05", 1, -1.0),
                         ("2024-01-06", "2024-01-07", 1, -1.0), ("2024-07-02", "2024-07-03", 1, 1.0)])
    m = metrics(tr, 0.5)
    assert m["n_trades"] == 4 and m["total_r"] == pytest.approx(1.0)
    assert m["max_dd_pct"] == pytest.approx(1.0)            # +1% then down 1%
    assert m["profit_factor"] == pytest.approx(1.5)
    folds = calendar_folds(tr, "2024-01-01", "2024-12-31", 6, 0.5)
    assert [f.test_trades for f in folds] == [3, 1]
    assert [f.test_profitable for f in folds] == [False, True]   # first half sums to 0R
    mo = monthly(tr, 0.5)
    assert len(mo) == 7 and mo["total"].sum() == pytest.approx(0.5)


def test_engine_trades_in_exit_order_and_ftmo_starts():
    rows = [(f"2024-01-{d:02d} 10:00", f"2024-01-{d:02d} 15:00", 1, 1.0) for d in range(2, 31)]
    tr = _leg("a", "X", rows)
    trades = to_engine_trades(tr.iloc[::-1], 1.0, 100000)
    assert trades[0].exit_time < trades[-1].exit_time and trades[0].pnl_usd == pytest.approx(1000.0)
    ft = ftmo_starts(trades, FTMO)
    assert ft["starts"] == 1 and ft["pass_rate"] == 1.0      # +10% in 10 trades, then +5% in 5


def test_run_portfolio_report():
    cfg = {"window": ["2024-01-01", "2024-12-31"], "initial_balance": 100000, "risk_per_trade_pct": 0.5, "folds_months": 6,
           "legs": {"a": {}, "b": {}}, "variants": {"independent": {}, "same_side_gate": {"skip_same_symbol_side": True}}}
    legs = {"a": _leg("a", "X", [("2024-01-02", "2024-01-03", 1, 1.0), ("2024-08-02", "2024-08-03", 1, -1.0)]),
            "b": _leg("b", "Y", [("2024-02-02", "2024-02-03", -1, 0.5), ("2024-09-02", "2024-09-03", -1, 0.5)])}
    res = run_portfolio(legs, cfg, FTMO)
    assert res["variants"]["independent"]["metrics"]["n_trades"] == 4
    assert set(res["legs"]) == {"a", "b"} and set(res["corr"]) == {"a", "b"}
    md = render_markdown(res, cfg)
    assert "**independent**" in md and "a alone" in md and "Monthly correlation" in md
