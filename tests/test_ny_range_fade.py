from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_strategy_config
from src.strategy.ny_range_fade import completed_atr, simulate_symbol, summarize

TZ = "America/New_York"
ZERO_COSTS = {
    "pip_size": {"EURUSD": 0.0001},
    "default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0},
    "pairs": {},
}


def _day_bars(day: str, overrides: dict[str, tuple[float, float, float, float]] | None = None) -> pd.DataFrame:
    """Flat M1 day: range bars span 1.0995-1.1005 (10 pips), other bars stay inside it."""
    idx = pd.date_range(f"{day} 08:00", f"{day} 16:30", freq="1min", tz=TZ, inclusive="left")
    t = idx.hour * 60 + idx.minute
    in_range = (t >= 8 * 60 + 12) & (t < 9 * 60 + 12)
    half = pd.Series(0.0002, index=idx).where(~in_range, 0.0005)
    df = pd.DataFrame(
        {"open": 1.1, "high": 1.1 + half, "low": 1.1 - half, "close": 1.1, "volume": 1.0},
        index=idx,
    )
    for hhmm, (o, h, lo, c) in (overrides or {}).items():
        df.loc[pd.Timestamp(f"{day} {hhmm}", tz=TZ), ["open", "high", "low", "close"]] = [o, h, lo, c]
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df.reset_index()


def _cfg(**signal) -> dict:
    cfg = load_strategy_config("strategy_ny_range_fade.yaml")
    cfg["signal"].update(signal)
    return cfg


def _run(bars: pd.DataFrame, **signal):
    trades, _ = simulate_symbol(bars, "EURUSD", _cfg(**signal), ZERO_COSTS)
    return trades


@pytest.mark.parametrize("day", ["2024-01-10", "2024-07-10"])  # EST and EDT
def test_range_is_ny_local_and_first_entry_bar_is_0912(day):
    bars = _day_bars(
        day,
        {
            "08:11": (1.1, 1.1050, 1.0950, 1.1),  # just before the range: ignored
            "09:12": (1.1, 1.1012, 1.1, 1.1010),  # first bar after the range closes above
            "10:00": (1.1, 1.1, 1.0990, 1.0998),
        },
    )
    (t,) = _run(bars)
    assert t.range_high == pytest.approx(1.1005)
    assert t.range_low == pytest.approx(1.0995)
    assert t.entry_time == pd.Timestamp(f"{day} 09:12", tz=TZ).tz_convert("UTC")
    assert t.side == -1


def test_sell_on_close_above_takes_profit_at_range_low():
    bars = _day_bars(
        "2024-03-20",
        {"09:30": (1.1, 1.1012, 1.1, 1.1010), "10:00": (1.1, 1.1, 1.0990, 1.0998)},
    )
    (t,) = _run(bars)
    assert (t.side, t.exit_reason) == (-1, "tp")
    assert t.take_profit == pytest.approx(1.0995)
    assert t.gross_ru == pytest.approx(1.5)
    assert t.exit_time == pd.Timestamp("2024-03-20 10:00", tz=TZ).tz_convert("UTC")


def test_buy_on_close_below_takes_profit_at_range_high():
    bars = _day_bars(
        "2024-03-20",
        {"09:30": (1.1, 1.1, 1.0988, 1.0990), "10:00": (1.1, 1.1010, 1.1, 1.1002)},
    )
    (t,) = _run(bars)
    assert (t.side, t.exit_reason) == (1, "tp")
    assert t.gross_ru == pytest.approx(1.5)


def test_no_stop_holds_to_flat_time():
    later = {f"{h:02d}:{m:02d}": (1.1030, 1.1032, 1.1028, 1.1030) for h in range(9, 16) for m in range(60)}
    later = {k: v for k, v in later.items() if k > "09:30"}
    bars = _day_bars("2024-03-20", {"09:30": (1.1, 1.1012, 1.1, 1.1010), **later})
    (t,) = _run(bars)
    assert t.exit_reason == "time"
    assert t.exit_time == pd.Timestamp("2024-03-20 15:59", tz=TZ).tz_convert("UTC")
    assert t.gross_ru == pytest.approx(-2.0)
    assert t.mae_ru == pytest.approx(2.2)


def test_range_mult_stop():
    bars = _day_bars(
        "2024-03-20",
        {
            "09:30": (1.1, 1.1012, 1.1, 1.1010),
            "09:31": (1.1010, 1.1016, 1.1008, 1.1012),  # stop = 1.1005 + 1.0 * 0.0010
            "10:00": (1.1, 1.1, 1.0990, 1.0998),
        },
    )
    (t,) = _run(bars, stop_mode="range_mult", stop_range_mult=1.0)
    assert t.exit_reason == "stop"
    assert t.stop == pytest.approx(1.1015)
    assert t.gross_ru == pytest.approx(-0.5)


def test_completed_atr_only_uses_closed_bars():
    bars = _day_bars("2024-03-20", {"09:30": (1.1, 1.1012, 1.1, 1.1010)})
    a = completed_atr(bars, "15min", 2)
    at = lambda hhmm: a.loc[pd.Timestamp(f"2024-03-20 {hhmm}", tz=TZ).tz_convert("UTC")]
    # M15 bars 09:00 (TR 0.0010, holds range bars) and 09:15 (TR 0.0004) complete at 09:30
    assert at("09:30") == pytest.approx(0.0007)
    assert at("09:44") == pytest.approx(0.0007)  # the 09:30 bar is still open
    assert at("09:45") == pytest.approx((0.0004 + 0.0014) / 2)  # 09:30 bar spans 1.0998-1.1012


@pytest.mark.parametrize("mult, reason, gross", [(1.0, "stop", -0.7), (2.0, "tp", 1.5)])
def test_atr_stop_from_entry(mult, reason, gross):
    bars = _day_bars(
        "2024-03-20",
        {
            "09:30": (1.1, 1.1012, 1.1, 1.1010),
            "09:31": (1.1010, 1.1018, 1.1008, 1.1012),  # ATR(2) = 0.0007 -> 1x stop at 1.1017
            "10:00": (1.1, 1.1, 1.0990, 1.0998),
        },
    )
    (t,) = _run(bars, stop_mode="atr", atr_timeframe="15min", atr_period=2, stop_atr_mult=mult)
    assert t.exit_reason == reason
    assert t.stop == pytest.approx(1.1010 + mult * 0.0007)
    assert t.stop_ru == pytest.approx(mult * 0.7)
    assert t.gross_ru == pytest.approx(gross)


def test_touch_entry_fills_at_edge_but_close_mode_waits():
    bars = _day_bars(
        "2024-03-20",
        {"09:30": (1.1, 1.1008, 1.1, 1.1), "10:00": (1.1, 1.1, 1.0990, 1.0998)},
    )
    assert _run(bars) == []
    (t,) = _run(bars, entry_mode="touch")
    assert t.side == -1
    assert t.gross_ru == pytest.approx(1.0)


def test_one_trade_per_day_unless_reentry_enabled():
    bars = _day_bars(
        "2024-03-20",
        {
            "09:30": (1.1, 1.1012, 1.1, 1.1010),
            "10:00": (1.1, 1.1, 1.0990, 1.0998),
            "11:00": (1.1, 1.1, 1.0988, 1.0990),
            "11:30": (1.1, 1.1010, 1.1, 1.1002),
        },
    )
    assert len(_run(bars)) == 1
    trades = _run(bars, max_trades_per_day=2)
    assert [t.side for t in trades] == [-1, 1]
    assert [t.exit_reason for t in trades] == ["tp", "tp"]


def test_costs_reduce_pnl_and_summary():
    bars = _day_bars(
        "2024-03-20",
        {"09:30": (1.1, 1.1012, 1.1, 1.1010), "10:00": (1.1, 1.1, 1.0990, 1.0998)},
    )
    costs = {**ZERO_COSTS, "default": {"half_spread_pips": 0.5, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}}
    trades, counts = simulate_symbol(bars, "EURUSD", _cfg(), costs)
    (t,) = trades
    assert t.pnl_pips == pytest.approx(15.0 - 1.0)
    assert counts["days_with_range"] == 1
    s = summarize(pd.DataFrame([t.__dict__]))
    assert s["n_trades"] == 1 and s["win_rate"] == 1.0
