"""NY kill-zone breakout (NAS100) rule tests on synthetic bid/ask bars."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.nykz_breakout import find_setups, run, simulate

CFG = {
    "session": {
        "timezone": "America/New_York",
        "range_start": "07:00",
        "range_end": "10:00",
        "entry_cutoff": "16:55",
    },
    "signal": {"min_range_bars": 30, "stop_buffer_points": 2.0, "take_profit_r": 4.0},
}

FLAT = (100.0, 101.0, 99.0, 100.0)


def _bars(day: str, rows: list[tuple], start: str = "07:00", spread: float = 0.5) -> pd.DataFrame:
    """Consecutive 5-min bid bars from `start` NY time; ask = bid + spread."""
    idx = pd.date_range(f"{day} {start}", periods=len(rows), freq="5min", tz="America/New_York")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx.tz_convert("UTC"))
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[c] + spread
    return df


def _day(day: str, after_range: list[tuple], spread: float = 0.5) -> pd.DataFrame:
    """36 flat range bars (07:00-10:00 NY, high 101 / low 99) then `after_range`."""
    return _bars(day, [FLAT] * 36 + after_range, spread=spread)


def test_long_break_enters_next_close_at_ask_and_hits_4r():
    bars = _day(
        "2024-07-10",
        [
            (100, 103, 100, 102.5),  # 10:00 break: close > 101
            (102.5, 104, 102, 103.5),  # 10:05 entry bar
            (103.5, 129, 103, 128.5),  # bid high reaches target
        ],
    )
    trades, summary = run(bars, CFG)
    (t,) = trades
    assert t.side == 1
    assert (t.range_high, t.range_low) == (101, 99)
    assert t.entry == 104.0  # ask close of the entry bar
    assert t.stop == 98.0  # break bar low - 2
    assert t.risk_points == 6.0
    assert t.take_profit == 128.0
    assert t.exit_reason == "tp" and t.r_multiple == pytest.approx(4.0)
    assert t.break_time == pd.Timestamp("2024-07-10 10:00", tz="America/New_York")
    assert t.entry_time == pd.Timestamp("2024-07-10 10:10", tz="America/New_York")
    assert summary["n_trades"] == 1 and summary["win_rate"] == 1.0


def test_short_stop_triggers_on_ask():
    bars = _day(
        "2024-07-10",
        [
            (100, 100, 97, 98),  # break: close < 99
            (98, 98.5, 97.5, 98),  # entry: sell bid 98, stop 102, risk 4
            (98, 101.6, 97, 99),  # bid high 101.6 but ask high 102.1 >= stop
        ],
    )
    (t,) = run(bars, CFG)[0]
    assert t.side == -1
    assert (t.entry, t.stop, t.take_profit) == (98.0, 102.0, 82.0)
    assert t.exit_reason == "stop" and t.r_multiple == pytest.approx(-1.0)


def test_stop_wins_when_one_bar_spans_stop_and_target():
    bars = _day("2024-07-10", [(100, 103, 100, 102.5), (102.5, 104, 102, 103.5), (103.5, 130, 90, 100)])
    (t,) = run(bars, CFG)[0]
    assert t.exit_reason == "stop" and t.r_multiple == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    bars = _day("2024-07-10", [(100, 103, 100, 102.5), (102.5, 104, 102, 103.5), (95, 96, 94, 95)])
    (t,) = run(bars, CFG)[0]
    assert t.exit_reason == "stop_gap"
    assert t.exit_price == 95.0
    assert t.r_multiple == pytest.approx((95 - 104) / 6)


def test_entry_bar_closing_beyond_stop_skips_day_and_later_breaks():
    bars = _day(
        "2024-07-10",
        [
            (100, 103, 100, 102.5),  # long break, stop 98
            (102.5, 102.5, 96, 97),  # entry bar closes below the stop -> skip
            (97, 98, 90, 91),  # later short break is ignored (first break only)
            (91, 92, 90, 91),
        ],
    )
    assert find_setups(bars, CFG) == []


def test_entry_cutoff_on_entry_bar_close():
    # Break bar opens at 16:45 NY; entry bar closes 16:55 -> allowed.
    flat_to_1645 = [FLAT] * (36 + 81)
    ok = _bars("2024-07-10", flat_to_1645 + [(100, 103, 100, 102.5), (102.5, 104, 102, 103.5)])
    assert len(find_setups(ok, CFG)) == 1
    # One bar later: entry bar closes 17:00 -> past the cutoff.
    late = _bars("2024-07-10", flat_to_1645 + [FLAT, (100, 103, 100, 102.5), (102.5, 104, 102, 103.5)])
    assert find_setups(late, CFG) == []


@pytest.mark.parametrize("day, utc_break", [("2024-07-10", "14:00"), ("2024-01-10", "15:00")])
def test_range_follows_new_york_time_across_dst(day, utc_break):
    # A 06:55 NY spike sits just outside the range and must be ignored.
    bars = _bars(
        day,
        [(100, 150, 50, 100)] + [FLAT] * 36 + [(100, 103, 100, 102.5), (102.5, 104, 102, 103.5)],
        start="06:55",
    )
    (s,) = find_setups(bars, CFG)
    assert (s.range_high, s.range_low) == (101, 99)
    assert s.break_time == pd.Timestamp(f"{day} {utc_break}", tz="UTC")


def test_min_range_bars():
    bars = _bars("2024-07-10", [FLAT] * 20 + [(100, 103, 100, 102.5), (102.5, 104, 102, 103.5)])
    assert find_setups(bars, CFG) == []


def test_one_trade_at_a_time_skips_days_while_open():
    trade_day = [(100, 103, 100, 102.5), (102.5, 104, 102, 103.5), (103.5, 105, 103, 104)]
    bars = pd.concat([_day("2024-07-10", trade_day), _day("2024-07-11", trade_day)])
    setups = find_setups(bars, CFG)
    assert len(setups) == 2
    trades, skipped = simulate(bars, setups)
    assert skipped == 1
    (t,) = trades
    assert t.day == "2024-07-10" and t.exit_reason == "data_end"


def test_config_file_matches_spec():
    from src.config import load_strategy_config

    cfg = load_strategy_config("strategy_nykz_breakout.yaml")
    assert cfg["strategy"] == "nykz_breakout"
    assert (cfg["symbol"], cfg["timeframe"]) == ("NAS100", "M5")
    assert cfg["session"] == CFG["session"]
    assert cfg["signal"] == CFG["signal"]


def test_ask_parquet_path_suffix():
    from src.data.dukascopy_ingest import parquet_path

    assert parquet_path("NAS100", "M5", Path("x")).name == "NAS100_M5.parquet"
    assert parquet_path("NAS100", "M5", Path("x"), side="ask").name == "NAS100_M5_ask.parquet"
