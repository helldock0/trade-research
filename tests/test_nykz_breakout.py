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
    assert cfg["signal"] == {**CFG["signal"], "confirm_close": True, "atr_period": 14, "atr_stop_mult": 2.0}


def test_ask_parquet_path_suffix():
    from src.data.dukascopy_ingest import parquet_path

    assert parquet_path("NAS100", "M5", Path("x")).name == "NAS100_M5.parquet"
    assert parquet_path("NAS100", "M5", Path("x"), side="ask").name == "NAS100_M5_ask.parquet"


def _cfg(**signal) -> dict:
    return {"session": CFG["session"], "signal": {**CFG["signal"], **signal}}


def test_min_break_points_ignores_shallow_closes():
    bars = _day(
        "2024-07-10",
        [
            (100, 103, 100, 101.5),  # closes 0.5 above the high: too shallow
            (101.5, 102, 100, 100.5),
            (100.5, 106, 100.5, 105),  # closes 4 above the high
            (105, 106, 104, 105.5),  # entry bar
        ],
    )
    (s,) = find_setups(bars, _cfg(min_break_points=3))
    assert s.break_time == pd.Timestamp("2024-07-10 10:10", tz="America/New_York")
    assert s.stop == 98.5 and s.entry == 106.0


def test_min_break_pct_scales_with_price():
    bars = _day("2024-07-10", [(100, 103, 100, 101.5), (101.5, 102, 101, 101.8)])
    assert find_setups(bars, _cfg(min_break_pct=1.0)) == []  # needs ~1 point
    assert len(find_setups(bars, _cfg(min_break_pct=0.1))) == 1  # needs ~0.1 point


def test_confirm_close_waits_for_a_confirmed_break():
    bars = _day(
        "2024-07-10",
        [
            (100, 103, 100, 102),  # break
            (102, 102, 99.5, 100.5),  # back inside: no confirmation
            (100.5, 104, 100.5, 103),  # second break
            (103, 105, 102.5, 104.5),  # confirms: entry at its close
        ],
    )
    assert find_setups(bars, CFG)[0].break_time == pd.Timestamp("2024-07-10 10:00", tz="America/New_York")
    (s,) = find_setups(bars, _cfg(confirm_close=True))
    assert s.break_time == pd.Timestamp("2024-07-10 10:10", tz="America/New_York")
    assert s.entry == 105.0 and s.stop == 98.5


def test_min_body_vs_range_needs_a_big_candle():
    rows = [(99.5, 101, 99, 100.5)] * 36  # range bars: 1-point bodies
    small = (101.0, 101.6, 100.9, 101.2)  # body 0.2, closes above 101
    big = (101.2, 104, 101.2, 103.5)  # body 2.3
    bars = _bars("2024-07-10", rows + [small, (101.2, 101.4, 101, 101.3), big, (103.5, 104, 103, 103.8)])
    (s,) = find_setups(bars, _cfg(min_body_vs_range=1.5))
    assert s.break_time == pd.Timestamp("2024-07-10 10:10", tz="America/New_York")


def test_stop_includes_prior_bar():
    bars = _day("2024-07-10", [(100, 100.5, 94, 99.5), (99.5, 100, 97, 98), (98, 98.5, 97.5, 98)])
    # Break bar is 10:05 (close 98 < 99); prior bar's high 100.5 beats the break bar's 100.
    assert find_setups(bars, CFG)[0].stop == 102.0
    assert find_setups(bars, _cfg(stop_includes_prior_bar=True))[0].stop == 102.5


# Every bar below has a true range of 2, so ATR(14) = 2 at entry.
ATR_BREAK = [(100.5, 102, 100, 101.5), (101.5, 102.5, 100.5, 102)]  # entry: ask 102.5, candle stop 98


def test_atr_stop_replaces_candle_stop():
    (s,) = find_setups(_day("2024-07-10", ATR_BREAK), _cfg(atr_stop_mult=1.5))
    assert s.atr_points == pytest.approx(2.0)
    assert (s.entry, s.stop, s.risk_points) == (102.5, pytest.approx(99.5), pytest.approx(3.0))
    assert s.take_profit == pytest.approx(114.5)


def test_min_stop_atr_only_widens_tight_stops():
    bars = _day("2024-07-10", ATR_BREAK)
    assert find_setups(bars, _cfg(min_stop_atr=2))[0].stop == 98.0  # 4.5 away >= 4
    assert find_setups(bars, _cfg(min_stop_atr=3))[0].stop == pytest.approx(96.5)  # widened to 6


def test_breakeven_after_1r():
    bars = _day("2024-07-10", ATR_BREAK + [(102, 107.5, 101.5, 107), (107, 107.5, 102, 102.5)])
    assert run(bars, CFG)[0][0].exit_reason == "data_end"
    (t,) = run(bars, _cfg(breakeven_at_r=1.0))[0]
    assert (t.exit_reason, t.exit_price, t.r_multiple) == ("breakeven", 102.5, 0.0)


def test_atr_trail_without_target_applies_from_next_bar():
    bars = _day(
        "2024-07-10",
        ATR_BREAK
        + [
            (102, 110, 101.5, 109.5),  # +1.7R: trail to 108 from the next bar, not this one
            (109.5, 112, 109, 111.5),  # trail to 110
            (111.5, 111.6, 109.5, 110),  # hits 110
        ],
    )
    (t,) = run(bars, _cfg(trail_atr_mult=1.0, trail_start_r=1.0, take_profit_r=0))[0]
    assert t.take_profit is None
    assert (t.exit_reason, t.exit_price) == ("trail", 110.0)
    assert t.r_multiple == pytest.approx((110 - 102.5) / 4.5)


# Break bar + confirming bar. The last range bar (high 101) and the confirming
# bar (low 101.8) leave a bullish FVG of 101-101.8 around the break bar.
FVG_BREAK = [(100, 103, 100, 102), (102, 104, 101.8, 103.5), (103.5, 104, 102.5, 103)]
FVG_CFG = {"confirm_close": True, "entry_mode": "fvg_retest"}


def test_fvg_retest_enters_on_close_back_out_of_the_gap():
    bars = _day("2024-07-10", FVG_BREAK + [(103, 103.2, 101.5, 102.4)])  # dips to 101.5, closes 102.4
    (s,) = find_setups(bars, _cfg(**FVG_CFG))
    assert s.entry_time == pd.Timestamp("2024-07-10 10:20", tz="America/New_York")
    assert s.entry == pytest.approx(102.9)  # ask close of the retest bar
    assert s.stop == 98.0  # break bar low - 2
    (s,) = find_setups(bars, _cfg(**FVG_CFG, stop_ref="fvg"))
    assert s.stop == 97.0  # gap's first bar (last range bar) low - 2


def test_fvg_retest_waits_for_the_confirming_close():
    bars = _day("2024-07-10", FVG_BREAK + [(103, 103.2, 101.2, 101.5), (101.5, 102.5, 101.4, 102.2)])
    (s,) = find_setups(bars, _cfg(**FVG_CFG))
    assert s.entry_time == pd.Timestamp("2024-07-10 10:25", tz="America/New_York")


def test_fvg_close_through_the_gap_cancels_the_day():
    bars = _day("2024-07-10", FVG_BREAK + [(103, 103.2, 100.2, 100.5), (100.5, 102.5, 100.4, 102.2)])
    assert find_setups(bars, _cfg(**FVG_CFG)) == []


def test_fvg_retest_short():
    bars = _day(
        "2024-07-10",
        [
            (100, 100, 97, 98),  # break below 99
            (98, 98.2, 96, 96.5),  # confirms; low-of-range-bar 99 > this high 98.2: gap 98.2-99
            (96.5, 97, 95, 96),
            (96, 98.6, 95.8, 97.5),  # back into the gap, closes back below 98.2
        ],
    )
    (s,) = find_setups(bars, _cfg(**FVG_CFG))
    assert (s.side, s.entry, s.stop) == (-1, 97.5, 102.0)


LIMIT_CFG = {"confirm_close": True, "entry_mode": "fvg_limit"}


def test_fvg_limit_fills_at_the_near_edge_on_ask():
    # Gap 101-101.8; ask low of the dip bar is 101.5 (bid 101.0 + 0.5 spread).
    bars = _day("2024-07-10", FVG_BREAK + [(103, 103.2, 101.0, 102.4)])
    (s,) = find_setups(bars, _cfg(**LIMIT_CFG))
    assert s.intrabar_fill
    assert (s.entry, s.stop) == (101.8, 98.0)
    assert s.take_profit == pytest.approx(101.8 + 4 * 3.8)
    # A midpoint limit (101.4) is never reached by the ask.
    assert find_setups(bars, _cfg(**LIMIT_CFG, fvg_limit_level=0.5)) == []


def test_fvg_limit_gap_open_fills_at_the_open():
    bars = _day("2024-07-10", FVG_BREAK + [(101.0, 101.5, 100.8, 101.2)])  # ask opens 101.5
    assert find_setups(bars, _cfg(**LIMIT_CFG))[0].entry == 101.5


def test_fvg_limit_fill_bar_that_reaches_the_stop_is_a_loss():
    bars = _day("2024-07-10", FVG_BREAK + [(103, 103.2, 96.5, 102.4), (102.4, 120, 102, 119)])
    (t,) = run(bars, _cfg(**LIMIT_CFG, stop_ref="fvg"))[0]  # stop 97, filled 101.8
    assert (t.exit_reason, t.exit_price) == ("stop", 97.0)
    assert t.r_multiple == pytest.approx(-1.0)
    assert t.exit_time == pd.Timestamp("2024-07-10 10:15", tz="America/New_York")
