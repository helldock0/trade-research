from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import pip_size, round_trip_cost_price
from src.research.box_sweep import config_table, config_trades, ist_label, sweep_symbol
from src.strategy.rolling_box import (
    REASON_SESSION,
    REASON_STOP,
    REASON_TIME,
    REASON_TP,
    REASON_TRAIL,
    breakout_candidates,
    build_boxes,
    m1_arrays,
    net_r,
    one_at_a_time,
    parse_exit,
    parse_stop,
    simulate_candidates,
)

DAY = pd.Timestamp("2024-03-04", tz="UTC")  # Monday


def _m1(rows: list[tuple[str, float, float, float, float]], fill_until: str = "20:00") -> pd.DataFrame:
    """M1 frame from (HH:MM, o, h, l, c) rows; gaps are filled flat at the last close."""
    idx = pd.date_range(DAY, DAY + pd.Timedelta(fill_until + ":00"), freq="1min", inclusive="left")
    df = pd.DataFrame(index=idx, columns=["open", "high", "low", "close"], dtype=float)
    for hhmm, o, h, lo, c in rows:
        df.loc[DAY + pd.Timedelta(hhmm + ":00")] = [o, h, lo, c]
    df["close"] = df["close"].ffill().fillna(1.1000)
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = 1.0
    df.index.name = "timestamp"
    return df.reset_index()


def _box_day() -> list[tuple[str, float, float, float, float]]:
    # 00:00-00:15 box: high 1.1010, low 1.0990
    return [("00:00", 1.1000, 1.1010, 1.0990, 1.1000)]


def _sim(m1: pd.DataFrame, stops: list[str], exits: list[str], confirm: int = 1, window: int = 60):
    m1a = m1_arrays(m1)
    boxes = build_boxes(m1, time(0, 0), time(19, 30))
    cand = breakout_candidates(m1, m1a, boxes, confirm, window)
    return cand, simulate_candidates(m1a, cand, stops, exits)


def test_parse_specs():
    assert parse_stop("opposite") == (0, 0.0)
    assert parse_stop("atr:1.5") == (2, 1.5)
    assert parse_exit("tp:2") == (0, 2.0, 0.0)
    assert parse_exit("time:15") == (1, 15.0, 0.0)
    assert parse_exit("trail_r:1:0.5") == (3, 1.0, 0.5)
    with pytest.raises(ValueError):
        parse_exit("nope")


def test_build_boxes_tokyo_to_ny():
    boxes = build_boxes(_m1([]), time(0, 0), time(19, 30))
    assert len(boxes) == 78
    first = pd.Timestamp(int(boxes["box_start"].iloc[0]) * 60, unit="s", tz="UTC")
    last = pd.Timestamp(int(boxes["box_end"].iloc[-1]) * 60, unit="s", tz="UTC")
    assert first == DAY
    assert last == DAY + pd.Timedelta("19:30:00")
    assert (boxes["sess_end"] == boxes["box_end"].iloc[-1]).all()
    assert ist_label(0) == "05:30" and ist_label(19) == "00:30"


def test_breakout_needs_close_beyond_box_and_respects_window():
    rows = _box_day() + [
        ("00:20", 1.1000, 1.1014, 1.1000, 1.1009),  # wick through, close inside -> no entry
        ("00:22", 1.1009, 1.1016, 1.1008, 1.1015),  # close above high -> entry at 00:23
    ]
    m1 = _m1(rows)
    m1a = m1_arrays(m1)
    boxes = build_boxes(m1, time(0, 0), time(19, 30))
    cand = breakout_candidates(m1, m1a, boxes, 1, 60)
    first = cand.loc[cand["box_start"] == boxes["box_start"].iloc[0]].iloc[0]
    assert first["side"] == 1
    assert pd.Timestamp(int(first["entry_t"]) * 60, unit="s", tz="UTC") == DAY + pd.Timedelta("00:23:00")
    assert first["entry_px"] == pytest.approx(1.1015)
    assert first["delay"] == 8

    # M15 confirmation: the 00:15-00:30 candle closes at 1.1015 -> entry at 00:30
    c15 = breakout_candidates(m1, m1a, boxes, 15, 15)
    first15 = c15.loc[c15["box_start"] == boxes["box_start"].iloc[0]].iloc[0]
    assert first15["delay"] == 15

    # Breakout 40 min after the box: invalid with a 15 min window
    late = _m1(_box_day() + [("00:55", 1.1000, 1.1020, 1.1000, 1.1018)])
    late_a = m1_arrays(late)
    late_boxes = build_boxes(late, time(0, 0), time(19, 30))
    b0 = late_boxes["box_start"].iloc[0]
    assert b0 not in set(breakout_candidates(late, late_a, late_boxes, 1, 15)["box_start"])
    assert b0 in set(breakout_candidates(late, late_a, late_boxes, 1, 60)["box_start"])


def _first(cand, arr):
    i = int(np.flatnonzero(cand["box_start"].to_numpy() == cand["box_start"].min())[0])
    return i, arr


def test_long_take_profit_and_stop_first_rule():
    rows = _box_day() + [
        ("00:20", 1.1000, 1.1015, 1.1000, 1.1015),  # entry 1.1015, opposite stop 1.0990 (R=25 pips)
        ("00:30", 1.1015, 1.1041, 1.1015, 1.1030),  # +1R target 1.1040 hit
    ]
    cand, (pnl, exit_t, reason, risk) = _sim(_m1(rows), ["opposite"], ["tp:1"])
    i, _ = _first(cand, pnl)
    assert risk[i, 0] == pytest.approx(0.0025)
    assert pnl[i, 0, 0] == pytest.approx(0.0025)
    assert reason[i, 0, 0] == REASON_TP

    # Same bar touches stop and target -> counted as a stop
    rows_both = _box_day() + [
        ("00:20", 1.1000, 1.1015, 1.1000, 1.1015),
        ("00:30", 1.1015, 1.1041, 1.0985, 1.1000),
    ]
    cand, (pnl, _, reason, _) = _sim(_m1(rows_both), ["opposite"], ["tp:1"])
    i, _ = _first(cand, pnl)
    assert reason[i, 0, 0] == REASON_STOP
    assert pnl[i, 0, 0] == pytest.approx(-0.0025)


def test_gap_through_stop_fills_at_open_and_mid_stop():
    rows = _box_day() + [
        ("00:20", 1.1000, 1.1015, 1.1000, 1.1015),
        ("00:30", 1.0980, 1.0982, 1.0975, 1.0978),  # gaps below both stops
    ]
    cand, (pnl, _, reason, risk) = _sim(_m1(rows), ["opposite", "mid"], ["session"])
    i, _ = _first(cand, pnl)
    assert risk[i, 1] == pytest.approx(0.0015)  # 1.1015 - mid 1.1000
    assert pnl[i, 0, 0] == pytest.approx(1.0980 - 1.1015)
    assert pnl[i, 1, 0] == pytest.approx(1.0980 - 1.1015)
    assert reason[i, 0, 0] == REASON_STOP


def test_short_side_is_mirrored():
    rows = _box_day() + [
        ("00:20", 1.1000, 1.1000, 1.0985, 1.0985),  # close below low -> short at 1.0985, stop 1.1010
        ("00:30", 1.0985, 1.0986, 1.0934, 1.0940),  # 2R target 1.0935 hit
    ]
    cand, (pnl, _, reason, risk) = _sim(_m1(rows), ["opposite"], ["tp:2"])
    i, _ = _first(cand, pnl)
    assert cand["side"].iloc[i] == -1
    assert risk[i, 0] == pytest.approx(0.0025)
    assert pnl[i, 0, 0] == pytest.approx(0.0050)
    assert reason[i, 0, 0] == REASON_TP


def test_time_session_and_trailing_exits():
    rows = _box_day() + [
        ("00:20", 1.1000, 1.1015, 1.1000, 1.1015),  # entry at 00:21 close 1.1015, R=25 pips
        ("00:25", 1.1015, 1.1020, 1.1015, 1.1020),
        ("00:35", 1.1020, 1.1022, 1.1018, 1.1021),  # bar ending 00:36 = entry + 15
        ("01:00", 1.1021, 1.1041, 1.1021, 1.1040),  # +1R reached -> trail 1R behind 1.1041
        ("01:10", 1.1040, 1.1040, 1.1010, 1.1012),  # falls back through the 1.1016 trail
    ]
    cand, (pnl, exit_t, reason, _) = _sim(_m1(rows), ["opposite"], ["time:15", "trail_r:1:1", "session"])
    i, _ = _first(cand, pnl)
    assert reason[i, 0, 0] == REASON_TIME
    assert pnl[i, 0, 0] == pytest.approx(1.1021 - 1.1015)
    assert pd.Timestamp(int(exit_t[i, 0, 0]) * 60, unit="s", tz="UTC") == DAY + pd.Timedelta("00:36:00")
    assert reason[i, 0, 1] == REASON_TRAIL
    assert pnl[i, 0, 1] == pytest.approx(1.1016 - 1.1015)
    # Session exit: flat at the 19:29 bar close (19:30 UTC = 01:00 IST)
    assert reason[i, 0, 2] == REASON_SESSION
    assert pd.Timestamp(int(exit_t[i, 0, 2]) * 60, unit="s", tz="UTC") == DAY + pd.Timedelta("19:30:00")


def test_one_at_a_time_blocks_overlap():
    entry = np.array([10, 20, 30, 50], np.int64)
    exit_ = np.array([35, 25, 40, 60], np.int64)
    keep = one_at_a_time(entry, exit_, np.ones(4, bool))
    assert keep.tolist() == [True, False, False, True]


def test_costs_in_r_and_index_pips():
    costs = {"default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7.0}}
    side, rt = round_trip_cost_price("EURUSD", costs)
    assert side == pytest.approx(0.00007)
    assert rt == pytest.approx(0.00021)  # 2 x 0.7 pip + 0.7 pip commission
    # +1R gross on a 10-pip stop nets (10 - 2.1) / 10.7
    assert net_r(0.0010, 0.0010, side, rt) == pytest.approx(7.9 / 10.7)
    assert pip_size("US30", {}) == 1.0
    no_comm = {"default": {"half_spread_pips": 1.0, "slippage_pips": 0.5, "commission_per_lot_usd": 0.0}}
    assert round_trip_cost_price("US30", no_comm) == (1.5, 3.0)


def _random_walk_m1(days: int = 15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-12-19", periods=days * 1440, freq="1min", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    close = 1.1 + np.cumsum(rng.normal(0, 0.00012, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.00008, len(idx)))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": 1.0,
        }
    )


def _small_cfg() -> dict:
    return {
        "sessions": {"start": "00:00", "end": "19:30"},
        "signal": {"box_minutes": 15, "atr_period": 14},
        "sweep": {
            "confirm_tf_minutes": [1, 15],
            "window_minutes": [15, 60],
            "stops": ["opposite", "atr:1"],
            "exits": ["tp:1", "trail_atr:1"],
            "concurrency": ["one", "multi"],
            "min_risk_cost": [0, 3],
        },
        "validation": {"oos_start": "2023-01-01"},
        "data": {"timeframe": "M1"},
    }


def test_sweep_matches_single_config_trades():
    cfg = _small_cfg()
    costs = {"default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7.0}}
    m1 = _random_walk_m1()
    summary, by_hour, by_year = sweep_symbol("EURUSD", m1, cfg, costs)
    table = config_table(cfg)
    assert len(summary) == len(table) == 2 * 2 * 2 * 2 * 2 * 2
    # Hour and year buckets add back up to the totals
    hours = by_hour.groupby("cfg_id")[["n_is", "n_oos"]].sum().sum(axis=1)
    assert (hours.reindex(summary["cfg_id"]).fillna(0).to_numpy() == summary["n"].to_numpy()).all()
    years = by_year.groupby("cfg_id")["sum_r"].sum()
    np.testing.assert_allclose(years.reindex(summary["cfg_id"]).fillna(0), summary["sum_r"], atol=1e-9)
    assert (summary["n_is"] <= summary["n"]).all() and (summary["n_is"] > 0).any()
    # Zero-cost scenario reproduces the gross R
    free = sweep_symbol("EURUSD", m1, {**cfg, "validation": {**cfg["validation"], "low_cost_mult": 0.0}}, costs)[0]
    np.testing.assert_allclose(free["sum_r_low_is"] + free["sum_r_low_oos"], free["sum_r_gross"], atol=1e-9)

    for _, row in table.sample(6, random_state=1).iterrows():
        trades = config_trades(
            "EURUSD",
            m1,
            cfg,
            costs,
            confirm_tf=row.confirm_tf,
            stop=row.stop,
            exit=row.exit,
            window=row.window,
            min_risk_cost=row.min_risk_cost,
            concurrency=row.concurrency,
        )
        s = summary.loc[summary["cfg_id"] == row.cfg_id].iloc[0]
        assert len(trades) == s["n"]
        assert trades["r_net"].sum() == pytest.approx(s["sum_r"])
        # No lookahead: entries only after the box closes, exits by 19:30 UTC
        assert (trades["entry_time"] >= trades["box_start"] + pd.Timedelta(minutes=15)).all()
        tod = trades["exit_time"].dt.hour * 60 + trades["exit_time"].dt.minute
        assert (tod <= 19 * 60 + 30).all()
        if row.concurrency == "one" and len(trades) > 1:
            assert (trades["entry_time"].iloc[1:].to_numpy() >= trades["exit_time"].iloc[:-1].to_numpy()).all()


def test_index_symbols_resolve():
    from src.data.dukascopy_ingest import FULL_UNIVERSE, INDICES, _resolve_instrument

    assert set(INDICES) <= set(FULL_UNIVERSE)
    assert _resolve_instrument("US30") == "USA30.IDX/USD"
    assert _resolve_instrument("GER40") == "DEU.IDX/EUR"
