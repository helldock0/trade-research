from __future__ import annotations

import io
import lzma
import tarfile

import numpy as np
import pandas as pd
import pytest

from src.research.jpy_recheck import (
    Costs,
    Minutes,
    box_trades,
    decode_candles,
    find_stop_entry,
    gotobi_trades,
    is_gotobi,
    load_datastore,
    manage,
    match_log,
    net_r,
    ny_daily,
    period_stats,
    render_markdown,
)

SPREAD = 0.003
GOT = {"entry_utc": "00:55", "exit_utc": "06:00", "sl_atr": 0.25}
BOX = {"box_start_utc": "23:30", "box_minutes": 60, "orders_until_utc": "05:30", "sl_box_mult": 1.0, "rr": 9.0,
       "be_trigger_r": 3.0, "be_offset_box": 1 / 6, "reentry_wait_minutes": 6}
FREE = Costs(0.0, 0.0, 0.0, None)


def _minutes(path: np.ndarray, start: str, wick: float = 0.001) -> Minutes:
    """Bid bars open at path[i] and close at path[i+1]; the ask sits SPREAD above."""
    o = np.asarray(path, float)
    c = np.append(o[1:], o[-1])
    h, low = np.maximum(o, c) + wick, np.minimum(o, c) - wick
    return Minutes(pd.Timestamp(start), o, h, low, c, o + SPREAD, h + SPREAD, low + SPREAD, c + SPREAD, np.ones(len(o)))


def _records(rows: list[tuple[int, int, int, int, int, float]]) -> bytes:
    raw = b"".join(np.array([r[:5]], dtype=">u4").tobytes() + np.array([r[5]], dtype=">f4").tobytes() for r in rows)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_decode_candles_reads_big_endian_records():
    out = decode_candles(_records([(0, 150000, 150010, 149990, 150020, 2.5), (60, 150010, 150005, 150000, 150030, 1.0)]), 0.001)
    assert out.shape == (2, 6)
    assert out[1, 0] == 1                                    # seconds -> minute
    assert out[0, 1:5].tolist() == pytest.approx([150.0, 150.02, 149.99, 150.01])   # open, high, low, close
    assert decode_candles(b"", 0.001).shape == (0, 6)


def test_load_datastore_reads_tar_members_onto_a_minute_grid(tmp_path):
    tp = tmp_path / "USDJPY_2024.tar"
    with tarfile.open(tp, "w") as tf:
        for side, base in (("BID", 150000), ("ASK", 150003)):
            blob = _records([(0, base, base + 5, base - 5, base + 10, 1.0), (120, base + 5, base, base, base + 5, 1.0)])
            info = tarfile.TarInfo(f"USDJPY_20240102_{side}.bi5")
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    m = load_datastore(tmp_path, "USDJPY", "2024-01-02", "2024-01-02")
    assert m.n == 1440 and m.t0 == pd.Timestamp("2024-01-02")
    assert m.bo[0] == pytest.approx(150.0) and m.ao[0] == pytest.approx(150.003)
    assert m.bo[1] == m.bo[0] and m.vol[1] == 0              # empty minute: forward-filled, no ticks
    assert m.bo[2] == pytest.approx(150.005) and m.vol[2] > 0


def test_is_gotobi_moves_weekend_dates_to_the_previous_weekday():
    assert is_gotobi(pd.Timestamp("2024-01-05")) and is_gotobi(pd.Timestamp("2024-01-31"))
    assert is_gotobi(pd.Timestamp("2024-02-09"))             # 10th is a Saturday
    assert not is_gotobi(pd.Timestamp("2024-02-10"))
    assert is_gotobi(pd.Timestamp("2024-03-29"))             # month-end on a Sunday
    assert not is_gotobi(pd.Timestamp("2024-01-08"))


def test_ny_daily_atr_is_known_before_the_day():
    rng = np.random.default_rng(0)
    m = _minutes(150 + np.cumsum(rng.normal(0, 0.01, 40 * 1440)), "2024-01-01")
    d = ny_daily(m, median_days=5)
    day = d.index[25]
    cut = m.idx(day - pd.Timedelta(hours=2))                 # 17:00 New York in January = 22:00 UTC the day before
    f = m.frame()
    f.iloc[cut:, :8] = f.iloc[cut:, :8] * 1.05               # change everything from the day's open on
    d2 = ny_daily(Minutes.from_frame(f), median_days=5)
    assert d2.loc[day, "atr20"] == pytest.approx(d.loc[day, "atr20"])
    assert d2.loc[day, "atr_med"] == pytest.approx(d.loc[day, "atr_med"])
    assert d2.loc[d.index[27], "atr20"] != pytest.approx(d.loc[d.index[27], "atr20"])


def test_rollovers_and_net_r():
    c = Costs(7.0, 0.2, 1.0, None)
    assert c.rollovers(pd.Timestamp("2024-01-05 00:55"), pd.Timestamp("2024-01-05 06:00")) == 0
    assert c.rollovers(pd.Timestamp("2024-01-02 23:00"), pd.Timestamp("2024-01-04 23:00")) == 4   # Wednesday counts 3
    gross, net = net_r(-1, 150.0, 149.9, 0.1, 1, pd.Timestamp("2024-01-05 01:00"), pd.Timestamp("2024-01-05 05:00"), 0.01, c)
    pv_x, pv_e = 1000 / 149.9, 1000 / 150.0
    assert gross == pytest.approx(1.0)
    assert net == pytest.approx((10 * pv_x - 7.0 - 0.2 * pv_x) / (10 * pv_e))


def _gotobi_path(jump_at: int | None = None) -> Minutes:
    p = np.full(1440, 150.0)
    p[56:361] = np.linspace(150.0, 149.8, 305)               # drifts down from the fix to 06:00
    p[361:] = 149.8
    if jump_at is not None:
        p[jump_at:] = 150.3
    m = _minutes(p, "2024-01-05")
    if jump_at is not None:                                  # a real gap: the bar before stays below the stop
        k = jump_at - 1
        m.bc[k], m.bh[k] = m.bo[k], m.bo[k] + 0.001
        m.ac[k], m.ah[k] = m.bc[k] + SPREAD, m.bh[k] + SPREAD
    return m


def test_gotobi_time_exit_and_gapped_stop():
    daily = pd.DataFrame({"atr20": [0.4], "atr_med": [0.3]}, index=[pd.Timestamp("2024-01-05")])
    t = gotobi_trades(_gotobi_path(), daily, GOT, FREE).iloc[0]
    assert t["reason"] == "TIME" and t["entry_time"] == pd.Timestamp("2024-01-05 00:55") and t["hot"]
    assert t["entry"] == pytest.approx(150.0) and t["risk_pips"] == pytest.approx(10.0)
    assert t["exit"] == pytest.approx(149.8 + SPREAD)        # buys back on the ask at 06:00
    assert t["gross_r"] == pytest.approx((150.0 - 149.803) / 0.1)
    s = gotobi_trades(_gotobi_path(jump_at=120), daily, GOT, FREE).iloc[0]
    assert s["reason"] == "SL" and s["exit"] == pytest.approx(150.3 + SPREAD)   # gapped through: fills at the open
    assert gotobi_trades(_gotobi_path(), daily.set_axis([pd.Timestamp("2024-01-08")]), GOT, FREE).empty


def test_stop_entry_and_manage_rules():
    p = np.full(300, 150.0)
    m = _minutes(p, "2024-01-03")
    assert find_stop_entry(m, 0, 300, 150.05, 149.95) is None
    p[100:] = 150.06
    side, i, px = find_stop_entry(_minutes(p, "2024-01-03"), 0, 300, 150.05, 149.95)
    assert (side, i) == (1, 99) and px == pytest.approx(150.05)   # bar 99 closes at 150.06: ask high crosses
    # one bar that touches both the stop and the target: the stop wins
    q = np.full(50, 150.0)
    mm = _minutes(q, "2024-01-03")
    mm.bl[10], mm.bh[10] = 149.0, 152.0
    assert manage(mm, 1, 0, 150.0, 0.1, 9.0, 3.0, 0.01)[2] == "SL"
    # after 3R the stop sits at entry + offset
    r = np.concatenate([np.linspace(150.0, 150.35, 20), np.linspace(150.35, 149.9, 30)])
    _, xp, why = manage(_minutes(r, "2024-01-03"), 1, 0, 150.0, 0.1, 9.0, 3.0, 0.02)
    assert why == "BE" and xp == pytest.approx(150.02)


def _box_day(after: np.ndarray) -> Minutes:
    """Minutes from 2024-01-02 00:00: a 149.95-150.05 box at 23:30-00:29 UTC, then `after` from 00:30 on 2024-01-03."""
    p = np.full(2 * 1440, 150.0)
    b0 = 1410                                                # 2024-01-02 23:30
    p[b0:b0 + 60] = 150.0
    m = _minutes(np.concatenate([p[: 1440 + 30], after, np.full(2 * 1440 - 1470 - len(after), after[-1])]), "2024-01-02")
    m.bh[b0 + 10], m.bl[b0 + 40] = 150.05, 149.95
    return m


def test_box_breakout_hits_target():
    after = np.concatenate([np.full(30, 150.0), np.linspace(150.0, 151.1, 120)])
    tr = box_trades(_box_day(after), pd.DataFrame(columns=["atr20", "atr_med"]), BOX, FREE, end="2024-01-03")
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["side"] == 1 and t["reason"] == "TP" and t["entry"] == pytest.approx(150.05)
    assert t["box_pips"] == pytest.approx(10.0) and t["gross_r"] == pytest.approx(9.0)
    net = box_trades(_box_day(after), pd.DataFrame(columns=["atr20", "atr_med"]), BOX, Costs(7.0, 0.2, 1.0, None), end="2024-01-03")
    assert net.iloc[0]["r"] < t["gross_r"]


def test_box_reentry_after_a_stop():
    up = np.linspace(150.0, 150.06, 10)
    down = np.linspace(150.06, 149.90, 10)                   # through the long's stop
    back = np.full(10, 150.0)                                # back inside the range
    fall = np.linspace(150.0, 149.80, 20)                    # through the range low: sell stop
    tr = box_trades(_box_day(np.concatenate([up, down, back, fall])), pd.DataFrame(columns=["atr20", "atr_med"]), BOX, FREE, end="2024-01-03")
    assert tr["num"].tolist() == [1, 2] and tr["side"].tolist() == [1, -1]
    assert tr.iloc[0]["reason"] == "SL" and tr.iloc[1]["entry"] == pytest.approx(149.95)


def test_match_log_and_report():
    mine = pd.DataFrame({"day": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-15"]), "r": [0.5, -1.0, 0.2]})
    log = pd.DataFrame({"day": ["2024-01-05", "2024-01-10"], "R": [0.51, -0.9]})
    mt = match_log(mine, log)
    assert (mt["logged"], mt["found"], mt["within_tol"]) == (2, 2, 1) and mt["share"] == pytest.approx(0.5)
    assert mt["extra_here"] == 0
    st = period_stats(mine, {"all": ["2024-01-01", "2024-12-31"], "none": ["2025-01-01", "2025-12-31"]})
    assert st["all"]["n"] == 3 and st["none"]["n"] == 0
    s = {"n": 1, "avg_r": 0.1, "t": 0.0, "win": 1.0}
    res = {"verdict": {"gotobi_matches": True, "vol_box_matches": False, "vol-gotobi": True, "vol-box": False},
           "gotobi_match": mt, "last3y": {"vol-box": {"mine": s, "theirs": {"n": 359, "avg_r": 0.24}}},
           "periods": {"p": ["2024-01-01", "2024-12-31"]},
           "stats": {"gotobi": {"p": s, "2018+": s, "2018+ gross": s, "exits": {"TIME": 1}}}}
    md = render_markdown(res)
    assert "| vol-box edge confirmed" in md and "NO |" in md
