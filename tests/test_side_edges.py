import numpy as np
import pandas as pd
import pytest

from src.strategy.side_edges import cash_sessions, dip_schedule, dip_trades, gotobi_trades, is_gotobi

LOT = 100_000


def _jpy_bars(day: str, bid: float = 150.0, spread: float = 0.02) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 00:00", f"{day} 07:00", freq="5min", tz="UTC")
    b = pd.DataFrame({c: bid for c in ("open", "high", "low", "close")}, index=idx)
    for c in ("open", "high", "low", "close"):
        b[f"ask_{c}"] = bid + spread
    return b


def _jpy_daily(day: str, atr20: float = 0.8, atr_med: float = 0.7) -> pd.DataFrame:
    return pd.DataFrame({"high": [0.0], "low": [0.0], "close": [0.0], "atr20": [atr20], "atr_med": [atr_med]}, index=pd.DatetimeIndex([day]))


def _r_short(entry: float, exit_: float, risk: float, fills: int) -> float:
    pnl = (entry - exit_) * LOT / exit_ - 7.0 - 0.002 * fills * LOT / exit_
    return pnl / (risk * LOT / entry)


def test_gotobi_days():
    assert is_gotobi(pd.Timestamp("2026-11-05"))
    assert not is_gotobi(pd.Timestamp("2026-11-06"))
    assert is_gotobi(pd.Timestamp("2026-10-23"))  # the 25th is a Sunday
    assert is_gotobi(pd.Timestamp("2026-10-30"))  # month-end (31st) is a Saturday
    assert is_gotobi(pd.Timestamp("2026-11-30"))


def test_gotobi_time_exit():
    trades = gotobi_trades(_jpy_bars("2024-03-05"), _jpy_daily("2024-03-05"), {"trade_from": "2024-01-01"})
    assert len(trades) == 1
    t = trades[0]
    assert t.side == -1 and t.entry == 150.0 and t.stop == pytest.approx(150.2)
    assert t.entry_time == pd.Timestamp("2024-03-05 00:55", tz="UTC")
    assert t.exit_time == pd.Timestamp("2024-03-05 06:00", tz="UTC")
    assert t.exit_reason == "time" and t.exit_price == pytest.approx(150.02)
    assert t.r_multiple == pytest.approx(_r_short(150.0, 150.02, 0.2, 1))


def test_gotobi_stop_and_gap():
    bars = _jpy_bars("2024-03-05")
    bars.loc[pd.Timestamp("2024-03-05 02:00", tz="UTC"), "ask_high"] = 150.25
    t = gotobi_trades(bars, _jpy_daily("2024-03-05"), {"trade_from": "2024-01-01"})[0]
    assert t.exit_reason == "stop" and t.exit_price == pytest.approx(150.2)
    assert t.exit_time == pd.Timestamp("2024-03-05 02:00", tz="UTC")
    assert t.r_multiple == pytest.approx(_r_short(150.0, 150.2, 0.2, 2))
    assert t.mae_r == pytest.approx(-t.r_multiple)  # the loss after costs is a bit over 1R

    bars.loc[pd.Timestamp("2024-03-05 02:00", tz="UTC"), ["ask_open", "ask_high"]] = [150.3, 150.35]
    t = gotobi_trades(bars, _jpy_daily("2024-03-05"), {"trade_from": "2024-01-01"})[0]
    assert t.exit_reason == "stop_gap" and t.exit_price == pytest.approx(150.3)


def test_gotobi_filters():
    bars = _jpy_bars("2024-03-05")
    quiet = _jpy_daily("2024-03-05", atr20=0.6, atr_med=0.7)
    assert gotobi_trades(bars, quiet, {"trade_from": "2024-01-01"}) == []
    assert len(gotobi_trades(bars, quiet, {"trade_from": "2024-01-01", "vol_filter": False})) == 1
    assert gotobi_trades(_jpy_bars("2024-03-06"), _jpy_daily("2024-03-06"), {"trade_from": "2024-01-01"}) == []


def test_cash_sessions_drop_early_closes():
    # 2024-07-03: 16:00 New York close; 2024-07-04: the CFD stops at 13:00
    idx = pd.date_range("2024-07-02 20:00", "2024-07-04 17:00", freq="5min", tz="UTC")
    b = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    s = cash_sessions(b)
    assert list(s.index) == [pd.Timestamp("2024-07-03")]
    assert s["close_ts"].iloc[0] == pd.Timestamp("2024-07-03 20:00", tz="UTC")


# Steady +10 closes, then four -10 closes: RSI(2) ends at 6.25, still above the 20-day average.
CLOSES = [10.0 * k for k in range(1, 21)] + [190.0, 180.0, 170.0, 160.0, 165.0, 180.0, 190.0]
SIG = {"trend_sma": 20, "rsi_period": 2, "rsi_entry": 10, "exit_sma": 5, "max_days": 10, "trade_from": "2000-01-01"}


def test_dip_schedule():
    assert dip_schedule(pd.Series(CLOSES), SIG) == [(23, 25)]  # 180 is the first close above its 5-day average
    assert dip_schedule(pd.Series(CLOSES), {**SIG, "max_days": 1}) == [(23, 24)]


def _dip_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.bdate_range("2024-01-02", periods=len(CLOSES))
    c = np.array(CLOSES)
    daily = pd.DataFrame({"open": c, "high": c + 5, "low": c - 5, "close": c}, index=days)
    daily["close_ts"] = [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=21) for d in days]  # 16:00 EST
    idx = pd.DatetimeIndex(daily["close_ts"] - pd.Timedelta(minutes=5))
    bars = pd.DataFrame({k: c for k in ("open", "high", "low", "close")}, index=idx)
    for k in ("open", "high", "low", "close"):
        bars[f"ask_{k}"] = c + 1
    return bars, daily


def test_dip_trade_exit_and_stop():
    bars, daily = _dip_data()
    fin = 0.06 / 365
    t = dip_trades(bars, daily, SIG)[0]
    # true range 15 every day -> ATR 15, stop 2.5 x 15 below the 161 ask fill
    assert t.side == 1 and t.entry == 161.0 and t.risk_points == pytest.approx(37.5)
    assert t.exit_reason == "exit_sma" and t.exit_price == 180.0
    held = (daily.index[25] - daily.index[23]).days
    assert t.r_multiple == pytest.approx((180.0 - 161.0 - 161.0 * fin * held) / 37.5)

    gap = pd.DataFrame({"open": [150.0], "high": [150.0], "low": [120.0], "close": [130.0]}, index=[bars.index[23] + pd.Timedelta(hours=3)])
    for k in ("open", "high", "low", "close"):
        gap[f"ask_{k}"] = gap[k] + 1
    t = dip_trades(pd.concat([bars, gap]).sort_index(), daily, SIG)[0]
    assert t.exit_reason == "stop" and t.exit_price == pytest.approx(123.5)
    assert t.r_multiple == pytest.approx((123.5 - 161.0 - 161.0 * fin * 0) / 37.5)
