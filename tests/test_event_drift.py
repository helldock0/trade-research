from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.data.calendar import load_event_calendar, normalize_calendar, parse_number, save_calendar
from src.features.event_table import (
    bar_minutes,
    build_event_table,
    label_at_or_after,
    label_at_or_before,
    trade_side,
)
from src.features.news import load_news_calendar
from src.features.nowcast import gasoline_cpi_nowcast, gasoline_mom_for_events, month_average_before
from src.features.surprise import (
    add_standardized_surprises,
    add_surprise_momentum,
    event_polarity,
    filter_events,
    polarity_for_rows,
)
from src.research.event_study import render_markdown, run_event_study, summarize_event_table, write_event_study_report
from src.strategy.cssm import Signal
from src.strategy.event_drift import generate_signals, prepare_events, signals_from_table
from src.strategy.registry import resolve_signal_fn

COSTS = {
    "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7.0},
    "pairs": {},
    "pip_size": {"EURUSD": 0.0001},
}

CPI = "Consumer Price Index (MoM)"


def _cfg(**overrides):
    from src.config import load_strategy_config

    cfg = load_strategy_config("strategy_event_drift.yaml")
    cfg["universe"] = ["EURUSD"]
    cfg["nowcast"]["enabled"] = False
    cfg["events"]["list"] = [{"currency": "USD", "name": CPI}]
    cfg["events"]["currency_pairs"] = {"USD": ["EURUSD"]}
    cfg["signal"]["entry_blackout_min"] = 0
    for k, v in overrides.items():
        cfg["signal"][k] = v
    return cfg


# ---------------------------------------------------------------- calendar


def test_parse_number_variants():
    assert parse_number("0.2%") == pytest.approx(0.2)
    assert parse_number("190K") == pytest.approx(190_000)
    assert parse_number("-775M") == pytest.approx(-775e6)
    assert parse_number(1.5) == 1.5
    assert math.isnan(parse_number(""))
    assert math.isnan(parse_number(None))
    assert math.isnan(parse_number("n/a"))


def test_normalize_calendar_fxstreet_and_forexfactory(tmp_path):
    fx = pd.DataFrame(
        [
            {
                "dateUtc": "2024-01-11T13:30:00Z",
                "currencyCode": "USD",
                "countryCode": "US",
                "name": CPI,
                "volatility": "HIGH",
                "actual": 0.3,
                "consensus": 0.2,
                "previous": 0.1,
                "revised": None,
                "unit": "%",
                "periodDateUtc": "2023-12-22T00:00:00Z",
                "eventId": "abc",
            }
        ]
    )
    cal = normalize_calendar(fx)
    assert list(cal.columns)[:4] == ["timestamp", "currency", "country", "event"]
    assert cal.loc[0, "impact"] == "high"
    assert cal.loc[0, "forecast"] == pytest.approx(0.2)
    assert cal.loc[0, "period"] == pd.Timestamp("2023-12-22")
    assert str(cal.loc[0, "timestamp"].tz) == "UTC"

    ff = pd.DataFrame(
        [
            {"title": "Non-Farm Employment Change", "country": "USD", "date": "2024-01-05T13:30:00Z", "impact": "High", "actual": "216K", "forecast": "168K", "previous": "173K"},
            {"title": "CPI m/m", "country": "USD", "date": "2024-01-11 13:30", "impact": "High", "actual": "0.3%", "forecast": "0.2%", "previous": "0.1%"},
        ]
    )
    cal2 = normalize_calendar(ff)
    assert len(cal2) == 2  # mixed timestamp layouts both parse
    assert cal2.loc[1, "timestamp"] == pd.Timestamp("2024-01-11 13:30", tz="UTC")
    assert cal2.loc[0, "event"] == "Non-Farm Employment Change"
    assert cal2.loc[0, "actual"] == pytest.approx(216_000)
    assert cal2.loc[0, "impact"] == "high"

    # Round trip through save/load and compatibility with the blackout loader
    p = tmp_path / "cal.csv"
    save_calendar(cal, p)
    back = load_event_calendar(p)
    assert len(back) == 1 and back.loc[0, "actual"] == pytest.approx(0.3)
    blackout = load_news_calendar(p)
    assert set(blackout.columns) == {"timestamp", "currency", "impact"}
    assert blackout.loc[0, "impact"] == "high"


# ---------------------------------------------------------------- surprises


def _monthly_cpi(n: int, surprises: list[float], start="2023-01-12 13:30") -> pd.DataFrame:
    ts = [pd.Timestamp(start, tz="UTC") + pd.DateOffset(months=i) for i in range(n)]
    forecast = [0.2] * n
    actual = [0.2 + s for s in surprises]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "currency": "USD",
            "country": "US",
            "event": CPI,
            "impact": "high",
            "actual": actual,
            "forecast": forecast,
            "previous": [np.nan] + actual[:-1],
            "revised": np.nan,
            "unit": "%",
            "period": [t.normalize().tz_localize(None) - pd.DateOffset(months=1) for t in ts],
            "event_id": "cpi",
        }
    )


def test_standardized_surprises_are_point_in_time():
    surprises = [0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.3, -0.3, 0.1, 0.2]
    cal = _monthly_cpi(len(surprises), surprises)
    x = add_standardized_surprises(cal, min_history=6, window=36)
    x = add_surprise_momentum(x, k=2)
    # First 6 rows have no history std; row 6 uses exactly rows 0..5
    assert x["hist_std"].iloc[:6].isna().all()
    expected = float(np.std(surprises[:6], ddof=1))
    assert x["hist_std"].iloc[6] == pytest.approx(expected)
    assert x["surprise_z"].iloc[6] == pytest.approx(surprises[6] / expected)
    # Momentum at row 8 averages z of rows 6 and 7 only (nothing from row 8)
    z6, z7 = x["surprise_z"].iloc[6], x["surprise_z"].iloc[7]
    assert x["surprise_mom"].iloc[8] == pytest.approx((z6 + z7) / 2)
    assert math.isnan(x["surprise_mom"].iloc[6])  # no prior z yet
    # hist_std at row 9 must not include row 9's own surprise
    assert x["hist_std"].iloc[9] == pytest.approx(float(np.std(surprises[:9], ddof=1)))


def test_polarity_and_side_mapping():
    assert event_polarity("Nonfarm Payrolls") == 1
    assert event_polarity("Unemployment Rate") == -1
    assert event_polarity("Initial Jobless Claims") == -1
    cal = pd.DataFrame({"currency": ["USD", "USD"], "event": ["Unemployment Rate", CPI]})
    pol = polarity_for_rows(cal, [{"currency": "USD", "name": "Unemployment Rate", "polarity": 1}])
    assert list(pol) == [1, 1]  # explicit override wins
    # USD-positive signal: short EURUSD, long USDJPY; polarity flips it
    assert trade_side("USD", "EURUSD", 1, 1.2) == -1
    assert trade_side("USD", "USDJPY", 1, 1.2) == 1
    assert trade_side("USD", "EURUSD", -1, 1.2) == 1
    assert trade_side("EUR", "EURUSD", 1, -0.8) == -1
    assert trade_side("USD", "EURUSD", 1, 0.0) == 0


def test_filter_events_by_list_and_impact():
    cal = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
            "currency": ["USD", "USD", "EUR"],
            "event": [CPI, "Something Low", "HICP"],
            "impact": ["high", "low", "high"],
        }
    )
    assert len(filter_events(cal, [{"currency": "USD", "name": CPI}])) == 1
    assert len(filter_events(cal, None, "high")) == 2


# ---------------------------------------------------------------- bar snapping


def test_label_snapping_and_bar_minutes():
    idx = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", tz="UTC")
    t = pd.Timestamp("2024-01-01 00:40", tz="UTC")
    assert label_at_or_before(idx, t) == pd.Timestamp("2024-01-01 00:30", tz="UTC")
    assert label_at_or_after(idx, t) == pd.Timestamp("2024-01-01 00:45", tz="UTC")
    assert label_at_or_before(idx, pd.Timestamp("2023-12-31 23:00", tz="UTC")) is None
    assert label_at_or_after(idx, pd.Timestamp("2024-01-01 02:00", tz="UTC")) is None
    assert bar_minutes("M15") == 15 and bar_minutes("H1") == 60
    with pytest.raises(ValueError):
        bar_minutes("W1")


# ---------------------------------------------------------------- engine


def _flat_bars(start: str, periods: int, px: float = 1.1) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": idx, "open": px, "high": px + 0.0005, "low": px - 0.0005, "close": px, "volume": 1.0}
    )


def test_engine_time_exit_closes_at_first_bar_at_or_after_exit_time():
    bars = _flat_bars("2024-01-02 00:00", 200)
    cfg = _cfg()
    entry_ts = pd.Timestamp("2024-01-02 01:00", tz="UTC")
    exit_time = pd.Timestamp("2024-01-02 03:07", tz="UTC")  # between labels -> 03:15
    sig = Signal(
        timestamp=entry_ts, symbol="EURUSD", side=1, entry=1.1, stop=1.09, take_profit=1.2,
        risk_price=0.01, session="event_pre", range_high=1.11, range_low=1.09, reason="t", exit_time=exit_time,
    )
    res = run_backtest([sig], {"EURUSD": bars}, cfg, COSTS)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.exit_reason == "time_exit"
    assert tr.exit_time == pd.Timestamp("2024-01-02 03:15", tz="UTC")


def test_signal_without_exit_time_is_unchanged():
    bars = _flat_bars("2024-01-02 00:00", 50)
    cfg = _cfg(max_hold_days=0)
    sig = Signal(
        timestamp=pd.Timestamp("2024-01-02 01:00", tz="UTC"), symbol="EURUSD", side=1, entry=1.1, stop=1.09,
        take_profit=1.2, risk_price=0.01, session="x", range_high=1.11, range_low=1.09, reason="t",
    )
    assert sig.exit_time is None
    res = run_backtest([sig], {"EURUSD": bars}, cfg, COSTS)
    assert res.trades[0].exit_reason == "eod_force"


# ---------------------------------------------------------------- synthetic end-to-end

EVENT_TIMES = ["2024-01-11 13:30", "2024-02-13 13:30", "2024-03-12 13:30", "2024-04-10 13:30"]


def _path_price(t: pd.Timestamp, events: list[pd.Timestamp], base: float = 1.1) -> float:
    """Piecewise path: drift down into each print, drop at the print, partial rebound after."""
    px = base
    for T in events:
        ramp0, ramp1 = T - pd.Timedelta(hours=48), T - pd.Timedelta(minutes=30)
        if t <= ramp0:
            continue
        if t <= ramp1:
            px -= 0.0050 * (t - ramp0) / (ramp1 - ramp0)
            continue
        px -= 0.0050
        if t < T:
            continue
        px -= 0.0030
        reb0, reb1 = T + pd.Timedelta(minutes=15), T + pd.Timedelta(minutes=255)
        if t < reb0:
            continue
        if t <= reb1:
            px += 0.0020 * (t - reb0) / (reb1 - reb0)
        else:
            px += 0.0020
    return px


def _synthetic_bars() -> pd.DataFrame:
    events = [pd.Timestamp(t, tz="UTC") for t in EVENT_TIMES]
    idx = pd.date_range("2023-12-13", "2024-04-20", freq="15min", tz="UTC")
    bar_len = pd.Timedelta(minutes=15)
    close = np.array([_path_price(L + bar_len, events) for L in idx])
    opn = np.array([_path_price(L, events) for L in idx])
    hi = np.maximum(opn, close) + 0.0003
    lo = np.minimum(opn, close) - 0.0003
    return pd.DataFrame({"timestamp": idx, "open": opn, "high": hi, "low": lo, "close": close, "volume": 1.0})


def _synthetic_calendar() -> pd.DataFrame:
    # 12 monthly prints before the bar window, then the 4 in-window events; surprises mostly hot
    pre = [0.1, 0.2, 0.1, 0.15, 0.2, 0.1, 0.25, 0.1, 0.2, 0.15, 0.1, 0.2]
    cal = _monthly_cpi(len(pre), pre, start="2023-01-12 13:30")
    tail = _monthly_cpi(4, [0.1, 0.2, 0.1, 0.2], start="2024-01-11 13:30")
    tail["timestamp"] = [pd.Timestamp(t, tz="UTC") for t in EVENT_TIMES]
    # add an unrelated high-impact release inside the window to exercise filtering
    other = tail.iloc[[0]].copy()
    other["event"] = "ISM Manufacturing PMI"
    other["timestamp"] = pd.Timestamp("2024-02-01 15:00", tz="UTC")
    return pd.concat([cal, tail, other], ignore_index=True)


def test_event_table_synthetic_drift_and_variants():
    bars = {"EURUSD": _synthetic_bars()}
    cal = _synthetic_calendar()
    cfg = _cfg()
    events = prepare_events(cal, cfg)
    assert set(events["event"]) == {CPI}
    in_window = events[events["timestamp"] >= pd.Timestamp("2024-01-01", tz="UTC")]
    assert len(in_window) == 4
    assert (in_window["signal_z"] > 0).all()

    table = build_event_table(events, bars, cfg, COSTS)
    assert len(table) == 4
    assert (table["symbol"] == "EURUSD").all()
    assert (table["side"] == -1).all()  # hot-CPI signal -> short EURUSD
    assert table["pre_clean"].all()
    # pre-exit bar closes before the print margin, entry 48h before
    for r in table.itertuples():
        assert r.entry_label == r.timestamp - pd.Timedelta(hours=48)
        assert r.pre_label + pd.Timedelta(minutes=15) <= r.timestamp - pd.Timedelta(minutes=30)
        assert r.post_label >= r.timestamp + pd.Timedelta(minutes=60)
        assert r.fade_entry_label >= r.timestamp + pd.Timedelta(minutes=15)
    assert (table["pre_ret"] > 0.0045).all()
    assert (table["post_ret"] > table["pre_ret"]).all()
    assert (table["reaction"] < 0).all()
    assert (table["fade_side"] == 1).all()
    assert (table["fade_ret"] > 0.0015).all()
    assert (table["pre_r"] > 0).all() and (table["fade_r"] > 0).all()
    assert (table["pred_move_atr"] > 0).all()

    # Signals per variant carry the scheduled exit and the right side
    pre = signals_from_table(table, "pre", cfg, COSTS, cal)
    post = signals_from_table(table, "post", cfg, COSTS, cal)
    fade = signals_from_table(table, "fade", cfg, COSTS, cal)
    assert len(pre) == len(post) == len(fade) == 4
    assert all(s.side == -1 and s.exit_time < s_t for s, s_t in zip(pre, table["timestamp"]))
    assert all(s.exit_time == t + pd.Timedelta(minutes=60) for s, t in zip(post, table["timestamp"]))
    assert all(s.side == 1 and s.session == "event_fade" for s in fade)
    assert all(s.stop > s.entry for s in pre)  # short: stop above
    with pytest.raises(ValueError):
        signals_from_table(table, "nope", cfg, COSTS, cal)

    # Engine honors the scheduled exits and books the drift as profit
    res = run_backtest(pre, bars, cfg, COSTS)
    assert len(res.trades) == 4
    assert all(t.exit_reason == "time_exit" for t in res.trades)
    assert all(t.pnl_usd > 0 for t in res.trades)
    assert all(t.exit_time < T for t, T in zip(res.trades, table["timestamp"]))


def test_generate_signals_registry_and_blackout():
    bars = {"EURUSD": _synthetic_bars()}
    cal = _synthetic_calendar()
    cfg = _cfg()
    assert resolve_signal_fn(cfg).__module__.endswith("event_drift")
    sigs = generate_signals(bars, cfg, COSTS, calendar_df=cal)
    assert len(sigs) == 4
    # An unrelated high-impact release at the entry bar blocks the entry
    cfg_b = _cfg(entry_blackout_min=60)
    clash = cal.iloc[[-1]].copy()
    clash["timestamp"] = pd.Timestamp(EVENT_TIMES[0], tz="UTC") - pd.Timedelta(hours=48)
    clash["event"] = "FOMC Minutes"
    cal_b = pd.concat([cal, clash], ignore_index=True)
    sigs_b = generate_signals(bars, cfg_b, COSTS, calendar_df=cal_b)
    assert len(sigs_b) == 3


def test_run_event_study_summary_and_report(tmp_path):
    bars = {"EURUSD": _synthetic_bars()}
    cal = _synthetic_calendar()
    cfg = _cfg()
    ftmo = {
        "account": {"initial_balance": 100000},
        "phase1": {"profit_target_pct": 10, "max_daily_loss_pct": 5, "max_loss_pct": 10, "min_trading_days": 4},
        "phase2": {"profit_target_pct": 5, "max_daily_loss_pct": 5, "max_loss_pct": 10, "min_trading_days": 4},
        "daily_reset": {"timezone": "Europe/Prague"},
    }
    result = run_event_study(bars, cfg, COSTS, ftmo, calendar_df=cal)
    assert result["n_table_rows"] == 4
    raw = result["raw"]
    assert raw["variants"]["pre"]["r"]["n"] == 4
    assert raw["variants"]["pre"]["r"]["mean"] > 0
    assert raw["variants"]["fade"]["r"]["hit_rate"] == 1.0
    assert raw["forecast"]["surprise_hit_rate"] == 1.0
    assert raw["forecast"]["reaction"]["mean"] > 0
    assert {r["event"] for r in raw["by_event"]} == {CPI}
    assert set(result["engine"]) == {"pre", "post", "fade"}
    assert result["engine"]["pre"]["metrics"]["n_trades"] == 4
    assert "ftmo" in result["engine"]["pre"]
    md = render_markdown(result)
    assert "## Raw returns" in md and "| pre |" in md and "## Engine runs" in md
    paths = write_event_study_report(result, tmp_path, tag="t")
    assert paths["json"].is_file() and paths["md"].is_file() and paths["table"].is_file()
    assert paths["trades_pre"].is_file()

    # Empty calendar degrades gracefully
    empty = run_event_study(bars, cfg, COSTS, None, calendar_df=pd.DataFrame(columns=cal.columns))
    assert empty["n_table_rows"] == 0
    assert summarize_event_table(empty["table"])["n_rows"] == 0
    assert "pre" in render_markdown(empty)


# ---------------------------------------------------------------- nowcast


def test_gasoline_nowcast_is_point_in_time():
    weeks = pd.date_range("2023-10-02", "2024-01-29", freq="7D", tz="UTC")
    vals = []
    for w in weeks:
        vals.append(3.0 if w.month in (10, 11) else (3.3 if w.month == 12 else 9.9))
    gas = pd.Series(vals, index=weeks)
    release = pd.Timestamp("2024-01-11 13:30", tz="UTC")
    # Reference month Dec 2023: avg 3.3; Nov: 3.0 -> +10%
    assert month_average_before(gas, pd.Timestamp("2023-12-01"), release) == pytest.approx(3.3)
    assert month_average_before(gas, pd.Timestamp("2023-11-01"), release) == pytest.approx(3.0)
    # January observations after the release are invisible
    assert math.isnan(month_average_before(gas, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01", tz="UTC")))

    ev = pd.DataFrame(
        {
            "timestamp": [release],
            "period": [pd.Timestamp("2023-12-22")],
            "actual": [0.3],
            "forecast": [0.2],
        }
    )
    mom = gasoline_mom_for_events(ev, gas)
    assert mom.iloc[0] == pytest.approx(10.0)

    # Nowcast = trailing mean of prior prints + weight * (gas_mom - trailing gas_mom)
    n = 8
    ts = [pd.Timestamp("2023-06-12 13:30", tz="UTC") + pd.DateOffset(months=i) for i in range(n)]
    ev2 = pd.DataFrame(
        {
            "timestamp": ts,
            "period": [t.normalize().tz_localize(None) - pd.DateOffset(months=1) for t in ts],
            "actual": [0.2] * n,
            "forecast": [0.2] * n,
        }
    )
    flat_gas = pd.Series(3.0, index=pd.date_range("2023-01-02", "2024-03-01", freq="7D", tz="UTC"))
    now = gasoline_cpi_nowcast(ev2, flat_gas, weight=0.035, baseline_n=6)
    assert math.isnan(now.iloc[0])
    assert now.iloc[-1] == pytest.approx(0.2)  # flat gas, flat prints


def test_event_drift_config_loads():
    from src.config import load_strategy_config

    cfg = load_strategy_config("strategy_event_drift.yaml")
    assert cfg["strategy"] == "event_drift"
    assert cfg["signal"]["variant"] in ("pre", "post", "fade")
    assert cfg["signal"]["exit_mode"] == "swing"
    assert cfg["events"]["list"]
    assert all({"currency", "name"} <= set(e) for e in cfg["events"]["list"])
    assert set(cfg["events"]["currency_pairs"]) >= {"USD", "EUR", "GBP"}
    for pairs in cfg["events"]["currency_pairs"].values():
        assert all(p in cfg["universe"] for p in pairs)


def test_load_fred_series_from_csv(tmp_path):
    from src.features.nowcast import load_fred_series

    (tmp_path / "GASREGW.csv").write_text(
        "observation_date,GASREGW\n2024-01-01,3.1\n2024-01-08,.\n2024-01-15,3.3\n", encoding="utf-8"
    )
    s = load_fred_series("GASREGW", cache_dir=tmp_path)
    assert len(s) == 2
    assert str(s.index.tz) == "UTC"
    assert s.iloc[-1] == pytest.approx(3.3)
    assert load_fred_series("NOPE", cache_dir=tmp_path).empty


def test_forecast_quality_calendar_only():
    from src.research.event_study import forecast_quality, render_forecast_markdown

    cal = _synthetic_calendar()
    events = prepare_events(cal, _cfg())
    fq = forecast_quality(events)
    assert fq["n"] > 0
    assert fq["surprise_hit_rate"] == 1.0  # every surprise in the fixture is hot
    assert fq["by_event"][0]["event"] == CPI
    md = render_forecast_markdown(fq)
    assert "hit rate" in md and CPI in md
    assert forecast_quality(events.iloc[0:0])["n"] == 0


def test_filter_events_by_country_and_preliminary():
    cal = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-16", "2024-01-17"], utc=True),
            "currency": ["EUR"] * 4,
            "country": ["DE", "EMU", "DE", "EMU"],
            "event": ["HICP"] * 4,
            "impact": ["high"] * 4,
            "preliminary": [True, True, False, False],
            "actual": [1.0, 1.1, 1.0, 1.1],
            "forecast": [0.9, 1.0, 1.0, 1.1],
        }
    )
    flash_emu = filter_events(cal, [{"currency": "EUR", "name": "HICP", "country": "EMU", "preliminary": True}])
    assert len(flash_emu) == 1 and flash_emu.iloc[0]["country"] == "EMU"
    any_emu = filter_events(cal, [{"currency": "EUR", "name": "HICP", "country": "EMU"}])
    assert len(any_emu) == 2
    assert len(filter_events(cal, [{"currency": "EUR", "name": "HICP"}])) == 4
    # Series key separates flash and final: each has its own surprise history
    x = add_standardized_surprises(cal, min_history=1, window=10)
    assert x["hist_std"].isna().all()  # one observation per series -> no history yet
    pol = polarity_for_rows(cal, [{"currency": "EUR", "name": "HICP", "country": "DE", "polarity": -1}])
    assert pol[cal["country"].eq("DE")].eq(-1).all() and pol[cal["country"].eq("EMU")].eq(1).all()


def test_normalize_calendar_preliminary_flag_and_series_dedupe():
    rows = [
        {"dateUtc": "2024-01-05T10:00:00Z", "currencyCode": "EUR", "countryCode": "DE", "name": "HICP", "volatility": "HIGH", "actual": 2.9, "consensus": 3.0, "previous": 2.4, "isPreliminary": True},
        {"dateUtc": "2024-01-05T10:00:00Z", "currencyCode": "EUR", "countryCode": "EMU", "name": "HICP", "volatility": "HIGH", "actual": 2.9, "consensus": 2.9, "previous": 2.4, "isPreliminary": False},
    ]
    cal = normalize_calendar(pd.DataFrame(rows))
    assert len(cal) == 2  # same timestamp/currency/name but different country are distinct series
    assert cal["preliminary"].tolist() == [True, False]
    assert "preliminary" in cal.columns
    legacy = normalize_calendar(pd.DataFrame([{"timestamp": "2024-01-01T00:00:00Z", "currency": "USD", "event": "x"}]))
    assert legacy["preliminary"].tolist() == [False]


def test_external_nowcast_attach_and_override(tmp_path):
    from src.features.nowcast import attach_external_nowcast, load_external_nowcast

    cal = _synthetic_calendar()
    cal["nowcast"] = 0.0  # model value to be overridden on one row
    p = tmp_path / "ext.csv"
    p.write_text(
        "timestamp,currency,event,nowcast\n"
        f"{EVENT_TIMES[0].replace(' ', 'T')}Z,USD,{CPI},0.45\n"
        "2099-01-01T00:00:00Z,USD,Nope,1.0\n",
        encoding="utf-8",
    )
    ext = load_external_nowcast(p)
    assert len(ext) == 2
    out = attach_external_nowcast(cal, ext)
    hit = out[out["timestamp"] == pd.Timestamp(EVENT_TIMES[0], tz="UTC")]
    assert hit["nowcast"].tolist() == [0.45]  # CPI row overridden
    assert (out.loc[out["timestamp"] != pd.Timestamp(EVENT_TIMES[0], tz="UTC"), "nowcast"] == 0.0).all()
    assert out.loc[out["event"] == "ISM Manufacturing PMI", "nowcast"].eq(0.0).all()  # other series untouched
    assert len(out) == len(cal)  # merge never duplicates rows
    assert load_external_nowcast(tmp_path / "missing.csv").empty
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,value\n2024-01-01,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_external_nowcast(bad)

    # Wired through prepare_events via nowcast.external_csv: only the supplied
    # release gets a nowcast, so it is the only row with a nowcast_gap signal
    cfg = _cfg(signal_mode="nowcast_gap", min_abs_z=0.0)
    cfg["nowcast"]["external_csv"] = str(p)
    ev = prepare_events(_synthetic_calendar(), cfg)
    assert len(ev) == 1 and ev.iloc[0]["nowcast"] == pytest.approx(0.45)
    assert ev.iloc[0]["signal_z"] > 0
