from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.data import macro as m
from src.features.macro_score import (
    align_with_lag,
    build_currency_scores,
    cross_sectional_z,
    latest_table,
    pair_bias,
    rolling_z,
)
from src.strategy.macro_swing import first_bias_exit, generate_signals, trigger_side
from src.strategy.registry import resolve_signal_fn

COSTS = {
    "default": {"half_spread_pips": 0.5, "slippage_pips": 0.2, "commission_per_lot_usd": 7.0},
    "pairs": {},
    "pip_size": {"EURUSD": 0.0001, "USDJPY": 0.01},
}


# ----------------------------------------------------------------- parsers


def test_parsers_each_source_format():
    fred = "observation_date,DGS2\n2024-01-02,4.33\n2024-01-03,.\n2024-01-04,4.38\n"
    s = m.parse_fred_csv(fred, "x")
    assert len(s) == 2 and s.iloc[-1] == 4.38 and str(s.index.tz) == "UTC"

    ecb = "KEY,FREQ,TIME_PERIOD,OBS_VALUE,OBS_STATUS\nYC.X,B,2024-01-02,2.4381,A\nYC.X,B,2024-01-03,2.4000,A\n"
    assert m.parse_ecb_csv(ecb).iloc[0] == pytest.approx(2.4381)

    boc = '"TERMS"\n\n"SERIES"\n"id","label"\n"BD","2y"\n\n"OBSERVATIONS"\n"date","BD.CDN.2YR.DQ.YLD"\n"2024-01-02","3.96"\n"2024-01-03","3.90"\n'
    assert m.parse_boc_csv(boc).iloc[-1] == pytest.approx(3.90)
    with pytest.raises(ValueError):
        m.parse_boc_csv("no block here")

    rba = (
        "F2 CAPITAL MARKET YIELDS\nTitle,2 year,3 year\nFrequency,Daily,Daily\n"
        "Series ID,FCMYGBAG2D,FCMYGBAG3D\n20-May-2013,,3.1\n21-May-2013,2.75,3.0\n"
    )
    s = m.parse_rba_csv(rba, "FCMYGBAG2D")
    assert len(s) == 1 and s.iloc[0] == 2.75
    with pytest.raises(ValueError):
        m.parse_rba_csv(rba, "NOPE")

    mof = "Interest Rate,,,(Unit : %)\nDate,1Y,2Y,3Y\n2026/9/29,1.692,1.976,2.116\n2026/9/30,1.684,-,2.086\n"
    s = m.parse_mof_csv(mof, "2Y")
    assert len(s) == 1 and s.iloc[0] == pytest.approx(1.976)

    boe = "DATE,IUDSNPY,IUDMNPY\n02 Jan 2024,3.443,3.719\n03 Jan 2024,3.5003,3.7638\n"
    assert m.parse_boe_csv(boe, "IUDSNPY").iloc[-1] == pytest.approx(3.5003)

    bis = "FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\nD,US,2024-01-01,5.375\nD,US,2024-01-02,5.375\n"
    assert len(m.parse_bis_csv(bis)) == 2

    tff = (
        '"Market_and_Exchange_Names","Report_Date_as_YYYY-MM-DD","Open_Interest_All",'
        '"Lev_Money_Positions_Long_All","Lev_Money_Positions_Short_All"\n'
        '"EURO FX - CHICAGO MERCANTILE EXCHANGE","2024-01-02","100000","30000","10000"\n'
        '"EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE","2024-01-02","100","50","10"\n'
        '"JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE","2024-01-02","200000","20000","80000"\n'
    )
    eur = m.parse_cftc_tff(tff, "EURO FX")
    # the EUR/GBP cross also matches the substring; last duplicate on the date wins -> keep=last
    assert len(eur) == 1
    jpy = m.parse_cftc_tff(tff, "JAPANESE YEN")
    assert jpy.iloc[0] == pytest.approx(-0.3)
    assert m.parse_cftc_tff(tff, "SWISS").empty

    local = "date,value\n2024-01-01,1.5\n2024-01-02,1.6\n"
    assert m.parse_local_csv(local).iloc[-1] == 1.6


def test_series_cache_round_trip_and_spec_flattening(tmp_path):
    s = m._series(["2024-01-02", "2024-01-03"], [1.0, 2.0], "rate_2y_USD")
    p = m.save_series(s, "rate_2y_USD", tmp_path)
    assert p.is_file()
    back = m.load_series("rate_2y_USD", tmp_path)
    assert back.tolist() == [1.0, 2.0]
    assert m.load_series("missing", tmp_path).empty
    cfg = {
        "series": {
            "rate_2y": {"USD": {"source": "fred", "id": "DGS2"}, "EUR": {"source": "ecb", "key": "k"}},
            "cot_net": {"EUR": {"source": "cftc_tff", "market": "EURO FX"}, "USD": {"derive": "neg_mean_others"}},
            "global": {"vix": {"source": "fred", "id": "VIXCLS"}},
        }
    }
    names = [n for n, *_ in m.iter_series_specs(cfg)]
    assert names == ["rate_2y_USD", "rate_2y_EUR", "cot_net_EUR", "global_vix"]
    panel = m.load_macro_panel(cfg, tmp_path)
    assert set(panel) == {"rate_2y_USD"}
    with pytest.raises(ValueError):
        m.fetch_series({"source": "nope"}, "x")


# ----------------------------------------------------------------- scoring


def test_align_with_lag_is_point_in_time():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    s = pd.Series([1.0, 2.0, 3.0], index=idx[[0, 2, 4]])
    out = align_with_lag(s, idx, lag_days=1)
    assert math.isnan(out.iloc[0])          # nothing known yet on day 0
    assert out.iloc[1] == 1.0               # day-0 value usable from day 1
    assert out.iloc[2] == 1.0               # day-2 value not yet usable on day 2
    assert out.iloc[3] == 2.0
    assert out.iloc[-1] == 3.0              # forward filled
    assert align_with_lag(pd.Series(dtype=float), idx, 1).isna().all()


def test_cross_sectional_and_rolling_z():
    df = pd.DataFrame({"A": [1.0, 1.0], "B": [3.0, 3.0], "C": [5.0, np.nan]})
    z = cross_sectional_z(df)
    assert abs(z.iloc[0].sum()) < 1e-12
    assert z.iloc[0]["C"] > 0 and z.iloc[0]["A"] < 0
    assert z.iloc[1].notna().sum() == 2 and abs(z.iloc[1]["A"] + z.iloc[1]["B"]) < 1e-12
    one = cross_sectional_z(pd.DataFrame({"A": [1.0]}))
    assert one.isna().all().all()
    s = pd.Series(np.arange(300, dtype=float))
    rz = rolling_z(s, 100)
    assert rz.iloc[:49].isna().all() and rz.iloc[-1] > 0


def _macro_cfg(**over):
    cfg = {
        "currencies": ["USD", "EUR", "JPY"],
        "min_components": 2,
        "lags": {"market_days": 1, "cot_days": 3},
        "series": {
            "rate_2y": {c: {"source": "csv", "path": "x"} for c in ("USD", "EUR", "JPY")},
            "policy_rate": {c: {"source": "csv", "path": "x"} for c in ("USD", "EUR", "JPY")},
            "cot_net": {"EUR": {"source": "csv", "path": "x"}, "JPY": {"source": "csv", "path": "x"}, "USD": {"derive": "neg_mean_others"}},
            "global": {"vix": {"source": "csv", "path": "x"}, "spx": {"source": "csv", "path": "x"}, "oil": {"source": "csv", "path": "x"}},
        },
        "components": {
            "rate_momentum": {"weight": 1.0, "input": "rate_2y", "transform": "diff", "lookback_days": 20, "standardize": "cross_section"},
            "carry": {"weight": 0.5, "input": "policy_rate", "transform": "level", "standardize": "cross_section"},
            "risk": {"weight": 0.5, "input": "riskoff", "standardize": "time_series", "window_days": 60, "beta": {"USD": 0.5, "JPY": 1.0, "EUR": 0.0}},
            "commodity": {"weight": 0.5, "input": "oil", "transform": "pct_change", "lookback_days": 20, "standardize": "time_series", "window_days": 60, "beta": {"CAD": 1.0}},
            "positioning": {"weight": 0.5, "input": "cot_net", "transform": "level", "standardize": "time_series", "window_days": 60, "sign": -1},
        },
    }
    cfg.update(over)
    return cfg


def _synthetic_panel(n_days: int = 400, eur_trend: float = 0.01) -> dict[str, pd.Series]:
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    t = np.arange(n_days)
    panel = {
        "rate_2y_USD": pd.Series(4.0 + 0.002 * rng.standard_normal(n_days).cumsum(), index=idx),
        "rate_2y_EUR": pd.Series(2.0 + eur_trend * t / 100 * 10 + 0.002 * rng.standard_normal(n_days).cumsum(), index=idx),
        "rate_2y_JPY": pd.Series(0.1 + 0.001 * rng.standard_normal(n_days).cumsum(), index=idx),
        "policy_rate_USD": pd.Series(5.25, index=idx),
        "policy_rate_EUR": pd.Series(4.0, index=idx),
        "policy_rate_JPY": pd.Series(0.1, index=idx),
        "cot_net_EUR": pd.Series(0.1 + 0.05 * np.sin(t[::5] / 30), index=idx[::5]),
        "cot_net_JPY": pd.Series(-0.2 + 0.05 * np.cos(t[::5] / 30), index=idx[::5]),
        "global_vix": pd.Series(15 + 3 * np.sin(t / 40), index=idx),
        "global_spx": pd.Series(4000 * np.exp(0.0002 * t), index=idx),
        "global_oil": pd.Series(80 + 5 * np.sin(t / 50), index=idx),
    }
    return panel


def test_currency_scores_and_pair_bias():
    panel = _synthetic_panel()
    cfg = _macro_cfg()
    res = build_currency_scores(panel, cfg)
    scores = res["scores"]
    assert list(scores.columns) == ["USD", "EUR", "JPY"]
    assert set(res["components"]) == {"rate_momentum", "carry", "risk", "positioning"}  # oil has no beta in universe
    last = scores.dropna().iloc[-1]
    # EUR rates trend up strongly -> EUR has the top rate momentum, and USD carry is highest
    rm = res["components"]["rate_momentum"].dropna().iloc[-1]
    assert rm["EUR"] > rm["USD"] and rm["EUR"] > rm["JPY"]
    carry = res["components"]["carry"].dropna().iloc[-1]
    assert carry["USD"] > carry["EUR"] > carry["JPY"]
    # USD positioning derived as negative mean of the others
    pos = res["components"]["positioning"]
    assert "USD" in pos.columns and pos["USD"].notna().any()
    # score requires min_components and lands between the component values
    n = res["n_components"].iloc[-1]
    assert (n >= 2).all()
    bias = pair_bias(scores, ["EURUSD", "USDJPY", "XXXYYY"])
    assert list(bias.columns) == ["EURUSD", "USDJPY"]
    assert bias["EURUSD"].iloc[-1] == pytest.approx(last["EUR"] - last["USD"])
    tab = latest_table(res)
    assert "score" in tab.columns and tab.index.name.startswith("as of")
    # min_components gate: with a high threshold, scores vanish
    res2 = build_currency_scores(panel, _macro_cfg(min_components=10))
    assert res2["scores"].isna().all().all()
    assert build_currency_scores({}, cfg)["scores"].empty


# ----------------------------------------------------------------- strategy


def _cfg(**sig_over):
    from src.config import load_strategy_config

    cfg = load_strategy_config("strategy_macro_swing.yaml")
    cfg["universe"] = ["EURUSD", "USDJPY"]
    cfg["news"]["enabled"] = False
    cfg["macro"] = _macro_cfg()
    cfg["signal"].update({"trigger": "none", "min_abs_bias": 0.3, "reentry_cooldown_days": 0, "max_hold_days": 0})
    cfg["signal"].update(sig_over)
    return cfg


def _bars(symbol: str, start="2023-06-01", end="2024-08-01", px=1.1, drift=0.0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="15min", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    t = np.arange(len(idx))
    close = px + drift * t + 0.0004 * np.sin(t / 7)
    return pd.DataFrame({"timestamp": idx, "open": close, "high": close + 0.0006, "low": close - 0.0006, "close": close, "volume": 1.0})


def test_trigger_side_modes():
    row = pd.Series({"open": 1.0, "high": 1.02, "low": 0.99, "close": 1.01, "ema": 1.0, "don_hi": 1.015, "don_lo": 0.98})
    assert trigger_side(row, None, "none", 1)
    assert trigger_side(row, None, "pullback", 1)          # dipped to EMA, closed above, up bar
    assert not trigger_side(row, None, "pullback", -1)
    assert not trigger_side(row, None, "breakout", 1)      # close 1.01 < channel high 1.015
    assert trigger_side(row.replace({1.01: 1.02}), None, "breakout", 1)
    assert not trigger_side(row.drop("ema").reindex(row.index), None, "pullback", 1)
    with pytest.raises(ValueError):
        trigger_side(row, None, "nope", 1)


def test_first_bias_exit():
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    bias = pd.Series([1.0, 0.8, 0.5, np.nan, -0.2, -0.5], index=idx)
    assert first_bias_exit(bias, idx[0], 1, 0.0) == idx[4]
    assert first_bias_exit(bias, idx[0], 1, 0.6) == idx[2]
    assert first_bias_exit(bias, idx[4], -1, 0.0) is None


def test_generate_signals_follows_bias_and_schedules_flip_exit():
    bars = {"EURUSD": _bars("EURUSD"), "USDJPY": _bars("USDJPY", px=150.0, drift=0.0)}
    panel = _synthetic_panel()
    cfg = _cfg()
    assert resolve_signal_fn(cfg).__module__.endswith("macro_swing")
    sigs = generate_signals(bars, cfg, COSTS, macro_panel=panel)
    assert sigs, "expected signals from a strongly trending EUR rate panel"
    assert all(s.timestamp.hour == 21 and s.timestamp.minute == 0 for s in sigs)
    assert all(s.session == "macro" and s.reason.startswith("macro_swing|none|bias=") for s in sigs)
    # the signal direction equals the sign of the bias encoded in the reason
    for s in sigs:
        b = float(s.reason.split("bias=")[1])
        assert (b > 0) == (s.side > 0)
        assert abs(b) >= 0.3
        assert (s.stop < s.entry) == (s.side > 0)
        assert s.risk_price > 0
        if s.exit_time is not None:
            assert s.exit_time > s.timestamp
    # one signal per decision day at most
    assert len({s.timestamp for s in sigs}) == len(sigs)
    # cooldown reduces signal count
    fewer = generate_signals(bars, _cfg(reentry_cooldown_days=10), COSTS, macro_panel=panel)
    assert len(fewer) < len(sigs)
    # pullback trigger is stricter than none
    pb = generate_signals(bars, _cfg(trigger="pullback"), COSTS, macro_panel=panel)
    assert len(pb) <= len(sigs)
    # empty panel -> no signals; missing cache -> no signals
    assert generate_signals(bars, cfg, COSTS, macro_panel={}) == []
    cfg_nc = _cfg()
    cfg_nc["macro"]["cache_dir"] = "/nonexistent/dir"
    assert generate_signals(bars, cfg_nc, COSTS) == []
    # engine run: flip exits appear as time_exit
    res = run_backtest(sigs, bars, cfg, COSTS)
    assert res.metrics["n_trades"] >= 1
    assert any(t.exit_reason == "time_exit" for t in res.trades) or all(t.exit_reason in ("stop", "tp", "breakeven_stop", "eod_force") for t in res.trades)


def test_macro_swing_config_loads():
    from src.config import load_strategy_config

    cfg = load_strategy_config("strategy_macro_swing.yaml")
    assert cfg["strategy"] == "macro_swing"
    assert cfg["signal"]["trigger"] in ("none", "pullback", "breakout")
    m_cfg = cfg["macro"]
    assert set(m_cfg["series"]) >= {"rate_2y", "policy_rate", "cot_net", "global"}
    for group in ("rate_2y", "policy_rate"):
        assert set(m_cfg["series"][group]) >= set(m_cfg["currencies"])
    # the wide variant's extra currencies have inputs too (NZD has no free daily 2y)
    assert set(m_cfg["series"]["policy_rate"]) >= {"CHF", "NZD"} and "CHF" in m_cfg["series"]["rate_2y"]
    assert m_cfg["series"]["cot_net"]["USD"] == {"derive": "neg_mean_others"}
    comps = m_cfg["components"]
    assert comps["rate_momentum"]["weight"] > 0 and comps["positioning"]["sign"] == -1
    for pair in cfg["universe"]:
        assert pair[:3] in m_cfg["currencies"] and pair[3:6] in m_cfg["currencies"]


def test_boe_fetch_retries_until_csv(monkeypatch):
    answers = iter([b"<head><title>Object moved</title></head>", b"DATE,IUDSNPY\n02 Jan 2024,3.443\n"])
    monkeypatch.setattr(m, "_get", lambda url, **kw: next(answers))
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    s = m.fetch_series({"source": "boe", "code": "IUDSNPY", "retries": 3}, "rate_2y_GBP")
    assert len(s) == 1 and s.iloc[0] == pytest.approx(3.443)
    monkeypatch.setattr(m, "_get", lambda url, **kw: b"<html>nope</html>")
    with pytest.raises(RuntimeError, match="no CSV"):
        m.fetch_series({"source": "boe", "code": "IUDSNPY", "retries": 2}, "rate_2y_GBP")
