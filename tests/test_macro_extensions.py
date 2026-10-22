from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import is_cross
from src.backtest.engine import run_backtest, usd_per_unit
from src.data import macro as m
from src.data.bars import cross_bars, resample_ohlc
from src.features.macro_score import build_currency_scores
from src.research import macro_ic as mic
from src.research.macro_compare import variant_config
from src.strategy.cssm import Signal

FREE = {"default": {"half_spread_pips": 0.0, "slippage_pips": 0.0, "commission_per_lot_usd": 0.0}, "pairs": {}}


# ----------------------------------------------------------------- sources


def test_snb_parser_filters_dimensions():
    snb = (
        '﻿"CubeId";"rendeiduebd"\n"PublishingDate";"2026-10-01 14:30"\n\n'
        '"Date";"D0";"D1";"Value"\n'
        '"2026-09-29";"CHF";"2J";"0.345"\n"2026-09-29";"CHF";"5J";"0.6"\n'
        '"2026-09-29";"EUR";"2J";"2.1"\n"2026-09-30";"CHF";"2J";"0.287"\n"2026-10-01";"CHF";"2J";""\n'
    )
    s = m.parse_snb_csv(snb, {"D0": "CHF", "D1": "2J"}, "rate_2y_CHF")
    assert s.tolist() == [0.345, 0.287] and str(s.index.tz) == "UTC"
    with pytest.raises(ValueError):
        m.parse_snb_csv(snb, {"D9": "x"})
    with pytest.raises(ValueError):
        m.parse_snb_csv("no header", {})


def test_cftc_parser_reads_old_date_column_and_renamed_markets():
    old = (
        '"Market_and_Exchange_Names","Report_Date_as_MM_DD_YYYY","Open_Interest_All",'
        '"Lev_Money_Positions_Long_All","Lev_Money_Positions_Short_All"\n'
        '"NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE ","2012-01-03","1000","300","100"\n'
    )
    new = (
        '"Market_and_Exchange_Names","Report_Date_as_YYYY-MM-DD","Open_Interest_All",'
        '"Lev_Money_Positions_Long_All","Lev_Money_Positions_Short_All"\n'
        '"NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE","2024-01-02","1000","100","300"\n'
    )
    names = ["NEW ZEALAND DOLLAR", "NZ DOLLAR"]
    assert m.parse_cftc_tff(old, names).iloc[0] == pytest.approx(0.2)
    assert m.parse_cftc_tff(new, names).iloc[0] == pytest.approx(-0.2)
    assert m.parse_cftc_tff(new, "NEW ZEALAND DOLLAR").empty


# ----------------------------------------------------------------- scoring and variants


def test_subtract_component_is_spread_of_two_inputs():
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    panel = {
        "rate_2y_USD": pd.Series(4.5, index=idx), "policy_rate_USD": pd.Series(5.0, index=idx),   # cuts priced: -0.5
        "rate_2y_EUR": pd.Series(3.0, index=idx), "policy_rate_EUR": pd.Series(2.0, index=idx),   # hikes priced: +1.0
        "rate_2y_JPY": pd.Series(0.3, index=idx), "policy_rate_JPY": pd.Series(0.1, index=idx),   # +0.2
        "policy_rate_NZD": pd.Series(5.5, index=idx),                                              # no 2y: excluded
    }
    cfg = {
        "min_components": 1,
        "lags": {"market_days": 0},
        "series": {"rate_2y": {c: {} for c in ("USD", "EUR", "JPY")}, "policy_rate": {c: {} for c in ("USD", "EUR", "JPY", "NZD")}},
        "components": {"rate_expectations": {"weight": 1.0, "input": "rate_2y", "subtract": "policy_rate", "transform": "level", "standardize": "cross_section"}},
    }
    res = build_currency_scores(panel, cfg, currencies=["USD", "EUR", "JPY", "NZD"])
    last = res["components"]["rate_expectations"].iloc[-1]
    assert last["EUR"] > last["JPY"] > last["USD"] and np.isnan(last["NZD"])
    assert abs(last[["USD", "EUR", "JPY"]].sum()) < 1e-9


def test_variant_extends_chain_and_cycle():
    cfg = {
        "universe": ["EURUSD"],
        "risk": {"max_open_positions": 1},
        "macro": {"components": {"carry": {"weight": 0.5}}},
        "validation": {"baseline": "a"},
        "variants": {
            "a": {"risk": {"max_open_positions": 3}},
            "b": {"extends": "a", "universe": ["EURUSD", "EURJPY"], "macro": {"components": {"new": {"weight": 1.0}}}},
            "x": {"extends": "y"},
            "y": {"extends": "x"},
        },
    }
    b = variant_config(cfg, "b")
    assert b["risk"]["max_open_positions"] == 3 and b["universe"] == ["EURUSD", "EURJPY"]
    assert set(b["macro"]["components"]) == {"carry", "new"} and "validation" not in b and "extends" not in b
    with pytest.raises(ValueError, match="cycle"):
        variant_config(cfg, "x")


def test_validate_variants_applies_the_preregistered_rule(monkeypatch):
    ics = {  # (variant, window) -> IC
        ("base", "dev"): 0.08, ("base", "holdout"): -0.05,
        ("better", "dev"): 0.10, ("better", "holdout"): 0.02,
        ("dev_only", "dev"): 0.12, ("dev_only", "holdout"): -0.08,
        ("less_bad", "dev"): 0.09, ("less_bad", "holdout"): -0.01,   # beats base on both but holdout still negative
    }

    def fake(panel, cfg, fx, start, end, pairs=None, h=20):
        window = "holdout" if start == "2012-01-01" else "dev"
        return {"ic": ics[(cfg["name"], window)], "t": 1.0, "by_year": [], "verdict": "WEAK", "pairs": []}

    monkeypatch.setattr(mic, "signal_ic", fake)
    configs = {n: {"name": n, "universe": ["EURUSD"]} for n in ("base", "better", "dev_only", "less_bad")}
    res = mic.validate_variants({}, configs, {}, "base", ("2018-01-01", None), ("2012-01-01", "2017-12-31"))
    kept = {r["variant"]: r["kept"] for r in res["rows"]}
    assert kept == {"base": None, "better": True, "dev_only": False, "less_bad": False}
    md = mic.render_validation_markdown(res)
    assert "| better | 1 |" in md and "**yes**" in md


# ----------------------------------------------------------------- crosses


def _m1(px: list[float], start: str = "2024-01-02 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(px), freq="1min", tz="UTC")
    p = np.asarray(px, dtype=float)
    return pd.DataFrame({"timestamp": idx, "open": np.r_[p[0], p[:-1]], "high": p, "low": p, "close": p, "volume": 1.0})


def test_cross_bars_from_usd_legs():
    eur = _m1([1.10, 1.11, 1.12, 1.09])
    jpy = _m1([150.0, 150.0, 151.0, 150.0])
    gbp = _m1([1.25, 1.25, 1.26, 1.24])
    ej = cross_bars("EURJPY", {"EURUSD": eur, "USDJPY": jpy})
    assert ej["close"].tolist() == pytest.approx([165.0, 166.5, 169.12, 163.5])
    assert (ej["high"] >= ej[["open", "close"]].max(axis=1)).all()
    eg = cross_bars("EURGBP", {"EURUSD": eur, "GBPUSD": gbp})
    assert eg["close"].iloc[2] == pytest.approx(1.12 / 1.26)
    m15 = resample_ohlc(ej.reset_index(), "15min")
    assert len(m15) == 1 and m15["high"].iloc[0] == pytest.approx(169.12) and m15["low"].iloc[0] == pytest.approx(163.5)
    with pytest.raises(ValueError):
        cross_bars("EURCHF", {"EURUSD": eur})
    assert is_cross("EURJPY") and is_cross("eurgbp") and not is_cross("USDJPY") and not is_cross("XAUUSD")


def test_cross_trade_is_sized_and_booked_in_usd():
    idx = pd.date_range("2024-01-02", periods=4, freq="15min", tz="UTC")
    ej = pd.DataFrame({"timestamp": idx, "open": 160.0, "high": [160.0, 160.5, 160.2, 158.6], "low": [160.0, 159.8, 159.0, 158.0], "close": [160.0, 160.2, 159.5, 158.2], "volume": 1.0})
    uj = pd.DataFrame({"timestamp": idx, "open": 150.0, "high": 150.0, "low": 150.0, "close": 150.0, "volume": 1.0})
    cfg = {
        "risk": {"risk_per_trade_pct": 0.5, "max_open_positions": 1, "soft_daily_halt_pct": 50.0},
        "sessions": {"london_end": "11:00", "ny_end": "16:00"},
        "signal": {"exit_mode": "swing", "take_profit_r": 3.0, "breakeven_at_r": 0.0, "allow_weekend_hold": True},
    }
    costs = {**FREE, "pip_size": {"EURJPY": 0.01}}
    sig = Signal(idx[0], "EURJPY", 1, 160.0, 158.4, 164.8, 1.6, "macro", 161.6, 158.4, "t")
    t = run_backtest([sig], {"EURJPY": ej, "USDJPY": uj}, cfg, costs).trades[0]
    assert t.exit_reason == "stop"
    assert t.r_multiple == pytest.approx(-1.0, abs=0.01)    # a full stop costs 1R, not EURJPY/USDJPY of it
    assert usd_per_unit("JPY", idx[1], {"USDJPY": uj.set_index("timestamp")["close"]}) == pytest.approx(1 / 150.0)
    with pytest.raises(ValueError, match="convert"):
        run_backtest([sig], {"EURJPY": ej}, cfg, costs)
