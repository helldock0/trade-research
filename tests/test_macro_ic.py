from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.research import macro_ic as mic
from src.research.macro_ic import (
    component_pair_signals,
    fetch_fx,
    forward_log_return,
    ic_frame,
    ic_stats,
    load_fx,
    newey_west_t,
    nonoverlap_t,
    phase_a_verdict,
    run_macro_ic,
    spearman,
    threshold_split,
    write_macro_ic_report,
)


def _walk(n: int, seed: int, start: str = "2019-01-01", px0: float = 1.1) -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(px0 * np.exp(np.cumsum(0.005 * rng.standard_normal(n))), index=idx)


def _calendar_daily(s: pd.Series) -> pd.Series:
    """Spread a business-day series onto a calendar-daily index (as the bias is)."""
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D", tz="UTC")
    return s.reindex(idx).ffill()


# ----------------------------------------------------------------- forward returns


def test_forward_return_is_positional_and_forward_only():
    px = _walk(60, 1)
    px = px.drop(px.index[[10, 11, 30]])  # holidays: positions, not calendar days, define h
    h = 5
    fwd = forward_log_return(px, h)
    for i in (0, 9, 10, 25, 40):
        assert fwd.iloc[i] == pytest.approx(math.log(px.iloc[i + h] / px.iloc[i]))
    assert fwd.iloc[-h:].isna().all() and fwd.iloc[:-h].notna().all()
    # changing any price before t leaves the return from t untouched
    bumped = px.copy()
    bumped.iloc[:20] *= 1.5
    assert forward_log_return(bumped, h).iloc[20:].equals(fwd.iloc[20:])


def test_ic_frame_pairs_signal_known_on_t_with_return_from_t():
    px = _walk(80, 2)
    h = 10
    # the signal value dated d is the realized return from the fix on d to the fix h days later
    sig = pd.DataFrame({"EURUSD": _calendar_daily(forward_log_return(px, h))})
    fr = ic_frame(sig, {"EURUSD": px}, h)
    assert len(fr) == len(px) - h
    assert np.allclose(fr["signal"], fr["fwd"])
    # a signal built from the past h-day return lines up with the past, not the future
    past = pd.DataFrame({"EURUSD": _calendar_daily(np.log(px / px.shift(h)))})
    fr_past = ic_frame(past, {"EURUSD": px}, h)
    row = fr_past.iloc[20]
    i = px.index.get_loc(row["date"])
    assert row["signal"] == pytest.approx(math.log(px.iloc[i] / px.iloc[i - h]))
    assert row["fwd"] == pytest.approx(math.log(px.iloc[i + h] / px.iloc[i]))
    # extra_lag uses the previous FX day's signal
    lag = ic_frame(sig, {"EURUSD": px}, h, extra_lag=1)
    assert lag.iloc[0]["signal"] == pytest.approx(fr.iloc[0]["signal"])
    assert lag.iloc[0]["date"] == fr.iloc[1]["date"]
    # start / end filter the decision dates
    cut = ic_frame(sig, {"EURUSD": px}, h, start=px.index[30], end=px.index[50])
    assert cut["date"].min() == px.index[30] and cut["date"].max() == px.index[50]
    assert ic_frame(sig, {"GBPUSD": px}, h).empty


# ----------------------------------------------------------------- IC and t


def _three_pairs(n: int = 1500) -> dict[str, pd.Series]:
    return {"EURUSD": _walk(n, 10), "USDJPY": _walk(n, 11, px0=150.0), "AUDUSD": _walk(n, 12, px0=0.7)}


def test_ic_is_one_for_known_future_and_zero_for_noise():
    fx = _three_pairs()
    h = 20
    rng = np.random.default_rng(3)
    future = pd.DataFrame({p: _calendar_daily(forward_log_return(px, h)) for p, px in fx.items()})
    st = ic_stats(ic_frame(future, fx, h), h)
    assert st["ic"] == pytest.approx(1.0)
    assert st["t"] > 10 and st["n"] == 3 * (1500 - h) and st["n_dates"] == 1500 - h
    noisy = future + 0.01 * pd.DataFrame(rng.standard_normal(future.shape), index=future.index, columns=future.columns)
    assert ic_stats(ic_frame(noisy, fx, h), h)["ic"] > 0.4
    rand = pd.DataFrame(rng.standard_normal(future.shape), index=future.index, columns=future.columns)
    st0 = ic_stats(ic_frame(rand, fx, h), h)
    assert abs(st0["ic"]) < 0.05 and abs(st0["t"]) < 2.5
    assert math.isnan(spearman(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0])))  # n < 3
    assert math.isnan(ic_stats(ic_frame(rand, fx, h).iloc[:2], h)["ic"])


def test_threshold_split_sums_to_full_sample():
    fx = _three_pairs(600)
    rng = np.random.default_rng(4)
    idx = pd.date_range("2019-01-01", "2021-12-31", freq="D", tz="UTC")
    sig = pd.DataFrame(rng.standard_normal((len(idx), 3)), index=idx, columns=list(fx))
    fr = ic_frame(sig, fx, 10)
    sp = threshold_split(fr, 10, 0.75)
    assert sp["above"]["n"] + sp["below"]["n"] == sp["all"]["n"] == len(fr)
    w = (sp["above"]["n"] * sp["above"]["mean_bp"] + sp["below"]["n"] * sp["below"]["mean_bp"]) / sp["all"]["n"]
    assert w == pytest.approx(sp["all"]["mean_bp"])
    assert 0.0 <= sp["above"]["hit"] <= 1.0
    empty = threshold_split(fr, 10, 99.0)["above"]
    assert empty["n"] == 0 and math.isnan(empty["mean_bp"])


def test_nonoverlap_t_is_near_zero_on_noise_and_controls_overlap():
    rng = np.random.default_rng(5)
    assert abs(nonoverlap_t(pd.Series(rng.standard_normal(4000)), 20)) < 2.0
    assert nonoverlap_t(pd.Series(0.5 + rng.standard_normal(4000)), 20) > 3.0
    assert nonoverlap_t(pd.Series([1.0, np.nan]), 5) == 0.0
    # overlapping 10-day returns times a persistent signal: the naive t rejects far
    # too often under the null, the non-overlap t does not
    h, n, naive_rej, no_rej = 10, 1500, 0, 0
    for seed in range(100):
        r = np.random.default_rng(100 + seed)
        daily = pd.Series(r.standard_normal(n + h))
        fwd = daily[::-1].rolling(h).sum()[::-1].shift(-1).iloc[:n]
        sig = np.sign(pd.Series(r.standard_normal(n)).rolling(60, min_periods=1).mean())
        prod = (sig * fwd).dropna()
        naive = prod.mean() / (prod.std(ddof=1) / math.sqrt(len(prod)))
        naive_rej += abs(naive) > 1.96
        no_rej += abs(nonoverlap_t(prod, h)) > 1.96
    assert no_rej <= 12
    assert naive_rej >= 25


def test_newey_west_t_matches_naive_at_lag_zero_and_shrinks_overlap():
    rng = np.random.default_rng(6)
    x = pd.Series(0.1 + rng.standard_normal(3000))
    naive = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert newey_west_t(x, 0) == pytest.approx(naive)
    assert abs(newey_west_t(pd.Series(rng.standard_normal(3000)), 19)) < 2.0
    # a 20-day moving sum of noise is strongly autocorrelated: HAC t is far smaller than naive
    ma = pd.Series(rng.standard_normal(3020)).rolling(20).sum().dropna() + 0.5
    naive_ma = ma.mean() / (ma.std(ddof=0) / math.sqrt(len(ma)))
    assert newey_west_t(ma, 19) < 0.5 * naive_ma
    assert newey_west_t(pd.Series([1.0, 2.0]), 5) == 0.0


# ----------------------------------------------------------------- components and verdict


def test_component_pair_signals_handles_partial_exposure():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    cad_only = pd.DataFrame({"USD": np.nan, "EUR": np.nan, "CAD": [0.1, 0.2, np.nan, 0.4, 0.5]}, index=idx)
    full = pd.DataFrame({"USD": [1.0] * 5, "EUR": [0.5] * 5, "CAD": [0.0] * 5}, index=idx)
    out = component_pair_signals({"oil": cad_only, "carry": full, "none": cad_only[["USD", "EUR"]]}, ["EURUSD", "USDCAD"])
    assert set(out) == {"oil", "carry"}
    assert list(out["oil"].columns) == ["USDCAD"]                     # EURUSD has no oil exposure
    assert out["oil"]["USDCAD"].iloc[1] == pytest.approx(-0.2)        # 0 - CAD
    assert math.isnan(out["oil"]["USDCAD"].iloc[2])                   # CAD unavailable that day
    assert out["carry"]["EURUSD"].iloc[0] == pytest.approx(-0.5)


def test_phase_a_verdict_rows():
    years = [{"year": 2018 + i, "ic": 0.06, "partial": False} for i in range(8)] + [{"year": 2026, "ic": -0.2, "partial": True}]
    assert phase_a_verdict({"h": 20, "ic": 0.06, "t": 2.5}, years)["verdict"] == "PROCEED"
    v = phase_a_verdict({"h": 20, "ic": 0.06, "t": 2.5}, years)
    assert v["years_same_sign"] == 8 and v["years_full"] == 8      # partial year excluded
    assert phase_a_verdict({"h": 20, "ic": 0.06, "t": 1.5}, years)["verdict"] == "WEAK"
    mixed = [dict(y, ic=(0.06 if i < 4 else -0.01)) for i, y in enumerate(years[:8])]
    assert phase_a_verdict({"h": 20, "ic": 0.06, "t": 2.5}, mixed)["verdict"] == "WEAK"   # 4 of 8 years
    assert phase_a_verdict({"h": 20, "ic": 0.03, "t": 5.0}, years)["verdict"] == "WEAK"
    assert phase_a_verdict({"h": 20, "ic": 0.01, "t": 5.0}, years)["verdict"] == "NO-GO"
    assert phase_a_verdict({"h": 20, "ic": -0.08, "t": -4.0}, years)["verdict"] == "NO-GO"
    assert phase_a_verdict({"h": 20, "ic": float("nan"), "t": 0.0}, [])["verdict"] == "NO-GO"


# ----------------------------------------------------------------- end to end


def _panel_and_cfg(n: int = 900):
    idx = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    t = np.arange(n)
    panel = {
        "rate_2y_USD": pd.Series(2.0 + 0.02 * rng.standard_normal(n).cumsum(), index=idx),
        "rate_2y_EUR": pd.Series(0.0 + 0.02 * rng.standard_normal(n).cumsum(), index=idx),
        "rate_2y_JPY": pd.Series(0.0 + 0.01 * rng.standard_normal(n).cumsum(), index=idx),
        "policy_rate_USD": pd.Series(np.where(t < 400, 2.0, 4.0), index=idx),
        "policy_rate_EUR": pd.Series(0.5, index=idx),
        "policy_rate_JPY": pd.Series(-0.1, index=idx),
        "global_oil": pd.Series(70 + 5 * np.sin(t / 50), index=idx),
    }
    cfg = {
        "universe": ["EURUSD", "USDJPY"],
        "signal": {"min_abs_bias": 0.75},
        "data": {"start": "2019-06-01", "end": None},
        "macro": {
            "currencies": ["USD", "EUR", "JPY"],
            "min_components": 2,
            "lags": {"market_days": 1},
            "series": {
                "rate_2y": {c: {"source": "csv", "path": "x"} for c in ("USD", "EUR", "JPY")},
                "policy_rate": {c: {"source": "csv", "path": "x"} for c in ("USD", "EUR", "JPY")},
                "global": {"oil": {"source": "csv", "path": "x"}},
            },
            "components": {
                "rate_momentum": {"weight": 1.0, "input": "rate_2y", "transform": "diff", "lookback_days": 20, "standardize": "cross_section"},
                "carry": {"weight": 0.5, "input": "policy_rate", "transform": "level", "standardize": "cross_section"},
                "commodity_energy": {"weight": 0.5, "input": "oil", "transform": "pct_change", "lookback_days": 20, "standardize": "time_series", "window_days": 60, "beta": {"JPY": -1.0}},
            },
        },
    }
    fx = {"EURUSD": _walk(n, 20), "USDJPY": _walk(n, 21, px0=110.0)}
    return panel, cfg, fx


def test_run_macro_ic_end_to_end_and_report(tmp_path):
    panel, cfg, fx = _panel_and_cfg()
    res = run_macro_ic(panel, cfg, fx, horizons=(5, 20))
    meta = res["meta"]
    assert meta["pairs"] == ["EURUSD", "USDJPY"] and meta["currencies"] == ["EUR", "JPY", "USD"]
    assert meta["first_date"] >= "2019-06-01"
    assert [r["h"] for r in res["pooled"]] == [5, 20]
    assert {(r["pair"], r["h"]) for r in res["per_pair"]} == {(p, h) for p in ("EURUSD", "USDJPY") for h in (5, 20)}
    comps = {r["component"]: r for r in res["components"] if r["h"] == 20}
    assert set(comps) == {"rate_momentum", "carry", "commodity_energy"}
    assert comps["commodity_energy"]["pairs"] == ["USDJPY"]
    for r in res["pooled"]:
        assert r["above"]["n"] + r["below"]["n"] == r["n"]
        assert -1.0 <= r["ic"] <= 1.0
    assert [y["year"] for y in res["by_year"]] == [2019, 2020, 2021, 2022]
    assert res["by_year"][0]["partial"] and not res["by_year"][1]["partial"]
    assert res["verdict"]["verdict"] in ("PROCEED", "WEAK", "NO-GO")
    paths = write_macro_ic_report(res, tmp_path)
    md = paths["md"].read_text(encoding="utf-8")
    assert "## Verdict (plan A3)" in md and "## Per component" in md and "| rate_momentum |" in md
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))  # strict JSON: NaN written as null
    assert payload["verdict"]["verdict"] == res["verdict"]["verdict"]
    assert write_macro_ic_report(res, tmp_path, tag="x")["md"].name == "macro_ic_x.md"


def test_fetch_fx_caches_fred_series(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(spec, name, start="2010-01-01"):
        calls.append(spec["id"])
        if spec["id"] == "DEXJPUS":
            raise RuntimeError("boom")
        return _walk(5, 1).rename(name)

    monkeypatch.setattr(mic, "fetch_series", fake_fetch)
    status = fetch_fx(["eurusd", "USDJPY", "EURGBP"], tmp_path)
    assert status["EURUSD"]["cached"] is False and status["EURUSD"]["n"] == 5
    assert "boom" in status["USDJPY"]["error"] and "no FRED" in status["EURGBP"]["error"]
    assert (tmp_path / "fx_EURUSD.csv").is_file()
    assert fetch_fx(["EURUSD"], tmp_path)["EURUSD"]["cached"] is True
    assert calls == ["DEXUSEU", "DEXJPUS"]
    assert list(load_fx(["EURUSD", "USDJPY"], tmp_path)) == ["EURUSD"]
