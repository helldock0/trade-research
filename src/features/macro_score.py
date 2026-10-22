"""Per-currency macro scores and pair bias from a panel of macro series.

Pipeline (all point-in-time):
    1. Align every series onto a daily calendar, forward-filled, then shift it
       by its availability lag (market data: next day; COT: report + 3 days).
    2. For each component, build a raw value per currency:
         - per-currency inputs (rate_2y, policy_rate, cot_net): transform of
           that currency's series (level / diff / pct_change over lookback),
           optionally minus another per-currency input (`subtract`)
         - global inputs (riskoff, oil, copper): transform of the global
           series times a per-currency beta
    3. Standardize: cross_section (z across currencies each day) or
       time_series (rolling z per currency).
    4. score[c] = weighted mean of available component z's (needs
       min_components); bias[pair] = score[base] - score[quote].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")


def _daily_index(series: list[pd.Series]) -> pd.DatetimeIndex:
    lo = min(s.index.min() for s in series if len(s))
    hi = max(s.index.max() for s in series if len(s))
    return pd.date_range(lo.normalize(), hi.normalize(), freq="D", tz="UTC")


def align_with_lag(s: pd.Series, index: pd.DatetimeIndex, lag_days: int, ffill_limit: int | None = None) -> pd.Series:
    """Forward-fill `s` onto `index` after shifting its dates by `lag_days`.

    The value observed on date d becomes usable from d + lag_days onwards.
    """
    if s is None or s.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    x = s.copy()
    x.index = x.index + pd.Timedelta(days=int(lag_days))
    x = x[~x.index.duplicated(keep="last")].sort_index()
    out = x.reindex(index.union(x.index)).ffill(limit=ffill_limit).reindex(index)
    return out.astype(float)


def build_panel_frames(
    panel: dict[str, pd.Series],
    macro_cfg: dict[str, Any],
    currencies: tuple[str, ...] | list[str] = DEFAULT_CURRENCIES,
) -> dict[str, Any]:
    """Return {'index', 'groups': {group: DataFrame[date x currency]}, 'global': {key: Series}}."""
    if not panel:
        return {"index": pd.DatetimeIndex([], tz="UTC"), "groups": {}, "global": {}}
    lags = macro_cfg.get("lags", {}) or {}
    market_lag = int(lags.get("market_days", 1))
    cot_lag = int(lags.get("cot_days", 3))
    ffill_limit = lags.get("ffill_limit_days", 45)
    index = _daily_index(list(panel.values()))
    series_cfg = macro_cfg.get("series", {}) or {}

    groups: dict[str, pd.DataFrame] = {}
    glob: dict[str, pd.Series] = {}
    for group, entries in series_cfg.items():
        if group == "global":
            for key, spec in (entries or {}).items():
                s = panel.get(f"global_{key}")
                lag = int((spec or {}).get("lag_days", market_lag))
                glob[key] = align_with_lag(s, index, lag, ffill_limit)
            continue
        cols: dict[str, pd.Series] = {}
        for cur in currencies:
            spec = (entries or {}).get(cur)
            s = panel.get(f"{group}_{cur}")
            if s is None or s.empty:
                continue
            default_lag = cot_lag if group.startswith("cot") else market_lag
            lag = int((spec or {}).get("lag_days", default_lag))
            cols[cur] = align_with_lag(s, index, lag, ffill_limit)
        df = pd.DataFrame(cols, index=index)
        # Derived columns (e.g. USD positioning = negative mean of the others)
        for cur, spec in (entries or {}).items():
            cur = str(cur).upper()
            if spec and spec.get("derive") == "neg_mean_others" and cur not in df.columns and not df.empty:
                df[cur] = -df.mean(axis=1, skipna=True)
        groups[group] = df.reindex(columns=[c for c in currencies if c in df.columns])
    return {"index": index, "groups": groups, "global": glob}


def riskoff_series(glob: dict[str, pd.Series], cfg: dict[str, Any] | None = None) -> pd.Series:
    """Risk-off gauge: mean of z(VIX level) and z(-equity return over lookback)."""
    cfg = cfg or {}
    win = int(cfg.get("window_days", 252))
    look = int(cfg.get("equity_lookback_days", 20))
    parts = []
    vix = glob.get("vix")
    if vix is not None and vix.notna().any():
        parts.append(rolling_z(vix, win))
    spx = glob.get("spx")
    if spx is not None and spx.notna().any():
        ret = spx.pct_change(look, fill_method=None)
        parts.append(rolling_z(-ret, win))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    minp = max(20, window // 2)
    mu = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std(ddof=0)
    return ((s - mu) / sd.where(sd > 0)).astype(float)


def cross_sectional_z(df: pd.DataFrame) -> pd.DataFrame:
    n = df.notna().sum(axis=1)
    mu = df.mean(axis=1, skipna=True)
    sd = df.std(axis=1, ddof=0, skipna=True)
    z = df.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    return z.where(n.ge(2), np.nan)


def transform(df: pd.DataFrame | pd.Series, mode: str, lookback: int) -> pd.DataFrame | pd.Series:
    mode = (mode or "level").lower()
    if mode == "level":
        return df
    if mode == "diff":
        return df - df.shift(lookback)
    if mode == "pct_change":
        return df.pct_change(lookback, fill_method=None)
    raise ValueError(f"Unknown transform: {mode}")


def component_frame(
    name: str,
    ccfg: dict[str, Any],
    frames: dict[str, Any],
    currencies: tuple[str, ...] | list[str],
) -> pd.DataFrame | None:
    """Standardized component values, DataFrame[date x currency] (NaN = unavailable)."""
    inp = str(ccfg.get("input", "")).lower()
    look = int(ccfg.get("lookback_days", 1))
    std_mode = str(ccfg.get("standardize", "cross_section")).lower()
    win = int(ccfg.get("window_days", 252))
    sign = float(ccfg.get("sign", 1.0))
    index = frames["index"]
    beta = {str(k).upper(): float(v) for k, v in (ccfg.get("beta") or {}).items()}

    if inp == "riskoff":
        base = riskoff_series(frames["global"], ccfg)
        if base.empty:
            return None
        z = rolling_z(base, win) if std_mode == "time_series" else base
        out = pd.DataFrame({c: z * b for c, b in beta.items() if c in currencies}, index=index)
        return out * sign if not out.empty else None

    if inp in frames["global"]:
        base = transform(frames["global"][inp], ccfg.get("transform", "level"), look)
        z = rolling_z(base, win) if std_mode == "time_series" else base
        out = pd.DataFrame({c: z * b for c, b in beta.items() if c in currencies}, index=index)
        return out * sign if not out.empty else None

    df = frames["groups"].get(inp)
    if df is None or df.empty:
        return None
    sub = ccfg.get("subtract")
    if sub:
        # spread of two per-currency inputs, e.g. 2y yield minus policy rate (market-implied rate path)
        other = frames["groups"].get(str(sub).lower())
        if other is None or other.empty:
            return None
        df = df - other.reindex(index=df.index, columns=df.columns)
    raw = transform(df, ccfg.get("transform", "level"), look)
    if std_mode == "cross_section":
        z = cross_sectional_z(raw)
    elif std_mode == "time_series":
        z = raw.apply(lambda col: rolling_z(col, win))
    else:
        raise ValueError(f"Unknown standardize: {std_mode}")
    return z.reindex(columns=[c for c in currencies if c in z.columns]) * sign


def build_currency_scores(
    panel: dict[str, pd.Series],
    macro_cfg: dict[str, Any],
    currencies: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Return {'scores': DataFrame[date x currency], 'n_components': DataFrame, 'components': {name: DataFrame}}."""
    currencies = tuple(c.upper() for c in (currencies or macro_cfg.get("currencies") or DEFAULT_CURRENCIES))
    frames = build_panel_frames(panel, macro_cfg, currencies)
    index = frames["index"]
    comps: dict[str, pd.DataFrame] = {}
    weights: dict[str, float] = {}
    for name, ccfg in (macro_cfg.get("components") or {}).items():
        w = float(ccfg.get("weight", 1.0))
        if w == 0:
            continue
        cf = component_frame(name, ccfg, frames, currencies)
        if cf is None:
            continue
        comps[name] = cf.reindex(index=index, columns=currencies)
        weights[name] = w
    if not comps:
        empty = pd.DataFrame(index=index, columns=list(currencies), dtype=float)
        return {"scores": empty, "n_components": empty.fillna(0), "components": {}, "weights": {}}

    num = sum(comps[n].fillna(0.0) * weights[n] for n in comps)
    den = sum(comps[n].notna().astype(float) * weights[n] for n in comps)
    n_comp = sum(comps[n].notna().astype(int) for n in comps)
    min_c = int(macro_cfg.get("min_components", 2))
    scores = (num / den.where(den > 0)).where(n_comp >= min_c)
    return {"scores": scores, "n_components": n_comp, "components": comps, "weights": weights}


def pair_bias(scores: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    out = {}
    for p in pairs:
        b, q = p.upper()[:3], p.upper()[3:6]
        if b in scores.columns and q in scores.columns:
            out[p.upper()] = scores[b] - scores[q]
    return pd.DataFrame(out, index=scores.index)


def latest_table(result: dict[str, Any]) -> pd.DataFrame:
    """Last available row of each component plus the score, for a quick look."""
    scores = result["scores"]
    if scores.empty:
        return pd.DataFrame()
    last = scores.dropna(how="all").index[-1] if scores.notna().any().any() else scores.index[-1]
    rows = {name: df.loc[last] for name, df in result["components"].items()}
    rows["score"] = scores.loc[last]
    rows["n"] = result["n_components"].loc[last]
    tab = pd.DataFrame(rows)
    tab.index.name = f"as of {last.date()}"
    return tab.sort_values("score", ascending=False)
