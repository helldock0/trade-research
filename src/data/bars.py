from __future__ import annotations

import pandas as pd


def ensure_utc_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if ts_col in out.columns:
        out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
        out = out.set_index(ts_col)
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC without lookahead (label=left, closed=left)."""
    x = ensure_utc_index(df)
    ohlc = x.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return ohlc.dropna(subset=["open", "high", "low", "close"])


def align_multi_symbol(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join close prices; keep full OHLC in MultiIndex columns."""
    pieces = []
    for symbol, df in frames.items():
        x = ensure_utc_index(df)
        x = x[["open", "high", "low", "close", "volume"]].copy()
        x.columns = pd.MultiIndex.from_product([[symbol], x.columns])
        pieces.append(x)
    if not pieces:
        return pd.DataFrame()
    out = pieces[0]
    for p in pieces[1:]:
        out = out.join(p, how="outer")
    return out.sort_index()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def cross_bars(pair: str, legs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Synthetic cross bars (EURJPY from EURUSD and USDJPY) on the legs' common timestamps.

    Each output bar uses only the legs' opens and closes (high/low = max/min of
    them), so it misses moves inside a leg bar: build from M1 and resample up.
    """
    pair = pair.upper()

    def usd_value(ccy: str) -> pd.DataFrame:
        if f"{ccy}USD" in legs:
            return ensure_utc_index(legs[f"{ccy}USD"])[["open", "close"]]
        if f"USD{ccy}" in legs:
            return 1.0 / ensure_utc_index(legs[f"USD{ccy}"])[["open", "close"]]
        raise ValueError(f"{pair}: no {ccy}USD or USD{ccy} leg")

    b, q = usd_value(pair[:3]), usd_value(pair[3:6])
    both = b.join(q, how="inner", lsuffix="_b", rsuffix="_q").dropna()
    o = both["open_b"] / both["open_q"]
    c = both["close_b"] / both["close_q"]
    out = pd.DataFrame({"open": o, "high": pd.concat([o, c], axis=1).max(axis=1), "low": pd.concat([o, c], axis=1).min(axis=1), "close": c, "volume": 0.0})
    out.index.name = "timestamp"
    return out


def daily_ohlc(m15: pd.DataFrame, close_hour_utc: int = 21) -> pd.DataFrame:
    """Weekday bars that roll at `close_hour_utc` UTC: OHLC plus the timestamp of each day's last bar.

    A day d covers [d-1 close_hour, d close_hour) UTC, so FX, gold and index CFDs share one daily clock.
    """
    x = ensure_utc_index(m15)
    rule = dict(rule="24h", offset=f"{int(close_hour_utc)}h", label="right")
    out = pd.DataFrame(
        {
            "open": x["open"].resample(**rule).first(),
            "high": x["high"].resample(**rule).max(),
            "low": x["low"].resample(**rule).min(),
            "close": x["close"].resample(**rule).last(),
            "last_ts": pd.Series(x.index, index=x.index).resample(**rule).last(),
        }
    )
    out.index = out.index.normalize().tz_localize(None)
    out.index.name = "date"
    return out[(out.index.dayofweek < 5)].dropna(subset=["close"])
