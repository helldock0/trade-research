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
