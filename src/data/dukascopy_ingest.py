from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import ROOT, ensure_dirs, load_strategy_config

# Map our symbols to dukascopy-python instrument constants when available.
SYMBOL_TO_INSTRUMENT_ATTR = {
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "USDJPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "AUDUSD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "USDCAD": "INSTRUMENT_FX_MAJORS_USD_CAD",
}


def parquet_path(symbol: str, interval: str, cache_dir: Path | None = None) -> Path:
    ensure_dirs()
    base = cache_dir or (ROOT / "data" / "parquet")
    return base / f"{symbol}_{interval}.parquet"


def _resolve_instrument(symbol: str):
    from dukascopy_python import instruments as inst

    attr = SYMBOL_TO_INSTRUMENT_ATTR.get(symbol.upper())
    if attr is None or not hasattr(inst, attr):
        mapping = {
            "EURUSD": "EUR/USD",
            "GBPUSD": "GBP/USD",
            "USDJPY": "USD/JPY",
            "AUDUSD": "AUD/USD",
            "USDCAD": "USD/CAD",
        }
        return mapping[symbol.upper()]
    return getattr(inst, attr)


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        # try common column names
        for col in ("timestamp", "time", "Date", "date"):
            if col in out.columns:
                out = out.set_index(col)
                break
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()

    rename = {c: c.lower() for c in out.columns}
    out = out.rename(columns=rename)
    # Dukascopy sometimes uses bid/ask only for ticks
    if "close" not in out.columns and "bidprice" in out.columns:
        mid = (out["bidprice"] + out.get("askprice", out["bidprice"])) / 2.0
        out["open"] = mid
        out["high"] = mid
        out["low"] = mid
        out["close"] = mid
        out["volume"] = out.get("bidvolume", 0) + out.get("askvolume", 0)

    needed = ["open", "high", "low", "close"]
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"Missing OHLC column '{c}' in Dukascopy frame: {list(out.columns)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out[needed + ["volume"]].astype(float)
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"
    return out.reset_index()


def fetch_symbol_ohlc(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "M15",
) -> pd.DataFrame:
    """Fetch OHLC from Dukascopy and return normalized frame."""
    import dukascopy_python

    interval_map = {
        "M1": getattr(dukascopy_python, "INTERVAL_MIN_1", None),
        "M15": getattr(dukascopy_python, "INTERVAL_MIN_15", None),
        "H1": getattr(dukascopy_python, "INTERVAL_HOUR_1", None),
    }
    # Some versions expose INTERVAL_MINUTE_15 etc.
    if interval_map.get("M15") is None:
        for cand in ("INTERVAL_MINUTE_15", "INTERVAL_MINUTES_15", "INTERVAL_15_MIN"):
            if hasattr(dukascopy_python, cand):
                interval_map["M15"] = getattr(dukascopy_python, cand)
                break
    if interval_map.get("H1") is None:
        for cand in ("INTERVAL_HOUR_1", "INTERVAL_HOURS_1", "INTERVAL_1_HOUR"):
            if hasattr(dukascopy_python, cand):
                interval_map["H1"] = getattr(dukascopy_python, cand)
                break
    if interval_map.get("M1") is None and hasattr(dukascopy_python, "INTERVAL_MINUTE_1"):
        interval_map["M1"] = dukascopy_python.INTERVAL_MINUTE_1

    duka_interval = interval_map.get(interval)
    if duka_interval is None:
        raise RuntimeError(
            f"Could not resolve Dukascopy interval for {interval}. "
            f"Available attrs sample: {[a for a in dir(dukascopy_python) if 'INTERVAL' in a][:30]}"
        )

    instrument = _resolve_instrument(symbol)
    offer = getattr(dukascopy_python, "OFFER_SIDE_BID", "B")

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # Chunk by year to avoid timeouts / huge payloads
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc), end)
        try:
            df = dukascopy_python.fetch(
                instrument,
                duka_interval,
                offer,
                cursor.replace(tzinfo=None),
                chunk_end.replace(tzinfo=None),
            )
        except TypeError:
            # Alternate signature: keyword args
            df = dukascopy_python.fetch(
                instrument=instrument,
                interval=duka_interval,
                offer_side=offer,
                start=cursor.replace(tzinfo=None),
                end=chunk_end.replace(tzinfo=None),
            )
        part = _normalize_ohlc(df)
        if not part.empty:
            frames.append(part)
        cursor = chunk_end

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def download_universe(
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    intervals: Iterable[str] = ("M15",),
    force: bool = False,
) -> dict[str, Path]:
    """Download and cache Dukascopy OHLC for the strategy universe.

    Default: M15 only (H1 features are derived from M15 in the strategy).
    """
    cfg = load_strategy_config()
    symbols = list(symbols or cfg["universe"])
    start_s = start or cfg["data"]["start"]
    end_s = end or cfg["data"].get("end")
    start_dt = datetime.fromisoformat(start_s).replace(tzinfo=timezone.utc)
    if end_s:
        end_dt = datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)

    cache_dir = ROOT / cfg["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for symbol in symbols:
        for interval in intervals:
            path = parquet_path(symbol, interval, cache_dir)
            key = f"{symbol}_{interval}"
            if path.exists() and not force:
                print(f"[skip] {key} exists at {path}")
                paths[key] = path
                continue
            print(f"[fetch] {key} {start_dt.date()} -> {end_dt.date()}")
            df = fetch_symbol_ohlc(symbol, start_dt, end_dt, interval=interval)
            if df.empty:
                raise RuntimeError(f"No data returned for {key}")
            save_parquet(df, path)
            print(f"[saved] {key}: {len(df)} bars -> {path}")
            paths[key] = path
    return paths


def load_symbol_bars(symbol: str, interval: str = "M15") -> pd.DataFrame:
    cfg = load_strategy_config()
    path = parquet_path(symbol, interval, ROOT / cfg["data"]["cache_dir"])
    return load_parquet(path)
