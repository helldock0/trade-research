from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import ROOT, ensure_dirs, load_strategy_config

# Majors (USD crosses) + typical major crosses (minors) + XAUUSD.
MAJORS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
]

MINORS = [
    "EURGBP",
    "EURJPY",
    "EURCHF",
    "EURAUD",
    "EURCAD",
    "EURNZD",
    "GBPJPY",
    "GBPCHF",
    "GBPAUD",
    "GBPCAD",
    "GBPNZD",
    "AUDJPY",
    "AUDCHF",
    "AUDCAD",
    "AUDNZD",
    "CADJPY",
    "CADCHF",
    "CHFJPY",
    "NZDJPY",
    "NZDCHF",
    "NZDCAD",
]

METALS = ["XAUUSD"]

FULL_FX_UNIVERSE = MAJORS + MINORS + METALS

# Canonical bar timeframes exposed by dukascopy-python (MT5-style names).
ALL_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")

# Map our symbols to dukascopy-python instrument constants when available.
SYMBOL_TO_INSTRUMENT_ATTR = {
    # Majors
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "USDJPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "USDCHF": "INSTRUMENT_FX_MAJORS_USD_CHF",
    "AUDUSD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "USDCAD": "INSTRUMENT_FX_MAJORS_USD_CAD",
    "NZDUSD": "INSTRUMENT_FX_MAJORS_NZD_USD",
    # Crosses / minors
    "EURGBP": "INSTRUMENT_FX_CROSSES_EUR_GBP",
    "EURJPY": "INSTRUMENT_FX_CROSSES_EUR_JPY",
    "EURCHF": "INSTRUMENT_FX_CROSSES_EUR_CHF",
    "EURAUD": "INSTRUMENT_FX_CROSSES_EUR_AUD",
    "EURCAD": "INSTRUMENT_FX_CROSSES_EUR_CAD",
    "EURNZD": "INSTRUMENT_FX_CROSSES_EUR_NZD",
    "GBPJPY": "INSTRUMENT_FX_CROSSES_GBP_JPY",
    "GBPCHF": "INSTRUMENT_FX_CROSSES_GBP_CHF",
    "GBPAUD": "INSTRUMENT_FX_CROSSES_GBP_AUD",
    "GBPCAD": "INSTRUMENT_FX_CROSSES_GBP_CAD",
    "GBPNZD": "INSTRUMENT_FX_CROSSES_GBP_NZD",
    "AUDJPY": "INSTRUMENT_FX_CROSSES_AUD_JPY",
    "AUDCHF": "INSTRUMENT_FX_CROSSES_AUD_CHF",
    "AUDCAD": "INSTRUMENT_FX_CROSSES_AUD_CAD",
    "AUDNZD": "INSTRUMENT_FX_CROSSES_AUD_NZD",
    "CADJPY": "INSTRUMENT_FX_CROSSES_CAD_JPY",
    "CADCHF": "INSTRUMENT_FX_CROSSES_CAD_CHF",
    "CHFJPY": "INSTRUMENT_FX_CROSSES_CHF_JPY",
    "NZDJPY": "INSTRUMENT_FX_CROSSES_NZD_JPY",
    "NZDCHF": "INSTRUMENT_FX_CROSSES_NZD_CHF",
    "NZDCAD": "INSTRUMENT_FX_CROSSES_NZD_CAD",
    # Metals
    "XAUUSD": "INSTRUMENT_FX_METALS_XAU_USD",
    # Indices (CFD)
    "NAS100": "INSTRUMENT_IDX_AMERICA_E_NQ_100",
}

# Fallback slash notation if instrument attr is missing.
SYMBOL_TO_SLASH = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD",
    "EURGBP": "EUR/GBP",
    "EURJPY": "EUR/JPY",
    "EURCHF": "EUR/CHF",
    "EURAUD": "EUR/AUD",
    "EURCAD": "EUR/CAD",
    "EURNZD": "EUR/NZD",
    "GBPJPY": "GBP/JPY",
    "GBPCHF": "GBP/CHF",
    "GBPAUD": "GBP/AUD",
    "GBPCAD": "GBP/CAD",
    "GBPNZD": "GBP/NZD",
    "AUDJPY": "AUD/JPY",
    "AUDCHF": "AUD/CHF",
    "AUDCAD": "AUD/CAD",
    "AUDNZD": "AUD/NZD",
    "CADJPY": "CAD/JPY",
    "CADCHF": "CAD/CHF",
    "CHFJPY": "CHF/JPY",
    "NZDJPY": "NZD/JPY",
    "NZDCHF": "NZD/CHF",
    "NZDCAD": "NZD/CAD",
    "XAUUSD": "XAU/USD",
}

INTERVAL_ATTR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "M1": ("INTERVAL_MIN_1", "INTERVAL_MINUTE_1"),
    "M5": ("INTERVAL_MIN_5", "INTERVAL_MINUTE_5"),
    "M15": ("INTERVAL_MIN_15", "INTERVAL_MINUTE_15", "INTERVAL_MINUTES_15"),
    "M30": ("INTERVAL_MIN_30", "INTERVAL_MINUTE_30"),
    "H1": ("INTERVAL_HOUR_1", "INTERVAL_HOURS_1", "INTERVAL_1_HOUR"),
    "H4": ("INTERVAL_HOUR_4", "INTERVAL_HOURS_4", "INTERVAL_4_HOUR"),
    "D1": ("INTERVAL_DAY_1", "INTERVAL_DAYS_1", "INTERVAL_1_DAY"),
    "W1": ("INTERVAL_WEEK_1", "INTERVAL_WEEKS_1", "INTERVAL_1_WEEK"),
    "MN1": ("INTERVAL_MONTH_1", "INTERVAL_MONTHS_1", "INTERVAL_1_MONTH"),
}


def parquet_path(
    symbol: str, interval: str, cache_dir: Path | None = None, side: str = "bid"
) -> Path:
    """Bid bars are `{SYMBOL}_{TF}.parquet`; ask bars add an `_ask` suffix."""
    ensure_dirs()
    base = cache_dir or (ROOT / "data" / "parquet")
    suffix = "_ask" if side == "ask" else ""
    return base / f"{symbol}_{interval}{suffix}.parquet"


def _resolve_instrument(symbol: str):
    from dukascopy_python import instruments as inst

    sym = symbol.upper()
    attr = SYMBOL_TO_INSTRUMENT_ATTR.get(sym)
    if attr is not None and hasattr(inst, attr):
        return getattr(inst, attr)
    if sym in SYMBOL_TO_SLASH:
        return SYMBOL_TO_SLASH[sym]
    raise KeyError(f"Unknown Dukascopy symbol mapping for {symbol}")


def _resolve_interval(interval: str):
    import dukascopy_python

    key = interval.upper()
    for cand in INTERVAL_ATTR_CANDIDATES.get(key, ()):
        if hasattr(dukascopy_python, cand):
            return getattr(dukascopy_python, cand)
    raise RuntimeError(
        f"Could not resolve Dukascopy interval for {interval}. "
        f"Available attrs sample: {[a for a in dir(dukascopy_python) if 'INTERVAL' in a][:30]}"
    )


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
    sleep_s: float = 0.15,
    side: str = "bid",
) -> pd.DataFrame:
    """Fetch OHLC from Dukascopy and return normalized frame."""
    import dukascopy_python

    duka_interval = _resolve_interval(interval)
    instrument = _resolve_instrument(symbol)
    if side == "ask":
        offer = getattr(dukascopy_python, "OFFER_SIDE_ASK", "A")
    else:
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
        if sleep_s > 0:
            time.sleep(sleep_s)

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
    sleep_s: float = 0.15,
    continue_on_error: bool = True,
    side: str = "bid",
) -> dict[str, Path]:
    """Download and cache Dukascopy OHLC for the strategy universe.

    Default: M15 only (H1 features are derived from M15 in the strategy).
    Skips existing parquet files unless force=True.
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
    failures: dict[str, str] = {}
    interval_list = list(intervals)
    total = len(symbols) * len(interval_list)
    done = 0

    # Interval-outer so all symbols get coarse TFs before any M1/M5 crawl.
    for interval in interval_list:
        for symbol in symbols:
            done += 1
            path = parquet_path(symbol, interval, cache_dir, side=side)
            key = f"{symbol}_{interval}" + ("_ask" if side == "ask" else "")
            if path.exists() and not force:
                print(f"[skip] ({done}/{total}) {key} exists at {path}", flush=True)
                paths[key] = path
                continue
            print(
                f"[fetch] ({done}/{total}) {key} {start_dt.date()} -> {end_dt.date()}",
                flush=True,
            )
            try:
                df = fetch_symbol_ohlc(
                    symbol, start_dt, end_dt, interval=interval, sleep_s=sleep_s, side=side
                )
                if df.empty:
                    raise RuntimeError(f"No data returned for {key}")
                save_parquet(df, path)
                print(f"[saved] {key}: {len(df)} bars -> {path}", flush=True)
                paths[key] = path
            except Exception as exc:  # noqa: BLE001 - resilient bulk download
                msg = f"{type(exc).__name__}: {exc}"
                failures[key] = msg
                print(f"[fail] {key}: {msg}", flush=True)
                if not continue_on_error:
                    raise

    if failures:
        print(f"[summary] ok={len(paths)} fail={len(failures)}", flush=True)
        for k, v in failures.items():
            print(f"  FAIL {k}: {v}", flush=True)
    else:
        print(f"[summary] ok={len(paths)} fail=0", flush=True)
    return paths


def load_symbol_bars(symbol: str, interval: str = "M15") -> pd.DataFrame:
    cfg = load_strategy_config()
    path = parquet_path(symbol, interval, ROOT / cfg["data"]["cache_dir"])
    return load_parquet(path)
