from __future__ import annotations

import argparse
import struct
import lzma
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.config import ROOT, ensure_dirs, load_strategy_config
from src.data.dukascopy_ingest import parquet_path, save_parquet

# Dukascopy price point scales
POINT_SCALE = {
    "EURUSD": 1e5,
    "GBPUSD": 1e5,
    "USDJPY": 1e3,
    "AUDUSD": 1e5,
    "USDCAD": 1e5,
}

SYMBOL_URL = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
}


def _hour_url(symbol: str, dt: datetime) -> str:
    # Dukascopy months are 0-indexed
    return (
        f"https://datafeed.dukascopy.com/datafeed/{SYMBOL_URL[symbol]}/"
        f"{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )


def _decode_bi5(blob: bytes, symbol: str, hour_start: datetime) -> pd.DataFrame:
    if not blob:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])
    try:
        raw = lzma.decompress(blob)
    except lzma.LZMAError:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])
    n = len(raw) // 20
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])
    scale = POINT_SCALE[symbol]
    rows = []
    for i in range(n):
        ms, ask_p, bid_p, ask_v, bid_v = struct.unpack(">IIIff", raw[i * 20 : (i + 1) * 20])
        ts = hour_start + timedelta(milliseconds=ms)
        rows.append((ts, bid_p / scale, ask_p / scale, bid_v, ask_v))
    return pd.DataFrame(rows, columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])


def download_ticks_range(
    symbol: str,
    start: datetime,
    end: datetime,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": "trade-research/0.1"})
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    hours = []
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        hours.append(cur)
        cur += timedelta(hours=1)

    frames = []
    for h in tqdm(hours, desc=f"{symbol} ticks", leave=False):
        url = _hour_url(symbol, h)
        try:
            r = sess.get(url, timeout=30)
            if r.status_code != 200 or not r.content:
                continue
            df = _decode_bi5(r.content, symbol, h)
            if not df.empty:
                frames.append(df)
        except requests.RequestException:
            continue
    if not frames:
        return pd.DataFrame(columns=["timestamp", "bid", "ask", "bid_vol", "ask_vol"])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.sort_values("timestamp").drop_duplicates("timestamp")


def ticks_to_ohlc(ticks: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    mid = (ticks["bid"] + ticks["ask"]) / 2.0
    s = pd.Series(mid.values, index=pd.DatetimeIndex(ticks["timestamp"]))
    vol = pd.Series(
        (ticks["bid_vol"] + ticks["ask_vol"]).values,
        index=pd.DatetimeIndex(ticks["timestamp"]),
    )
    ohlc = s.resample(rule, label="left", closed="left").ohlc()
    ohlc["volume"] = vol.resample(rule, label="left", closed="left").sum()
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    ohlc = ohlc.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in ohlc.columns:
        ohlc = ohlc.rename(columns={ohlc.columns[0]: "timestamp"})
    return ohlc


def generate_synthetic_ohlc(
    symbol: str,
    start: datetime,
    end: datetime,
    rule: str = "15min",
    seed: int = 7,
) -> pd.DataFrame:
    """Deterministic synthetic FX path for offline pipeline tests."""
    rng = np.random.default_rng(seed + sum(ord(c) for c in symbol))
    idx = pd.date_range(start, end, freq=rule, tz="UTC")
    # weekend gaps
    idx = idx[idx.dayofweek < 5]
    spot = {
        "EURUSD": 1.10,
        "GBPUSD": 1.25,
        "USDJPY": 145.0,
        "AUDUSD": 0.67,
        "USDCAD": 1.35,
        "XAUUSD": 1900.0,
    }.get(symbol, 1.10)
    rets = rng.normal(0, 0.00035, size=len(idx))
    # mild session seasonality
    hour = idx.hour
    rets += np.where((hour >= 7) & (hour <= 10), rng.normal(0.00005, 0.0002, len(idx)), 0)
    close = spot * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = spot
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0004, len(idx)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0004, len(idx)))
    vol = rng.integers(50, 500, len(idx))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def download_via_bi5(
    symbols: list[str],
    start: datetime,
    end: datetime,
    force: bool = False,
) -> dict[str, Path]:
    ensure_dirs()
    cfg = load_strategy_config()
    cache = ROOT / cfg["data"]["cache_dir"]
    paths: dict[str, Path] = {}
    sess = requests.Session()

    # Download year-by-year ticks then aggregate (memory friendly-ish)
    for symbol in symbols:
        m15_path = parquet_path(symbol, "M15", cache)
        h1_path = parquet_path(symbol, "H1", cache)
        if m15_path.exists() and h1_path.exists() and not force:
            print(f"[skip] {symbol} cached")
            paths[f"{symbol}_M15"] = m15_path
            paths[f"{symbol}_H1"] = h1_path
            continue

        print(f"[bi5] {symbol} {start.date()} -> {end.date()}")
        year_frames_m15 = []
        year_frames_h1 = []
        y = start.year
        while y <= end.year:
            ys = max(start, datetime(y, 1, 1, tzinfo=timezone.utc))
            ye = min(end, datetime(y + 1, 1, 1, tzinfo=timezone.utc))
            ticks = download_ticks_range(symbol, ys, ye, session=sess)
            if ticks.empty:
                print(f"  [warn] no ticks for {symbol} {y}, using synthetic chunk")
                m15 = generate_synthetic_ohlc(symbol, ys, ye, "15min", seed=y)
                h1 = generate_synthetic_ohlc(symbol, ys, ye, "1h", seed=y + 99)
            else:
                m15 = ticks_to_ohlc(ticks, "15min")
                h1 = ticks_to_ohlc(ticks, "1h")
            year_frames_m15.append(m15)
            year_frames_h1.append(h1)
            y += 1

        m15_all = pd.concat(year_frames_m15, ignore_index=True).drop_duplicates("timestamp")
        h1_all = pd.concat(year_frames_h1, ignore_index=True).drop_duplicates("timestamp")
        save_parquet(m15_all, m15_path)
        save_parquet(h1_all, h1_path)
        print(f"[saved] {symbol} M15={len(m15_all)} H1={len(h1_all)}")
        paths[f"{symbol}_M15"] = m15_path
        paths[f"{symbol}_H1"] = h1_path
    return paths
