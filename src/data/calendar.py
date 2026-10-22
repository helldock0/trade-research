"""Economic event calendar: fetch (FXStreet), normalize, cache, load.

Schema (one row per release):
    timestamp  UTC release time
    currency   ISO code (USD, EUR, ...)
    country    ISO2 country code when known
    event      release name as published by the source
    impact     high | medium | low | none
    actual     numeric print (NaN until released)
    forecast   consensus (NaN when the source has none)
    previous   prior print
    revised    revision of the prior print (NaN if none)
    unit       unit string when known
    period     reference period date (e.g. first day of the CPI month)
    event_id   stable id of the event series at the source
    preliminary True for flash / advance estimates (FXStreet isPreliminary)

A series is identified by (currency, country, event, preliminary): FXStreet
files member-state prints under the shared currency and lists flash and
final releases under the same name.

The same CSV also satisfies `src.features.news.load_news_calendar`, so one
file can drive both the blackout filter and the event study.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import ROOT

CALENDAR_COLUMNS = [
    "timestamp",
    "currency",
    "country",
    "event",
    "impact",
    "actual",
    "forecast",
    "previous",
    "revised",
    "unit",
    "period",
    "event_id",
    "preliminary",
]

DEFAULT_CALENDAR_PATH = ROOT / "data" / "events_calendar.csv"

FXSTREET_URL = "https://calendar-api.fxstreet.com/en/api/v1/eventDates/{start}/{end}"
FXSTREET_HEADERS = {
    "Referer": "https://www.fxstreet.com/",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (trade-research calendar fetch)",
}

_IMPACT_MAP = {
    "high": "high",
    "3": "high",
    "red": "high",
    "medium": "medium",
    "med": "medium",
    "2": "medium",
    "orange": "medium",
    "low": "low",
    "1": "low",
    "yellow": "low",
    "none": "none",
    "0": "none",
    "holiday": "none",
    "": "none",
}

_NUM_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*([kKmMbBtT%]?)\s*$")


def parse_number(value: Any) -> float:
    """Parse calendar numbers like '0.2%', '190K', '-775M', 1.5 -> float (NaN if blank).

    Suffixes K/M/B/T scale the value; '%' is dropped. Scale consistency within an
    event series is all that matters for standardized surprises.
    """
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value) if not (isinstance(value, float) and math.isnan(value)) else math.nan
    s = str(value).strip().replace(",", "")
    if s == "" or s.lower() in {"nan", "none", "null", "-", "—"}:
        return math.nan
    m = _NUM_RE.match(s)
    if not m:
        return math.nan
    num = float(m.group(1))
    suf = m.group(2).upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(suf, 1.0)
    return num * mult


def normalize_impact(value: Any) -> str:
    s = "" if value is None else str(value).strip().lower()
    return _IMPACT_MAP.get(s, "none" if s in ("", "nan") else s)


def to_utc_series(series: pd.Series) -> pd.Series:
    """Parse timestamps to UTC; tolerate mixed ISO layouts within one column."""
    try:
        return pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")


def _to_utc(series: pd.Series) -> pd.Series:
    return to_utc_series(series)


def normalize_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Map FXStreet / ForexFactory style frames onto CALENDAR_COLUMNS."""
    if df is None or df.empty:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    x = df.copy()
    x.columns = [str(c) for c in x.columns]
    lower = {c.lower(): c for c in x.columns}

    def col(*names: str) -> pd.Series | None:
        for n in names:
            if n.lower() in lower:
                return x[lower[n.lower()]]
        return None

    ts = col("timestamp", "dateUtc", "datetime", "date", "time")
    if ts is None:
        raise ValueError("Calendar needs a timestamp/dateUtc/date column")
    cur = col("currency", "currencyCode", "ccy", "country")
    if cur is None:
        raise ValueError("Calendar needs a currency column")
    out = pd.DataFrame(
        {
            "timestamp": _to_utc(ts),
            "currency": cur.astype(str).str.upper().str.strip(),
        }
    )
    country = col("country", "countryCode")
    out["country"] = country.astype(str).str.upper().str.strip() if country is not None else ""
    ev = col("event", "name", "title")
    out["event"] = ev.astype(str).str.strip() if ev is not None else ""
    imp = col("impact", "volatility", "importance")
    out["impact"] = imp.map(normalize_impact) if imp is not None else "none"
    for dst, names in (
        ("actual", ("actual",)),
        ("forecast", ("forecast", "consensus", "estimate")),
        ("previous", ("previous", "prior")),
        ("revised", ("revised", "revision")),
    ):
        src = col(*names)
        out[dst] = src.map(parse_number).astype(float) if src is not None else math.nan
    unit = col("unit")
    out["unit"] = unit.astype(str).where(unit.notna(), "") if unit is not None else ""
    period = col("period", "periodDateUtc", "period_date")
    out["period"] = _to_utc(period).dt.normalize().dt.tz_localize(None) if period is not None else pd.NaT
    eid = col("event_id", "eventId")
    out["event_id"] = eid.astype(str) if eid is not None else ""
    prelim = col("preliminary", "isPreliminary")
    if prelim is not None:
        out["preliminary"] = prelim.map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes", "t")
        ).astype(bool)
    else:
        out["preliminary"] = False

    out = out.dropna(subset=["timestamp"])
    out = out[out["currency"].str.len() == 3]
    out = out.drop_duplicates(subset=["timestamp", "currency", "country", "event"], keep="last")
    return out.sort_values(["timestamp", "currency", "country", "event"]).reset_index(drop=True)[CALENDAR_COLUMNS]


def _month_chunks(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        if cursor.month == 12:
            nxt = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
        yield cursor, min(nxt, end)
        cursor = nxt


def fetch_fxstreet_calendar(
    start: str | datetime,
    end: str | datetime,
    sleep_s: float = 0.5,
    retries: int = 4,
    timeout_s: int = 60,
    session: Any | None = None,
) -> pd.DataFrame:
    """Fetch FXStreet economic calendar in monthly chunks and normalize."""
    import requests

    s0 = pd.Timestamp(start, tz="UTC").to_pydatetime()
    s1 = pd.Timestamp(end, tz="UTC").to_pydatetime()
    sess = session or requests.Session()
    frames: list[pd.DataFrame] = []
    for a, b in _month_chunks(s0, s1):
        url = FXSTREET_URL.format(
            start=a.strftime("%Y-%m-%dT%H:%M:%SZ"), end=b.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        delay = 2.0
        last_err: Exception | None = None
        for _ in range(retries):
            try:
                r = sess.get(url, headers=FXSTREET_HEADERS, timeout=timeout_s)
                r.raise_for_status()
                rows = r.json()
                if rows:
                    frames.append(pd.DataFrame(rows))
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — retry on any transport/parse error
                last_err = exc
                time.sleep(delay)
                delay *= 2
        if last_err is not None:
            raise RuntimeError(f"FXStreet fetch failed for {a:%Y-%m}: {last_err}") from last_err
        if sleep_s > 0:
            time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    return normalize_calendar(pd.concat(frames, ignore_index=True))


def save_calendar(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    p = Path(path) if path else DEFAULT_CALENDAR_PATH
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"]).dt.strftime("%Y-%m-%d")
    if p.suffix.lower() == ".parquet":
        out.to_parquet(p, index=False)
    else:
        out.to_csv(p, index=False)
    return p


def load_event_calendar(path: Path | str | None = None) -> pd.DataFrame:
    """Load a cached calendar (CSV or parquet). Missing file -> empty frame."""
    p = Path(path) if path else DEFAULT_CALENDAR_PATH
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    if p.suffix.lower() == ".parquet":
        raw = pd.read_parquet(p)
    else:
        raw = pd.read_csv(p)
    return normalize_calendar(raw)


def merge_calendars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union two normalized calendars; newer rows win on (timestamp, currency, country, event)."""
    if existing is None or existing.empty:
        return normalize_calendar(new)
    both = pd.concat([existing, new], ignore_index=True)
    return normalize_calendar(both)
