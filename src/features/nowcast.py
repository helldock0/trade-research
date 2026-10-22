"""Point-in-time headline CPI nowcast from retail gasoline prices.

Model (parameter-light, no fitting):
    nowcast_mom = trailing_mean(prior headline MoM prints, n)
                + weight * (gas_mom - trailing_mean(prior gas_mom, n))

gas_mom is the % change of the average weekly retail gasoline price in the
reference month vs the prior month, using only observations dated before the
release. `weight` approximates the motor-fuel share of the CPI basket.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd

from src.config import ROOT
from src.data.calendar import to_utc_series

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DEFAULT_MACRO_DIR = ROOT / "data" / "macro"


def macro_path(series_id: str, cache_dir: Path | None = None) -> Path:
    base = cache_dir or DEFAULT_MACRO_DIR
    return base / f"{series_id.upper()}.csv"


def fetch_fred_series(series_id: str, cache_dir: Path | None = None, force: bool = False, retries: int = 3) -> pd.Series:
    """Download a FRED series as a date-indexed float Series, caching to CSV."""
    import requests

    p = macro_path(series_id, cache_dir)
    if p.exists() and not force:
        return load_fred_series(series_id, cache_dir)
    url = FRED_CSV_URL.format(series=series_id.upper())
    delay = 2.0
    last: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(r.text, encoding="utf-8")
            last = None
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(delay)
            delay *= 2
    if last is not None:
        raise RuntimeError(f"FRED fetch failed for {series_id}: {last}") from last
    return load_fred_series(series_id, cache_dir)


def load_fred_series(series_id: str, cache_dir: Path | None = None) -> pd.Series:
    p = macro_path(series_id, cache_dir)
    if not p.exists():
        return pd.Series(dtype=float, name=series_id.upper())
    df = pd.read_csv(p)
    date_col = df.columns[0]
    val_col = df.columns[1]
    idx = pd.DatetimeIndex(pd.to_datetime(df[date_col], errors="coerce"))
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=idx,
        name=series_id.upper(),
    )
    s = s[~s.index.isna()]
    return s.dropna().sort_index()


def month_average_before(series: pd.Series, month_start: pd.Timestamp, asof: pd.Timestamp) -> float:
    """Average of observations inside [month_start, month_end] dated strictly before `asof`."""
    if series is None or series.empty:
        return math.nan
    m0 = pd.Timestamp(month_start)
    m0 = m0.tz_localize("UTC") if m0.tzinfo is None else m0.tz_convert("UTC")
    m1 = m0 + pd.offsets.MonthBegin(1)
    cut = pd.Timestamp(asof)
    cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
    win = series[(series.index >= m0) & (series.index < m1) & (series.index < cut)]
    if win.empty:
        return math.nan
    return float(win.mean())


def gasoline_mom_for_events(events: pd.DataFrame, gas: pd.Series) -> pd.Series:
    """Per-event gasoline MoM (%) for the reference month, point-in-time."""
    out = []
    for ts, period in zip(events["timestamp"], events["period"]):
        if pd.isna(period):
            out.append(math.nan)
            continue
        m = pd.Timestamp(period).normalize().replace(day=1)
        cur = month_average_before(gas, m, ts)
        prev = month_average_before(gas, m - pd.offsets.MonthBegin(1), ts)
        if math.isnan(cur) or math.isnan(prev) or prev <= 0:
            out.append(math.nan)
        else:
            out.append((cur / prev - 1.0) * 100.0)
    return pd.Series(out, index=events.index, dtype=float)


def gasoline_cpi_nowcast(
    cpi_events: pd.DataFrame,
    gas: pd.Series,
    weight: float = 0.035,
    baseline_n: int = 6,
) -> pd.Series:
    """Nowcast headline CPI MoM for each event row (same units as `actual`).

    Only prior prints and gasoline observations before each release are used.
    """
    x = cpi_events.sort_values("timestamp").copy()
    x["gas_mom"] = gasoline_mom_for_events(x, gas)
    base = x["actual"].shift(1).rolling(baseline_n, min_periods=max(2, baseline_n // 2)).mean()
    gas_base = x["gas_mom"].shift(1).rolling(baseline_n, min_periods=max(2, baseline_n // 2)).mean()
    now = base + weight * (x["gas_mom"] - gas_base)
    return now.reindex(cpi_events.index)


def attach_gasoline_nowcast(
    cal: pd.DataFrame,
    gas: pd.Series,
    currency: str = "USD",
    event_name: str = "Consumer Price Index (MoM)",
    weight: float = 0.035,
    baseline_n: int = 6,
) -> pd.DataFrame:
    """Return calendar with a `nowcast` column filled for the headline CPI series."""
    x = cal.copy()
    if "nowcast" not in x.columns:
        x["nowcast"] = math.nan
    mask = x["currency"].eq(currency.upper()) & x["event"].str.strip().str.lower().eq(event_name.strip().lower())
    if mask.any() and gas is not None and not gas.empty:
        now = gasoline_cpi_nowcast(x[mask], gas, weight=weight, baseline_n=baseline_n)
        x.loc[mask, "nowcast"] = now.values
    return x


def load_external_nowcast(path: Path | str | None) -> pd.DataFrame:
    """Load a user-supplied nowcast table (CSV/parquet).

    Columns: timestamp (release time, UTC), currency, event, nowcast; optional
    country and preliminary. Each row must be a forecast that existed *before*
    the release it is attached to (e.g. a Cleveland Fed vintage or a
    prediction-market implied value captured the day before).
    """
    if path is None:
        return pd.DataFrame(columns=["timestamp", "currency", "event", "nowcast"])
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "currency", "event", "nowcast"])
    raw = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    raw.columns = [str(c).lower() for c in raw.columns]
    need = {"timestamp", "currency", "event", "nowcast"}
    if not need <= set(raw.columns):
        raise ValueError(f"External nowcast needs columns {sorted(need)}; got {list(raw.columns)}")
    out = pd.DataFrame(
        {
            "timestamp": to_utc_series(raw["timestamp"]),
            "currency": raw["currency"].astype(str).str.upper().str.strip(),
            "event": raw["event"].astype(str).str.strip(),
            "nowcast": pd.to_numeric(raw["nowcast"], errors="coerce"),
        }
    )
    if "country" in raw.columns:
        out["country"] = raw["country"].astype(str).str.upper().str.strip()
    if "preliminary" in raw.columns:
        out["preliminary"] = raw["preliminary"].map(lambda v: str(v).strip().lower() in ("true", "1", "yes", "t"))
    return out.dropna(subset=["timestamp", "nowcast"]).reset_index(drop=True)


def attach_external_nowcast(cal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Fill `nowcast` from an external table on (timestamp, currency, event[, country, preliminary]).

    External values override any model nowcast already present for the same row.
    """
    x = cal.copy()
    if "nowcast" not in x.columns:
        x["nowcast"] = math.nan
    if external is None or external.empty:
        return x
    keys = ["timestamp", "currency", "event"]
    for k in ("country", "preliminary"):
        if k in external.columns and k in x.columns:
            keys.append(k)
    ext = external.drop_duplicates(subset=keys, keep="last")[keys + ["nowcast"]].rename(columns={"nowcast": "_ext"})
    left = x.reset_index().merge(ext, on=keys, how="left").set_index("index")
    left.index.name = None
    x["nowcast"] = left["_ext"].where(left["_ext"].notna(), x["nowcast"]).values
    return x
