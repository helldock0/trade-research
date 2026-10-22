"""Macro series: fetch from free official sources, cache as CSV, load as panels.

Each series is cached at data/macro/{name}.csv with columns date,value. Sources:
    fred        FRED fredgraph CSV (id)
    ecb         ECB Data Portal SDMX csvdata (key, e.g. YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y)
    boc         Bank of Canada Valet CSV (series)
    rba         RBA statistical table CSV (table, series_id)
    mof         Japan MOF JGB yield CSVs (column, e.g. "2Y")
    boe         Bank of England IADB CSV (code)
    bis_policy  BIS WS_CBPOL daily policy rates (area, e.g. US, XM, GB, JP, AU, CA, CH, NZ)
    cftc_tff    CFTC Traders in Financial Futures yearly zips (market substring) ->
                leveraged-money net position as a share of open interest
    csv         local drop-in file (path) with date,value columns

Parsers are pure functions over text so they can be unit-tested offline.
"""

from __future__ import annotations

import io
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ROOT

DEFAULT_MACRO_DIR = ROOT / "data" / "macro"
UA = {"User-Agent": "Mozilla/5.0 (trade-research macro fetch)"}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/{key}?format=csvdata&startPeriod={start}"
BOC_URL = "https://www.bankofcanada.ca/valet/observations/{series}/csv?start_date={start}"
RBA_URL = "https://www.rba.gov.au/statistics/tables/csv/{table}-data.csv"
MOF_HIST_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
MOF_CUR_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"
BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom={start}&Dateto=now&SeriesCodes={code}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)
BIS_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.{area}?format=csv&startPeriod={start}"
CFTC_TFF_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"


# ----------------------------------------------------------------- helpers


def _series(dates: pd.Series | list, values: pd.Series | list, name: str) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates), errors="coerce"))
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    vals = pd.to_numeric(pd.Series(values), errors="coerce").values
    s = pd.Series(vals, index=idx.normalize(), name=name)
    s = s[~s.index.isna()].dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _get(url: str, retries: int = 3, timeout_s: int = 90) -> bytes:
    import requests

    delay = 2.0
    last: Exception | None = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout_s)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"fetch failed: {url}: {last}") from last


# ----------------------------------------------------------------- parsers


def parse_fred_csv(text: str, name: str = "fred") -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    return _series(df.iloc[:, 0], df.iloc[:, 1], name)


def parse_ecb_csv(text: str, name: str = "ecb") -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    return _series(df["TIME_PERIOD"], df["OBS_VALUE"], name)


def parse_boc_csv(text: str, name: str = "boc") -> pd.Series:
    lines = text.replace("﻿", "").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().strip('"') == "OBSERVATIONS")
    except StopIteration as exc:
        raise ValueError("Bank of Canada CSV: no OBSERVATIONS block") from exc
    df = pd.read_csv(io.StringIO("\n".join(lines[start + 1 :])))
    return _series(df.iloc[:, 0], df.iloc[:, 1], name)


def parse_rba_csv(text: str, series_id: str, name: str = "rba") -> pd.Series:
    """RBA tables: metadata rows, a 'Series ID' row naming columns, then dated rows."""
    lines = text.replace("﻿", "").splitlines()
    header_i = next((i for i, ln in enumerate(lines) if ln.startswith("Series ID")), None)
    if header_i is None:
        raise ValueError("RBA CSV: no 'Series ID' row")
    header = [h.strip() for h in lines[header_i].split(",")]
    if series_id not in header:
        raise ValueError(f"RBA CSV: series {series_id} not in {header}")
    col = header.index(series_id)
    dates, vals = [], []
    for ln in lines[header_i + 1 :]:
        parts = ln.split(",")
        if len(parts) <= col or not parts[0].strip():
            continue
        dates.append(parts[0].strip())
        vals.append(parts[col].strip())
    return _series(pd.to_datetime(pd.Series(dates), format="%d-%b-%Y", errors="coerce"), vals, name)


def parse_mof_csv(text: str, column: str = "2Y", name: str = "mof") -> pd.Series:
    lines = text.replace("﻿", "").splitlines()
    header_i = next((i for i, ln in enumerate(lines) if ln.startswith("Date,")), None)
    if header_i is None:
        raise ValueError("MOF CSV: no 'Date,' header")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_i:])))
    df.columns = [str(c).strip() for c in df.columns]
    if column not in df.columns:
        raise ValueError(f"MOF CSV: column {column} not in {list(df.columns)}")
    vals = df[column].astype(str).str.strip().replace({"-": None})
    return _series(pd.to_datetime(df["Date"].astype(str).str.strip(), format="%Y/%m/%d", errors="coerce"), vals, name)


def parse_boe_csv(text: str, code: str, name: str = "boe") -> pd.Series:
    df = pd.read_csv(io.StringIO(text.replace("﻿", "")))
    df.columns = [str(c).strip() for c in df.columns]
    if code not in df.columns:
        raise ValueError(f"BoE CSV: code {code} not in {list(df.columns)}")
    return _series(pd.to_datetime(df["DATE"].astype(str).str.strip(), format="%d %b %Y", errors="coerce"), df[code], name)


def parse_bis_csv(text: str, name: str = "bis") -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    return _series(df["TIME_PERIOD"], df["OBS_VALUE"], name)


def parse_cftc_tff(text: str, market: str, name: str = "cot") -> pd.Series:
    """Leveraged-money net (long - short) as a share of open interest, by report date."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    names = df["Market_and_Exchange_Names"].astype(str).str.upper()
    sub = df[names.str.contains(market.upper(), regex=False)]
    if sub.empty:
        return pd.Series(dtype=float, name=name)
    oi = pd.to_numeric(sub["Open_Interest_All"], errors="coerce")
    net = pd.to_numeric(sub["Lev_Money_Positions_Long_All"], errors="coerce") - pd.to_numeric(
        sub["Lev_Money_Positions_Short_All"], errors="coerce"
    )
    share = (net / oi.where(oi > 0)).astype(float)
    return _series(sub["Report_Date_as_YYYY-MM-DD"], share, name)


def parse_local_csv(text: str, name: str = "csv") -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    return _series(df.iloc[:, 0], df.iloc[:, 1], name)


# ----------------------------------------------------------------- fetchers


def fetch_series(spec: dict[str, Any], name: str, start: str = "2010-01-01") -> pd.Series:
    """Fetch one series described by a config spec; returns a UTC date-indexed Series."""
    src = str(spec.get("source", "")).lower()
    if src == "fred":
        return parse_fred_csv(_get(FRED_URL.format(id=spec["id"])).decode("utf-8"), name)
    if src == "ecb":
        return parse_ecb_csv(_get(ECB_URL.format(key=spec["key"], start=start)).decode("utf-8"), name)
    if src == "boc":
        return parse_boc_csv(_get(BOC_URL.format(series=spec["series"], start=start)).decode("utf-8-sig"), name)
    if src == "rba":
        return parse_rba_csv(_get(RBA_URL.format(table=spec.get("table", "f2"))).decode("utf-8-sig"), spec["series_id"], name)
    if src == "mof":
        hist = parse_mof_csv(_get(MOF_HIST_URL).decode("utf-8-sig", errors="replace"), spec.get("column", "2Y"), name)
        try:
            cur = parse_mof_csv(_get(MOF_CUR_URL).decode("utf-8-sig", errors="replace"), spec.get("column", "2Y"), name)
            hist = pd.concat([hist, cur])
        except Exception:  # noqa: BLE001 — current-month file is optional
            pass
        hist = hist[~hist.index.duplicated(keep="last")].sort_index()
        return hist.rename(name)
    if src == "boe":
        # The IADB endpoint intermittently answers with a redirect to an HTML
        # error page instead of the CSV (throttling); retry until a CSV arrives.
        d0 = pd.Timestamp(start).strftime("%d/%b/%Y")
        url = BOE_URL.format(start=d0, code=spec["code"])
        delay = float(spec.get("retry_delay_s", 10.0))
        last_text = ""
        for _ in range(int(spec.get("retries", 8))):
            last_text = _get(url).decode("utf-8-sig", errors="replace")
            if last_text.lstrip().upper().startswith("DATE"):
                return parse_boe_csv(last_text, spec["code"], name)
            time.sleep(delay)
            delay = min(delay * 1.5, 60.0)
        raise RuntimeError(f"BoE IADB returned no CSV for {spec['code']}: {last_text[:120]!r}")
    if src == "bis_policy":
        return parse_bis_csv(_get(BIS_URL.format(area=spec["area"], start=start)).decode("utf-8"), name)
    if src == "cftc_tff":
        y0 = int(spec.get("start_year", pd.Timestamp(start).year))
        y1 = int(spec.get("end_year", datetime.now(timezone.utc).year))
        parts = []
        for year in range(y0, y1 + 1):
            try:
                blob = _get(CFTC_TFF_URL.format(year=year))
            except RuntimeError:
                continue  # current year may not exist yet early in January
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
                txt = zf.read(member).decode("utf-8", errors="replace")
            parts.append(parse_cftc_tff(txt, spec["market"], name))
        if not parts:
            return pd.Series(dtype=float, name=name)
        s = pd.concat(parts)
        return s[~s.index.duplicated(keep="last")].sort_index()
    if src == "csv":
        p = Path(spec["path"])
        if not p.is_absolute():
            p = ROOT / p
        return parse_local_csv(p.read_text(encoding="utf-8-sig"), name)
    raise ValueError(f"Unknown macro source: {src!r} for {name}")


# ----------------------------------------------------------------- cache


def series_path(name: str, cache_dir: Path | None = None) -> Path:
    return (cache_dir or DEFAULT_MACRO_DIR) / f"{name}.csv"


def save_series(s: pd.Series, name: str, cache_dir: Path | None = None) -> Path:
    p = series_path(name, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "value": s.values})
    out.to_csv(p, index=False)
    return p


def load_series(name: str, cache_dir: Path | None = None) -> pd.Series:
    p = series_path(name, cache_dir)
    if not p.exists():
        return pd.Series(dtype=float, name=name)
    df = pd.read_csv(p)
    return _series(df.iloc[:, 0], df.iloc[:, 1], name)


def iter_series_specs(macro_cfg: dict[str, Any]) -> list[tuple[str, str, str | None, dict[str, Any]]]:
    """Flatten config into (cache_name, group, currency_or_None, spec)."""
    out: list[tuple[str, str, str | None, dict[str, Any]]] = []
    for group, entries in (macro_cfg.get("series") or {}).items():
        if group == "global":
            for key, spec in (entries or {}).items():
                out.append((f"global_{key}", group, None, dict(spec)))
        else:
            for cur, spec in (entries or {}).items():
                if spec is None or "source" not in spec:
                    continue  # derived entries (e.g. derive: neg_mean_others)
                out.append((f"{group}_{str(cur).upper()}", group, str(cur).upper(), dict(spec)))
    return out


def download_macro(macro_cfg: dict[str, Any], cache_dir: Path | None = None, force: bool = False, start: str = "2010-01-01") -> dict[str, Any]:
    """Fetch every configured series into the cache. Returns per-series status."""
    cache_dir = cache_dir or Path(macro_cfg.get("cache_dir", DEFAULT_MACRO_DIR))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    status: dict[str, Any] = {}
    for name, _group, _cur, spec in iter_series_specs(macro_cfg):
        p = series_path(name, cache_dir)
        if p.exists() and not force:
            s = load_series(name, cache_dir)
            status[name] = {"cached": True, "n": int(len(s)), "last": str(s.index[-1].date()) if len(s) else None}
            continue
        try:
            s = fetch_series(spec, name, start=start)
            save_series(s, name, cache_dir)
            status[name] = {"cached": False, "n": int(len(s)), "last": str(s.index[-1].date()) if len(s) else None}
        except Exception as exc:  # noqa: BLE001 — report and continue with other series
            status[name] = {"error": str(exc)}
    return status


def load_macro_panel(macro_cfg: dict[str, Any], cache_dir: Path | None = None) -> dict[str, pd.Series]:
    """Load all cached series named by the config. Missing series are omitted."""
    cache_dir = cache_dir or Path(macro_cfg.get("cache_dir", DEFAULT_MACRO_DIR))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    panel: dict[str, pd.Series] = {}
    for name, _group, _cur, _spec in iter_series_specs(macro_cfg):
        s = load_series(name, cache_dir)
        if not s.empty:
            panel[name] = s
    return panel
