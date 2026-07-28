from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.config import ROOT


def load_news_calendar(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["timestamp", "currency", "impact"])
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        # Optional: strategy runs without news filter if missing
        return pd.DataFrame(columns=["timestamp", "currency", "impact"])
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("datetime") or cols.get("date")
    cur_col = cols.get("currency") or cols.get("ccy")
    impact_col = cols.get("impact") or cols.get("importance")
    if ts_col is None or cur_col is None:
        raise ValueError("News CSV needs timestamp/datetime and currency columns")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], utc=True),
            "currency": df[cur_col].astype(str).str.upper().str.strip(),
            "impact": df[impact_col].astype(str).str.lower() if impact_col else "high",
        }
    )
    return out.sort_values("timestamp").reset_index(drop=True)


def pair_currencies(pair: str) -> set[str]:
    p = pair.upper()
    return {p[:3], p[3:6]}


def is_news_blackout(
    ts: pd.Timestamp,
    pair: str,
    calendar: pd.DataFrame,
    blackout_minutes: int = 60,
) -> bool:
    if calendar is None or calendar.empty:
        return False
    currencies = pair_currencies(pair) | {"USD"}
    window = timedelta(minutes=blackout_minutes)
    t0, t1 = ts - window, ts + window
    hit = calendar[
        (calendar["timestamp"] >= t0)
        & (calendar["timestamp"] <= t1)
        & (calendar["currency"].isin(currencies))
    ]
    if hit.empty:
        return False
    # Only high impact if column present
    if "impact" in hit.columns:
        high = hit["impact"].isin(["high", "3", "red"])
        if high.any():
            return True
        # if all impacts unknown, treat as blackout
        if hit["impact"].isna().all():
            return True
        return False
    return True


def write_empty_news_template(path: Path | None = None) -> Path:
    p = path or (ROOT / "data" / "news_calendar.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        pd.DataFrame(columns=["timestamp", "currency", "impact"]).to_csv(p, index=False)
    return p
