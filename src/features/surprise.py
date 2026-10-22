"""Point-in-time standardized surprises and pre-release signals.

All quantities used as a *signal* for release i are computed from releases
strictly before i within the same (currency, event) series, so nothing known
only at or after the print leaks into the decision.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

GROUP_KEYS = ["currency", "country", "event", "preliminary"]

# Events where a higher-than-expected print is *bad* for the currency.
DEFAULT_INVERTED_PATTERNS = (
    r"unemployment",
    r"jobless",
    r"claims",
    r"budget deficit",
    r"trade deficit",
)


def event_polarity(event_name: str, inverted_patterns: list[str] | tuple[str, ...] | None = None) -> int:
    """+1 if a positive surprise is bullish for the currency, -1 if bearish."""
    pats = tuple(inverted_patterns) if inverted_patterns is not None else DEFAULT_INVERTED_PATTERNS
    name = (event_name or "").lower()
    for p in pats:
        if re.search(p, name):
            return -1
    return 1


def ensure_series_keys(cal: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the series key columns exist (older CSVs lack country/preliminary)."""
    x = cal.copy()
    if "country" not in x.columns:
        x["country"] = ""
    if "preliminary" not in x.columns:
        x["preliminary"] = False
    x["country"] = x["country"].fillna("").astype(str)
    x["preliminary"] = x["preliminary"].fillna(False).astype(bool)
    return x


def add_standardized_surprises(
    cal: pd.DataFrame,
    min_history: int = 6,
    window: int = 36,
) -> pd.DataFrame:
    """Add surprise, hist_std (point-in-time), surprise_z columns.

    hist_std for release i is the std of surprises of releases < i in the same
    series (rolling `window`, at least `min_history` observations).
    """
    x = ensure_series_keys(cal).sort_values(GROUP_KEYS + ["timestamp"]).reset_index(drop=True)
    x["surprise"] = x["actual"] - x["forecast"]
    g = x.groupby(GROUP_KEYS, sort=False)["surprise"]
    x["hist_std"] = g.transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_history).std()
    )
    x["hist_mean"] = g.transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_history).mean()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = x["surprise"] / x["hist_std"]
    x["surprise_z"] = z.where(x["hist_std"] > 0)
    return x


def add_surprise_momentum(cal: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Mean of the previous k standardized surprises (known before release i)."""
    x = ensure_series_keys(cal)
    if "surprise_z" not in x.columns:
        x = add_standardized_surprises(x)
    x = x.sort_values(GROUP_KEYS + ["timestamp"]).reset_index(drop=True)
    g = x.groupby(GROUP_KEYS, sort=False)["surprise_z"]
    x["surprise_mom"] = g.transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean())
    x["n_prior"] = g.transform(lambda s: s.shift(1).notna().cumsum())
    return x


def add_nowcast_gap(cal: pd.DataFrame) -> pd.DataFrame:
    """(nowcast - forecast) / hist_std when a `nowcast` column is present."""
    x = cal.copy()
    if "hist_std" not in x.columns:
        x = add_standardized_surprises(x)
    if "nowcast" not in x.columns:
        x["nowcast"] = math.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = (x["nowcast"] - x["forecast"]) / x["hist_std"]
    x["nowcast_gap_z"] = gap.where(x["hist_std"] > 0)
    return x


def select_signal(cal: pd.DataFrame, mode: str = "surprise_momentum") -> pd.Series:
    """Pick the pre-release signal column according to mode."""
    mode = (mode or "surprise_momentum").lower()
    if mode == "surprise_momentum":
        return cal["surprise_mom"]
    if mode == "nowcast_gap":
        return cal["nowcast_gap_z"]
    if mode == "combined":
        return cal["nowcast_gap_z"].where(cal["nowcast_gap_z"].notna(), cal["surprise_mom"])
    raise ValueError(f"Unknown signal mode: {mode}")


def _entry_mask(x: pd.DataFrame, e: dict[str, Any]) -> pd.Series:
    """Rows matching one config entry: currency + name, optional country / preliminary."""
    cur = str(e.get("currency", "")).upper()
    name = str(e.get("name", "")).strip().lower()
    m = x["currency"].eq(cur) & x["event"].str.strip().str.lower().eq(name)
    if e.get("country") is not None:
        m &= x["country"].astype(str).str.upper().eq(str(e["country"]).upper())
    if e.get("preliminary") is not None:
        m &= x["preliminary"].astype(bool).eq(bool(e["preliminary"]))
    return m


def filter_events(cal: pd.DataFrame, events_cfg: list[dict[str, Any]] | None, min_impact: str = "high") -> pd.DataFrame:
    """Keep configured series (currency + name, optional country / preliminary); or all at/above min_impact."""
    x = ensure_series_keys(cal)
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    if events_cfg:
        keep = pd.Series(False, index=x.index)
        for e in events_cfg:
            keep |= _entry_mask(x, e)
        x = x[keep]
    else:
        x = x[x["impact"].map(lambda v: rank.get(str(v), 0)) >= rank.get(min_impact, 3)]
    return x.reset_index(drop=True)


def polarity_for_rows(cal: pd.DataFrame, events_cfg: list[dict[str, Any]] | None, inverted_patterns=None) -> pd.Series:
    """Per-row polarity: explicit config override, else pattern default."""
    x = ensure_series_keys(cal)
    out = pd.Series([event_polarity(n, inverted_patterns) for n in x["event"]], index=x.index, dtype=int)
    for e in events_cfg or []:
        if "polarity" in e:
            out[_entry_mask(x, e)] = int(e["polarity"])
    return out
