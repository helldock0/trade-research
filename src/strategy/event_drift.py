"""Event drift — trade scheduled macro releases on a pre-release signal.

Hypothesis: the market prices an expected surprise *before* the print. The
pre-release signal is either surprise momentum (consensus anchoring makes
surprises autocorrelated), a point-in-time nowcast gap, or the combination.

Variants (config `signal.variant`):
    pre   enter N hours before the release, exit before the print (drift only)
    post  same entry, hold through the print, exit after it (drift + surprise)
    fade  enter after the print against the reaction when it went the signal's way
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.data.calendar import load_event_calendar, normalize_calendar
from src.features.event_table import build_event_table
from src.features.news import is_news_blackout
from src.features.nowcast import (
    attach_external_nowcast,
    attach_gasoline_nowcast,
    load_external_nowcast,
    load_fred_series,
)
from src.features.surprise import (
    add_nowcast_gap,
    add_standardized_surprises,
    add_surprise_momentum,
    filter_events,
    polarity_for_rows,
    select_signal,
)
from src.strategy.cssm import Signal

VARIANTS = ("pre", "post", "fade")


def load_calendar_for(strategy_cfg: dict[str, Any], calendar_df: pd.DataFrame | None = None) -> pd.DataFrame:
    ev_cfg = strategy_cfg.get("events", {})
    if calendar_df is not None:
        return normalize_calendar(calendar_df)
    return load_event_calendar(ev_cfg.get("calendar_csv"))


def prepare_events(cal: pd.DataFrame, strategy_cfg: dict[str, Any]) -> pd.DataFrame:
    """Filter configured releases and attach point-in-time signal columns."""
    sig = strategy_cfg.get("signal", {})
    ev_cfg = strategy_cfg.get("events", {})
    now_cfg = strategy_cfg.get("nowcast", {}) or {}
    if cal is None or cal.empty:
        return pd.DataFrame(
            columns=["timestamp", "currency", "event", "actual", "forecast", "surprise_z", "polarity", "signal_z"]
        )
    ev = filter_events(cal, ev_cfg.get("list"), str(ev_cfg.get("min_impact", "high")))
    ev = add_standardized_surprises(
        ev, min_history=int(sig.get("min_history", 6)), window=int(sig.get("std_window", 36))
    )
    ev = add_surprise_momentum(ev, k=int(sig.get("surprise_k", 2)))
    if now_cfg.get("enabled", False):
        gas = load_fred_series(str(now_cfg.get("gasoline_series", "GASREGW")))
        ev = attach_gasoline_nowcast(
            ev,
            gas,
            currency=str(now_cfg.get("currency", "USD")),
            event_name=str(now_cfg.get("cpi_event", "Consumer Price Index (MoM)")),
            weight=float(now_cfg.get("weight", 0.035)),
            baseline_n=int(now_cfg.get("baseline_n", 6)),
        )
    if now_cfg.get("external_csv"):
        ev = attach_external_nowcast(ev, load_external_nowcast(now_cfg["external_csv"]))
    ev = add_nowcast_gap(ev)
    ev["polarity"] = polarity_for_rows(ev, ev_cfg.get("list"), ev_cfg.get("inverted_patterns"))
    ev["signal_z"] = select_signal(ev, str(sig.get("signal_mode", "surprise_momentum")))
    min_abs = float(sig.get("min_abs_z", 0.0))
    ev = ev[ev["signal_z"].notna() & (ev["signal_z"].abs() >= min_abs) & (ev["signal_z"] != 0)]
    return ev.sort_values("timestamp").reset_index(drop=True)


def _dedupe_entries(table: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """One signal per (symbol, entry bar): keep the strongest |signal_z|."""
    if table.empty:
        return table
    x = table.copy()
    x["_abs"] = x["signal_z"].abs()
    x = x.sort_values(["symbol", label_col, "_abs"], ascending=[True, True, False])
    x = x.drop_duplicates(subset=["symbol", label_col], keep="first").drop(columns="_abs")
    return x.sort_values(label_col).reset_index(drop=True)


def signals_from_table(
    table: pd.DataFrame,
    variant: str,
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    calendar: pd.DataFrame | None = None,
) -> list[Signal]:
    variant = str(variant).lower()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
    sig = strategy_cfg.get("signal", {})
    tp_r = float(sig.get("take_profit_r", 50.0))
    blackout_min = int(sig.get("entry_blackout_min", 0) or 0)
    high_cal = None
    if calendar is not None and not calendar.empty and blackout_min > 0 and variant != "fade":
        high_cal = calendar[calendar["impact"].eq("high")][["timestamp", "currency", "impact"]]

    if table is None or table.empty:
        return []

    if variant == "pre":
        rows = table[table["pre_clean"].astype(bool)]
        label_col, px_col, exit_col, side_col = "entry_label", "entry_px", "pre_exit_time", "side"
    elif variant == "post":
        rows = table[table["post_label"].notna()]
        label_col, px_col, exit_col, side_col = "entry_label", "entry_px", "post_exit_time", "side"
    else:
        rows = table[(table["fade_side"] != 0) & table["fade_entry_label"].notna()]
        label_col, px_col, exit_col, side_col = "fade_entry_label", "fade_entry_px", "fade_exit_time", "fade_side"

    rows = _dedupe_entries(rows, label_col)
    out: list[Signal] = []
    for r in rows.itertuples(index=False):
        side = int(getattr(r, side_col))
        if side == 0:
            continue
        ts = pd.Timestamp(getattr(r, label_col))
        entry = float(getattr(r, px_col))
        risk = float(r.stop_dist)
        if math.isnan(entry) or math.isnan(risk) or risk <= 0:
            continue
        if high_cal is not None and is_news_blackout(ts, r.symbol, high_cal, blackout_min):
            continue
        stop = entry - side * risk
        tp = entry + side * tp_r * risk
        out.append(
            Signal(
                timestamp=ts,
                symbol=str(r.symbol),
                side=side,
                entry=entry,
                stop=stop,
                take_profit=tp,
                risk_price=risk,
                session=f"event_{variant}",
                range_high=entry + risk,
                range_low=entry - risk,
                reason=f"event_{variant}|{r.currency}|{r.event}|z={float(r.signal_z):+.2f}",
                exit_time=pd.Timestamp(getattr(r, exit_col)),
            )
        )
    out.sort(key=lambda s: s.timestamp)
    return out


def generate_signals(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    calendar_df: pd.DataFrame | None = None,
) -> list[Signal]:
    """Registry entry point: build the event table and emit the configured variant."""
    cal = load_calendar_for(strategy_cfg, calendar_df)
    events = prepare_events(cal, strategy_cfg)
    table = build_event_table(events, bars_by_symbol, strategy_cfg, costs_cfg)
    variant = str(strategy_cfg.get("signal", {}).get("variant", "pre"))
    return signals_from_table(table, variant, strategy_cfg, costs_cfg, cal)
