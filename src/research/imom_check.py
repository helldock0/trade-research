"""Intraday-momentum signal check on bars: last-half-hour return in the signal's direction, net of costs.

No stops, one trade per day: the raw edge before the engine adds stops and
sizing. Per window (dev / holdout): mean net return, simple t (daily trades do
not overlap), hit rate, and the always-long placebo over the same half hour.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import pair_cost, pip_size
from src.strategy.imom import signal_frame


def round_trip_cost(symbol: str, price: pd.Series, costs_cfg: dict[str, Any]) -> pd.Series:
    """Spread + slippage on entry and exit, as a log-return fraction of the price."""
    c = pair_cost(symbol, costs_cfg)
    pts = 2.0 * (c["half_spread_pips"] + c["slippage_pips"]) * pip_size(symbol, costs_cfg)
    return pts / price


def _stats(x: pd.Series) -> dict[str, Any]:
    v = pd.Series(x, dtype=float).dropna()
    n = int(len(v))
    if n < 2:
        return {"n": n, "mean_bp": float("nan"), "t": 0.0, "hit": float("nan")}
    sd = float(v.std(ddof=1))
    return {
        "n": n,
        "mean_bp": float(v.mean() * 1e4),
        "t": float(v.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0,
        "hit": float((v > 0).mean()),
    }


def _window(t: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        t = t[t.index >= pd.Timestamp(start)]
    if end is not None:
        t = t[t.index <= pd.Timestamp(end)]
    return t


def net_returns(m15: pd.DataFrame, symbol: str, cfg: dict[str, Any], costs_cfg: dict[str, Any], calendar=None) -> pd.DataFrame:
    """Per traded day: gross and net last-half-hour return in the signal's direction, and always-long net."""
    t = signal_frame(m15, cfg, calendar)
    cost = round_trip_cost(symbol, t["entry"], costs_cfg)
    traded = t["side"] != 0
    out = pd.DataFrame(index=t.index)
    out["gross"] = (t["side"] * t["last_ret"]).where(traded)
    out["net"] = out["gross"] - cost.where(traded)
    out["long_net"] = t["last_ret"] - cost
    out["signal"] = t["signal"]
    out["last_ret"] = t["last_ret"]
    return out


def run_imom_check(bars: dict[str, pd.DataFrame], cfg: dict[str, Any], costs_cfg: dict[str, Any], calendar=None) -> dict[str, Any]:
    val = cfg["validation"]
    primary = cfg["universe"][0].upper()
    symbols = [primary] + [s.upper() for s in val.get("cross_checks", []) if s.upper() in bars]
    variants = {
        "day": {},
        "strong": {"strong_only": True},
        "first": {"mode": "first"},
    }
    rows, years = [], []
    for sym in symbols:
        for vname, over in variants.items():
            vcfg = deepcopy(cfg)
            vcfg["signal"].update(over)
            nr = net_returns(bars[sym], sym, vcfg, costs_cfg, calendar)
            row: dict[str, Any] = {"symbol": sym, "variant": vname}
            for label in ("dev", "holdout"):
                w = _window(nr, *val[label])
                row[label] = {**_stats(w["net"]), "gross_bp": float(w["gross"].mean() * 1e4), "long_net_bp": float(w["long_net"].mean() * 1e4), "long_t": _stats(w["long_net"])["t"], "corr": float(w["signal"].corr(w["last_ret"]))}
            rows.append(row)
            if sym == primary and vname == "day":
                for y, g in nr.groupby(nr.index.year):
                    years.append({"year": int(y), **_stats(g["net"]), "long_net_bp": float(g["long_net"].mean() * 1e4)})
    head = next(r for r in rows if r["symbol"] == primary and r["variant"] == "day")
    min_t = float(val.get("min_t", 2.0))
    checks = {
        "dev_positive": head["dev"]["mean_bp"] > 0,
        "holdout_positive_t": head["holdout"]["mean_bp"] > 0 and head["holdout"]["t"] >= min_t,
        "holdout_beats_long": head["holdout"]["mean_bp"] > head["holdout"]["long_net_bp"],
    }
    return {
        "primary": primary,
        "rows": rows,
        "by_year": years,
        "verdict": {"verdict": "PASS" if all(checks.values()) else "NO-GO", "checks": checks, "min_t": min_t, "dev": val["dev"], "holdout": val["holdout"]},
    }


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_markdown(res: dict[str, Any]) -> str:
    v = res["verdict"]
    lines = ["# Intraday momentum signal check (last half hour, net of costs)", ""]
    lines.append(f"- dev {v['dev'][0]} → {v['dev'][1]} · holdout {v['holdout'][0]} → {v['holdout'][1] or 'end'} · one trade per day, no stops")
    lines.append(
        f"- rule (fixed before the run), on {res['primary']} / `day`: dev net mean > 0; holdout net mean > 0 with t ≥ {v['min_t']}; "
        "holdout beats always-long over the same half hour"
    )
    ok = ", ".join(f"{k} {'ok' if b else 'fails'}" for k, b in v["checks"].items())
    lines += ["", f"**{v['verdict']}**: {ok}.", ""]
    lines.append("| symbol | variant | window | days | net bp | t | hit | gross bp | always-long net bp (t) | corr(signal, last 30m) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        for label in ("dev", "holdout"):
            s = r[label]
            lines.append(
                f"| {r['symbol']} | {r['variant']} | {label} | {s['n']} | {_f(s['mean_bp'])} | {_f(s['t'])} | {_f(s['hit'])} | "
                f"{_f(s['gross_bp'])} | {_f(s['long_net_bp'])} ({_f(s['long_t'])}) | {_f(s['corr'], 3)} |"
            )
    lines += ["", f"## By year ({res['primary']}, `day`)", "", "| year | days | net bp | t | hit | always-long net bp |", "|---|---|---|---|---|---|"]
    for y in res["by_year"]:
        lines.append(f"| {y['year']} | {y['n']} | {_f(y['mean_bp'])} | {_f(y['t'])} | {_f(y['hit'])} | {_f(y['long_net_bp'])} |")
    lines.append("")
    return "\n".join(lines)


def write_report(res: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "imom_check.json"
    p_json.write_text(json.dumps(res, indent=2, default=lambda o: None), encoding="utf-8")
    p_md = out_dir / "imom_check.md"
    p_md.write_text(render_markdown(res), encoding="utf-8")
    return {"json": p_json, "md": p_md}
