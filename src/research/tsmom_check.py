"""Time-series momentum signal check on daily history (no bars).

Per instrument, at each month end m: position = sign(log return over the past
`lookback` daily closes); the holding return is the log return to the next
month end. Net = position x return - (1 bp a side per unit of position change)
- financing for the month; then divided by the trailing `vol_window`-day
volatility scaled to a month, so every instrument contributes in risk units.
The portfolio is the mean across instruments available that calendar month
(rows are aligned by month, since markets end their months on different
days). The final (partial) month of each series is dropped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INDEX_SYMBOLS = {"US500", "NAS100", "US30", "US2000", "GER40", "UK100", "JP225"}


def monthly_tsmom(
    close: pd.Series,
    lookback: int = 252,
    vol_window: int = 60,
    cost_bp: float = 2.0,
    financing_pct: float = 0.0,
) -> pd.DataFrame:
    """Rows = decision month ends: position, holding return, net and vol-scaled net."""
    c = pd.Series(close, dtype=float).dropna()
    c = c[c > 0].sort_index()
    if c.index.tz is not None:
        c.index = c.index.tz_localize(None)
    logc = np.log(c)
    daily = logc.diff()
    past = logc - logc.shift(int(lookback))
    vol = daily.rolling(int(vol_window), min_periods=int(vol_window) // 2).std() * math.sqrt(21)
    month = c.index.to_period("M")
    ends = pd.Series(c.index, index=c.index).groupby(month).max().to_numpy()
    ends = pd.DatetimeIndex(ends)[:-1]                     # the last month may be partial
    t = pd.DataFrame(index=ends)
    t["pos"] = np.sign(past.reindex(ends))
    t["vol"] = vol.reindex(ends)
    lc = logc.reindex(ends)
    t["ret"] = lc.shift(-1) - lc
    t = t.dropna(subset=["pos", "vol", "ret"])
    t["pos"] = t["pos"].astype(int)
    change = (t["pos"] - t["pos"].shift(1).fillna(0)).abs()
    t["gross"] = t["pos"] * t["ret"]
    t["net"] = t["gross"] - change * (cost_bp / 2.0) / 1e4 - t["pos"].abs() * financing_pct / 100.0 / 12.0
    t["scaled"] = t["net"] / t["vol"].where(t["vol"] > 0)
    return t


def _stats(x: pd.Series) -> dict[str, Any]:
    v = pd.Series(x, dtype=float).dropna()
    n = int(len(v))
    if n < 3:
        return {"n": n, "mean": float("nan"), "t": 0.0, "sharpe": float("nan"), "hit": float("nan")}
    sd = float(v.std(ddof=1))
    return {
        "n": n,
        "mean": float(v.mean()),
        "t": float(v.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0,
        "sharpe": float(v.mean() / sd * math.sqrt(12)) if sd > 0 else float("nan"),
        "hit": float((v > 0).mean()),
    }


def _cut(x: pd.Series, start: str | None, end: str | None) -> pd.Series:
    if start is not None:
        x = x[x.index >= pd.Timestamp(start)]
    if end is not None:
        x = x[x.index <= pd.Timestamp(end)]
    return x


def run_tsmom_check(history: dict[str, pd.Series], cfg: dict[str, Any]) -> dict[str, Any]:
    sig, val = cfg["signal"], cfg["validation"]
    per: dict[str, pd.Series] = {}
    rows = []
    for sym, name in val["history"].items():
        s = history.get(name)
        if s is None or s.empty:
            rows.append({"symbol": sym, "series": name, "missing": True})
            continue
        fin = float(val["index_financing_pct_per_year"] if sym in INDEX_SYMBOLS else val["fx_financing_pct_per_year"])
        t = monthly_tsmom(s, int(sig.get("lookback_days", 252)), int(val.get("vol_window_days", 60)), float(val.get("round_trip_cost_bp", 2.0)), fin)
        # align by calendar month, not date: markets end their months on different days (holidays, time zones)
        per[sym] = t["scaled"].set_axis(t.index.to_period("M").to_timestamp())
        rows.append({
            "symbol": sym, "series": name, "missing": False, "first": str(t.index.min().date()) if len(t) else None,
            "holdout": _stats(_cut(t["scaled"], *val["holdout"])), "dev": _stats(_cut(t["scaled"], *val["dev"])),
        })
    panel = pd.DataFrame(per).sort_index()
    port = panel.mean(axis=1, skipna=True).dropna()
    counts = panel.notna().sum(axis=1)
    out = {
        "rows": rows,
        "portfolio": {label: {**_stats(_cut(port, *val[label])), "avg_instruments": float(_cut(counts, *val[label]).mean())} for label in ("holdout", "dev")},
        "by_decade": [],
        "by_year_dev": [],
    }
    for dec in sorted({d.year // 10 * 10 for d in port.index}):
        out["by_decade"].append({"decade": dec, **_stats(port[(port.index.year >= dec) & (port.index.year < dec + 10)])})
    for y, g in _cut(port, *val["dev"]).groupby(lambda d: d.year):
        out["by_year_dev"].append({"year": int(y), **_stats(g), "sum": float(g.sum())})
    h, d = out["portfolio"]["holdout"], out["portfolio"]["dev"]
    min_t = float(val.get("min_t", 2.0))
    ok_h = bool(h["mean"] > 0 and h["t"] >= min_t)
    ok_d = bool(d["mean"] > 0)
    out["verdict"] = {"verdict": "PASS" if ok_h and ok_d else "NO-GO", "holdout_ok": ok_h, "dev_ok": ok_d,
                      "rule": f"holdout portfolio mean > 0 with t >= {min_t}; dev portfolio mean > 0"}
    return out


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_markdown(res: dict[str, Any], cfg: dict[str, Any]) -> str:
    val, v = cfg["validation"], res["verdict"]
    lines = ["# Time-series momentum signal check (daily history, monthly, vol-scaled)", ""]
    lines.append(f"- holdout {val['holdout']} · dev {val['dev']} · returns in monthly-volatility units, net of {val['round_trip_cost_bp']} bp round trips and financing")
    lines.append(f"- rule (fixed before the run): {v['rule']}")
    lines += ["", f"**{v['verdict']}**: holdout {'ok' if v['holdout_ok'] else 'fails'}, dev {'ok' if v['dev_ok'] else 'fails'}.", ""]
    lines.append("| portfolio | months | avg markets | mean per month | t | Sharpe (yr) | hit |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in ("holdout", "dev"):
        p = res["portfolio"][label]
        lines.append(f"| {label} | {p['n']} | {_f(p['avg_instruments'], 1)} | {_f(p['mean'])} | {_f(p['t'], 2)} | {_f(p['sharpe'], 2)} | {_f(p['hit'], 2)} |")
    lines += ["", "## Per market", "", "| market | history | from | holdout mean | t | dev mean | t |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if r.get("missing"):
            lines.append(f"| {r['symbol']} | {r['series']} | missing | | | | |")
            continue
        lines.append(f"| {r['symbol']} | {r['series']} | {r['first']} | {_f(r['holdout']['mean'])} | {_f(r['holdout']['t'], 2)} | {_f(r['dev']['mean'])} | {_f(r['dev']['t'], 2)} |")
    lines += ["", "## Portfolio by decade", "", "| decade | months | mean | t | Sharpe |", "|---|---|---|---|---|"]
    for d in res["by_decade"]:
        lines.append(f"| {d['decade']}s | {d['n']} | {_f(d['mean'])} | {_f(d['t'], 2)} | {_f(d['sharpe'], 2)} |")
    lines += ["", "## Dev window by year", "", "| year | months | sum (vol units) | hit |", "|---|---|---|---|"]
    for y in res["by_year_dev"]:
        lines.append(f"| {y['year']} | {y['n']} | {_f(y['sum'], 2)} | {_f(y['hit'], 2)} |")
    lines.append("")
    return "\n".join(lines)


def write_report(res: dict[str, Any], cfg: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "tsmom_check.json"
    p_json.write_text(json.dumps(res, indent=2, default=lambda o: None), encoding="utf-8")
    p_md = out_dir / "tsmom_check.md"
    p_md.write_text(render_markdown(res, cfg), encoding="utf-8")
    return {"json": p_json, "md": p_md}
