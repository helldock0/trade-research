"""Relative-value signal check on FRED daily rates: band mean-reversion trades, net of costs.

Trades come from src.strategy.rv.band_trades on each cross (built from its USD
legs) and are assigned to a window by entry date. Net = side x log(exit /
entry) - round-trip cost - financing per calendar day held. Within a pair
trades never overlap; the pooled t treats the pairs' trades as one sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.rv import band_trades


def pair_trades(close: pd.Series, sig: dict[str, Any], cost_bp: float, financing_pct: float) -> pd.DataFrame:
    c = pd.Series(close, dtype=float).dropna()
    c = c[c > 0]
    if c.index.tz is not None:
        c.index = c.index.tz_localize(None)
    tr = band_trades(c, sig)
    tr["gross"] = tr["side"] * np.log(tr["exit_px"] / tr["entry_px"])
    tr["cal_days"] = (tr["exit_date"] - tr["entry_date"]).dt.days
    tr["net"] = tr["gross"] - cost_bp / 1e4 - financing_pct / 100.0 * tr["cal_days"] / 365.0
    return tr


def _stats(x: pd.Series) -> dict[str, Any]:
    v = pd.Series(x, dtype=float).dropna()
    n = int(len(v))
    if n < 3:
        return {"n": n, "mean_pct": float("nan"), "t": 0.0, "hit": float("nan"), "worst_pct": float("nan")}
    sd = float(v.std(ddof=1))
    return {
        "n": n,
        "mean_pct": float(v.mean() * 100),
        "t": float(v.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0,
        "hit": float((v > 0).mean()),
        "worst_pct": float(v.min() * 100),
    }


def _cut(tr: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    m = pd.Series(True, index=tr.index)
    if start is not None:
        m &= tr["entry_date"] >= pd.Timestamp(start)
    if end is not None:
        m &= tr["entry_date"] <= pd.Timestamp(end)
    return tr[m]


def run_rv_check(fx: dict[str, pd.Series], cfg: dict[str, Any]) -> dict[str, Any]:
    sig, val = cfg["signal"], cfg["validation"]
    fin = float(val.get("financing_pct_per_year", 0.0))
    rows, hold_all, dev_all, years = [], [], [], []
    for pair, pv in val["pairs"].items():
        s = fx.get(pair)
        if s is None or s.empty:
            rows.append({"pair": pair, "missing": True})
            continue
        tr = pair_trades(s, sig, float(pv.get("round_trip_cost_bp", 2.0)), fin)
        h, d = _cut(tr, *pv["holdout"]), _cut(tr, *val["dev"])
        hold_all.append(h)
        dev_all.append(d)
        rows.append({"pair": pair, "missing": False, "first": str(s.index.min().date()), "holdout_window": pv["holdout"],
                     "holdout": {**_stats(h["net"]), "avg_days": float(h["days"].mean()) if len(h) else float("nan")},
                     "dev": {**_stats(d["net"]), "avg_days": float(d["days"].mean()) if len(d) else float("nan")},
                     "by_decade": [{"decade": dec, **_stats(g["net"])} for dec, g in tr.groupby(tr["entry_date"].dt.year // 10 * 10)]})
        for y, g in d.groupby(d["entry_date"].dt.year):
            years.append({"pair": pair, "year": int(y), "n": int(len(g)), "sum_pct": float(g["net"].sum() * 100)})
    ph = _stats(pd.concat(hold_all)["net"]) if hold_all else _stats(pd.Series(dtype=float))
    pd_ = _stats(pd.concat(dev_all)["net"]) if dev_all else _stats(pd.Series(dtype=float))
    min_t = float(val.get("min_t", 2.0))
    ok_h = bool(ph["n"] > 2 and ph["mean_pct"] > 0 and ph["t"] >= min_t)
    ok_d = bool(pd_["n"] > 2 and pd_["mean_pct"] > 0)
    return {
        "rows": rows,
        "pooled": {"holdout": ph, "dev": pd_},
        "dev_by_year": years,
        "verdict": {"verdict": "PASS" if ok_h and ok_d else "NO-GO", "holdout_ok": ok_h, "dev_ok": ok_d,
                    "rule": f"pooled holdout mean net > 0 with t >= {min_t}; pooled dev mean net > 0"},
    }


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_markdown(res: dict[str, Any], cfg: dict[str, Any]) -> str:
    v = res["verdict"]
    lines = ["# Relative-value signal check (FRED daily, band mean reversion, net of costs)", ""]
    lines.append(f"- rule (fixed before the run): {v['rule']}")
    lines += ["", f"**{v['verdict']}**: holdout {'ok' if v['holdout_ok'] else 'fails'}, dev {'ok' if v['dev_ok'] else 'fails'}.", ""]
    lines.append("| pooled | trades | mean net % | t | hit | worst % |")
    lines.append("|---|---|---|---|---|---|")
    for label in ("holdout", "dev"):
        p = res["pooled"][label]
        lines.append(f"| {label} | {p['n']} | {_f(p['mean_pct'], 3)} | {_f(p['t'])} | {_f(p['hit'])} | {_f(p['worst_pct'])} |")
    lines += ["", "## Per pair", "", "| pair | holdout window | trades | mean net % | t | hit | avg days | dev trades | dev mean % | t | hit |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if r.get("missing"):
            lines.append(f"| {r['pair']} | missing | | | | | | | | | |")
            continue
        h, d = r["holdout"], r["dev"]
        lines.append(
            f"| {r['pair']} | {r['holdout_window'][0]} → {r['holdout_window'][1]} | {h['n']} | {_f(h['mean_pct'], 3)} | {_f(h['t'])} | {_f(h['hit'])} | {_f(h['avg_days'], 1)} | "
            f"{d['n']} | {_f(d['mean_pct'], 3)} | {_f(d['t'])} | {_f(d['hit'])} |"
        )
    lines += ["", "## By decade (all trades)", "", "| pair | decade | trades | mean net % | t | hit |", "|---|---|---|---|---|---|"]
    for r in res["rows"]:
        for dd in r.get("by_decade", []):
            lines.append(f"| {r['pair']} | {dd['decade']}s | {dd['n']} | {_f(dd['mean_pct'], 3)} | {_f(dd['t'])} | {_f(dd['hit'])} |")
    lines += ["", "## Dev by year", "", "| pair | year | trades | sum net % |", "|---|---|---|---|"]
    for y in res["dev_by_year"]:
        lines.append(f"| {y['pair']} | {y['year']} | {y['n']} | {_f(y['sum_pct'])} |")
    lines.append("")
    return "\n".join(lines)


def write_report(res: dict[str, Any], cfg: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "rv_check.json"
    p_json.write_text(json.dumps(res, indent=2, default=lambda o: None), encoding="utf-8")
    p_md = out_dir / "rv_check.md"
    p_md.write_text(render_markdown(res, cfg), encoding="utf-8")
    return {"json": p_json, "md": p_md}
