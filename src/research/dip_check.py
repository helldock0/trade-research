"""Dip-buying signal check on FRED daily closes: trade returns vs simply holding in an uptrend.

Trades come from src.strategy.dip.dip_trades on the full series (so the
averages are warm) and are assigned to a window by entry date. Net return =
log(exit / entry) - round-trip spread (configs/costs.yaml, `cost_symbol`) -
financing per calendar day held. Excess = net - (mean daily log return on the
window's uptrend days) x trading days held. Trades never overlap, so a simple
t on the per-trade excess is fair.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import pair_cost, pip_size
from src.strategy.dip import dip_trades


def _t(x: pd.Series) -> float:
    v = pd.Series(x, dtype=float).dropna()
    if len(v) < 3 or v.std(ddof=1) == 0:
        return 0.0
    return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v))))


def trade_table(close: pd.Series, cfg: dict[str, Any], costs_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """All trades with gross / net log returns, plus the daily returns flagged as uptrend days."""
    sig, val = cfg["signal"], cfg["validation"]
    c = pd.Series(close, dtype=float).dropna()
    c = c[c > 0]
    c.index = pd.DatetimeIndex(c.index).tz_localize(None) if c.index.tz is not None else c.index
    tr = dip_trades(c, sig)
    fin = float(val.get("financing_pct_per_year", 0.0)) / 100.0
    tr["gross"] = np.log(tr["exit_px"] / tr["entry_px"])
    tr["cal_days"] = (tr["exit_date"] - tr["entry_date"]).dt.days
    if val.get("round_trip_cost_bp") is not None:
        # spreads scale with the index level: a fixed fraction of price
        cost = float(val["round_trip_cost_bp"]) / 1e4
    else:
        # fixed index points from costs.yaml: only right near today's price level
        sym = val.get("cost_symbol")
        pts = 2.0 * sum(pair_cost(sym, costs_cfg)[k] for k in ("half_spread_pips", "slippage_pips")) * pip_size(sym, costs_cfg) if sym else 0.0
        cost = pts / tr["entry_px"]
    tr["net"] = tr["gross"] - cost - fin * tr["cal_days"] / 365.0
    ret = np.log(c / c.shift(1))
    up = (c > c.rolling(int(sig.get("trend_sma", 200))).mean()).shift(1)  # the day's return earned while in an uptrend
    return tr, ret[up.fillna(False).astype(bool)]


def window_stats(tr: pd.DataFrame, up_ret: pd.Series, start: str | None, end: str | None) -> dict[str, Any]:
    def cut(x, col=None):
        idx = x[col] if col else x.index
        m = pd.Series(True, index=x.index)
        if start is not None:
            m &= idx >= pd.Timestamp(start)
        if end is not None:
            m &= idx <= pd.Timestamp(end)
        return x[m]

    t, u = cut(tr, "entry_date"), cut(up_ret)
    base = float(u.mean()) if len(u) else float("nan")
    ex = t["net"] - base * t["days"]
    years = max((pd.Timestamp(end or u.index.max()) - pd.Timestamp(start or u.index.min())).days / 365.25, 1e-9) if len(u) else float("nan")
    return {
        "n": int(len(t)),
        "per_year": float(len(t) / years) if len(u) else float("nan"),
        "mean_net_pct": float(t["net"].mean() * 100) if len(t) else float("nan"),
        "mean_gross_pct": float(t["gross"].mean() * 100) if len(t) else float("nan"),
        "hit": float((t["net"] > 0).mean()) if len(t) else float("nan"),
        "avg_days": float(t["days"].mean()) if len(t) else float("nan"),
        "baseline_pct_per_day": base * 100 if not math.isnan(base) else float("nan"),
        "mean_excess_pct": float(ex.mean() * 100) if len(t) else float("nan"),
        "t": _t(ex),
        "worst_pct": float(t["net"].min() * 100) if len(t) else float("nan"),
    }


def run_dip_check(series: dict[str, pd.Series], cfg: dict[str, Any], costs_cfg: dict[str, Any]) -> dict[str, Any]:
    val = cfg["validation"]
    names = [n for n in val["series"] if n in series]
    rows, decades = [], []
    for name in names:
        tr, up = trade_table(series[name], cfg, costs_cfg)
        row: dict[str, Any] = {"series": name}
        for label in ("context", "holdout", "dev"):
            row[label] = window_stats(tr, up, *val[label])
        rows.append(row)
        for dec in sorted({d.year // 10 * 10 for d in tr["entry_date"]}):
            decades.append({"series": name, "decade": dec, **window_stats(tr, up, f"{dec}-01-01", f"{dec + 9}-12-31")})
    primary = next((r for r in rows if r["series"] == val["series"][0]), None)
    min_t = float(val.get("min_t", 2.0))
    hold_ok = bool(primary and primary["holdout"]["mean_excess_pct"] > 0 and primary["holdout"]["t"] >= min_t)
    dev_ok = bool(rows) and all(r["dev"]["mean_excess_pct"] > 0 for r in rows)
    return {
        "rows": rows,
        "decades": decades,
        "verdict": {
            "verdict": "PASS" if hold_ok and dev_ok else "NO-GO",
            "holdout_ok": hold_ok,
            "dev_ok": dev_ok,
            "rule": f"holdout mean excess > 0 with t >= {min_t} on {val['series'][0]}; dev mean excess > 0 on every series",
        },
    }


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_markdown(res: dict[str, Any], cfg: dict[str, Any]) -> str:
    v, val = res["verdict"], cfg["validation"]
    lines = ["# Buy-the-dip signal check (daily closes, net of spread and financing)", ""]
    lines.append(f"- windows: context {val['context']} · holdout {val['holdout']} · dev {val['dev']}")
    lines.append(f"- rule (fixed before the run): {v['rule']}")
    lines.append("- excess = net trade return - mean uptrend-day return x trading days held")
    lines += ["", f"**{v['verdict']}**: holdout {'ok' if v['holdout_ok'] else 'fails'}, dev {'ok' if v['dev_ok'] else 'fails'}.", ""]
    lines.append("| series | window | trades | per year | net % | gross % | hit | avg days | hold-baseline %/day | excess % | t | worst % |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        for label in ("context", "holdout", "dev"):
            s = r[label]
            lines.append(
                f"| {r['series']} | {label} | {s['n']} | {_f(s['per_year'], 1)} | {_f(s['mean_net_pct'])} | {_f(s['mean_gross_pct'])} | {_f(s['hit'])} | "
                f"{_f(s['avg_days'], 1)} | {_f(s['baseline_pct_per_day'], 3)} | {_f(s['mean_excess_pct'])} | {_f(s['t'])} | {_f(s['worst_pct'])} |"
            )
    lines += ["", "## By decade", "", "| series | decade | trades | net % | hit | excess % | t |", "|---|---|---|---|---|---|---|"]
    for d in res["decades"]:
        lines.append(f"| {d['series']} | {d['decade']}s | {d['n']} | {_f(d['mean_net_pct'])} | {_f(d['hit'])} | {_f(d['mean_excess_pct'])} | {_f(d['t'])} |")
    lines.append("")
    return "\n".join(lines)


def write_report(res: dict[str, Any], cfg: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "dip_check.json"
    p_json.write_text(json.dumps(res, indent=2, default=lambda o: None), encoding="utf-8")
    p_md = out_dir / "dip_check.md"
    p_md.write_text(render_markdown(res, cfg), encoding="utf-8")
    return {"json": p_json, "md": p_md}
