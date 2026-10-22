"""One account running several frozen strategy legs (configs/portfolio_combo.yaml).

Each leg brings a trade list with net R per trade, already charged with its own costs. The account
risks a fixed share of the initial balance per 1R on every leg (no compounding) and books P&L at
each exit. It is judged by the repo's promotion gates: the folds are consecutive calendar windows
(nothing is trained), and the Monte Carlo and FTMO replay are the same as elsewhere.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import Trade
from src.prop.ftmo_simulator import simulate_ftmo_path
from src.research.promote import evaluate_promotion
from src.research.walk_forward import FoldResult

COLS = ["leg", "symbol", "side", "entry_time", "exit_time", "r"]


def _naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert("UTC").dt.tz_localize(None) if s.dt.tz is not None else s


def leg_frame(leg: str, symbol: str, tr: pd.DataFrame, window: list[str] | None = None) -> pd.DataFrame:
    """Normalise one leg's trades to COLS (naive UTC times), keeping entries inside the window."""
    if tr is None or not len(tr):
        return pd.DataFrame(columns=COLS)
    r = tr["r"] if "r" in tr else tr["r_multiple"]
    out = pd.DataFrame({"leg": leg, "symbol": symbol, "side": tr["side"].astype(int).to_numpy(),
                        "entry_time": _naive_utc(tr["entry_time"]).to_numpy(), "exit_time": _naive_utc(tr["exit_time"]).to_numpy(),
                        "r": r.astype(float).to_numpy()})
    if window:
        a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1]) + pd.Timedelta(days=1)
        out = out[(out["entry_time"] >= a) & (out["entry_time"] < b)]
    return out.reset_index(drop=True)


def combine(legs: list[pd.DataFrame], skip_same_symbol_side: bool = False) -> pd.DataFrame:
    """All legs' trades by entry time; optionally skip one while a kept trade holds the same symbol and side."""
    tr = pd.concat([x for x in legs if len(x)], ignore_index=True) if any(len(x) for x in legs) else pd.DataFrame(columns=COLS)
    tr = tr.sort_values(["entry_time", "leg"], kind="stable").reset_index(drop=True)
    if not skip_same_symbol_side or not len(tr):
        return tr
    keep: list[int] = []
    open_: list[int] = []
    for i, t in tr.iterrows():
        open_ = [j for j in open_ if tr.at[j, "exit_time"] > t["entry_time"]]
        if any(tr.at[j, "symbol"] == t["symbol"] and tr.at[j, "side"] == t["side"] for j in open_):
            continue
        keep.append(i)
        open_.append(i)
    return tr.loc[keep].reset_index(drop=True)


def to_engine_trades(tr: pd.DataFrame, risk_pct: float, initial: float) -> list[Trade]:
    """Engine Trade records in exit order (P&L = R x risk), for the gates and the FTMO replay."""
    risk = initial * risk_pct / 100.0
    out = []
    for t in tr.sort_values("exit_time", kind="stable").itertuples():
        out.append(Trade(symbol=t.symbol, side=int(t.side), entry_time=pd.Timestamp(t.entry_time, tz="UTC"),
                         exit_time=pd.Timestamp(t.exit_time, tz="UTC"), entry_price=0.0, exit_price=0.0, stop=0.0,
                         take_profit=0.0, lots=0.0, pnl_usd=float(t.r) * risk, r_multiple=float(t.r), session=t.leg,
                         exit_reason="", commission=0.0))
    return out


def metrics(tr: pd.DataFrame, risk_pct: float) -> dict[str, Any]:
    """Per-trade and closed-equity stats; equity in % of the initial balance, booked at exits."""
    if not len(tr):
        return {"n_trades": 0, "expectancy_r": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0,
                "total_r": 0.0, "t": 0.0}
    x = tr.sort_values("exit_time", kind="stable")
    r = x["r"].to_numpy(float)
    eq = np.cumsum(r) * risk_pct
    dd = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return {
        "n_trades": len(r),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0),
        "win_rate": float((r > 0).mean()),
        "max_dd_pct": float(dd.max()) if len(dd) else 0.0,
        "total_r": float(r.sum()),
        "t": float(r.mean() / (sd / math.sqrt(len(r)))) if sd > 0 else 0.0,
    }


def calendar_folds(tr: pd.DataFrame, start: str, end: str, months: int, risk_pct: float) -> list[FoldResult]:
    """Consecutive `months`-long test windows by exit time; a fold is profitable when its R sums above 0."""
    folds, a, end_ts, k = [], pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), 0
    while a < end_ts:
        b = min(a + pd.DateOffset(months=months), end_ts)
        w = tr[(tr["exit_time"] >= a) & (tr["exit_time"] < b)]
        m = metrics(w, risk_pct)
        folds.append(FoldResult(k, a, a, a, b, {}, m, m["n_trades"], m["total_r"] > 0))
        a, k = b, k + 1
    return folds


def monthly(tr: pd.DataFrame, risk_pct: float) -> pd.DataFrame:
    """Return % per calendar month (by exit) for each leg and the total."""
    if not len(tr):
        return pd.DataFrame()
    x = tr.assign(month=tr["exit_time"].dt.to_period("M"), pct=tr["r"] * risk_pct)
    m = x.pivot_table(index="month", columns="leg", values="pct", aggfunc="sum").fillna(0.0)
    full = pd.period_range(m.index.min(), m.index.max(), freq="M")
    m = m.reindex(full, fill_value=0.0)
    m["total"] = m.sum(axis=1)
    return m


def ftmo_starts(trades: list[Trade], ftmo: dict[str, Any]) -> dict[str, Any]:
    """FTMO 2-step replayed from the first trade (in exit order) of every calendar month."""
    if not trades:
        return {"starts": 0, "pass_rate": 0.0, "fail_rate": 0.0, "median_days_to_pass": None}
    seen, starts = set(), []
    for i, t in enumerate(trades):
        key = (t.exit_time.year, t.exit_time.month)
        if key not in seen:
            seen.add(key)
            starts.append(i)
    res, days = [], []
    for i in starts:
        sim = simulate_ftmo_path(trades[i:], ftmo)
        if sim.overall_pass:
            res.append("pass")
            used = sim.phase1.trades_used + (sim.phase2.trades_used if sim.phase2 else 0)
            days.append((trades[i + used - 1].exit_time - trades[i].entry_time) / pd.Timedelta(days=1))
        elif "profit_target_not_reached" in str(sim.phase1.breach or "") or (sim.phase2 and "profit_target_not_reached" in str(sim.phase2.breach or "")):
            res.append("open")
        else:
            res.append("fail")
    n = len(res)
    return {"starts": n, "pass_rate": res.count("pass") / n, "fail_rate": res.count("fail") / n,
            "open_rate": res.count("open") / n, "median_days_to_pass": float(np.median(days)) if days else None}


def evaluate(tr: pd.DataFrame, cfg: dict[str, Any], ftmo: dict[str, Any] | None) -> dict[str, Any]:
    risk, init = float(cfg["risk_per_trade_pct"]), float(cfg["initial_balance"])
    a, b = cfg["window"]
    folds = calendar_folds(tr, a, b, int(cfg.get("folds_months", 6)), risk)
    trades = to_engine_trades(tr, risk, init)
    promo = evaluate_promotion(folds, trades)
    mo = monthly(tr, risk)
    years = (pd.Timestamp(b) - pd.Timestamp(a)) / pd.Timedelta(days=365.25)
    full = metrics(tr, risk)
    out = {
        "metrics": full,
        "trades_per_year": full["n_trades"] / years,
        "return_pct_per_year": full["total_r"] * risk / years,
        "months_positive": float((mo["total"] > 0).mean()) if len(mo) else 0.0,
        "worst_month_pct": float(mo["total"].min()) if len(mo) else 0.0,
        "verdict": promo["verdict"],
        "checks": promo["checks"],
        "oos": {k: v for k, v in promo["oos"].items() if k != "folds"},
        "folds": [{"start": str(f.test_start.date()), "n": f.test_trades, "total_r": f.test_metrics["total_r"]} for f in folds],
        "monte_carlo": promo["monte_carlo"],
        "ftmo": ftmo_starts(trades, ftmo) if ftmo else None,
        "by_year": {int(y): float(g["total"].sum()) for y, g in mo.groupby(mo.index.year)} if len(mo) else {},
    }
    return out


def run_portfolio(legs: dict[str, pd.DataFrame], cfg: dict[str, Any], ftmo: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every variant on the combined list, each leg alone, and the legs' monthly correlation."""
    risk = float(cfg["risk_per_trade_pct"])
    frames = [legs[k] for k in cfg["legs"] if k in legs]
    rows = {}
    for name, over in (cfg.get("variants") or {"independent": {}}).items():
        tr = combine(frames, bool((over or {}).get("skip_same_symbol_side")))
        rows[name] = evaluate(tr, cfg, ftmo) | {"by_leg_n": tr["leg"].value_counts().to_dict()}
    alone = {k: evaluate(legs[k], cfg, ftmo) for k in cfg["legs"] if k in legs}
    mo = monthly(combine(frames), risk)
    corr = mo.drop(columns="total").corr() if len(mo) else pd.DataFrame()
    return {"variants": rows, "legs": alone, "corr": {a: {b: float(corr.at[a, b]) for b in corr.columns} for a in corr.index}}


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _ok(b: bool) -> str:
    return "pass" if b else "FAIL"


def render_markdown(res: dict[str, Any], cfg: dict[str, Any]) -> str:
    risk = cfg["risk_per_trade_pct"]
    lines = ["# Portfolio: " + " + ".join(cfg["legs"]), "",
             (f"Rules: configs/portfolio_combo.yaml (fixed before the run). {risk}% of the initial balance per 1R on every leg, "
              f"trades entered {cfg['window'][0]} → {cfg['window'][1]}, each leg net of its own costs."), ""]
    lines += ["## Summary", "", "| book | trades | trades/yr | avg R | t | PF | win | total R | return %/yr | max DD % | months + | worst month % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    books = [(f"**{k}**", v) for k, v in res["variants"].items()] + [(f"{k} alone", v) for k, v in res["legs"].items()]
    for name, v in books:
        m = v["metrics"]
        lines.append(f"| {name} | {m['n_trades']} | {_f(v['trades_per_year'], 0)} | {_f(m['expectancy_r'])} | {_f(m['t'], 2)} | "
                     f"{_f(m['profit_factor'], 2)} | {_f(m['win_rate'], 2)} | {_f(m['total_r'], 1)} | {_f(v['return_pct_per_year'], 1)} | "
                     f"{_f(m['max_dd_pct'], 1)} | {_f(v['months_positive'], 2)} | {_f(v['worst_month_pct'], 1)} |")
    lines += ["", "## Promotion gates (6-month folds, Monte Carlo, FTMO)", "",
              "| book | fold exp R | fold PF | fold DD % | folds + | MC success | FTMO pass (from each month) | median days | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, v in books:
        o, c, mc, ft = v["oos"], v["checks"], v["monte_carlo"], v.get("ftmo") or {}
        lines.append(f"| {name} | {_f(o['expectancy_r'])} {_ok(c['expectancy_r'])} | {_f(o['profit_factor'], 2)} {_ok(c['profit_factor'])} | "
                     f"{_f(o['max_dd_pct'], 1)} {_ok(c['max_dd_pct'])} | {o['profitable_folds']}/{o['n_folds']} {_ok(c['profitable_folds'] and c['n_folds'])} | "
                     f"{_f(mc['success_rate'], 2)} {_ok(c['monte_carlo_success'])} | {_f(ft.get('pass_rate'), 2)} | {_f(ft.get('median_days_to_pass'), 0)} | **{v['verdict']}** |")
    lines += ["", ("Gates: mean fold expectancy > 0.15R, mean fold PF > 1.25, worst fold DD < 8%, ≥ 3 profitable folds of ≥ 4, "
                   "Monte Carlo success ≥ 0.80."), ""]
    lines += ["## By year (return %)", "", "| year | " + " | ".join(n for n, _ in books) + " |", "|---|" + "---|" * len(books)]
    years = sorted({y for _, v in books for y in v["by_year"]})
    for y in years:
        lines.append(f"| {y} | " + " | ".join(_f(v["by_year"].get(y, 0.0), 1) for _, v in books) + " |")
    if res["corr"]:
        legs = list(res["corr"])
        lines += ["", "## Monthly correlation between legs", "", "| | " + " | ".join(legs) + " |", "|---|" + "---|" * len(legs)]
        for a in legs:
            lines.append(f"| {a} | " + " | ".join(_f(res["corr"][a][b], 2) for b in legs) + " |")
    lines.append("")
    return "\n".join(lines)


def write_report(res: dict[str, Any], cfg: dict[str, Any], out_dir: Path, stem: str = "portfolio_combo") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / f"{stem}.json", "md": out_dir / f"{stem}.md"}
    paths["json"].write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    paths["md"].write_text(render_markdown(res, cfg), encoding="utf-8")
    return paths
