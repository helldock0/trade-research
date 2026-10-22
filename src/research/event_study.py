"""Event study: raw (unstopped) returns per variant plus engine runs with costs.

Raw returns answer "is there anything here" with one number per event and no
path dependence. Engine runs answer "does it survive stops, costs, one position
at a time and the FTMO rules". Both come from the same event table.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.features.event_table import build_event_table
from src.prop.ftmo_simulator import simulate_ftmo_path
from src.strategy.event_drift import (
    VARIANTS,
    load_calendar_for,
    prepare_events,
    signals_from_table,
)


def _stats(x: pd.Series) -> dict[str, Any]:
    v = pd.Series(x, dtype=float).dropna()
    n = int(len(v))
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "hit_rate": 0.0, "t_stat": 0.0, "sum": 0.0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    t = float(mean / (std / math.sqrt(n))) if (n > 1 and std > 0) else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": float(v.median()),
        "std": std,
        "hit_rate": float((v > 0).mean()),
        "t_stat": t,
        "sum": float(v.sum()),
    }


def _forecast_stats(table: pd.DataFrame) -> dict[str, Any]:
    """Does the pre-release signal predict the realized surprise and the reaction?"""
    out: dict[str, Any] = {}
    fz = table[["signal_z", "surprise_z"]].dropna()
    fz = fz[(fz["signal_z"] != 0) & (fz["surprise_z"] != 0)]
    out["n_surprise"] = int(len(fz))
    out["surprise_hit_rate"] = float((np.sign(fz["signal_z"]) == np.sign(fz["surprise_z"])).mean()) if len(fz) else 0.0
    out["corr_signal_surprise"] = float(fz["signal_z"].corr(fz["surprise_z"])) if len(fz) > 2 else 0.0
    pm = table["pred_move_atr"].dropna()
    out["reaction"] = _stats(pm)
    return out


def forecast_quality(events: pd.DataFrame) -> dict[str, Any]:
    """Calendar-only check: does the pre-release signal predict the realized surprise?

    Works without price bars. `events` is the output of prepare_events.
    """
    cols = ["timestamp", "currency", "country", "event", "preliminary", "signal_z", "surprise_z"]
    if events is None or events.empty or any(c not in events.columns for c in cols):
        return {"n": 0, "surprise_hit_rate": 0.0, "corr_signal_surprise": 0.0, "by_event": []}
    x = events[cols].dropna()
    x = x[(x["signal_z"] != 0) & (x["surprise_z"] != 0)]
    hit = (np.sign(x["signal_z"]) == np.sign(x["surprise_z"])).astype(float)
    out: dict[str, Any] = {
        "n": int(len(x)),
        "surprise_hit_rate": float(hit.mean()) if len(x) else 0.0,
        "hit_t_stat": _stats(hit - 0.5)["t_stat"] if len(x) > 1 else 0.0,
        "corr_signal_surprise": float(x["signal_z"].corr(x["surprise_z"])) if len(x) > 2 else 0.0,
        "by_event": [],
    }
    for (cur, country, ev, prelim), g in x.groupby(["currency", "country", "event", "preliminary"], sort=True):
        h = (np.sign(g["signal_z"]) == np.sign(g["surprise_z"])).astype(float)
        out["by_event"].append(
            {
                "currency": str(cur),
                "country": str(country),
                "event": str(ev),
                "preliminary": bool(prelim),
                "n": int(len(g)),
                "surprise_hit": float(h.mean()),
                "corr": float(g["signal_z"].corr(g["surprise_z"])) if len(g) > 2 else 0.0,
                "first": str(pd.Timestamp(g["timestamp"].min()).date()),
                "last": str(pd.Timestamp(g["timestamp"].max()).date()),
            }
        )
    return out


def render_forecast_markdown(fq: dict[str, Any], title: str = "Forecast quality (calendar only)") -> str:
    lines = [f"# {title}", ""]
    lines.append(
        f"- events: {fq['n']} · surprise direction hit rate: {_fmt(fq['surprise_hit_rate'])} "
        f"(t vs 0.5: {_fmt(fq.get('hit_t_stat', 0.0), 2)}) · corr(signal, surprise): {_fmt(fq['corr_signal_surprise'])}"
    )
    if fq.get("by_event"):
        lines += ["", "| currency | country | event | flash | n | hit | corr | first | last |", "|---|---|---|---|---|---|---|---|---|"]
        for r in fq["by_event"]:
            lines.append(
                f"| {r['currency']} | {r.get('country', '')} | {r['event']} | {'y' if r.get('preliminary') else ''} | {r['n']} | "
                f"{_fmt(r['surprise_hit'], 2)} | {_fmt(r['corr'], 2)} | {r['first']} | {r['last']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _by_group(table: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    if table.empty:
        return rows
    for key, g in table.groupby(keys, sort=True):
        key_t = key if isinstance(key, tuple) else (key,)
        row = {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in zip(keys, key_t)}
        row["n"] = int(len(g))
        for v in VARIANTS:
            s = _stats(g[f"{v}_r"])
            row[f"{v}_n"] = s["n"]
            row[f"{v}_mean_r"] = s["mean"]
            row[f"{v}_hit"] = s["hit_rate"]
            row[f"{v}_t"] = s["t_stat"]
        fz = g[["signal_z", "surprise_z"]].dropna()
        fz = fz[(fz["signal_z"] != 0) & (fz["surprise_z"] != 0)]
        row["surprise_hit"] = float((np.sign(fz["signal_z"]) == np.sign(fz["surprise_z"])).mean()) if len(fz) else 0.0
        rows.append(row)
    return rows


def summarize_event_table(table: pd.DataFrame) -> dict[str, Any]:
    if table is None or table.empty:
        return {"n_rows": 0, "variants": {v: _stats(pd.Series(dtype=float)) for v in VARIANTS}}
    out: dict[str, Any] = {"n_rows": int(len(table)), "variants": {}}
    for v in VARIANTS:
        out["variants"][v] = {
            "r": _stats(table[f"{v}_r"]),
            "atr": _stats(table[f"{v}_atr"]),
        }
    out["forecast"] = _forecast_stats(table)
    out["by_event"] = _by_group(table, ["currency", "country", "event", "preliminary"])
    x = table.copy()
    x["year"] = pd.to_datetime(x["timestamp"], utc=True).dt.year
    out["by_year"] = _by_group(x, ["year"])
    med = float(x["signal_z"].abs().median()) if len(x) else 0.0
    x["z_bucket"] = np.where(x["signal_z"].abs() >= med, "strong", "weak")
    out["abs_z_median"] = med
    out["by_z_bucket"] = _by_group(x, ["z_bucket"])
    return out


def run_event_study(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    ftmo_cfg: dict[str, Any] | None = None,
    calendar_df: pd.DataFrame | None = None,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    cal = load_calendar_for(strategy_cfg, calendar_df)
    events = prepare_events(cal, strategy_cfg)
    table = build_event_table(events, bars_by_symbol, strategy_cfg, costs_cfg)
    raw = summarize_event_table(table)

    engine: dict[str, Any] = {}
    trades: dict[str, pd.DataFrame] = {}
    for v in variants:
        cfg_v = deepcopy(strategy_cfg)
        cfg_v.setdefault("signal", {})["variant"] = v
        sigs = signals_from_table(table, v, cfg_v, costs_cfg, cal)
        res = run_backtest(sigs, bars_by_symbol, cfg_v, costs_cfg)
        row: dict[str, Any] = {"n_signals": len(sigs), "metrics": res.metrics}
        if ftmo_cfg:
            sim = simulate_ftmo_path(res.trades, ftmo_cfg)
            row["ftmo"] = {
                "overall_pass": sim.overall_pass,
                "notes": sim.notes,
                "phase1_breach": sim.phase1.breach,
                "phase2_breach": sim.phase2.breach if sim.phase2 else None,
            }
        engine[v] = row
        trades[v] = res.trades_frame()

    return {
        "n_calendar_rows": int(len(cal)),
        "n_events_with_signal": int(len(events)),
        "n_table_rows": int(len(table)),
        "raw": raw,
        "engine": engine,
        "table": table,
        "trades": trades,
    }


def _fmt(x: Any, nd: int = 3) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_markdown(result: dict[str, Any], title: str = "Event study") -> str:
    raw = result["raw"]
    lines = [f"# {title}", ""]
    lines.append(
        f"Calendar rows: {result['n_calendar_rows']} · events with signal: "
        f"{result['n_events_with_signal']} · event/pair rows: {result['n_table_rows']}"
    )
    lines += ["", "## Raw returns (no stops, net of round-trip cost, R units)", ""]
    lines.append("| variant | n | mean R | median R | hit | t-stat | sum R |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        s = raw.get("variants", {}).get(v, {}).get("r", {}) if raw.get("n_rows") else {}
        if not s:
            s = {"n": 0, "mean": 0.0, "median": 0.0, "hit_rate": 0.0, "t_stat": 0.0, "sum": 0.0}
        lines.append(
            f"| {v} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
            f"{_fmt(s['hit_rate'])} | {_fmt(s['t_stat'], 2)} | {_fmt(s['sum'], 2)} |"
        )
    fc = raw.get("forecast")
    if fc:
        lines += ["", "## Forecast quality", ""]
        lines.append(f"- surprise direction hit rate: {_fmt(fc['surprise_hit_rate'])} (n={fc['n_surprise']})")
        lines.append(f"- corr(signal, realized surprise): {_fmt(fc['corr_signal_surprise'])}")
        r = fc["reaction"]
        lines.append(
            f"- print reaction in signal direction (ATR): mean {_fmt(r['mean'])}, hit {_fmt(r['hit_rate'])}, "
            f"t {_fmt(r['t_stat'], 2)} (n={r['n']})"
        )
    eng = result.get("engine", {})
    if eng:
        lines += ["", "## Engine runs (stops, costs, one position, FTMO sim)", ""]
        lines.append("| variant | signals | trades | expectancy R | PF | win | max DD % | FTMO |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for v, row in eng.items():
            m = row["metrics"]
            ft = row.get("ftmo", {})
            ft_s = "pass" if ft.get("overall_pass") else (ft.get("phase1_breach") or ft.get("phase2_breach") or "-")
            lines.append(
                f"| {v} | {row['n_signals']} | {m.get('n_trades', 0)} | {_fmt(m.get('expectancy_r', 0.0))} | "
                f"{_fmt(m.get('profit_factor', 0.0), 2)} | {_fmt(m.get('win_rate', 0.0), 2)} | "
                f"{_fmt(m.get('max_dd_pct', 0.0), 2)} | {ft_s} |"
            )
    by_ev = raw.get("by_event") or []
    if by_ev:
        lines += ["", "## By event (mean R: pre / post / fade)", ""]
        lines.append("| currency | country | event | flash | n | pre | post | fade | surprise hit |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in by_ev:
            lines.append(
                f"| {r['currency']} | {r.get('country', '')} | {r['event']} | {'y' if r.get('preliminary') else ''} | {r['n']} | {_fmt(r['pre_mean_r'])} | "
                f"{_fmt(r['post_mean_r'])} | {_fmt(r['fade_mean_r'])} | {_fmt(r['surprise_hit'], 2)} |"
            )
    by_z = raw.get("by_z_bucket") or []
    if by_z:
        lines += ["", f"## By signal strength (|z| split at median {_fmt(raw.get('abs_z_median', 0.0), 2)})", ""]
        lines.append("| bucket | n | pre | post | fade |")
        lines.append("|---|---|---|---|---|")
        for r in by_z:
            lines.append(f"| {r['z_bucket']} | {r['n']} | {_fmt(r['pre_mean_r'])} | {_fmt(r['post_mean_r'])} | {_fmt(r['fade_mean_r'])} |")
    by_y = raw.get("by_year") or []
    if by_y:
        lines += ["", "## By year", ""]
        lines.append("| year | n | pre | post | fade |")
        lines.append("|---|---|---|---|---|")
        for r in by_y:
            lines.append(f"| {r['year']} | {r['n']} | {_fmt(r['pre_mean_r'])} | {_fmt(r['post_mean_r'])} | {_fmt(r['fade_mean_r'])} |")
    lines.append("")
    return "\n".join(lines)


def write_event_study_report(result: dict[str, Any], out_dir: Path, tag: str = "event_drift") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    payload = {k: v for k, v in result.items() if k not in ("table", "trades")}
    p_json = out_dir / f"event_study_{tag}.json"
    p_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths["json"] = p_json
    p_md = out_dir / f"event_study_{tag}.md"
    p_md.write_text(render_markdown(result, title=f"Event study: {tag}"), encoding="utf-8")
    paths["md"] = p_md
    table = result.get("table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        p_tab = out_dir / f"event_table_{tag}.csv"
        table.to_csv(p_tab, index=False)
        paths["table"] = p_tab
    for v, tf in (result.get("trades") or {}).items():
        if isinstance(tf, pd.DataFrame) and not tf.empty:
            p = out_dir / f"event_trades_{tag}_{v}.csv"
            tf.to_csv(p, index=False)
            paths[f"trades_{v}"] = p
    return paths
