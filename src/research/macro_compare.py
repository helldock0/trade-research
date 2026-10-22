"""Compare macro_swing variants (trade management, triggers) on the same bars.

Each variant is a dict deep-merged over the strategy config (`variants:` in
configs/strategy_macro_swing.yaml). Per variant: full backtest, rolling
walk-forward, the promotion gates (incl. Monte Carlo) and the FTMO path
replay. Parameters are frozen; nothing is optimized.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.backtest.engine import Trade, run_backtest
from src.prop.ftmo_simulator import simulate_ftmo_path
from src.research.promote import evaluate_promotion
from src.research.walk_forward import walk_forward
from src.strategy.registry import resolve_signal_fn


def variants_universe(strategy_cfg: dict[str, Any], names: list[str] | None = None) -> list[str]:
    """Every pair any of the chosen variants trades (bars to load once)."""
    names = list(names or (strategy_cfg.get("variants") or {}))
    pairs = [p for n in names for c in [variant_config(strategy_cfg, n)] for p in [*c["universe"], *(c.get("conversion_pairs") or [])]]
    return list(dict.fromkeys(p.upper() for p in pairs))


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else deepcopy(v)
    return out


def variant_config(strategy_cfg: dict[str, Any], name: str, _chain: tuple[str, ...] = ()) -> dict[str, Any]:
    """Config for one variant; `extends: other` merges onto that variant instead of the base."""
    variants = strategy_cfg.get("variants") or {}
    if name not in variants:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(variants)}")
    if name in _chain:
        raise ValueError(f"variant cycle: {' -> '.join(_chain + (name,))}")
    over = dict(variants[name] or {})
    parent = over.pop("extends", None)
    if parent:
        base = variant_config(strategy_cfg, parent, _chain + (name,))
    else:
        base = {k: v for k, v in strategy_cfg.items() if k not in ("variants", "validation")}
    return deep_merge(base, over)


def max_concurrent(trades: list[Trade]) -> int:
    """Most positions open at once (an exit and an entry on the same bar do not overlap)."""
    events = [(pd.Timestamp(t.exit_time), 0) for t in trades] + [(pd.Timestamp(t.entry_time), 1) for t in trades]
    n = peak = 0
    for _, kind in sorted(events):
        n += 1 if kind else -1
        peak = max(peak, n)
    return peak


def trade_stats(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {"avg_hold_days": 0.0, "max_concurrent": 0, "exit_reasons": {}}
    hold = [(pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)) / pd.Timedelta(days=1) for t in trades]
    return {
        "avg_hold_days": float(sum(hold) / len(hold)),
        "max_concurrent": max_concurrent(trades),
        "exit_reasons": dict(Counter(t.exit_reason for t in trades).most_common()),
    }


def run_variant(
    name: str,
    cfg: dict[str, Any],
    bars: dict[str, pd.DataFrame],
    costs: dict[str, Any],
    ftmo: dict[str, Any] | None,
    signal_fn: Callable | None = None,
    train_months: int = 18,
    test_months: int = 6,
) -> dict[str, Any]:
    gen = signal_fn or resolve_signal_fn(cfg)
    signals = gen(bars, cfg, costs)
    full = run_backtest(signals, bars, cfg, costs)
    folds = walk_forward(bars, cfg, costs, train_months=train_months, test_months=test_months, signal_fn=gen)
    promo = evaluate_promotion(folds, full.trades)
    span = full.equity_curve.index.max() - full.equity_curve.index.min() if full.equity_curve is not None and len(full.equity_curve) else pd.Timedelta(0)
    years = span / pd.Timedelta(days=365.25)
    out: dict[str, Any] = {
        "variant": name,
        "universe": sorted(cfg.get("universe") or bars),
        "traded": sorted({t.symbol for t in full.trades}),
        "n_signals": len(signals),
        "metrics": full.metrics,
        "return_pct_per_year": float(full.metrics.get("total_return_pct", 0.0) / years) if years > 0 else 0.0,
        **trade_stats(full.trades),
        "verdict": promo["verdict"],
        "checks": promo["checks"],
        "oos": {k: v for k, v in promo["oos"].items() if k != "folds"},
        "monte_carlo": promo["monte_carlo"],
        "ftmo": None,
    }
    if ftmo is not None:
        sim = simulate_ftmo_path(full.trades, ftmo)
        out["ftmo"] = {
            "overall_pass": sim.overall_pass,
            "phase1_passed": sim.phase1.passed,
            "phase1_days": sim.phase1.trading_days,
            "phase1_breach": sim.phase1.breach,
            "notes": sim.notes,
        }
    return out


def compare_variants(
    strategy_cfg: dict[str, Any],
    bars: dict[str, pd.DataFrame],
    costs: dict[str, Any],
    ftmo: dict[str, Any] | None = None,
    names: list[str] | None = None,
    signal_fn: Callable | None = None,
) -> dict[str, Any]:
    names = list(names or (strategy_cfg.get("variants") or {}))
    rows = []
    for n in names:
        cfg = variant_config(strategy_cfg, n)
        # each variant trades its own universe: the signal derives its currencies from the bars it gets;
        # `conversion_pairs` ride along so the engine can convert non-USD P&L
        need = [*cfg["universe"], *(cfg.get("conversion_pairs") or [])]
        missing = [s for s in need if s not in bars]
        if missing:
            raise ValueError(f"variant {n}: no bars for {missing}")
        rows.append(run_variant(n, cfg, {s: bars[s] for s in dict.fromkeys(need)}, costs, ftmo, signal_fn))
    return {"strategy": strategy_cfg.get("strategy"), "variants": {n: (strategy_cfg.get("variants") or {})[n] for n in names}, "rows": rows}


# ----------------------------------------------------------------- report


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ok(b: bool) -> str:
    return "pass" if b else "FAIL"


def render_markdown(result: dict[str, Any], title: str = "macro_swing variant comparison") -> str:
    lines = [f"# {title}", ""]
    for n, over in result["variants"].items():
        lines.append(f"- `{n}`: {json.dumps(over, sort_keys=True)}")
    lines += ["", "## Full sample", ""]
    lines.append("| variant | signals | trades | exp R | PF | win | max DD % | return %/yr | avg hold d | max open |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in result["rows"]:
        m = r["metrics"]
        lines.append(
            f"| {r['variant']} | {r['n_signals']} | {m.get('n_trades', 0)} | {_f(m.get('expectancy_r'))} | "
            f"{_f(m.get('profit_factor'), 2)} | {_f(m.get('win_rate'), 2)} | {_f(m.get('max_dd_pct'), 1)} | "
            f"{_f(r['return_pct_per_year'], 2)} | {_f(r['avg_hold_days'], 1)} | {r['max_concurrent']} |"
        )
    lines += ["", "## Promotion gates (walk-forward OOS + Monte Carlo)", ""]
    lines.append("| variant | OOS exp R | OOS PF | OOS DD % | folds + | MC success | MC ruin | FTMO path | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in result["rows"]:
        o, mc, c = r["oos"], r["monte_carlo"], r["checks"]
        ft = r.get("ftmo") or {}
        ft_s = "—" if not ft else ("pass" if ft["overall_pass"] else f"fail ({ft.get('phase1_breach') or 'phase2'})")
        lines.append(
            f"| {r['variant']} | {_f(o['expectancy_r'])} {_ok(c['expectancy_r'])} | {_f(o['profit_factor'], 2)} {_ok(c['profit_factor'])} | "
            f"{_f(o['max_dd_pct'], 1)} {_ok(c['max_dd_pct'])} | {o['profitable_folds']}/{o['n_folds']} {_ok(c['profitable_folds'] and c['n_folds'])} | "
            f"{_f(mc['success_rate'], 2)} {_ok(c['monte_carlo_success'])} | {_f(mc.get('ruin_rate'), 2)} | {ft_s} | **{r['verdict']}** |"
        )
    lines.append("")
    lines.append(
        "Gates: OOS expectancy > 0.15R, PF > 1.25, max DD < 8%, ≥ 3 profitable folds of ≥ 4, Monte Carlo success ≥ 0.80. "
        "The Monte Carlo shuffles trades one at a time, so it scores the average trade, not how many run in parallel."
    )
    lines += ["", "## Exit reasons", ""]
    lines.append("| variant | exits |")
    lines.append("|---|---|")
    for r in result["rows"]:
        lines.append(f"| {r['variant']} | " + ", ".join(f"{k} {v}" for k, v in r["exit_reasons"].items()) + " |")
    lines.append("")
    return "\n".join(lines)


def write_compare_report(
    result: dict[str, Any], out_dir: Path, tag: str | None = None, stem: str = "macro_compare", title: str | None = None
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stem}_{tag}" if tag else stem
    p_json = out_dir / f"{stem}.json"
    p_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    p_md = out_dir / f"{stem}.md"
    p_md.write_text(render_markdown(result, title=title) if title else render_markdown(result), encoding="utf-8")
    return {"json": p_json, "md": p_md}
