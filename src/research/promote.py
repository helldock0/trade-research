from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import ROOT, ensure_dirs
from src.research.monte_carlo import monte_carlo_ftmo
from src.research.walk_forward import FoldResult, aggregate_oos


GATES = {
    "min_expectancy_r": 0.15,
    "min_profit_factor": 1.25,
    "max_dd_pct": 8.0,
    "min_profitable_folds": 3,
    "min_folds": 4,
    "min_mc_success": 0.80,
}


def evaluate_promotion(
    folds: list[FoldResult],
    trades_for_mc: list,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    g = {**GATES, **(gates or {})}
    oos = aggregate_oos(folds)
    mc = monte_carlo_ftmo(trades_for_mc)

    checks = {
        "expectancy_r": oos["expectancy_r"] > g["min_expectancy_r"],
        "profit_factor": oos["profit_factor"] > g["min_profit_factor"],
        "max_dd_pct": oos["max_dd_pct"] < g["max_dd_pct"],
        "profitable_folds": oos["profitable_folds"] >= g["min_profitable_folds"],
        "n_folds": oos["n_folds"] >= g["min_folds"],
        "monte_carlo_success": mc["success_rate"] >= g["min_mc_success"],
    }
    go = all(checks.values())
    verdict = {
        "verdict": "GO" if go else "NO-GO",
        "checks": checks,
        "gates": g,
        "oos": oos,
        "monte_carlo": mc,
    }
    return verdict


def write_promotion_report(verdict: dict[str, Any], path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (ROOT / "reports" / "promotion_verdict.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, default=str)

    md = path.with_suffix(".md")
    lines = [
        f"# Promotion Verdict: **{verdict['verdict']}**",
        "",
        "## Checks",
    ]
    for k, v in verdict["checks"].items():
        lines.append(f"- `{k}`: {'PASS' if v else 'FAIL'}")
    lines += [
        "",
        "## OOS summary",
        f"- folds: {verdict['oos'].get('n_folds')}",
        f"- profitable folds: {verdict['oos'].get('profitable_folds')}",
        f"- expectancy R: {verdict['oos'].get('expectancy_r'):.4f}",
        f"- profit factor: {verdict['oos'].get('profit_factor'):.4f}",
        f"- max DD %: {verdict['oos'].get('max_dd_pct'):.4f}",
        "",
        "## Monte Carlo",
        f"- success rate: {verdict['monte_carlo'].get('success_rate'):.4f}",
        f"- ruin rate: {verdict['monte_carlo'].get('ruin_rate'):.4f}",
        "",
    ]
    if verdict["verdict"] != "GO":
        lines.append("MT5 execution stub **not** authorized (NO-GO).")
    else:
        lines.append("MT5 execution stub authorized.")
    md.write_text("\n".join(lines), encoding="utf-8")
    return path
