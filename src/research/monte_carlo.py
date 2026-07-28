from __future__ import annotations

from typing import Any

import numpy as np

from src.backtest.engine import Trade


def monte_carlo_ftmo(
    trades: list[Trade],
    n_paths: int = 1000,
    target_pct: float = 10.0,
    ruin_pct: float = 10.0,
    initial: float = 100000.0,
    slippage_shock_r: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Shuffle trade R-multiples with mild slippage shock; measure P(hit target before ruin)."""
    if not trades:
        return {
            "n_paths": n_paths,
            "success_rate": 0.0,
            "median_ending_pct": 0.0,
            "paths_hit_target": 0,
            "paths_ruined": 0,
        }

    rng = np.random.default_rng(seed)
    rs = np.array([t.r_multiple for t in trades], dtype=float)
    # Convert R to pct at 0.5R => 0.5% if risk is 0.5% — use empirical pnl scaling
    risk_pct = 0.5  # matches strategy config default
    pcts = rs * risk_pct  # each trade return in % of equity approx

    hits = 0
    ruined = 0
    endings = []

    for _ in range(n_paths):
        order = rng.permutation(len(pcts))
        shock = rng.normal(0.0, slippage_shock_r, size=len(pcts))
        path = pcts[order] - np.abs(shock) * risk_pct
        equity_pct = 0.0
        hit = False
        dead = False
        for p in path:
            equity_pct += p
            if equity_pct <= -ruin_pct:
                dead = True
                ruined += 1
                break
            if equity_pct >= target_pct:
                hit = True
                hits += 1
                break
        endings.append(equity_pct if not dead else -ruin_pct)
        if not hit and not dead:
            # exhausted trades without target or ruin
            pass

    return {
        "n_paths": n_paths,
        "success_rate": hits / n_paths,
        "ruin_rate": ruined / n_paths,
        "median_ending_pct": float(np.median(endings)),
        "paths_hit_target": hits,
        "paths_ruined": ruined,
    }
