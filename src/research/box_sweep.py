"""Parameter sweep for the rolling 15-minute box breakout.

For each symbol, breakout candidates are found once per confirmation
timeframe and every (stop, exit) variant is simulated on M1 in one pass.
Window, cost filter and concurrency are cheap post-filters on those results,
so the full grid costs little more than the simulation itself.

Results are kept as sums (not means) so they can be pooled across symbols,
split in-sample / out-of-sample, by UTC hour of the box and by year.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import round_trip_cost_price
from src.data.dukascopy_ingest import load_parquet, parquet_path
from src.features.sessions import parse_hhmm
from src.strategy.rolling_box import (
    REASON_NAMES,
    breakout_candidates,
    build_boxes,
    m1_arrays,
    max_drawdown,
    net_r,
    one_at_a_time,
    simulate_candidates,
)

GRID_KEYS = ("confirm_tf", "stop", "exit", "window", "min_risk_cost", "concurrency")
FIRST_YEAR = 2018
N_YEARS = 12
IST_OFFSET_MIN = 330


def ist_label(hour_utc: int) -> str:
    """UTC hour bucket -> IST start label (e.g. 0 -> '05:30')."""
    m = (hour_utc * 60 + IST_OFFSET_MIN) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def grid_lists(cfg: dict[str, Any]) -> dict[str, list]:
    sw = cfg["sweep"]
    return {
        "confirm_tf": [int(x) for x in sw["confirm_tf_minutes"]],
        "stop": [str(x) for x in sw["stops"]],
        "exit": [str(x) for x in sw["exits"]],
        "window": sorted(int(x) for x in sw["window_minutes"]),
        "min_risk_cost": [float(x) for x in sw["min_risk_cost"]],
        "concurrency": [str(x) for x in sw["concurrency"]],
    }


def config_table(cfg: dict[str, Any]) -> pd.DataFrame:
    """All grid combinations; row order defines cfg_id (matches sweep loops)."""
    g = grid_lists(cfg)
    rows = list(itertools.product(*(g[k] for k in GRID_KEYS)))
    out = pd.DataFrame(rows, columns=list(GRID_KEYS))
    out.insert(0, "cfg_id", np.arange(len(out)))
    return out


def _minutes(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 60_000_000_000)


def sweep_symbol(
    symbol: str,
    m1: pd.DataFrame,
    cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (summary, by_hour, by_year) frames keyed by cfg_id for one symbol."""
    g = grid_lists(cfg)
    sess, sig = cfg["sessions"], cfg.get("signal", {})
    oos_start = _minutes(cfg["validation"]["oos_start"])
    m1a = m1_arrays(m1)
    boxes = build_boxes(
        m1,
        parse_hhmm(sess["start"]),
        parse_hhmm(sess["end"]),
        box_minutes=int(sig.get("box_minutes", 15)),
        atr_period=int(sig.get("atr_period", 14)),
    )
    day_min = boxes["box_start"] - boxes["box_start"] % 1440
    n_days = int(day_min.nunique())
    n_days_is = int(day_min[boxes["box_start"] < oos_start].nunique())
    side_cost, rt_cost = round_trip_cost_price(symbol, costs_cfg)

    summary_rows: list[tuple] = []
    hour_blocks: list[np.ndarray] = []
    year_blocks: list[np.ndarray] = []
    cfg_id = 0
    for confirm in g["confirm_tf"]:
        cand = breakout_candidates(m1, m1a, boxes, confirm, max(g["window"]))
        pnl, exit_t, _, risk = simulate_candidates(m1a, cand, g["stop"], g["exit"])
        entry_t = cand["entry_t"].to_numpy(np.int64)
        delay = cand["delay"].to_numpy()
        hour = cand["hour_utc"].to_numpy()
        year = np.clip(cand["year"].to_numpy() - FIRST_YEAR, 0, N_YEARS - 1)
        oos = cand["box_start"].to_numpy() >= oos_start
        for s_i in range(len(g["stop"])):
            r0 = risk[:, s_i]
            for e_i in range(len(g["exit"])):
                p = pnl[:, s_i, e_i]
                valid = np.isfinite(r0) & np.isfinite(p)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rn = net_r(p, r0, side_cost, rt_cost)
                    rg = p / r0
                xt = exit_t[:, s_i, e_i]
                for window in g["window"]:
                    in_window = valid & (delay <= window)
                    for k in g["min_risk_cost"]:
                        m = in_window & ((r0 + side_cost) >= k * rt_cost) if k > 0 else in_window
                        for conc in g["concurrency"]:
                            keep = one_at_a_time(entry_t, xt, m) if conc == "one" else m
                            r, o = rn[keep], oos[keep]
                            h, y = hour[keep], year[keep]
                            summary_rows.append(
                                (
                                    cfg_id,
                                    len(r),
                                    int((~o).sum()),
                                    float(r.sum()),
                                    float(r[~o].sum()),
                                    float(r[o].sum()),
                                    float(rg[keep].sum()),
                                    int((r > 0).sum()),
                                    float(r[r > 0].sum()),
                                    float(-r[r < 0].sum()),
                                    float(max_drawdown(r)) if len(r) else 0.0,
                                )
                            )
                            hb = np.zeros((24, 4))
                            hb[:, 0] = np.bincount(h[~o], minlength=24)
                            hb[:, 1] = np.bincount(h[~o], weights=r[~o], minlength=24)
                            hb[:, 2] = np.bincount(h[o], minlength=24)
                            hb[:, 3] = np.bincount(h[o], weights=r[o], minlength=24)
                            hour_blocks.append(hb)
                            yb = np.zeros((N_YEARS, 2))
                            yb[:, 0] = np.bincount(y, minlength=N_YEARS)
                            yb[:, 1] = np.bincount(y, weights=r, minlength=N_YEARS)
                            year_blocks.append(yb)
                            cfg_id += 1

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "cfg_id",
            "n",
            "n_is",
            "sum_r",
            "sum_r_is",
            "sum_r_oos",
            "sum_r_gross",
            "wins",
            "win_r",
            "loss_r",
            "max_dd_r",
        ],
    )
    summary.insert(0, "symbol", symbol)
    summary["n_days"] = n_days
    summary["n_days_is"] = n_days_is

    hours = np.stack(hour_blocks)  # (cfg, 24, 4)
    n_cfg = hours.shape[0]
    by_hour = pd.DataFrame(
        {
            "symbol": symbol,
            "cfg_id": np.repeat(np.arange(n_cfg), 24),
            "hour_utc": np.tile(np.arange(24), n_cfg),
            "n_is": hours[:, :, 0].ravel().astype(np.int64),
            "sum_r_is": hours[:, :, 1].ravel(),
            "n_oos": hours[:, :, 2].ravel().astype(np.int64),
            "sum_r_oos": hours[:, :, 3].ravel(),
        }
    )
    by_hour = by_hour.loc[(by_hour["n_is"] + by_hour["n_oos"]) > 0].reset_index(drop=True)

    years = np.stack(year_blocks)  # (cfg, N_YEARS, 2)
    by_year = pd.DataFrame(
        {
            "symbol": symbol,
            "cfg_id": np.repeat(np.arange(n_cfg), N_YEARS),
            "year": np.tile(np.arange(N_YEARS) + FIRST_YEAR, n_cfg),
            "n": years[:, :, 0].ravel().astype(np.int64),
            "sum_r": years[:, :, 1].ravel(),
        }
    )
    by_year = by_year.loc[by_year["n"] > 0].reset_index(drop=True)
    return summary, by_hour, by_year


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive expectancy / PF / win-rate columns from pooled sums."""
    out = df.copy()
    n_oos = out["n"] - out["n_is"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["exp_r"] = out["sum_r"] / out["n"]
        out["exp_r_is"] = out["sum_r_is"] / out["n_is"]
        out["exp_r_oos"] = out["sum_r_oos"] / n_oos
        out["exp_r_gross"] = out["sum_r_gross"] / out["n"]
        out["win_rate"] = out["wins"] / out["n"]
        out["profit_factor"] = out["win_r"] / out["loss_r"]
        if "n_days" in out:
            out["trades_per_day"] = out["n"] / out["n_days"]
    out["n_oos"] = n_oos
    return out


def portfolio_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Pool every symbol's trades per config (equal 1R risk per trade)."""
    cols = ["n", "n_is", "sum_r", "sum_r_is", "sum_r_oos", "sum_r_gross", "wins", "win_r", "loss_r"]
    pooled = summary.groupby("cfg_id", as_index=False)[cols].sum()
    pooled["n_symbols"] = summary.groupby("cfg_id")["symbol"].nunique().to_numpy()
    return add_metrics(pooled)


def _sweep_worker(
    symbol: str, cfg: dict[str, Any], costs_cfg: dict[str, Any], cache_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tf = cfg["data"].get("timeframe", "M1")
    m1 = load_parquet(parquet_path(symbol, tf, Path(cache_dir)))
    return sweep_symbol(symbol, m1, cfg, costs_cfg)


def run_sweep(
    symbols: list[str],
    cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    cache_dir: Path,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Sweep all symbols (one process per symbol). Missing data files are skipped."""
    tf = cfg["data"].get("timeframe", "M1")
    todo, missing = [], []
    for s in symbols:
        (todo if parquet_path(s, tf, cache_dir).exists() else missing).append(s)
    for s in missing:
        print(f"[skip] {s}: no {tf} parquet in {cache_dir}", flush=True)

    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    if workers <= 1:
        for s in todo:
            parts.append(_sweep_worker(s, cfg, costs_cfg, str(cache_dir)))
            print(f"[done] {s}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_sweep_worker, s, cfg, costs_cfg, str(cache_dir)): s for s in todo}
            for fut in as_completed(futs):
                parts.append(fut.result())
                print(f"[done] {futs[fut]}", flush=True)
    if not parts:
        raise FileNotFoundError(f"No {tf} data found for any symbol in {cache_dir}")
    summary = pd.concat([p[0] for p in parts], ignore_index=True)
    by_hour = pd.concat([p[1] for p in parts], ignore_index=True)
    by_year = pd.concat([p[2] for p in parts], ignore_index=True)
    return summary, by_hour, by_year, missing


def config_trades(
    symbol: str,
    m1: pd.DataFrame,
    cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    confirm_tf: int,
    stop: str,
    exit: str,
    window: int,
    min_risk_cost: float = 0.0,
    concurrency: str = "one",
) -> pd.DataFrame:
    """Trade list for a single configuration (for drill-down and charts)."""
    sess, sig = cfg["sessions"], cfg.get("signal", {})
    m1a = m1_arrays(m1)
    boxes = build_boxes(
        m1,
        parse_hhmm(sess["start"]),
        parse_hhmm(sess["end"]),
        box_minutes=int(sig.get("box_minutes", 15)),
        atr_period=int(sig.get("atr_period", 14)),
    )
    cand = breakout_candidates(m1, m1a, boxes, confirm_tf, window)
    pnl, exit_t, reason, risk = simulate_candidates(m1a, cand, [stop], [exit])
    p, xt, rs, r0 = pnl[:, 0, 0], exit_t[:, 0, 0], reason[:, 0, 0], risk[:, 0]
    side_cost, rt_cost = round_trip_cost_price(symbol, costs_cfg)
    m = np.isfinite(p) & np.isfinite(r0)
    if min_risk_cost > 0:
        m &= (r0 + side_cost) >= min_risk_cost * rt_cost
    entry_t = cand["entry_t"].to_numpy(np.int64)
    keep = one_at_a_time(entry_t, xt, m) if concurrency == "one" else m
    c = cand.loc[keep]
    to_ts = lambda mins: pd.to_datetime(np.asarray(mins, np.int64) * 60, unit="s", utc=True)  # noqa: E731
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "box_start": to_ts(c["box_start"]),
            "entry_time": to_ts(c["entry_t"]),
            "exit_time": to_ts(xt[keep]),
            "side": c["side"].to_numpy(),
            "box_high": c["high"].to_numpy(),
            "box_low": c["low"].to_numpy(),
            "atr": c["atr"].to_numpy(),
            "entry_px": c["entry_px"].to_numpy(),
            "risk_px": r0[keep],
            "pnl_px": p[keep],
            "r_gross": p[keep] / r0[keep],
            "r_net": net_r(p[keep], r0[keep], side_cost, rt_cost),
            "exit_reason": [REASON_NAMES[i] for i in rs[keep]],
            "hour_utc": c["hour_utc"].to_numpy(),
        }
    )
    out["box_start_ist"] = out["box_start"].dt.tz_convert("Asia/Kolkata")
    return out
