"""Build data/parquet/{PAIR}_M15.parquet from the repo's `data` branch (Git LFS), for cloud sessions.

    python scripts/bars_from_data_branch.py                       # the macro_swing universe (+ variants)
    python scripts/bars_from_data_branch.py --pairs EURUSD,EURJPY --force

USD pairs and index CFDs: the branch's M5 BID bars (M1 where there is no M5)
resampled to M15. Crosses (no USD leg): built minute by minute from the M1 BID
bars of their two USD legs, then resampled to M15. Only the needed LFS objects
are pulled into a worktree under data/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.costs import is_cross
from src.config import load_strategy_config
from src.data.bars import cross_bars, resample_ohlc
from src.data.dukascopy_ingest import load_parquet, save_parquet


def _legs(pair: str) -> list[str]:
    majors = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
    return [next(m for m in majors if ccy in m) for ccy in (pair[:3], pair[3:6])]


def _universe(cfg_name: str) -> list[str]:
    cfg = load_strategy_config(cfg_name)
    pairs = list(cfg["universe"])
    for v in (cfg.get("variants") or {}).values():
        pairs += list((v or {}).get("universe") or [])
    return list(dict.fromkeys(p.upper() for p in pairs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M15 bars from the data branch (Git LFS)")
    parser.add_argument("--pairs", default=None, help="comma list (default: macro_swing universe and variants)")
    parser.add_argument("--config", default="strategy_macro_swing.yaml")
    parser.add_argument("--branch", default="origin/data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    pairs = [p.strip().upper() for p in args.pairs.split(",")] if args.pairs else _universe(args.config)
    out_dir = ROOT / "data" / "parquet"
    todo = [p for p in pairs if args.force or not (out_dir / f"{p}_M15.parquet").exists()]
    if not todo:
        print("All M15 files exist; use --force to rebuild.")
        return
    wt = ROOT / "data" / "_data_branch"
    if not wt.exists():
        subprocess.run(["git", "fetch", "origin", args.branch.split("/", 1)[-1]], cwd=ROOT, check=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), args.branch], cwd=ROOT, check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    listed = set(subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=wt, check=True, capture_output=True, text=True).stdout.split())

    def source(pair: str) -> str:
        m5 = f"parquet/{pair}_M5.parquet"
        return m5 if m5 in listed else f"parquet/{pair}_M1.parquet"

    files = sorted({source(p) for p in todo if not is_cross(p)} | {f"parquet/{leg}_M1.parquet" for p in todo if is_cross(p) for leg in _legs(p)})
    subprocess.run(["git", "lfs", "pull", f"--include={','.join(files)}"], cwd=wt, check=True)

    for pair in todo:
        if is_cross(pair):
            legs = {leg: load_parquet(wt / "parquet" / f"{leg}_M1.parquet") for leg in _legs(pair)}
            m15 = resample_ohlc(cross_bars(pair, legs), "15min")
        else:
            m15 = resample_ohlc(load_parquet(wt / source(pair)), "15min")
        m15 = m15.rename_axis("timestamp").reset_index()
        save_parquet(m15, out_dir / f"{pair}_M15.parquet")
        print(f"{pair}: {len(m15)} M15 bars {m15['timestamp'].min()} -> {m15['timestamp'].max()}")


if __name__ == "__main__":
    main()
