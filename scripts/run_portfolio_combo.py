"""One account running USDJPY vol-gotobi + NAS100 NY breakout v3 + NAS100 buy-the-dip (configs/portfolio_combo.yaml).

    python -I scripts/run_portfolio_combo.py
    python -I scripts/run_portfolio_combo.py --datastore /path/to/gj-strat/datastore

Each leg runs its own frozen config and costs:
- gotobi: gj-strat's raw USDJPY minute files (src.research.jpy_recheck)
- nykz: data/parquet/NAS100_M5{,_ask}.parquet (bid and ask, from the data branch)
- dip: data/parquet/NAS100_M15.parquet through the engine
Writes reports/portfolio_combo.{json,md} and reports/portfolio_combo_trades.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.engine import run_backtest
from src.config import ROOT as PROJECT_ROOT
from src.config import load_costs_config, load_ftmo_config, load_strategy_config
from src.data.dukascopy_ingest import load_parquet, load_symbol_bars, parquet_path
from src.research import jpy_recheck as jr
from src.research.macro_compare import deep_merge
from src.research.portfolio import (
    combine,
    leg_frame,
    render_markdown,
    run_portfolio,
    write_report,
)
from src.strategy.nykz_breakout import merge_bid_ask, trades_frame
from src.strategy.nykz_breakout import run as run_nykz
from src.strategy.registry import resolve_signal_fn


def gotobi_leg(cfg_name: str, datastore: str | None) -> pd.DataFrame:
    cfg = load_strategy_config(cfg_name)
    m = jr.load_datastore(Path(datastore or cfg["datastore"]), cfg["symbol"], cfg["start"], cfg["end"])
    c = cfg["costs"]
    costs = jr.Costs(float(c["commission_usd_per_lot"]), float(c["slippage_pips"]), float(c["swap_markup_pct"]),
                     jr.load_rates(PROJECT_ROOT / "data" / "macro"))
    daily = jr.ny_daily(m, int(cfg["vol_filter"]["median_days"]))
    got = jr.gotobi_trades(m, daily, cfg["gotobi"], costs, float(cfg["pip"]))
    return got[got["hot"]].assign(side=-1)                    # vol-gotobi: sells only


def nykz_leg(cfg_name: str, start: str) -> pd.DataFrame:
    cfg = load_strategy_config(cfg_name)
    cache = PROJECT_ROOT / cfg["data"]["cache_dir"]
    sym, tf = cfg["symbol"], cfg["timeframe"]
    bars = merge_bid_ask(load_parquet(parquet_path(sym, tf, cache, side="bid")), load_parquet(parquet_path(sym, tf, cache, side="ask")))
    bars = bars.loc[pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=365):]
    trades, _ = run_nykz(bars, {**cfg, "signal": {**cfg["signal"], "trade_from": start}})
    return trades_frame(trades)


def dip_leg(cfg_name: str) -> pd.DataFrame:
    cfg = load_strategy_config(cfg_name)
    bars = {s: load_symbol_bars(s, cfg.get("timeframe", "M15")) for s in cfg["universe"]}
    costs = deep_merge(load_costs_config(), cfg.get("costs_overrides") or {})
    res = run_backtest(resolve_signal_fn(cfg)(bars, cfg, costs), bars, cfg, costs)
    return res.trades_frame()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="gotobi + NY breakout + dip portfolio")
    parser.add_argument("--config", default="portfolio_combo.yaml")
    parser.add_argument("--datastore", default=None, help="gj-strat datastore (default: configs/jpy_recheck.yaml)")
    parser.add_argument("--no-ftmo", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_strategy_config(args.config)
    win, lc = cfg["window"], cfg["legs"]
    build = {
        "gotobi": lambda: gotobi_leg(lc["gotobi"]["config"], args.datastore),
        "nykz": lambda: nykz_leg(lc["nykz"]["config"], win[0]),
        "dip": lambda: dip_leg(lc["dip"]["config"]),
    }
    legs = {k: leg_frame(k, lc[k]["symbol"], build[k](), win) for k in lc}
    for k, tr in legs.items():
        print(f"{k}: {len(tr)} trades, avg R {tr['r'].mean():+.3f}", flush=True)
    res = run_portfolio(legs, cfg, None if args.no_ftmo else load_ftmo_config())
    paths = write_report(res, cfg, PROJECT_ROOT / "reports")
    combine(list(legs.values())).to_csv(PROJECT_ROOT / "reports" / "portfolio_combo_trades.csv", index=False)
    print(render_markdown(res, cfg))
    print("Reports:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
