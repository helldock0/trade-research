# Agent notes

## Market data

All OHLC data lives in **`data/parquet`** (`D:\AI\trade\data\parquet`).

- Format: `{SYMBOL}_{TF}.parquet` (e.g. `EURUSD_M15.parquet`)
- Universe: majors + minors + XAUUSD, TFs M1–MN1, from ~2018-01-01
- Load via `src.data.dukascopy_ingest` / strategy configs (`data.cache_dir`)

Do not use another data folder unless explicitly asked. Prefer existing files over re-downloading.

Cloud sessions (no `D:` drive): `python scripts/bars_from_data_branch.py` builds `data/parquet/{PAIR}_M15.parquet`
from the repo's `data` branch (Git LFS M5 BID bars from 2018; crosses built from their USD legs' M1 bars).

## Event / macro data

- Economic calendar: `data/events_calendar.csv` (FXStreet, UTC release times, actual/forecast/previous).
  Refresh with `python scripts/download_calendar.py` (merges into the cache).
- FRED series cache: `data/macro/{SERIES}.csv` (e.g. `GASREGW` for the CPI nowcast).
- Macro swing inputs: `data/macro/{group}_{CCY}.csv` and `data/macro/global_{key}.csv`, fetched by
  `python scripts/download_macro.py` from the sources named in `configs/strategy_macro_swing.yaml`
  (`src.data.macro` has one parser per source; `--show` prints the live score table).
- FRED daily FX for the macro IC check: `data/macro/fx_{PAIR}.csv`, fetched and used by
  `python scripts/run_macro_ic.py` (`src.research.macro_ic`). No price bars needed.
- Daily index histories for signal checks (`run_tom_check.py`, `run_dip_check.py`, `run_dip_basket_check.py`):
  `data/macro/{NAME}.csv`, FRED ids or `YH_*` names fetched from Yahoo Finance (stooq is blocked from the cloud).
- The same calendar CSV drives the news blackout (`src.features.news`) and the event study
  (`src.strategy.event_drift`, `src.research.event_study`).
- The gj-strat re-check (`scripts/run_jpy_recheck.py`, owner request) reads Dukascopy 1-minute BID/ASK files from
  gj-strat's `datastore` (path in `configs/jpy_recheck.yaml`) with this repo's own decoder; run it with `python -I`.
- Signals must stay point-in-time: anything used before a release comes from releases strictly before it.

## Repo layout

- Code/configs/tests are tracked; `data/` and `reports/` are local-only.
- Research ledger: `docs/STATUS.md`
- Package name: `trade-research` (`pyproject.toml`)
