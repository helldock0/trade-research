# Agent notes

## Market data

All OHLC data lives in **`data/parquet`** (`D:\AI\trade\data\parquet`).

- Format: `{SYMBOL}_{TF}.parquet` (e.g. `EURUSD_M15.parquet`)
- Universe: majors + minors + XAUUSD, TFs M1–MN1, from ~2018-01-01
- Load via `src.data.dukascopy_ingest` / strategy configs (`data.cache_dir`)

Do not use another data folder unless explicitly asked. Prefer existing files over re-downloading.

## Event / macro data

- Economic calendar: `data/events_calendar.csv` (FXStreet, UTC release times, actual/forecast/previous).
  Refresh with `python scripts/download_calendar.py` (merges into the cache).
- FRED series cache: `data/macro/{SERIES}.csv` (e.g. `GASREGW` for the CPI nowcast).
- The same calendar CSV drives the news blackout (`src.features.news`) and the event study
  (`src.strategy.event_drift`, `src.research.event_study`).
- Signals must stay point-in-time: anything used before a release comes from releases strictly before it.

## Repo layout

- Code/configs/tests are tracked; `data/` and `reports/` are local-only.
- Research ledger: `docs/STATUS.md`
- Package name: `trade-research` (`pyproject.toml`)
