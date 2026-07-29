# Agent notes

## Market data

All OHLC data lives in **`data/parquet`** (`D:\AI\trade\data\parquet`).

- Format: `{SYMBOL}_{TF}.parquet` (e.g. `EURUSD_M15.parquet`)
- Universe: majors + minors + XAUUSD, TFs M1–MN1, from ~2018-01-01
- Load via `src.data.dukascopy_ingest` / strategy configs (`data.cache_dir`)

Do not use another data folder unless explicitly asked. Prefer existing files over re-downloading.
