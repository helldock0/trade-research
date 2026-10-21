# Agent notes

## Market data

All OHLC data lives in **`data/parquet`** (`D:\AI\trade\data\parquet`).

- Format: `{SYMBOL}_{TF}.parquet` (e.g. `EURUSD_M15.parquet`)
- Universe: majors + minors + XAUUSD, TFs M1–MN1, from ~2018-01-01
- Load via `src.data.dukascopy_ingest` / strategy configs (`data.cache_dir`)

Do not use another data folder unless explicitly asked. Prefer existing files over re-downloading.

No local data (e.g. cloud sessions)? M1 for the 7 majors, XAUUSD and NAS100/SPX500/US30/US2000
(2018-01-01 -> 2026-10-21) is on the orphan `data` branch via Git LFS; restore into `data/parquet`
per that branch's README before downloading.

## Repo layout

- Code/configs/tests are tracked; `data/` and `reports/` are local-only.
- Research ledger: `docs/STATUS.md`
- Package name: `trade-research` (`pyproject.toml`)
