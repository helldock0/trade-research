# trade-research market data

Orphan branch holding Dukascopy OHLC caches for `trade-research`, stored with **Git LFS**.
Code lives on `main`; nothing here is merged into it.

- Source: Dukascopy BID OHLC via `scripts/download_data.py --intervals M1` (`dukascopy-python`)
- Bars: M1, UTC bar-open timestamps, columns `timestamp, open, high, low, close, volume`
- Span: 2018-01-01 -> 2026-10-21 (US2000 history starts 2019)
- Indices are Dukascopy CFDs: NAS100 = `E_NQ-100`, SPX500 = `E_SandP-500`, US30 = `E_D&J-Ind`, US2000 = `USSC2000.IDX/USD`
- Re-encoded losslessly with zstd (level 9); tables read back identical to the downloads

## Restore into `data/parquet`

Requires `git lfs install` once per machine.

```bash
# from the trade-research checkout
git fetch origin data
git worktree add ../trade-data origin/data      # LFS files download on checkout
cp -n ../trade-data/parquet/*.parquet data/parquet/   # -n: keep any files you already have
git worktree remove ../trade-data
```

Or a standalone shallow copy:

```bash
git clone --branch data --single-branch --depth 1 https://github.com/helldock0/trade-research.git trade-data
```

## Files

| File | Rows | First bar (UTC) | Last bar (UTC) | MB | sha256 (prefix) |
|---|---:|---|---|---:|---|
| `AUDUSD_M1.parquet` | 3,273,309 | 2018-01-01 22:00 | 2026-10-21 00:00 | 38.8 | `1278f3fef3480b6f` |
| `EURUSD_M1.parquet` | 3,278,152 | 2018-01-01 22:00 | 2026-10-21 00:00 | 40.7 | `5c8c84fd4b980a36` |
| `GBPUSD_M1.parquet` | 3,278,534 | 2018-01-01 22:00 | 2026-10-21 00:00 | 40.7 | `466f9461b0e8c68f` |
| `NAS100_M1.parquet` | 2,974,244 | 2018-01-02 07:05 | 2026-10-21 00:00 | 54.1 | `84554e33cec1eaaf` |
| `NZDUSD_M1.parquet` | 3,268,244 | 2018-01-01 22:01 | 2026-10-21 00:00 | 38.3 | `60dc5f1a73452e0f` |
| `SPX500_M1.parquet` | 2,878,911 | 2018-01-02 07:05 | 2026-10-21 00:00 | 44.6 | `8e495f57ca279680` |
| `US2000_M1.parquet` | 2,632,337 | 2018-08-08 13:30 | 2026-10-21 00:00 | 37.8 | `fc966d71c1b156d3` |
| `US30_M1.parquet` | 2,972,545 | 2018-01-02 07:05 | 2026-10-21 00:00 | 47.5 | `8ff180031846b281` |
| `USDCAD_M1.parquet` | 3,273,150 | 2018-01-01 22:04 | 2026-10-21 00:00 | 39.4 | `7acf6cf494144646` |
| `USDCHF_M1.parquet` | 3,261,310 | 2018-01-01 22:04 | 2026-10-21 00:00 | 38.3 | `fd3d43dddcb56e4d` |
| `USDJPY_M1.parquet` | 3,277,820 | 2018-01-01 22:00 | 2026-10-21 00:00 | 40.5 | `35d18154cb0bd15a` |
| `XAUUSD_M1.parquet` | 3,119,579 | 2018-01-01 23:00 | 2026-10-21 00:00 | 46.8 | `bbeee2ac700576cd` |

## Adding files

```bash
git worktree add ../trade-data data
cp data/parquet/NEW_M1.parquet ../trade-data/parquet/
cd ../trade-data && git add parquet/NEW_M1.parquet README.md && git commit -m "Add NEW M1" && git push origin data
```
