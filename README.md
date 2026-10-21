# trade-research market data

Orphan branch holding Dukascopy OHLC caches for `trade-research`, stored with **Git LFS**.
Code lives on `main`; nothing here is merged into it.

- Source: Dukascopy BID OHLC via `scripts/download_data.py --intervals M1` (`dukascopy-python`)
- Bars: M1, UTC bar-open timestamps, columns `timestamp, open, high, low, close, volume`
- Also M5 **bid and ask** for 13 markets (`*_M5.parquet` bid, `*_M5_ask.parquet` ask; `--intervals M5 [--side ask]`), used by the NY kill-zone backtests for real-spread fills. M5 index names follow `src/data/dukascopy_ingest.py`: US500 = SPX500 (`E_SandP-500`), GER40 = `E_DAAX`, UK100 = `E_Futsee-100`, JP225 = `E_N225Jap`
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

### M1 (bid)

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

### M5 (bid and ask, added 2026-10-21)

| File | Rows | First bar (UTC) | Last bar (UTC) | MB | sha256 (prefix) |
|---|---:|---|---|---:|---|
| `AUDUSD_M5.parquet` | 657,961 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.0 | `a65d77be3271d959` |
| `AUDUSD_M5_ask.parquet` | 657,961 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.0 | `51483bb775c0967c` |
| `EURUSD_M5.parquet` | 658,080 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.6 | `44e50732665a1b63` |
| `EURUSD_M5_ask.parquet` | 658,080 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.6 | `f4aced2b893da5e0` |
| `GBPUSD_M5.parquet` | 657,917 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.6 | `7c358da05c70a1d0` |
| `GBPUSD_M5_ask.parquet` | 657,917 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.6 | `ff1880975617a2cc` |
| `GER40_M5.parquet` | 594,554 | 2018-01-02 07:05 | 2026-10-20 00:00 | 10.7 | `2cea7a0215cebca7` |
| `GER40_M5_ask.parquet` | 594,554 | 2018-01-02 07:05 | 2026-10-20 00:00 | 10.7 | `e3524d44e28eff13` |
| `JP225_M5.parquet` | 590,263 | 2018-01-02 01:05 | 2026-10-20 00:00 | 9.1 | `5c52b74178e5131c` |
| `JP225_M5_ask.parquet` | 590,263 | 2018-01-02 01:05 | 2026-10-20 00:00 | 9.1 | `94c7583d979bec22` |
| `NAS100_M5.parquet` | 595,155 | 2018-01-02 07:05 | 2026-10-20 10:50 | 12.3 | `ca207f4b50045cc6` |
| `NAS100_M5_ask.parquet` | 595,155 | 2018-01-02 07:05 | 2026-10-20 10:50 | 11.9 | `8978da2ff0676b9d` |
| `NZDUSD_M5.parquet` | 657,619 | 2018-01-01 22:00 | 2026-10-20 00:00 | 8.8 | `0afa3eb9f6c210da` |
| `NZDUSD_M5_ask.parquet` | 657,619 | 2018-01-01 22:00 | 2026-10-20 00:00 | 8.8 | `b18bf7a10ca022ef` |
| `UK100_M5.parquet` | 589,973 | 2018-01-02 07:05 | 2026-10-20 00:00 | 11.3 | `0ea70d49ece38e49` |
| `UK100_M5_ask.parquet` | 589,973 | 2018-01-02 07:05 | 2026-10-20 00:00 | 11.4 | `c828dc536bb75924` |
| `US30_M5.parquet` | 595,306 | 2018-01-02 07:05 | 2026-10-20 00:00 | 11.1 | `dca8b8a4a8c4a055` |
| `US30_M5_ask.parquet` | 595,306 | 2018-01-02 07:05 | 2026-10-20 00:00 | 11.2 | `e9cbc0afb9e44bd3` |
| `US500_M5.parquet` | 594,503 | 2018-01-02 07:05 | 2026-10-20 00:00 | 10.9 | `52bbe1202cf24c14` |
| `US500_M5_ask.parquet` | 594,503 | 2018-01-02 07:05 | 2026-10-20 00:00 | 11.0 | `9b3876a2e07d9ede` |
| `USDCAD_M5.parquet` | 657,831 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.1 | `e86635acd7c4b70c` |
| `USDCAD_M5_ask.parquet` | 657,831 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.1 | `976166a751dcbc4d` |
| `USDCHF_M5.parquet` | 657,497 | 2018-01-01 22:00 | 2026-10-20 00:00 | 8.8 | `30df984d456b1919` |
| `USDCHF_M5_ask.parquet` | 657,497 | 2018-01-01 22:00 | 2026-10-20 00:00 | 8.8 | `2daefe0b4ef471bf` |
| `USDJPY_M5.parquet` | 658,065 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.5 | `5b06cd5741789d20` |
| `USDJPY_M5_ask.parquet` | 658,065 | 2018-01-01 22:00 | 2026-10-20 00:00 | 9.5 | `284ffd6882b940d1` |

## Adding files

```bash
git worktree add ../trade-data data
cp data/parquet/NEW_M1.parquet ../trade-data/parquet/
cd ../trade-data && git add parquet/NEW_M1.parquet README.md && git commit -m "Add NEW M1" && git push origin data
```
