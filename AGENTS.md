# Agent notes

## Market data

All OHLC data lives in **`data/parquet`** (`D:\AI\trade\data\parquet`).

- Format: `{SYMBOL}_{TF}.parquet` (e.g. `EURUSD_M15.parquet`)
- Universe: majors + minors + XAUUSD, TFs M1–MN1, from ~2018-01-01
- Load via `src.data.dukascopy_ingest` / strategy configs (`data.cache_dir`)

Do not use another data folder unless explicitly asked. Prefer existing files over re-downloading.

## Repo layout

- Code/configs/tests are tracked; `data/` and `reports/` are local-only.
- Research ledger: `docs/STATUS.md`
- Package name: `trade-research` (`pyproject.toml`)

## Cursor Cloud specific instructions

Pure Python CLI/batch research harness — no web server, DB, or GUI. Standard commands live in `README.md` (Workflow section) and `pyproject.toml` `[project.scripts]`.

- Env: a virtualenv is provisioned at `.venv` by the update script. Activate with `source .venv/bin/activate` before running anything; scripts also self-add repo root to `sys.path`, so `python scripts/<name>.py ...` works.
- Tests: `python -m pytest -q` (unit tests use synthetic/in-memory data, no network or Parquet on disk needed). There is no separate lint config (no ruff/mypy/flake8 configs despite cache dirs in `.gitignore`).
- Market data is gitignored and absent on a fresh VM. Dukascopy downloads need internet; in the sandbox use the offline generator instead: `python scripts/download_data.py --synthetic --years 8` writes M15+H1 Parquet for the strategy universe into `data/parquet/`.
- Synthetic data only covers M15 and H1. Strategies that need H4/D1/other TFs (e.g. `htf_adx_pullback`, `d1_mom`) won't have inputs from `--synthetic` alone; `cssm` (M15) is a good smoke-test strategy.
- End-to-end smoke test after generating data: `python scripts/run_backtest.py --strategy cssm` then `run_prop_sim.py` / `run_walkforward.py`. Outputs land in `reports/` (gitignored). A full backtest over 8y synthetic data takes ~30-40s per strategy; walk-forward + Monte Carlo takes ~2min.
- All 11 strategies are currently NO-GO by design, so a NO-GO promotion verdict is the expected/correct result, not a failure.
