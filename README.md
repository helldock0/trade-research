# CSSM - Cross-Sectional Session Momentum

Greenfield research harness for an FTMO-style prop challenge strategy.

**Edge:** overnight FX relative strength + session-range break, volatility regime gated, same-day exits, hard Friday flat.

## Setup

```bash
cd D:\AI\trade
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

Optional env (Windows):

```powershell
$env:PYTHONPATH="D:\AI\trade"
$env:PYTHONIOENCODING="utf-8"
```

## Workflow

```bash
# 1) Download Dukascopy data (majors -> parquet)
python scripts/download_data.py

# Offline / CI fallback:
# python scripts/download_data.py --synthetic --years 8

# 2) Single-period backtest
python scripts/run_backtest.py

# 3) Walk-forward + Monte Carlo + promotion gates
python scripts/run_walkforward.py

# 4) FTMO 2-Step path simulation
python scripts/run_prop_sim.py

# 5) Unit tests
python -m pytest -q
```

## Current research status

See `reports/promotion_verdict.md` and `reports/diagnosis.md`.

**First frozen CSSM cycle: NO-GO** — do not trade live or buy a challenge on this version.

## Promotion

`src/research/promote.py` prints **GO** only if walk-forward / Monte Carlo gates pass. MT5 EA stubs under `mt5/` are written only after GO.

## Config

- `configs/strategy_cssm.yaml` - strategy and sessions
- `configs/ftmo_2step.yaml` - prop rules
- `configs/costs.yaml` - spreads / slippage / commission
