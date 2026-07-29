# trade-research

Private research harness for FTMO-style prop challenge strategies.

Local market data: `data/parquet` (gitignored). Run outputs: `reports/` (gitignored).

## Strategy status

| Id | Idea | Status |
|----|------|--------|
| `cssm` | Cross-sectional session momentum | NO-GO |
| `orb` | Session opening-range breakout | NO-GO |
| `csmo` | Cross-sectional momentum open | NO-GO |
| `d1_mom` | D1 20d cross-sectional momentum | NO-GO |
| `htf_pullback` | H4 EMA200 + M15 EMA21 pullback | NO-GO |
| `htf_adx_pullback` | H4 EMA200 + ADX gate + M15 reclaim | NO-GO |
| `fail_fade` | London fail fade of Asian range | NO-GO |
| `msnr` | H4 MSNR zones + M15 rejection | NO-GO |
| `vwap_fade` | Session VWAP stretch fade | NO-GO |
| `donchian_atr` | H1 Donchian breakout + ATR stop | NO-GO |
| `xau_htf_pullback` | XAUUSD H4/M15 EMA pullback | research / unproven |

## Setup

```bash
cd D:\AI\trade
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

Optional (Windows):

```powershell
$env:PYTHONPATH="D:\AI\trade"
$env:PYTHONIOENCODING="utf-8"
```

## Workflow

```bash
# 1) Dukascopy OHLC -> data/parquet
python scripts/download_data.py

# 2) Single strategy
python scripts/run_backtest.py --strategy xau_htf_pullback
python scripts/run_prop_sim.py --strategy xau_htf_pullback
python scripts/run_walkforward.py --strategy xau_htf_pullback

# 2a) Backtest + FTMO for all (or a subset)
python scripts/run_all.py
python scripts/run_all.py --strategies cssm,orb

# 3) Tests
python -m pytest -q
```

Outputs land under `reports/` (per-strategy JSON/CSV plus `batch_summary.json`).

## Layout

```
configs/          strategy + FTMO + cost YAMLs
scripts/          download, backtest, prop sim, walk-forward, batch
src/              data, features, strategies, backtest, prop, research
tests/
mt5/              EA stubs only after promotion GO
docs/             tracked research notes
data/             local parquet + logs (ignored)
reports/          local run outputs (ignored)
```

## Configs

- Strategies: `configs/strategy_<id>.yaml`
- Prop rules: `configs/ftmo_2step.yaml`
- Costs: `configs/costs.yaml`

## Promotion

`src/research/promote.py` prints **GO** only if gates pass. MT5 stubs under `mt5/` only after GO.

See [docs/STATUS.md](docs/STATUS.md) for the research ledger.
