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
| `xau_htf_pullback` | XAUUSD H4/M15 EMA pullback | NO-GO |
| `nykz_breakout` | NAS100 NY kill-zone (07-10 NY) range breakout, next-candle confirm, 2x ATR stop, 4R | v3 ~+0.136R, testing |

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
python scripts/run_backtest.py --strategy htf_adx_pullback
python scripts/run_prop_sim.py --strategy htf_adx_pullback
python scripts/run_walkforward.py --strategy htf_adx_pullback

# 2a) Backtest + FTMO for all (or a subset)
python scripts/run_all.py
python scripts/run_all.py --strategies cssm,orb

# 2b) NAS100 NY kill-zone breakout (standalone: bid/ask fills)
python scripts/download_data.py --symbols NAS100 --intervals M5 --start 2023-01-01
python scripts/download_data.py --symbols NAS100 --intervals M5 --start 2023-01-01 --side ask
python scripts/run_nykz_breakout.py
python scripts/compare_nykz_breakout.py --grid break   # clean-break / stop placement, 2023-26 + 2018-22
python scripts/compare_nykz_breakout.py --grid exit    # ATR stop size / breakeven / trailing
python scripts/compare_nykz_breakout.py --grid fvg     # FVG retest / limit entries vs next close
python scripts/compare_nykz_breakout.py --grid rr      # take-profit targets on the config's rules
python scripts/compare_nykz_breakout.py --grid weekday # skip one weekday at a time
python scripts/compare_nykz_breakout.py --grid session # earlier entry cutoffs (e.g. London overlap)

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
