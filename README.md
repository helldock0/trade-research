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
| `event_drift` | Pre-release drift / surprise / fade on macro releases | RESEARCH |
| `macro_swing` | Macro currency scores (rates, carry, risk, commodities, positioning) + D1 trigger | RESEARCH |

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

# 1b) Economic calendar (FXStreet) + FRED gasoline -> data/events_calendar.csv, data/macro/
python scripts/download_calendar.py --gasoline

# 1c) Macro series (FRED, ECB, BoC, RBA, MOF, BoE, BIS, CFTC) -> data/macro/
python scripts/download_macro.py --show

# 2) Single strategy
python scripts/run_backtest.py --strategy htf_adx_pullback
python scripts/run_prop_sim.py --strategy htf_adx_pullback
python scripts/run_walkforward.py --strategy htf_adx_pullback

# 1d) Macro IC check (FRED daily FX, no bars): does the macro_swing bias predict returns?
python scripts/run_macro_ic.py
python scripts/run_macro_ic.py --validate    # signal variants: dev 2018+ vs holdout 2012-2017

# cloud sessions without data/parquet: M15 bars from the data branch (Git LFS)
python scripts/bars_from_data_branch.py

# 1f) Turn of the month on index CFDs: FRED signal check first, then bars vs a mid-month placebo
python scripts/run_tom_check.py
python scripts/bars_from_data_branch.py --pairs NAS100,US500
python scripts/run_tom_backtest.py

# 1g) Intraday momentum on index CFDs (last half hour): bar signal check net of costs, then the engine
python scripts/bars_from_data_branch.py --pairs US500,NAS100,US30
python scripts/run_imom_check.py
python scripts/run_variants.py --config strategy_imom.yaml --stem imom_compare

# 1h) Buy the dip in an uptrend on index CFDs: FRED signal check, then the engine
python scripts/run_dip_check.py
python scripts/run_variants.py --config strategy_dip.yaml --stem dip_compare

# 1e) macro_swing variants side by side: fixed target vs trailing stop vs correlation-gated basket
python scripts/run_macro_compare.py

# 2b) Event study: pre / post / fade variants side by side
python scripts/run_event_study.py
python scripts/run_event_study.py --signal-mode nowcast_gap --tag cpi_nowcast

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
scripts/          download (bars, calendar, macro), backtest, prop sim, walk-forward, batch, event study, macro IC / compare
src/              data, features, strategies, backtest, prop, research
tests/
mt5/              EA stubs only after promotion GO
docs/             tracked research notes
data/             local parquet, events_calendar.csv, macro/ (ignored)
reports/          local run outputs (ignored)
```

## Configs

- Strategies: `configs/strategy_<id>.yaml`
- Prop rules: `configs/ftmo_2step.yaml`
- Costs: `configs/costs.yaml`

## Promotion

`src/research/promote.py` prints **GO** only if gates pass. MT5 stubs under `mt5/` only after GO.

See [docs/STATUS.md](docs/STATUS.md) for the research ledger.
