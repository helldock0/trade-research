# Prop challenge research harness

Greenfield research harness for FTMO-style prop challenge strategies.

**Strategies:**
- `cssm` — Cross-Sectional Session Momentum — **NO-GO**
- `orb` — Pure Session Opening Range Breakout — **NO-GO**
- `csmo` — Cross-Sectional Momentum Open — **NO-GO**
- `d1_mom` — D1 20d cross-sectional momentum (swing/weekend) — **NO-GO**
- `htf_pullback` — H4 EMA200 + M15 EMA21 pullback (swing/weekend) — **NO-GO**
- `fail_fade` — London fail fade of Asian range — **NO-GO**
- `msnr` — H4 swing MSNR zones + M15 rejection (swing/weekend) — **NO-GO**
- `vwap_fade` — Session VWAP stretch fade (London/NY) — **NO-GO**
- `donchian_atr` — H1 Donchian breakout + ATR stop (swing/weekend) — **NO-GO**
- `xau_htf_pullback` — XAUUSD H4 EMA200 + M15 EMA21 pullback (swing/weekend) — **research / unproven**
- `htf_adx_pullback` — H4 EMA200 + ADX gate + M15 deep EMA reclaim (swing/weekend) — **NO-GO**

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

# 2) Backtest / FTMO / walk-forward (single strategy)
python scripts/run_backtest.py --strategy htf_pullback
python scripts/run_prop_sim.py --strategy htf_pullback
python scripts/run_walkforward.py --strategy htf_pullback

# 2a) Backtest + FTMO for all strategies (skips walk-forward)
python scripts/run_all.py
# optional subset:
python scripts/run_all.py --strategies cssm,orb
# Walk-forward / promote remains single-strategy:
#   python scripts/run_walkforward.py --strategy <id>
# Outputs: per-strategy reports under reports/ plus aggregate reports/batch_summary.json

# 3) Unit tests
python -m pytest -q
```

## Current research status

See `reports/diagnosis.md`. CSSM, ORB, CSMO, HTF pullback, HTF ADX pullback, D1 mom, fail_fade, vwap_fade, msnr, and donchian_atr are **NO-GO**. Current cycle: XAUUSD HTF pullback.

## Promotion

`src/research/promote.py` prints **GO** only if gates pass. MT5 stubs under `mt5/` only after GO.

## Config

- `configs/strategy_d1_mom.yaml` - D1 CS momentum (rejected)
- `configs/strategy_htf_pullback.yaml` - HTF pullback
- `configs/strategy_orb.yaml` - Pure Session ORB (rejected)
- `configs/strategy_csmo.yaml` - CSMO (rejected)
- `configs/strategy_cssm.yaml` - CSSM (rejected)
- `configs/strategy_msnr.yaml` - MSNR v1 (rejected)
- `configs/strategy_vwap_fade.yaml` - Session VWAP fade (rejected)
- `configs/strategy_donchian_atr.yaml` - Donchian + ATR H1 (rejected)
- `configs/strategy_xau_htf_pullback.yaml` - XAUUSD HTF pullback (research)
- `configs/strategy_htf_adx_pullback.yaml` - HTF ADX pullback (rejected)
- `configs/ftmo_2step.yaml` - prop rules
- `configs/costs.yaml` - spreads / slippage / commission
