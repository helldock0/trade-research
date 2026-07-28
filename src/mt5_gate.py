from __future__ import annotations

from pathlib import Path

from src.config import ROOT, ensure_dirs


MT5_STUB = r'''//+------------------------------------------------------------------+
//| CSSM_EA.mq5 — stub (authorized only after research GO)           |
//| Cross-Sectional Session Momentum                                  |
//+------------------------------------------------------------------+
#property copyright "cssm-prop"
#property version   "1.00"
#property strict

input double RiskPercent = 0.5;
input double TakeProfitR = 1.5;
input int    StopBufferPips = 2;
input bool   EnableNYSession = true;

// Demo parity logging
int OnInit()
  {
   Print("CSSM EA stub initialized. Wire signals after demo parity checklist.");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) {}

void OnTick()
  {
   // TODO: port Python CSSM rules:
   // - Asian range, overnight cross-section ranks
   // - London/NY break in momentum direction
   // - Friday hard flat 20:00 UTC
   // - FTMO soft daily halt 2%
  }
'''


def maybe_write_mt5_stub(promoted: bool) -> Path | None:
    ensure_dirs()
    mt5_dir = ROOT / "mt5"
    mt5_dir.mkdir(parents=True, exist_ok=True)
    readme = mt5_dir / "README.md"
    if not promoted:
        readme.write_text(
            "# MT5\n\nPromotion verdict is **NO-GO**. EA stub not written.\n"
            "Re-run walk-forward after hypothesis changes.\n",
            encoding="utf-8",
        )
        ea = mt5_dir / "CSSM_EA.mq5"
        if ea.exists():
            ea.unlink()
        return None

    ea_path = mt5_dir / "CSSM_EA.mq5"
    ea_path.write_text(MT5_STUB, encoding="utf-8")
    readme.write_text(
        "# MT5\n\nPromotion **GO** — stub EA written to `CSSM_EA.mq5`.\n\n"
        "## Demo parity checklist\n"
        "1. Compare signal timestamps vs Python backtest on same week\n"
        "2. Log fill slippage vs `configs/costs.yaml`\n"
        "3. Paper trade ≥20 sessions before challenge size\n",
        encoding="utf-8",
    )
    return ea_path
