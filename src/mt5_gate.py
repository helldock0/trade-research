from __future__ import annotations

from pathlib import Path

from src.config import ROOT, ensure_dirs


CSSM_STUB = r'''//+------------------------------------------------------------------+
//| CSSM_EA.mq5 — stub (authorized only after research GO)           |
//+------------------------------------------------------------------+
#property copyright "cssm-prop"
#property version   "1.00"
#property strict

input double RiskPercent = 0.5;
input double TakeProfitR = 1.5;

int OnInit()
  {
   Print("CSSM EA stub initialized.");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) {}
void OnTick() {}
'''

ORB_STUB = r'''//+------------------------------------------------------------------+
//| ORB_EA.mq5 — stub (authorized only after research GO)            |
//+------------------------------------------------------------------+
#property copyright "orb-prop"
#property version   "1.00"
#property strict

input double RiskPercent = 0.5;
input double TakeProfitR = 1.5;

int OnInit()
  {
   Print("ORB EA stub initialized.");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) {}
void OnTick() {}
'''

HTF_STUB = r'''//+------------------------------------------------------------------+
//| HTF_PULLBACK_EA.mq5 — stub (authorized only after research GO)   |
//| H4 EMA200 bias + M15 EMA21 pullback reclaim                       |
//+------------------------------------------------------------------+
#property copyright "htf-prop"
#property version   "1.00"
#property strict

input double RiskPercent = 0.5;
input double TakeProfitR = 2.0;
input int    HtfEmaPeriod = 200;
input int    EntryEmaPeriod = 21;
input double AtrStopMult = 1.5;

int OnInit()
  {
   Print("HTF pullback EA stub initialized. Wire signals after demo parity.");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) {}

void OnTick()
  {
   // TODO: port Python HTF pullback rules:
   // - H4 close vs EMA200 bias (completed bars only)
   // - M15 EMA21 pullback reclaim in bias direction
   // - ATR stop, TP 2R, hold_to_ny, Friday hard flat
  }
'''


def _ea_filename(strategy_name: str) -> str:
    sid = strategy_name.lower()
    if sid == "orb":
        return "ORB_EA.mq5"
    if sid in ("htf_pullback", "htf"):
        return "HTF_PULLBACK_EA.mq5"
    return "CSSM_EA.mq5"


def _ea_stub(strategy_name: str) -> str:
    sid = strategy_name.lower()
    if sid == "orb":
        return ORB_STUB
    if sid in ("htf_pullback", "htf"):
        return HTF_STUB
    return CSSM_STUB


def _nogo_next_hint(strategy_name: str) -> str:
    sid = strategy_name.lower()
    if sid in ("htf_pullback", "htf"):
        return (
            "Next research cycle on NO-GO for HTF pullback: Donchian + ATR (H1) "
            "(do not stack HTF knobs).\n"
        )
    if sid == "orb":
        return (
            "Next research cycle on NO-GO for ORB: HTF trend + M15 pullback "
            "(do not stack ORB knobs).\n"
        )
    if sid == "msnr":
        return (
            "Next research cycle on NO-GO for MSNR: Donchian + ATR (H1) "
            "(do not stack MSNR knobs).\n"
        )
    if sid in ("vwap_fade", "vwap"):
        return (
            "Next research cycle on NO-GO for VWAP fade: Donchian + ATR (H1) "
            "(do not stack VWAP knobs).\n"
        )
    if sid in ("donchian_atr", "donchian"):
        return (
            "Next research cycle on NO-GO for Donchian: XAUUSD HTF pullback "
            "(do not stack Donchian knobs).\n"
        )
    return "Re-run walk-forward after hypothesis changes.\n"


def maybe_write_mt5_stub(promoted: bool, strategy_name: str = "cssm") -> Path | None:
    ensure_dirs()
    mt5_dir = ROOT / "mt5"
    mt5_dir.mkdir(parents=True, exist_ok=True)
    readme = mt5_dir / "README.md"
    ea_name = _ea_filename(strategy_name)
    ea = mt5_dir / ea_name

    if not promoted:
        readme.write_text(
            f"# MT5\n\nPromotion verdict for **{strategy_name}** is **NO-GO**. "
            f"EA stub `{ea_name}` not written.\n"
            "Re-run walk-forward after hypothesis changes.\n"
            + _nogo_next_hint(strategy_name),
            encoding="utf-8",
        )
        if ea.exists():
            ea.unlink()
        return None

    ea.write_text(_ea_stub(strategy_name), encoding="utf-8")
    readme.write_text(
        f"# MT5\n\nPromotion **GO** ({strategy_name}) — stub EA written to `{ea_name}`.\n\n"
        "## Demo parity checklist\n"
        "1. Compare signal timestamps vs Python backtest on same week\n"
        "2. Log fill slippage vs `configs/costs.yaml`\n"
        "3. Paper trade ≥20 sessions before challenge size\n",
        encoding="utf-8",
    )
    return ea
