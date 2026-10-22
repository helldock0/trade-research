from __future__ import annotations

from typing import Any


def pair_cost(symbol: str, costs_cfg: dict[str, Any]) -> dict[str, float]:
    base = dict(costs_cfg.get("default", {}))
    base.update(costs_cfg.get("pairs", {}).get(symbol, {}))
    return {
        "half_spread_pips": float(base.get("half_spread_pips", 0.6)),
        "slippage_pips": float(base.get("slippage_pips", 0.2)),
        "commission_per_lot_usd": float(base.get("commission_per_lot_usd", 7.0)),
    }


def pip_size(symbol: str, costs_cfg: dict[str, Any]) -> float:
    sym = symbol.upper()
    if "XAU" in sym:
        default = 0.01
    elif "JPY" in sym:
        default = 0.01
    else:
        default = 0.0001
    return float(costs_cfg.get("pip_size", {}).get(symbol, default))


def entry_fill_price(side: int, mid: float, symbol: str, costs_cfg: dict[str, Any]) -> float:
    c = pair_cost(symbol, costs_cfg)
    pip = pip_size(symbol, costs_cfg)
    adverse = (c["half_spread_pips"] + c["slippage_pips"]) * pip
    return mid + adverse if side > 0 else mid - adverse


def exit_fill_price(side: int, mid: float, symbol: str, costs_cfg: dict[str, Any]) -> float:
    c = pair_cost(symbol, costs_cfg)
    pip = pip_size(symbol, costs_cfg)
    adverse = (c["half_spread_pips"] + c["slippage_pips"]) * pip
    # exiting long sells into bid
    return mid - adverse if side > 0 else mid + adverse


def pip_value_per_lot(symbol: str, price: float) -> float:
    """Approximate USD pip value for 1.0 standard lot.

    FX: 100k base. XAUUSD: 100 oz contract, $1 per 0.01 move per lot.
    """
    sym = symbol.upper()
    if "XAU" in sym:
        return 1.0
    if sym.endswith("USD"):
        # EURUSD etc: 1 pip = 10 USD per lot for 0.0001 pip
        return 10.0 if "JPY" not in sym else 1000.0 / price  # shouldn't hit
    if sym.startswith("USD") and "JPY" in sym:
        # USDJPY: pip 0.01, pip value ≈ 100000 * 0.01 / price
        return 100000.0 * 0.01 / price
    if sym.startswith("USD"):
        # USDCAD: quote CAD, rough USD pip value
        return 10.0 / price
    return 10.0


def is_cross(symbol: str) -> bool:
    """FX pair without a USD leg (EURJPY, EURGBP, ...): its P&L is in a non-USD quote currency."""
    sym = symbol.upper()
    return len(sym) == 6 and sym.isalpha() and "USD" not in sym and "XAU" not in sym


def contract_size(symbol: str, costs_cfg: dict[str, Any]) -> float | None:
    """Units per lot for instruments under `contracts:` in costs.yaml (USD index CFDs: $ per point per lot).

    None means the FX / gold defaults (100k units, 100 oz).
    """
    v = (costs_cfg.get("contracts") or {}).get(symbol.upper())
    return float(v) if v is not None else None


def position_size_lots(
    equity: float,
    risk_pct: float,
    risk_price: float,
    symbol: str,
    price: float,
    costs_cfg: dict[str, Any],
    usd_per_quote: float | None = None,
    contract: float | None = None,
) -> float:
    """Lots risking risk_pct of equity over risk_price.

    `usd_per_quote` (USD value of one unit of the quote currency) and `contract`
    (units per lot) price the pip exactly; the engine passes them for crosses
    and index CFDs, where pip_value_per_lot cannot.
    """
    risk_cash = equity * (risk_pct / 100.0)
    pip = pip_size(symbol, costs_cfg)
    stop_pips = risk_price / pip
    if stop_pips <= 0:
        return 0.0
    if usd_per_quote or contract:
        pv = pip * (contract or 100000.0) * (usd_per_quote or 1.0)
    else:
        pv = pip_value_per_lot(symbol, price)
    if pv <= 0:
        return 0.0
    lots = risk_cash / (stop_pips * pv)
    return max(0.01, round(lots, 2))


def commission_usd(lots: float, symbol: str, costs_cfg: dict[str, Any]) -> float:
    c = pair_cost(symbol, costs_cfg)
    return abs(lots) * float(c["commission_per_lot_usd"])
