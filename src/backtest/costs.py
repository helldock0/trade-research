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
    return float(
        costs_cfg.get("pip_size", {}).get(
            symbol, 0.01 if "JPY" in symbol.upper() else 0.0001
        )
    )


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
    """Approximate USD pip value for 1.0 standard lot (100k base)."""
    sym = symbol.upper()
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


def position_size_lots(
    equity: float,
    risk_pct: float,
    risk_price: float,
    symbol: str,
    price: float,
    costs_cfg: dict[str, Any],
) -> float:
    risk_cash = equity * (risk_pct / 100.0)
    pip = pip_size(symbol, costs_cfg)
    stop_pips = risk_price / pip
    if stop_pips <= 0:
        return 0.0
    pv = pip_value_per_lot(symbol, price)
    if pv <= 0:
        return 0.0
    lots = risk_cash / (stop_pips * pv)
    return max(0.01, round(lots, 2))


def commission_usd(lots: float, symbol: str, costs_cfg: dict[str, Any]) -> float:
    c = pair_cost(symbol, costs_cfg)
    return abs(lots) * float(c["commission_per_lot_usd"])
