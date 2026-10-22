from __future__ import annotations

from typing import Any

# Quote currency of each index CFD (prices are in index points).
INDEX_QUOTE = {
    "US30": "USD",
    "NAS100": "USD",
    "SPX500": "USD",
    "GER40": "EUR",
    "UK100": "GBP",
    "JPN225": "JPY",
}

# Rough USD value of one unit of each quote currency (commission conversion only).
QUOTE_USD_APPROX = {
    "USD": 1.0,
    "EUR": 1.10,
    "GBP": 1.27,
    "JPY": 0.0075,
    "CHF": 1.12,
    "CAD": 0.74,
    "AUD": 0.67,
    "NZD": 0.61,
}


def is_index(symbol: str) -> bool:
    return symbol.upper() in INDEX_QUOTE


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
    if is_index(sym):
        default = 1.0
    elif "XAU" in sym:
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


def contract_size(symbol: str) -> float:
    """Units per 1.0 lot: FX 100k base, XAUUSD 100 oz, index CFDs 1 per point."""
    sym = symbol.upper()
    if is_index(sym):
        return 1.0
    if "XAU" in sym:
        return 100.0
    return 100000.0


def round_trip_cost_price(symbol: str, costs_cfg: dict[str, Any]) -> tuple[float, float]:
    """(entry-side cost, full round-trip cost) in price units for one trade.

    Spread + slippage are paid on both fills; commission per lot is converted to
    price units with an approximate quote->USD rate.
    """
    c = pair_cost(symbol, costs_cfg)
    pip = pip_size(symbol, costs_cfg)
    side_cost = (c["half_spread_pips"] + c["slippage_pips"]) * pip
    sym = symbol.upper()
    quote = INDEX_QUOTE.get(sym, sym[3:6])
    quote_usd = QUOTE_USD_APPROX.get(quote, 1.0)
    commission = c["commission_per_lot_usd"] / (contract_size(sym) * quote_usd)
    return side_cost, 2.0 * side_cost + commission
