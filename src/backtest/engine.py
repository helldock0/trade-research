from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import (
    commission_usd,
    entry_fill_price,
    exit_fill_price,
    position_size_lots,
)
from src.data.bars import ensure_utc_index
from src.features.sessions import is_friday, parse_hhmm
from src.strategy.cssm import Signal


@dataclass
class Trade:
    symbol: str
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop: float
    take_profit: float
    lots: float
    pnl_usd: float
    r_multiple: float
    session: str
    exit_reason: str
    commission: float


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    initial_balance: float = 100000.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])


def unrealized_r(side: int, entry: float, price: float, risk_price: float) -> float:
    """Signed R multiple from entry to price given initial stop distance."""
    if risk_price <= 0:
        return 0.0
    if side > 0:
        return (price - entry) / risk_price
    return (entry - price) / risk_price


def maybe_move_stop_to_breakeven(
    side: int,
    entry: float,
    stop: float,
    close: float,
    risk_price: float,
    breakeven_at_r: float,
    already_be: bool = False,
) -> tuple[float, bool]:
    """If close reaches +breakeven_at_r, move stop to entry. Returns (stop, is_be)."""
    if already_be or breakeven_at_r <= 0:
        return stop, already_be
    if unrealized_r(side, entry, close, risk_price) >= breakeven_at_r:
        return entry, True
    return stop, False


def stop_exit_reason(side: int, stop: float, entry: float, be_armed: bool) -> str:
    if be_armed and abs(stop - entry) < 1e-12:
        return "breakeven_stop"
    return "stop"


def run_backtest(
    signals: list[Signal],
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    initial_balance: float = 100000.0,
) -> BacktestResult:
    risk_pct = float(strategy_cfg["risk"]["risk_per_trade_pct"])
    max_open = int(strategy_cfg["risk"]["max_open_positions"])
    soft_daily = float(strategy_cfg["risk"]["soft_daily_halt_pct"])
    sessions_cfg = strategy_cfg.get("sessions", {})
    friday_flat = (
        parse_hhmm(sessions_cfg["friday_hard_flat"])
        if sessions_cfg.get("friday_hard_flat")
        else None
    )
    friday_no_new = (
        parse_hhmm(sessions_cfg["friday_no_new_entries"])
        if sessions_cfg.get("friday_no_new_entries")
        else None
    )
    london_end = parse_hhmm(sessions_cfg["london_end"])
    ny_end = parse_hhmm(sessions_cfg["ny_end"])
    exit_mode = str(strategy_cfg.get("signal", {}).get("exit_mode", "session"))
    breakeven_at_r = float(strategy_cfg.get("signal", {}).get("breakeven_at_r", 0.0))
    allow_weekend_hold = bool(strategy_cfg.get("signal", {}).get("allow_weekend_hold", False))
    max_hold_days = int(strategy_cfg.get("signal", {}).get("max_hold_days", 0) or 0)

    bars = {s: ensure_utc_index(df) for s, df in bars_by_symbol.items()}
    balance = initial_balance
    equity = initial_balance
    trades: list[Trade] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []

    # Max 1 global position
    position: dict[str, Any] | None = None

    day_start_equity = initial_balance
    current_day = None
    halted_today = False

    # Merge all timestamps
    all_index = sorted(set().union(*[set(b.index) for b in bars.values()]))
    sig_i = 0
    signals = sorted(signals, key=lambda s: s.timestamp)

    for ts in all_index:
        day = ts.normalize()
        if current_day is None or day != current_day:
            current_day = day
            day_start_equity = equity
            halted_today = False

        # Soft daily halt
        day_pnl_pct = (equity - day_start_equity) / initial_balance * 100.0
        if day_pnl_pct <= -soft_daily:
            halted_today = True

        # Manage open position
        if position is not None:
            sym = position["symbol"]
            bar = bars[sym].loc[ts] if ts in bars[sym].index else None
            if bar is not None:
                high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
                side = position["side"]
                entry = float(position["entry_price"])
                risk_price = float(position["risk_price"])
                tp = position["take_profit"]
                be_armed = bool(position.get("be_armed", False))
                stop = float(position["stop"])

                exit_mid = None
                reason = None

                # Intrabar stop/tp with stop from prior bars (no same-bar BE lookahead)
                if side > 0:
                    if low <= stop:
                        exit_mid, reason = stop, stop_exit_reason(side, stop, entry, be_armed)
                    elif high >= tp:
                        exit_mid, reason = tp, "tp"
                else:
                    if high >= stop:
                        exit_mid, reason = stop, stop_exit_reason(side, stop, entry, be_armed)
                    elif low <= tp:
                        exit_mid, reason = tp, "tp"

                # Session / Friday time stops (skipped in swing mode)
                sess = position["session"]
                if exit_mode != "swing":
                    if exit_mode == "hold_to_ny":
                        end_t = ny_end
                    else:
                        end_t = ny_end if sess == "ny" else london_end
                    if reason is None and ts.time() >= end_t:
                        exit_mid, reason = close, "session_end"
                    if (
                        reason is None
                        and not allow_weekend_hold
                        and friday_flat is not None
                        and is_friday(ts)
                        and ts.time() >= friday_flat
                    ):
                        exit_mid, reason = close, "friday_flat"

                # Max calendar-day hold (swing safety)
                if (
                    reason is None
                    and max_hold_days > 0
                    and (ts.normalize() - position["entry_time"].normalize()).days
                    >= max_hold_days
                ):
                    exit_mid, reason = close, "max_hold"

                # After surviving the bar, arm breakeven from close for subsequent bars
                if reason is None:
                    new_stop, be_armed = maybe_move_stop_to_breakeven(
                        side,
                        entry,
                        stop,
                        close,
                        risk_price,
                        breakeven_at_r,
                        already_be=be_armed,
                    )
                    position["stop"] = new_stop
                    position["be_armed"] = be_armed
                    stop = new_stop

                if reason is not None:
                    fill = exit_fill_price(side, exit_mid, sym, costs_cfg)
                    lots = position["lots"]
                    pnl = _pnl_usd(side, position["entry_price"], fill, lots, sym)
                    comm = commission_usd(lots, sym, costs_cfg)
                    pnl -= comm
                    r_mult = pnl / (initial_balance * risk_pct / 100.0)

                    trades.append(
                        Trade(
                            symbol=sym,
                            side=side,
                            entry_time=position["entry_time"],
                            exit_time=ts,
                            entry_price=position["entry_price"],
                            exit_price=fill,
                            stop=stop,
                            take_profit=tp,
                            lots=lots,
                            pnl_usd=pnl,
                            r_multiple=r_mult,
                            session=sess,
                            exit_reason=reason,
                            commission=comm,
                        )
                    )
                    balance += pnl
                    equity = balance
                    position = None

        equity_points.append((ts, equity))

        # Entries
        while sig_i < len(signals) and signals[sig_i].timestamp <= ts:
            sig = signals[sig_i]
            sig_i += 1
            if position is not None:
                continue
            if halted_today:
                continue
            if (
                not allow_weekend_hold
                and friday_no_new is not None
                and is_friday(ts)
                and ts.time() >= friday_no_new
            ):
                continue
            if sig.timestamp != ts:
                # fill only on signal bar close
                continue
            if max_open < 1:
                continue

            fill = entry_fill_price(sig.side, sig.entry, sig.symbol, costs_cfg)
            # Adjust stop/tp distance from actual fill
            risk_price = abs(fill - sig.stop)
            if risk_price <= 0:
                continue
            if sig.side > 0:
                tp = fill + float(strategy_cfg["signal"]["take_profit_r"]) * risk_price
            else:
                tp = fill - float(strategy_cfg["signal"]["take_profit_r"]) * risk_price

            lots = position_size_lots(
                equity, risk_pct, risk_price, sig.symbol, fill, costs_cfg
            )
            if lots <= 0:
                continue

            position = {
                "symbol": sig.symbol,
                "side": sig.side,
                "entry_time": ts,
                "entry_price": fill,
                "stop": sig.stop,
                "take_profit": tp,
                "lots": lots,
                "session": sig.session,
                "risk_price": risk_price,
                "be_armed": False,
            }

    # Force close leftover
    if position is not None:
        sym = position["symbol"]
        last_ts = bars[sym].index[-1]
        close = float(bars[sym].iloc[-1]["close"])
        fill = exit_fill_price(position["side"], close, sym, costs_cfg)
        lots = position["lots"]
        side = position["side"]
        pnl = _pnl_usd(side, position["entry_price"], fill, lots, sym)
        comm = commission_usd(lots, sym, costs_cfg)
        pnl -= comm
        trades.append(
            Trade(
                symbol=sym,
                side=side,
                entry_time=position["entry_time"],
                exit_time=last_ts,
                entry_price=position["entry_price"],
                exit_price=fill,
                stop=position["stop"],
                take_profit=position["take_profit"],
                lots=lots,
                pnl_usd=pnl,
                r_multiple=pnl / (initial_balance * risk_pct / 100.0),
                session=position["session"],
                exit_reason="eod_force",
                commission=comm,
            )
        )
        balance += pnl
        equity = balance

    curve = pd.Series(
        {t: e for t, e in equity_points},
        name="equity",
    ).sort_index()
    result = BacktestResult(
        trades=trades,
        equity_curve=curve,
        initial_balance=initial_balance,
        metrics=compute_metrics(trades, curve, initial_balance, risk_pct),
    )
    return result


def _pnl_usd(side: int, entry: float, exit_px: float, lots: float, symbol: str) -> float:
    """Approximate USD P&L (FX 100k lots; XAU 100 oz)."""
    sym = symbol.upper()
    diff = (exit_px - entry) if side > 0 else (entry - exit_px)
    if "XAU" in sym:
        return diff * lots * 100.0
    if sym.endswith("JPY"):
        # quote JPY: convert via exit price
        return diff * lots * 100000.0 / exit_px
    if sym.endswith("USD"):
        return diff * lots * 100000.0
    if sym.startswith("USD"):
        # USDCAD etc: P&L in quote, convert to USD
        return (diff * lots * 100000.0) / exit_px
    return diff * lots * 100000.0


def compute_metrics(
    trades: list[Trade],
    equity: pd.Series | None,
    initial_balance: float,
    risk_pct: float,
) -> dict[str, Any]:
    if not trades:
        return {
            "n_trades": 0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_dd_pct": 0.0,
            "net_pnl": 0.0,
            "total_return_pct": 0.0,
        }
    rs = np.array([t.r_multiple for t in trades], dtype=float)
    pnls = np.array([t.pnl_usd for t in trades], dtype=float)
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0

    max_dd = 0.0
    if equity is not None and not equity.empty:
        # downsample for speed
        eq = equity.iloc[:: max(1, len(equity) // 5000)]
        peak = eq.cummax()
        dd = (eq - peak) / initial_balance * 100.0
        max_dd = float((-dd).max()) if len(dd) else 0.0

    return {
        "n_trades": len(trades),
        "expectancy_r": float(rs.mean()),
        "profit_factor": pf,
        "win_rate": float((pnls > 0).mean()),
        "max_dd_pct": max_dd,
        "net_pnl": float(pnls.sum()),
        "total_return_pct": float(pnls.sum() / initial_balance * 100.0),
        "avg_r": float(rs.mean()),
        "risk_pct": risk_pct,
    }
