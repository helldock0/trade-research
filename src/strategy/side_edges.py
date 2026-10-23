"""Two small side edges to run next to the NY kill-zone breakout.

Both come from branch claude/swing-strategy-fundamentals-v5mb8y
(configs/jpy_recheck.yaml, configs/strategy_dip.yaml) with the same rules,
re-implemented here on M5 bid/ask bars. They return KZTrade records (net R,
mae_r / mfe_r) so the prop and instant simulations can replay them with v3.

USDJPY gotobi short (Tokyo fix flow):
- Gotobi days: the 5th, 10th, 15th, 20th, 25th and month-end, moved to the
  previous weekday when on a weekend (Japanese holidays ignored).
- Sell at the bid open of the 00:55 UTC bar (09:55 JST); exit at the ask
  open of the 06:00 UTC bar. Stop = entry + sl_atr x ATR20, checked on the
  ask from the entry bar on; a bar opening past it fills at its open.
- ATR20 = mean daily high-low of New York-close (17:00) bid bars, through
  the previous day. With vol_filter, only days whose ATR20 is at or above
  its rolling median_days median are traded.
- Costs: commission per lot round turn and slippage per stop fill (the
  entry counts as one). No rollover is held, so no swap.

NAS100 buy the dip:
- Daily bars per US cash session (16:00 New York close); only sessions that
  trade to the full close count as days. Signal at the close: RSI(2) below
  rsi_entry and close above its trend_sma average. Long at the ask close.
- Exit at the bid close of the first later session that closes above its
  exit_sma average, or after max_days sessions. One position at a time.
- Stop = entry - stop_atr_mult x ATR20 (daily, through the previous
  session), checked on the bid; this is 1R. Financing is charged per
  calendar day held.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.bars import atr, ensure_utc_index
from src.strategy.nykz_breakout import KZTrade

NY = "America/New_York"


def is_gotobi(day: pd.Timestamp) -> bool:
    """5th / 10th / 15th / 20th / 25th / month-end, moved to the previous weekday when on a weekend."""
    day = pd.Timestamp(day).normalize()
    targets = [day.replace(day=k) for k in (5, 10, 15, 20, 25)] + [(day + pd.offsets.MonthEnd(0)).normalize()]
    for t in targets:
        while t.dayofweek >= 5:
            t -= pd.Timedelta(days=1)
        if t == day:
            return True
    return False


def ny_close_daily(bars: pd.DataFrame, median_days: int = 250) -> pd.DataFrame:
    """New York-close (17:00) daily bid bars (weekdays) with ATR20 and its rolling median, both known before the day."""
    x = ensure_utc_index(bars)
    if "volume" in x:
        x = x[x["volume"] > 0]
    day = (x.index.tz_convert(NY) + pd.Timedelta(hours=7)).normalize().tz_localize(None)
    g = x.groupby(day)
    d = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    d = d[d.index.dayofweek < 5]
    d["atr20"] = (d["high"] - d["low"]).rolling(20).mean().shift(1)
    d["atr_med"] = d["atr20"].rolling(median_days, min_periods=median_days).median()
    return d


def _pos(index: pd.DatetimeIndex, ts: pd.Timestamp, within: pd.Timedelta) -> int:
    """Row of the first bar at or after ts, if it opens before ts + within; else -1."""
    i = int(index.searchsorted(ts))
    return i if i < len(index) and index[i] < ts + within else -1


def gotobi_trades(bars: pd.DataFrame, daily: pd.DataFrame, sig: dict[str, Any]) -> list[KZTrade]:
    """USDJPY gotobi shorts on merged M5 bid/ask bars; `daily` from ny_close_daily."""
    pip = float(sig.get("pip", 0.01))
    lot = float(sig.get("lot", 100_000))
    commission = float(sig.get("commission_usd_per_lot", 7.0))
    slip = float(sig.get("slippage_pips", 0.2)) * pip
    sl_atr = float(sig.get("sl_atr", 0.25))
    vol_filter = bool(sig.get("vol_filter", True))
    entry_t = pd.Timedelta(str(sig.get("entry_utc", "00:55")) + ":00")
    exit_t = pd.Timedelta(str(sig.get("exit_utc", "06:00")) + ":00")
    trade_from = pd.Timestamp(sig.get("trade_from", "2018-01-01"))
    idx = bars.index
    ao, ah, al = (bars[f"ask_{c}"].to_numpy() for c in ("open", "high", "low"))
    bo = bars["open"].to_numpy()
    window = pd.Timedelta(minutes=5)

    trades: list[KZTrade] = []
    for day, d in daily.loc[trade_from:].iterrows():
        a, med = float(d["atr20"]), float(d["atr_med"])
        if not is_gotobi(day) or not np.isfinite(a) or (vol_filter and not (np.isfinite(med) and a >= med)):
            continue
        utc_day = pd.Timestamp(day, tz="UTC")
        i, j = _pos(idx, utc_day + entry_t, window), _pos(idx, utc_day + exit_t, pd.Timedelta(hours=1))
        if i < 0 or j <= i:
            continue
        entry = float(bo[i])
        risk = sl_atr * a
        stop = entry + risk
        hit = np.flatnonzero(ah[i:j] >= stop)
        if len(hit):
            k = i + int(hit[0])
            exit_px, reason = (stop, "stop") if k == i else (max(stop, float(ao[k])), "stop" if ao[k] < stop else "stop_gap")
            last = k
        else:
            k, exit_px, reason, last = j, float(ao[j]), "time", j - 1
        fills = 2 if reason.startswith("stop") else 1
        # P&L in JPY converted at the exit price; 1R = the stop distance in USD at the entry price
        pnl = (entry - exit_px) * lot / exit_px - commission - slip * fills * lot / exit_px
        r = pnl / (risk * lot / entry)
        mae = (float(ah[i : last + 1].max()) - entry) / risk
        mfe = (entry - float(al[i : last + 1].min())) / risk
        trades.append(KZTrade(
            day=str(day.date()), side=-1, range_high=float("nan"), range_low=float("nan"),
            break_time=idx[i], entry_time=idx[i], entry=entry, stop=stop, take_profit=None,
            risk_points=risk, spread_points=float(ao[i] - bo[i]), atr_points=a,
            exit_time=idx[k], exit_price=exit_px, exit_reason=reason, r_multiple=float(r),
            mae_r=float(max(min(mae, 1.0), -r, 0.0)), mfe_r=float(max(mfe, r, 0.0)),
        ))
    return trades


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Wilder RSI."""
    d = pd.Series(close, dtype=float).diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return 100.0 - 100.0 / (1.0 + up / dn)


def cash_sessions(bars: pd.DataFrame, cash_close: str = "16:00") -> pd.DataFrame:
    """Daily bid OHLC per US cash session (prior close -> this close) and the close time (UTC).

    Sessions whose last bar does not end at the cash close (holidays, early
    closes) are dropped, as are weekends.
    """
    x = ensure_utc_index(bars)
    hh, mm = (int(v) for v in cash_close.split(":"))
    local = x.index.tz_convert(NY)
    day = (local + pd.Timedelta(hours=24 - hh, minutes=-mm)).normalize().tz_localize(None)
    g = x.assign(_ts=x.index).groupby(day)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), last_ts=("_ts", "last"))
    out["close_ts"] = out.pop("last_ts") + x.index.to_series().diff().median()
    ends = out["close_ts"].dt.tz_convert(NY)
    full = (ends.dt.hour == hh) & (ends.dt.minute == mm)
    return out[full.to_numpy() & (out.index.dayofweek < 5)]


def dip_schedule(close: pd.Series, sig: dict[str, Any]) -> list[tuple[int, int]]:
    """(entry, exit) positions of the rule on a daily close series."""
    c = pd.Series(close, dtype=float)
    entry_ok = ((c > c.rolling(int(sig.get("trend_sma", 200))).mean()) & (rsi(c, int(sig.get("rsi_period", 2))) < float(sig.get("rsi_entry", 10)))).to_numpy()
    above_exit = (c > c.rolling(int(sig.get("exit_sma", 5))).mean()).to_numpy()
    max_days = int(sig.get("max_days", 10))
    out, i, n = [], 0, len(c)
    while i < n:
        if not entry_ok[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not above_exit[j] and (j - i) < max_days:
            j += 1
        if j >= n:
            break  # still open when the data ends
        out.append((i, j))
        i = j + 1
    return out


def dip_trades(bars: pd.DataFrame, daily: pd.DataFrame, sig: dict[str, Any]) -> list[KZTrade]:
    """NAS100 dip longs filled on merged M5 bid/ask bars; signals on `daily` (cash_sessions, may start earlier)."""
    stop_mult = float(sig.get("stop_atr_mult", 2.5))
    fin = float(sig.get("financing_pct_per_year", 6.0)) / 100 / 365
    trade_from = pd.Timestamp(sig.get("trade_from", "2018-01-01"))
    a = atr(daily["high"], daily["low"], daily["close"], period=int(sig.get("atr_period", 20))).shift(1).to_numpy()
    idx = bars.index
    bo, bl, bc = (bars[c].to_numpy() for c in ("open", "low", "close"))
    bh = bars["high"].to_numpy()
    ac = bars["ask_close"].to_numpy()
    days = daily.index

    trades: list[KZTrade] = []
    for i, j in dip_schedule(daily["close"], sig):
        if days[i] < trade_from or not np.isfinite(a[i]):
            continue
        # fills at the close of the last bar before the cash close
        p, q = (int(idx.searchsorted(daily["close_ts"].iloc[x])) - 1 for x in (i, j))
        if min(p, q) < 0 or any(idx[x] < daily["close_ts"].iloc[y] - pd.Timedelta(minutes=15) for x, y in ((p, i), (q, j))):
            continue  # no bars near the close for this trade
        entry = float(ac[p])
        risk = stop_mult * float(a[i])
        stop = entry - risk
        k, exit_px, reason = q, float(bc[q]), "time" if j - i >= int(sig.get("max_days", 10)) else "exit_sma"
        hit = np.flatnonzero(bl[p + 1 : q + 1] <= stop)
        if len(hit):
            k = p + 1 + int(hit[0])
            exit_px, reason = (float(bo[k]), "stop_gap") if bo[k] <= stop else (stop, "stop")
        held = (idx[k].tz_convert(NY).normalize() - idx[p].tz_convert(NY).normalize()).days
        r = (exit_px - entry - entry * fin * held) / risk
        span = slice(p + 1, k + 1)
        mae = (entry - float(bl[span].min())) / risk if k > p else 0.0
        mfe = (float(bh[span].max()) - entry) / risk if k > p else 0.0
        trades.append(KZTrade(
            day=str(days[i].date()), side=1, range_high=float("nan"), range_low=float("nan"),
            break_time=idx[p], entry_time=idx[p], entry=entry, stop=stop, take_profit=None,
            risk_points=risk, spread_points=float(ac[p] - bc[p]), atr_points=float(a[i]),
            exit_time=idx[k], exit_price=exit_px, exit_reason=reason, r_multiple=float(r),
            mae_r=float(max(min(mae, 1.0), -r, 0.0)), mfe_r=float(max(mfe, r, 0.0)),
        ))
    return trades
