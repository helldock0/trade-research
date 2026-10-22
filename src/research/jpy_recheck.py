"""Independent re-check of gj-strat's USDJPY "gotobi" and "vol-box" edges (configs/jpy_recheck.yaml).

Everything here is written from the rules as described; nothing from gj-strat is imported.
Data: Dukascopy 1-minute BID/ASK candle files (.bi5) on a continuous UTC minute grid.
Fills follow the stated conventions: buys trigger and fill on the ask, sells on the bid; longs exit
on the bid and shorts on the ask; a stop gapped through fills at the bar's open; when one bar
touches both the stop and the target, the stop wins. Net R = (gross P&L - commission - slippage
+ swap) / the USD risked, per lot.
"""

from __future__ import annotations

import json
import lzma
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOT = 100_000
NY = "America/New_York"
_MEMBER = re.compile(r"^(?P<sym>[A-Z]+)_(?P<d>\d{8})_(?P<side>BID|ASK)\.bi5$")


# ----------------------------------------------------------------- data


def decode_candles(blob: bytes, point: float) -> np.ndarray:
    """Dukascopy day candle file -> array [minute, open, high, low, close, volume] (24-byte big-endian records)."""
    if not blob:
        return np.empty((0, 6))
    raw = lzma.decompress(blob)
    rec = np.frombuffer(raw, dtype=np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")]))
    out = np.empty((len(rec), 6))
    out[:, 0] = rec["t"] // 60
    out[:, 1], out[:, 2], out[:, 3], out[:, 4] = rec["o"] * point, rec["h"] * point, rec["l"] * point, rec["c"] * point
    out[:, 5] = rec["v"]
    return out


@dataclass
class Minutes:
    t0: pd.Timestamp                  # naive UTC of index 0
    bo: np.ndarray
    bh: np.ndarray
    bl: np.ndarray
    bc: np.ndarray
    ao: np.ndarray
    ah: np.ndarray
    al: np.ndarray
    ac: np.ndarray
    vol: np.ndarray                   # bid tick volume; 0 = no ticks that minute

    @property
    def n(self) -> int:
        return len(self.bo)

    def idx(self, ts: pd.Timestamp) -> int:
        return int((pd.Timestamp(ts) - self.t0) // pd.Timedelta(minutes=1))

    def ts(self, i: int) -> pd.Timestamp:
        return self.t0 + pd.Timedelta(minutes=int(i))

    def frame(self) -> pd.DataFrame:
        idx = pd.date_range(self.t0, periods=self.n, freq="min")
        return pd.DataFrame({k: getattr(self, k) for k in ("bo", "bh", "bl", "bc", "ao", "ah", "al", "ac", "vol")}, index=idx)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Minutes:
        return cls(pd.Timestamp(df.index[0]), *(df[k].to_numpy(dtype=float) for k in ("bo", "bh", "bl", "bc", "ao", "ah", "al", "ac", "vol")))


def load_datastore(datastore: Path, symbol: str, start: str, end: str, point: float = 0.001) -> Minutes:
    """Decode every {symbol}_*.tar day file in [start, end] onto one UTC minute grid (forward-filled, vol 0 where empty)."""
    d0, d1 = pd.Timestamp(start), pd.Timestamp(end)
    files: dict[tuple[pd.Timestamp, str], bytes] = {}
    for tp in sorted(Path(datastore).glob(f"{symbol}*.tar")):
        with tarfile.open(tp) as tf:
            for mem in tf.getmembers():
                mt = _MEMBER.match(Path(mem.name).name)
                if not mt or mt["sym"] != symbol or not mem.isfile():
                    continue
                day = pd.Timestamp(mt["d"])
                if d0 <= day <= d1:
                    files[(day, mt["side"])] = tf.extractfile(mem).read()
    n = int(((d1 + pd.Timedelta(days=1)) - d0) / pd.Timedelta(minutes=1))
    cols = {k: np.full(n, np.nan) for k in ("bo", "bh", "bl", "bc", "ao", "ah", "al", "ac")}
    vol = np.zeros(n)
    for (day, side), blob in files.items():
        c = decode_candles(blob, point)
        if not len(c):
            continue
        pos = (int((day - d0) / pd.Timedelta(minutes=1)) + c[:, 0]).astype(int)
        ok = (pos >= 0) & (pos < n)
        p = "b" if side == "BID" else "a"
        for j, k in enumerate("ohlc", start=1):
            cols[p + k][pos[ok]] = c[ok, j]
        if side == "BID":
            vol[pos[ok]] = c[ok, 5]
    df = pd.DataFrame(cols).ffill().bfill()
    return Minutes(d0, *(df[k].to_numpy() for k in ("bo", "bh", "bl", "bc", "ao", "ah", "al", "ac")), vol)


def ny_daily(m: Minutes, median_days: int = 250) -> pd.DataFrame:
    """New York-close (17:00) daily BID bars on minutes with ticks; ATR20 and its rolling median known before the day."""
    f = m.frame()
    f = f[f["vol"] > 0]
    local = f.index.tz_localize("UTC").tz_convert(NY)
    day = (local + pd.Timedelta(hours=7)).normalize().tz_localize(None)     # 17:00 NY rolls to the next date
    g = f.groupby(day)
    d = pd.DataFrame({"high": g["bh"].max(), "low": g["bl"].min(), "close": g["bc"].last()})
    d = d[d.index.dayofweek < 5]
    d["atr20"] = (d["high"] - d["low"]).rolling(20).mean().shift(1)
    d["atr_med"] = d["atr20"].rolling(median_days, min_periods=median_days).median()
    return d


# ----------------------------------------------------------------- calendars


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


# ----------------------------------------------------------------- costs


@dataclass
class Costs:
    commission: float = 7.0
    slippage_pips: float = 0.2
    swap_markup_pct: float = 1.0
    rates: pd.DataFrame | None = None     # columns USD, JPY: policy rates in % by date

    def rollovers(self, t_in: pd.Timestamp, t_out: pd.Timestamp) -> int:
        """17:00 New York rollovers crossed while holding; the Wednesday one counts three times."""
        a = pd.Timestamp(t_in).tz_localize("UTC").tz_convert(NY)
        b = pd.Timestamp(t_out).tz_localize("UTC").tz_convert(NY)
        n, k = 0, (a - pd.Timedelta(hours=17)).normalize() + pd.Timedelta(days=1, hours=17)
        while k <= b:
            if k.dayofweek < 5:
                n += 3 if k.dayofweek == 2 else 1
            k += pd.Timedelta(days=1)
        return n

    def swap_usd_per_lot(self, side: int, t_in: pd.Timestamp, t_out: pd.Timestamp) -> float:
        nights = self.rollovers(t_in, t_out)
        if nights == 0:
            return 0.0
        diff = 0.0
        if self.rates is not None and len(self.rates):
            r = self.rates.asof(pd.Timestamp(t_in).normalize())
            diff = float(r["USD"] - r["JPY"])
        return LOT * (side * diff - self.swap_markup_pct) / 100.0 / 360.0 * nights


def net_r(side: int, entry: float, exit_: float, risk: float, fills: int, t_in, t_out, pip: float, costs: Costs) -> tuple[float, float]:
    """(gross R, net R) per lot of USDJPY: P&L in JPY converted at the exit price, risk at the entry price."""
    move = (exit_ - entry) * side
    gross = move / risk
    pv_exit = LOT * pip / exit_                     # USD per pip per lot
    pv_entry = LOT * pip / entry
    pnl = move / pip * pv_exit - costs.commission - costs.slippage_pips * fills * pv_exit + costs.swap_usd_per_lot(side, t_in, t_out)
    return gross, pnl / (risk / pip * pv_entry)


# ----------------------------------------------------------------- gotobi


def gotobi_trades(m: Minutes, daily: pd.DataFrame, cfg: dict[str, Any], costs: Costs, pip: float = 0.01, pick=is_gotobi) -> pd.DataFrame:
    """SELL at the entry minute's bid open on days `pick` selects (gotobi days); SL on the ask; time exit at the exit minute's ask open."""
    eh, em = (int(x) for x in cfg["entry_utc"].split(":"))
    xh, xm = (int(x) for x in cfg["exit_utc"].split(":"))
    rows = []
    for day, r in daily.iterrows():
        if not pick(day) or not np.isfinite(r["atr20"]):
            continue
        i = m.idx(day + pd.Timedelta(hours=eh, minutes=em))
        j = m.idx(day + pd.Timedelta(hours=xh, minutes=xm))
        if i < 0 or j >= m.n or j <= i:
            continue
        entry = m.bo[i]
        risk = float(cfg["sl_atr"]) * float(r["atr20"])
        sl = entry + risk
        xi, xp, why = j, m.ao[j], "TIME"
        if m.ah[i] >= sl:
            xi, xp, why = i, sl, "SL"
        else:
            hit = np.flatnonzero(m.ah[i + 1 : j] >= sl)
            if len(hit):
                k = i + 1 + int(hit[0])
                xi, xp, why = k, (max(sl, m.ao[k])), "SL"
        gross, net = net_r(-1, entry, xp, risk, 1 + (why == "SL"), m.ts(i), m.ts(xi), pip, costs)
        rows.append({"day": day, "entry_time": m.ts(i), "entry": entry, "exit_time": m.ts(xi), "exit": xp,
                     "risk_pips": risk / pip, "reason": why, "gross_r": gross, "r": net,
                     "hot": bool(r["atr20"] >= r["atr_med"]) if np.isfinite(r["atr_med"]) else False})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- box (Version B)


def _first(mask: np.ndarray) -> int:
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else -1


def _scan(m: Minutes, start: int, fn) -> int:
    """First index >= start where fn(slice) is True, scanning in growing chunks; -1 if never."""
    s, chunk = start, 512
    while s < m.n:
        e = min(s + chunk, m.n)
        k = _first(fn(slice(s, e)))
        if k >= 0:
            return s + k
        s, chunk = e, chunk * 4
    return -1


def find_stop_entry(m: Minutes, a: int, b: int, buy: float | None, sell: float | None) -> tuple[int, int, float] | None:
    """First fill in [a, b) of a buy stop (ask >= buy) and/or a sell stop (bid <= sell): (side, index, price)."""
    if a >= b:
        return None
    bh = m.ah[a:b] >= buy if buy is not None else np.zeros(b - a, bool)
    sh = m.bl[a:b] <= sell if sell is not None else np.zeros(b - a, bool)
    k = _first(bh | sh)
    if k < 0:
        return None
    i = a + k
    if bh[k] and sh[k]:
        over_b, over_s = m.ao[i] - buy, sell - m.bo[i]
        if over_b >= 0 or over_s >= 0:
            side = 1 if over_b >= over_s else -1
        else:
            side = 1 if -over_b <= -over_s else -1
    else:
        side = 1 if bh[k] else -1
    if side > 0:
        return 1, i, max(buy, m.ao[i])
    return -1, i, min(sell, m.bo[i])


def manage(m: Minutes, side: int, ei: int, entry: float, risk: float, rr: float, be_r: float, be_off: float) -> tuple[int, float, str]:
    """Hold to SL, TP or (after be_r) the break-even stop; the stop wins a shared bar."""
    if side > 0:
        lo, hi, op = m.bl, m.bh, m.bo
        adverse = lambda s, lvl: lo[s] <= lvl
        favour = lambda s, lvl: hi[s] >= lvl
        gapped = lambda i, lvl: op[i] < lvl
    else:
        lo, hi, op = m.al, m.ah, m.ao
        adverse = lambda s, lvl: hi[s] >= lvl
        favour = lambda s, lvl: lo[s] <= lvl
        gapped = lambda i, lvl: op[i] > lvl
    sl, tp = entry - side * risk, entry + side * rr * risk
    trig, be_stop = entry + side * be_r * risk, entry + side * be_off
    if adverse(slice(ei, ei + 1), sl)[0]:
        return ei, sl, "SL"
    j = _scan(m, ei + 1, lambda s: adverse(s, sl) | favour(s, trig))
    if j < 0:
        return m.n - 1, (m.bc if side > 0 else m.ac)[m.n - 1], "OPEN"
    if gapped(j, sl):
        return j, op[j], "SL"
    if adverse(slice(j, j + 1), sl)[0]:
        return j, sl, "SL"
    if favour(slice(j, j + 1), tp)[0]:
        return j, tp, "TP"
    k = _scan(m, j + 1, lambda s: adverse(s, be_stop) | favour(s, tp))
    if k < 0:
        return m.n - 1, (m.bc if side > 0 else m.ac)[m.n - 1], "OPEN"
    if gapped(k, be_stop):
        return k, op[k], "BE"
    if adverse(slice(k, k + 1), be_stop)[0]:
        return k, be_stop, "BE"
    return k, tp, "TP"


def box_trades(m: Minutes, daily: pd.DataFrame, cfg: dict[str, Any], costs: Costs, pip: float = 0.01,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Version B box breakout per IST trading date (Mon-Fri), with one re-entry at the range level."""
    bh_, bm_ = (int(x) for x in cfg["box_start_utc"].split(":"))
    oh_, om_ = (int(x) for x in cfg["orders_until_utc"].split(":"))
    nbox, rr, be_r = int(cfg["box_minutes"]), float(cfg["rr"]), float(cfg["be_trigger_r"])
    wait = int(cfg["reentry_wait_minutes"])
    first = m.t0.normalize() + pd.Timedelta(days=1)
    last = (m.t0 + pd.Timedelta(minutes=m.n - 1)).normalize()
    days = [d for d in pd.date_range(first, last, freq="D") if d.dayofweek < 5]
    if start:
        days = [d for d in days if d >= pd.Timestamp(start)]
    if end:
        days = [d for d in days if d <= pd.Timestamp(end)]
    rows = []
    for day in days:                                         # IST trading date
        rs = m.idx(day - pd.Timedelta(days=1) + pd.Timedelta(hours=bh_, minutes=bm_))
        place = rs + nbox
        expiry = m.idx(day + pd.Timedelta(hours=oh_, minutes=om_))
        if rs < 0 or expiry >= m.n or (m.vol[rs:place] > 0).sum() < nbox:
            continue
        hi, lo = float(m.bh[rs:place].max()), float(m.bl[rs:place].min())
        if hi <= lo:
            continue
        sl_dist = float(cfg["sl_box_mult"]) * (hi - lo)
        be_off = sl_dist * float(cfg["be_offset_box"])
        hit = find_stop_entry(m, place, expiry, hi, lo)
        num = 1
        a = daily["atr20"].get(day, np.nan)
        md = daily["atr_med"].get(day, np.nan)
        hot = bool(np.isfinite(a) and np.isfinite(md) and a >= md)
        while hit is not None:
            side, ei, px = hit
            spread = max(m.ao[ei] - m.bo[ei], 0.0)
            risk = sl_dist + spread
            xi, xp, why = manage(m, side, ei, px, risk, rr, be_r, be_off)
            gross, net = net_r(side, px, xp, risk, 1 + (why in ("SL", "BE")), m.ts(ei), m.ts(xi), pip, costs)
            rows.append({"day": day, "num": num, "side": side, "entry_time": m.ts(ei), "entry": px, "exit_time": m.ts(xi),
                         "exit": xp, "risk_pips": risk / pip, "box_pips": (hi - lo) / pip, "reason": why,
                         "gross_r": gross, "r": net, "hot": hot})
            if why not in ("SL", "BE") or num == 2 or xi >= expiry:
                break
            q = xi + wait
            if q >= expiry:
                break
            if side > 0:                                     # stopped out long -> sell stop at the range low
                if m.bo[q] <= lo:
                    break
                hit = find_stop_entry(m, q, expiry, None, lo)
            else:                                            # stopped out short -> buy stop at the range high
                if m.ao[q] >= hi:
                    break
                hit = find_stop_entry(m, q, expiry, hi, None)
            num = 2
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- stats


def period_stats(tr: pd.DataFrame, periods: dict[str, list], col: str = "r") -> dict[str, dict[str, Any]]:
    out = {}
    for name, (a, b) in periods.items():
        x = tr[(tr["day"] >= pd.Timestamp(a)) & (tr["day"] <= pd.Timestamp(b))][col] if len(tr) else pd.Series(dtype=float)
        n = len(x)
        sd = float(x.std(ddof=1)) if n > 1 else float("nan")
        out[name] = {
            "n": n,
            "avg_r": float(x.mean()) if n else float("nan"),
            "t": float(x.mean() / (sd / math.sqrt(n))) if n > 2 and sd > 0 else 0.0,
            "win": float((x > 0).mean()) if n else float("nan"),
        }
    return out


def load_rates(macro_dir: Path) -> pd.DataFrame:
    """Daily policy rates (%) for USD and JPY from data/macro/policy_rate_*.csv."""
    cols = {}
    for c in ("USD", "JPY"):
        s = pd.read_csv(Path(macro_dir) / f"policy_rate_{c}.csv", parse_dates=["date"]).set_index("date")["value"]
        cols[c] = s.astype(float)
    return pd.DataFrame(cols).sort_index().ffill().dropna()


def match_log(mine: pd.DataFrame, log: pd.DataFrame, tol: float = 0.05) -> dict[str, Any]:
    """Share of the other side's logged trades found here on the same day with |net R difference| < tol."""
    theirs = pd.DataFrame({"day": pd.to_datetime(log["day"]), "their_r": log["R"].astype(float)})
    j = theirs.merge(mine[["day", "r"]], on="day", how="left")
    found = j["r"].notna()
    dr = (j["r"] - j["their_r"]).abs()
    span = mine[(mine["day"] >= theirs["day"].min()) & (mine["day"] <= theirs["day"].max())]
    return {
        "logged": len(j),
        "found": int(found.sum()),
        "within_tol": int((found & (dr < tol)).sum()),
        "share": float((found & (dr < tol)).mean()) if len(j) else 0.0,
        "median_abs_dr": float(dr[found].median()) if found.any() else float("nan"),
        "extra_here": int((~span["day"].isin(theirs["day"])).sum()),
        "their_avg_r": float(theirs["their_r"].mean()),
        "my_avg_r_same_days": float(j.loc[found, "r"].mean()) if found.any() else float("nan"),
    }


def window_stats(tr: pd.DataFrame, start: str, end: str, col: str = "r") -> dict[str, Any]:
    return period_stats(tr, {"w": [start, end]}, col)["w"]


def run_recheck(m: Minutes, cfg: dict[str, Any], rates: pd.DataFrame | None, gotobi_log: pd.DataFrame | None) -> dict[str, Any]:
    c = cfg["costs"]
    costs = Costs(float(c["commission_usd_per_lot"]), float(c["slippage_pips"]), float(c["swap_markup_pct"]), rates)
    pip = float(cfg["pip"])
    daily = ny_daily(m, int(cfg["vol_filter"]["median_days"]))
    got = gotobi_trades(m, daily, cfg["gotobi"], costs, pip)
    box = box_trades(m, daily, cfg["box"], costs, pip)
    periods = dict(cfg["periods"])
    span = [periods["check"][0], periods["holdout"][1]]
    lines = {"gotobi": got, "vol-gotobi": got[got["hot"]], "box": box, "vol-box": box[box["hot"]] if len(box) else box}
    stats = {}
    for name, tr in lines.items():
        s = period_stats(tr, periods)
        s["2018+"] = window_stats(tr, *span)
        s["2018+ gross"] = window_stats(tr, *span, col="gross_r")
        s["exits"] = tr["reason"].value_counts().to_dict() if len(tr) else {}
        stats[name] = s
    last3 = ["2023-10-07", "2026-10-07"]
    claims = {"vol-box": {"n": 359, "avg_r": 0.24}, "vol-gotobi": {"n": 92, "avg_r": 0.21}}
    last = {k: {"mine": window_stats(lines[k], *last3), "theirs": v} for k, v in claims.items()}
    match_g = match_log(got, gotobi_log) if gotobi_log is not None else None
    vb = last["vol-box"]["mine"]
    verdict = {
        "gotobi_matches": bool(match_g and match_g["share"] >= 0.95),
        "vol_box_matches": bool(abs(vb["avg_r"] - 0.24) <= 0.05 and abs(vb["n"] - 359) <= 0.05 * 359),
    }
    for k in ("vol-gotobi", "vol-box"):
        s = stats[k]
        verdict[k] = bool(s["check"]["avg_r"] > 0 and s["holdout"]["avg_r"] > 0 and s["2018+"]["t"] >= 2.0)
    out = {"periods": periods, "stats": stats, "last3y": last, "gotobi_match": match_g, "verdict": verdict,
           "trades": {"gotobi": got, "box": box}, "daily": daily}
    if cfg.get("robustness"):
        out["robustness"] = robustness(m, cfg, costs, daily, got, box)
    return out


def robustness(m: Minutes, cfg: dict[str, Any], costs: Costs, daily: pd.DataFrame, got: pd.DataFrame, box: pd.DataFrame) -> dict[str, Any]:
    """Descriptive stress checks (configs/jpy_recheck.yaml `robustness`); they do not change the verdicts."""
    rb, pip, periods = cfg["robustness"], float(cfg["pip"]), cfg["periods"]
    span = [periods["check"][0], periods["holdout"][1]]
    out: dict[str, Any] = {"slippage": [], "placebo": {}, "by_year": []}
    for slip in rb.get("slippage_pips") or []:
        c2 = Costs(costs.commission, float(slip), costs.swap_markup_pct, costs.rates)
        g2 = gotobi_trades(m, daily, cfg["gotobi"], c2, pip)
        b2 = box_trades(m, daily, cfg["box"], c2, pip)
        for k, tr in (("vol-gotobi", g2[g2["hot"]]), ("vol-box", b2[b2["hot"]])):
            out["slippage"].append({"line": k, "slippage_pips": float(slip), **period_stats(tr, {**periods, "2018+": span})})
    if rb.get("placebo"):
        pl = gotobi_trades(m, daily, cfg["gotobi"], costs, pip, pick=lambda d: not is_gotobi(d))
        for k, tr in (("gotobi", got), ("placebo", pl), ("vol-gotobi", got[got["hot"]]), ("vol-placebo", pl[pl["hot"]])):
            out["placebo"][k] = period_stats(tr, {**periods, "2018+": span})
    if rb.get("by_year"):
        a, b = (pd.Timestamp(x) for x in rb["by_year"])
        for k, tr in (("vol-gotobi", got[got["hot"]]), ("vol-box", box[box["hot"]])):
            w = tr[(tr["day"] >= a) & (tr["day"] <= b)]
            for y, g in w.groupby(w["day"].dt.year):
                out["by_year"].append({"line": k, "year": int(y), "n": len(g), "sum_r": float(g["r"].sum()), "avg_r": float(g["r"].mean())})
    return out


def unseen_market(m: Minutes, cfg: dict[str, Any], pip: float = 0.01) -> dict[str, Any]:
    """The same frozen rules on another market, gross R only (no costs, so no currency conversion needed)."""
    daily = ny_daily(m, int(cfg["vol_filter"]["median_days"]))
    free = Costs(0.0, 0.0, 0.0, None)
    got = gotobi_trades(m, daily, cfg["gotobi"], free, pip)
    box = box_trades(m, daily, cfg["box"], free, pip)
    periods = cfg["periods"]
    span = [periods["check"][0], periods["holdout"][1]]
    lines = {"gotobi": got, "vol-gotobi": got[got["hot"]], "box": box, "vol-box": box[box["hot"]] if len(box) else box}
    return {k: period_stats(tr, {**periods, "2018+": span}, col="gross_r") for k, tr in lines.items()}


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_markdown(res: dict[str, Any]) -> str:
    v, mg = res["verdict"], res["gotobi_match"]
    yes = lambda b: "yes" if b else "NO"
    lines = ["# USDJPY gotobi / vol-box re-check (independent code, gj-strat's raw minute data)", ""]
    lines.append("Rules and verdict thresholds: configs/jpy_recheck.yaml (fixed before the run). Net R after $7/lot, "
                 "0.2 pip per stop fill and carry swap minus 1%/yr.")
    lines += ["", "## Verdicts", "", "| check | result |", "|---|---|"]
    lines.append(f"| gotobi reproduces gj-strat's trade log (>= 95% same day, abs dR < 0.05) | {yes(v['gotobi_matches'])} |")
    lines.append(f"| vol-box reproduces Oct 2023 - Oct 2026 (359 trades, +0.24R) | {yes(v['vol_box_matches'])} |")
    lines.append(f"| vol-gotobi edge confirmed (check > 0, holdout > 0, t >= 2 over 2018+) | {yes(v['vol-gotobi'])} |")
    lines.append(f"| vol-box edge confirmed (check > 0, holdout > 0, t >= 2 over 2018+) | {yes(v['vol-box'])} |")
    if mg:
        lines += ["", "## gotobi trade-log match", "",
                  (f"- their trades {mg['logged']}, found here {mg['found']}, within 0.05R {mg['within_tol']} ({mg['share']:.1%}); "
                   f"median abs dR {_f(mg['median_abs_dr'])}; days traded here but not in their log {mg['extra_here']}"),
                  f"- avg R on their days: theirs {_f(mg['their_avg_r'])}, here {_f(mg['my_avg_r_same_days'])}"]
    lines += ["", "## Oct 2023 - Oct 2026 (gj-strat's basket window)", "", "| line | trades here | avg R here | t | gj-strat trades | gj-strat avg R |", "|---|---|---|---|---|---|"]
    for k, x in res["last3y"].items():
        lines.append(f"| {k} | {x['mine']['n']} | {_f(x['mine']['avg_r'])} | {_f(x['mine']['t'], 2)} | {x['theirs']['n']} | {_f(x['theirs']['avg_r'], 2)} |")
    lines += ["", "## By period (net R per trade)", "", "| line | period | trades | avg R | t | win |", "|---|---|---|---|---|---|"]
    for k, s in res["stats"].items():
        for p in (*res["periods"], "2018+", "2018+ gross"):
            x = s[p]
            lines.append(f"| {k} | {p} | {x['n']} | {_f(x['avg_r'])} | {_f(x['t'], 2)} | {_f(x['win'], 2)} |")
    lines += ["", "## Exits", "", "| line | exits |", "|---|---|"]
    for k, s in res["stats"].items():
        lines.append(f"| {k} | " + ", ".join(f"{a} {b}" for a, b in s["exits"].items()) + " |")
    rb = res.get("robustness")
    if rb:
        lines += ["", "## Robustness (descriptive; does not change the verdicts)", ""]
        lines += ["### Higher slippage per stop fill", "", "| line | slippage pips | check avg R | holdout avg R | 2018+ avg R | 2018+ t |", "|---|---|---|---|---|---|"]
        for r in rb["slippage"]:
            lines.append(f"| {r['line']} | {r['slippage_pips']:g} | {_f(r['check']['avg_r'])} | {_f(r['holdout']['avg_r'])} | {_f(r['2018+']['avg_r'])} | {_f(r['2018+']['t'], 2)} |")
        lines += ["", "### Gotobi placebo (same short on non-gotobi weekdays)", "", "| line | design avg R (t) | check avg R (t) | holdout avg R (t) | 2018+ trades | 2018+ avg R (t) |", "|---|---|---|---|---|---|"]
        for k, s in rb["placebo"].items():
            cell = lambda x: f"{_f(x['avg_r'])} ({_f(x['t'], 2)})"
            lines.append(f"| {k} | {cell(s['design'])} | {cell(s['check'])} | {cell(s['holdout'])} | {s['2018+']['n']} | {cell(s['2018+'])} |")
        lines += ["", "### By year (net R)", "", "| line | year | trades | sum R | avg R |", "|---|---|---|---|---|"]
        for r in rb["by_year"]:
            lines.append(f"| {r['line']} | {r['year']} | {r['n']} | {_f(r['sum_r'], 1)} | {_f(r['avg_r'])} |")
        if rb.get("unseen"):
            lines += ["", "### Unseen JPY markets (same rules, gross R before costs)", "", "| market | line | trades | design avg R (t) | check avg R (t) | holdout avg R (t) | 2018+ avg R (t) |", "|---|---|---|---|---|---|---|"]
            for mk, d in rb["unseen"].items():
                for k, s in d.items():
                    cell = lambda x: f"{_f(x['avg_r'])} ({_f(x['t'], 2)})"
                    n = s["design"]["n"] + s["check"]["n"] + s["holdout"]["n"]
                    lines.append(f"| {mk} | {k} | {n} | {cell(s['design'])} | {cell(s['check'])} | {cell(s['holdout'])} | {cell(s['2018+'])} |")
    lines.append("")
    return "\n".join(lines)


def write_report(res: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in res.items() if k not in ("trades", "daily")}
    paths = {"json": out_dir / "jpy_recheck.json", "md": out_dir / "jpy_recheck.md"}
    paths["json"].write_text(json.dumps(keep, indent=2, default=lambda o: None), encoding="utf-8")
    paths["md"].write_text(render_markdown(res), encoding="utf-8")
    for k, tr in res["trades"].items():
        paths[k] = out_dir / f"jpy_recheck_{k}.csv"
        tr.to_csv(paths[k], index=False)
    return paths
