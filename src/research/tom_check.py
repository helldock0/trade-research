"""Turn-of-month signal check on daily index closes (no bars).

A trading day is a turn-of-month (TOM) day when it is one of the last
`days_before` trading days of its month or one of the first `days_after`.
The day's return is log(close_t / close_t-1), so the TOM days of a month add
up to the return from the close of day -(days_before+1) to the close of day
+days_after: exactly what a trade entered and exited at those closes earns.

The test compares TOM-day returns with all other days (Welch t), overall and
by decade, and per month (window return vs the same number of other days).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def tom_mask(index: pd.DatetimeIndex, days_before: int = 1, days_after: int = 3) -> pd.Series:
    """True on the last `days_before` and first `days_after` trading days of each month."""
    idx = pd.DatetimeIndex(index)
    month = idx.to_period("M") if idx.tz is None else idx.tz_localize(None).to_period("M")
    pos = pd.Series(np.arange(len(idx)), index=idx).groupby(month.values)
    from_start = pos.cumcount() + 1
    from_end = pos.cumcount(ascending=False) + 1
    return ((from_start <= int(days_after)) | (from_end <= int(days_before))).rename("tom")


def full_months(s: pd.Series) -> pd.Series:
    """Drop the first and last calendar month (a partial month mislabels its first / last days)."""
    if s.empty:
        return s
    m = s.index.to_period("M") if s.index.tz is None else s.index.tz_localize(None).to_period("M")
    return s[(m != m.min()) & (m != m.max())]


def welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def tom_stats(close: pd.Series, days_before: int = 1, days_after: int = 3) -> dict[str, Any]:
    """TOM-day vs other-day daily log returns on one window of closes."""
    px = pd.Series(close, dtype=float).dropna()
    px = px[px > 0].sort_index()
    ret = np.log(px / px.shift(1)).dropna()
    mask = tom_mask(px.index, days_before, days_after).reindex(ret.index)
    ret = full_months(ret)
    mask = mask.reindex(ret.index).astype(bool)
    tom, other = ret[mask], ret[~mask]
    n_window = days_before + days_after
    # per month: the TOM window's return vs n_window average other days of the same sample
    m = ret.index.to_period("M") if ret.index.tz is None else ret.index.tz_localize(None).to_period("M")
    windows = tom.groupby(m[mask.values]).sum()
    out = {
        "n_days": int(len(ret)),
        "n_tom_days": int(len(tom)),
        "n_months": int(len(windows)),
        "tom_mean_bp": float(tom.mean() * 1e4) if len(tom) else float("nan"),
        "other_mean_bp": float(other.mean() * 1e4) if len(other) else float("nan"),
        "diff_bp": float((tom.mean() - other.mean()) * 1e4) if len(tom) and len(other) else float("nan"),
        "t": welch_t(tom, other),
        "window_mean_bp": float(windows.mean() * 1e4) if len(windows) else float("nan"),
        "window_hit": float((windows > 0).mean()) if len(windows) else float("nan"),
        "window_excess_bp": float((windows.mean() - n_window * other.mean()) * 1e4) if len(windows) else float("nan"),
        "share_of_return": float(tom.sum() / ret.sum()) if ret.sum() != 0 else float("nan"),
    }
    return out


def _window(s: pd.Series, start: str | None, end: str | None) -> pd.Series:
    if start is not None:
        s = s[s.index >= pd.Timestamp(start, tz=s.index.tz)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end, tz=s.index.tz)]
    return s


def run_tom_check(series: dict[str, pd.Series], cfg: dict[str, Any]) -> dict[str, Any]:
    sig, val = cfg["signal"], cfg["validation"]
    before, after = int(sig.get("days_before", 1)), int(sig.get("days_after", 3))
    names = [n for n in val["series"] if n in series]
    rows, decades = [], []
    for name in names:
        s = series[name]
        row = {"series": name, "first": str(s.index.min().date()), "last": str(s.index.max().date())}
        for label in ("holdout", "dev"):
            a, b = val[label]
            row[label] = tom_stats(_window(s, a, b), before, after)
        rows.append(row)
        years = s.index.year
        for dec in sorted({int(y) // 10 * 10 for y in years}):
            st = tom_stats(s[(years >= dec) & (years < dec + 10)], before, after)
            decades.append({"series": name, "decade": dec, **st})
    primary = next((r for r in rows if r["series"] == val["series"][0]), None)
    min_t = float(val.get("min_t", 2.0))
    hold_ok = bool(primary and primary["holdout"]["t"] >= min_t)
    dev_ok = bool(rows) and all(r["dev"]["diff_bp"] > 0 for r in rows)
    verdict = "PASS" if hold_ok and dev_ok else "NO-GO"
    return {
        "days_before": before,
        "days_after": after,
        "rows": rows,
        "decades": decades,
        "verdict": {
            "verdict": verdict,
            "holdout_t": primary["holdout"]["t"] if primary else float("nan"),
            "holdout_ok": hold_ok,
            "dev_ok": dev_ok,
            "rule": f"holdout Welch t >= {min_t} on {val['series'][0]} and dev TOM-minus-other > 0 on every series",
        },
    }


def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_markdown(res: dict[str, Any]) -> str:
    v = res["verdict"]
    lines = ["# Turn-of-month signal check (daily closes)", ""]
    lines.append(
        f"- TOM days: last {res['days_before']} + first {res['days_after']} trading days of each month; "
        "returns are daily log closes, first and last partial months dropped"
    )
    lines.append(f"- rule (fixed before the run): {v['rule']}")
    lines += ["", f"**{v['verdict']}**: holdout t {_f(v['holdout_t'])} ({'ok' if v['holdout_ok'] else 'fails'}), dev {'ok' if v['dev_ok'] else 'fails'}.", ""]
    lines.append("| series | window | days | TOM mean bp/day | other mean bp/day | diff bp | t | months | window mean bp | window hit | window excess bp | share of all return |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        for label in ("holdout", "dev"):
            s = r[label]
            lines.append(
                f"| {r['series']} | {label} | {s['n_days']} | {_f(s['tom_mean_bp'])} | {_f(s['other_mean_bp'])} | {_f(s['diff_bp'])} | "
                f"{_f(s['t'])} | {s['n_months']} | {_f(s['window_mean_bp'], 1)} | {_f(s['window_hit'])} | {_f(s['window_excess_bp'], 1)} | {_f(s['share_of_return'])} |"
            )
    lines += ["", "## By decade (TOM minus other, bp per day)", ""]
    lines.append("| series | decade | days | diff bp | t | window hit |")
    lines.append("|---|---|---|---|---|---|")
    for d in res["decades"]:
        lines.append(f"| {d['series']} | {d['decade']}s | {d['n_days']} | {_f(d['diff_bp'])} | {_f(d['t'])} | {_f(d['window_hit'])} |")
    lines.append("")
    return "\n".join(lines)


def write_report(res: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "tom_check.json"
    p_json.write_text(json.dumps(res, indent=2, default=lambda o: None), encoding="utf-8")
    p_md = out_dir / "tom_check.md"
    p_md.write_text(render_markdown(res), encoding="utf-8")
    return {"json": p_json, "md": p_md}
