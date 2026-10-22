"""Macro IC check: does the macro pair bias predict forward FX returns?

Phase A of docs/plans/2026-10-22-fundamentals-next-steps.md. FRED daily noon
buying rates (Fed H.10) stand in for the parquet bars, so the check runs
without them.

Alignment: the bias dated t only uses data published on or before t-1 (the
panel lags in src.features.macro_score), and the forward return runs from the
FX fix on t to the fix h business days later, by position in that pair's own
FX index. A positive bias means long the pair, so the signed return is
sign(bias) * forward return.

Overlap: h-day returns sampled daily overlap, so a naive t overstates
significance. Every t here is the median, across the h offsets, of the t-stat
on every h-th date of a per-date series (mean across pairs). Clustering by
date also keeps the five USD pairs from counting as independent draws. A
Newey-West t (lag h-1) on the same per-date series is reported alongside as a
cross-check; the verdict uses the non-overlap t.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ROOT
from src.data.macro import DEFAULT_MACRO_DIR, fetch_series, load_series, save_series, series_path
from src.features.macro_score import build_currency_scores, pair_bias

# pair -> FRED id; every series is already quoted as the pair (no inversion)
FRED_FX = {
    "EURUSD": "DEXUSEU",  # USD per EUR
    "GBPUSD": "DEXUSUK",  # USD per GBP
    "USDJPY": "DEXJPUS",  # JPY per USD
    "AUDUSD": "DEXUSAL",  # USD per AUD
    "USDCAD": "DEXCAUS",  # CAD per USD
    "NZDUSD": "DEXUSNZ",  # USD per NZD
    "USDCHF": "DEXSZUS",  # CHF per USD
}
HORIZONS = (5, 10, 20, 40)
HEADLINE_H = 20

# Phase A decision rule (plan section A3)
GO_IC = 0.05
WEAK_IC = 0.02
GO_T = 2.0
GO_YEAR_SHARE = 5 / 8


# ----------------------------------------------------------------- FX data


def fx_name(pair: str) -> str:
    return f"fx_{pair.upper()}"


def macro_cache_dir(macro_cfg: dict[str, Any]) -> Path:
    p = Path(macro_cfg.get("cache_dir", DEFAULT_MACRO_DIR))
    return p if p.is_absolute() else ROOT / p


def usd_legs(pair: str) -> list[str]:
    """FRED pairs needed for `pair`: itself, or the USD majors of both currencies of a cross."""
    pair = pair.upper()
    if pair in FRED_FX:
        return [pair]
    legs = []
    for ccy in (pair[:3], pair[3:6]):
        leg = next((p for p in FRED_FX if p in (f"{ccy}USD", f"USD{ccy}")), None)
        if leg is None:
            return []
        legs.append(leg)
    return legs


def _usd_value(ccy: str, fx: dict[str, pd.Series]) -> pd.Series | None:
    """USD per unit of `ccy`."""
    if ccy == "USD":
        return None
    if f"{ccy}USD" in fx:
        return fx[f"{ccy}USD"]
    if f"USD{ccy}" in fx:
        return 1.0 / fx[f"USD{ccy}"]
    return None


def synthetic_cross(pair: str, fx: dict[str, pd.Series]) -> pd.Series:
    """Cross rate from two USD majors on their common dates (EURJPY = EURUSD * USDJPY)."""
    pair = pair.upper()
    b, q = _usd_value(pair[:3], fx), _usd_value(pair[3:6], fx)
    if b is None or q is None:
        return pd.Series(dtype=float, name=pair)
    both = pd.concat([b, q], axis=1, join="inner").dropna()
    return (both.iloc[:, 0] / both.iloc[:, 1]).rename(pair)


def fetch_fx(pairs: list[str], cache_dir: Path | None = None, force: bool = False) -> dict[str, Any]:
    """Fetch FRED daily FX into data/macro/fx_{PAIR}.csv (crosses: their USD legs). Returns per-pair status."""
    status: dict[str, Any] = {}
    needed: list[str] = []
    for pair in (p.upper() for p in pairs):
        legs = usd_legs(pair)
        if not legs:
            status[pair] = {"error": "no FRED series for this pair or its USD legs"}
        needed += [leg for leg in legs if leg not in needed]
    for pair in needed:
        fid = FRED_FX[pair]
        name = fx_name(pair)
        if series_path(name, cache_dir).exists() and not force:
            s = load_series(name, cache_dir)
            status[pair] = {"cached": True, "n": int(len(s)), "last": str(s.index[-1].date()) if len(s) else None}
            continue
        try:
            s = fetch_series({"source": "fred", "id": fid}, name)
            save_series(s, name, cache_dir)
            status[pair] = {"cached": False, "n": int(len(s)), "last": str(s.index[-1].date()) if len(s) else None}
        except Exception as exc:  # noqa: BLE001 — report and continue with other pairs
            status[pair] = {"error": str(exc)}
    return status


def load_fx(pairs: list[str], cache_dir: Path | None = None) -> dict[str, pd.Series]:
    """Cached FRED pairs; crosses are built from their cached USD legs."""
    legs = {leg: load_series(fx_name(leg), cache_dir) for p in pairs for leg in usd_legs(p)}
    legs = {k: v for k, v in legs.items() if not v.empty}
    out: dict[str, pd.Series] = {}
    for pair in (p.upper() for p in pairs):
        s = legs.get(pair) if pair in FRED_FX else synthetic_cross(pair, legs)
        if s is not None and not s.empty:
            out[pair] = s
    return out


# ----------------------------------------------------------------- statistics


def forward_log_return(px: pd.Series, h: int) -> pd.Series:
    """log(px[t+h] / px[t]) by position in px's own index; NaN for the last h rows."""
    p = pd.Series(px, dtype=float).dropna()
    p = p[p > 0]
    return np.log(p.shift(-int(h)) / p)


def ic_frame(
    signal: pd.DataFrame,
    fx: dict[str, pd.Series],
    h: int,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    extra_lag: int = 0,
) -> pd.DataFrame:
    """Long frame [date, pair, signal, fwd]: signal known on date vs the h-day forward return.

    The signal is forward-filled onto each pair's FX dates; `extra_lag` shifts it
    by that many more FX days (an alignment robustness check).
    """
    parts = []
    for pair, px in fx.items():
        if pair not in signal.columns:
            continue
        p = pd.Series(px, dtype=float).dropna()
        p = p[p > 0].sort_index()
        fwd = forward_log_return(p, h)
        sig = signal[pair].sort_index().reindex(p.index, method="ffill")
        if extra_lag:
            sig = sig.shift(int(extra_lag))
        parts.append(pd.DataFrame({"date": p.index, "pair": pair, "signal": sig.values, "fwd": fwd.values}))
    if not parts:
        return pd.DataFrame(columns=["date", "pair", "signal", "fwd"])
    out = pd.concat(parts, ignore_index=True).dropna(subset=["signal", "fwd"])
    if start is not None:
        out = out[out["date"] >= _utc(start)]
    if end is not None:
        out = out[out["date"] <= _utc(end)]
    return out.sort_values(["date", "pair"]).reset_index(drop=True)


def _utc(x: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def spearman(x: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if len(d) < 3:
        return float("nan")
    return float(d["x"].rank().corr(d["y"].rank()))


def _rank_z(s: pd.Series) -> pd.Series:
    r = s.rank()
    sd = r.std(ddof=0)
    return (r - r.mean()) / sd if sd > 0 else r * np.nan


def nonoverlap_t(per_date: pd.Series, h: int) -> float:
    """Median over the h offsets of the t-stat of the mean on every h-th date.

    Rows h apart in a business-day index carry non-overlapping h-day returns.
    """
    v = pd.Series(per_date, dtype=float).dropna()
    h = max(1, int(h))
    ts = []
    for k in range(h):
        sub = v.iloc[k::h]
        n = len(sub)
        sd = float(sub.std(ddof=1)) if n > 2 else 0.0
        if sd > 0:
            ts.append(float(sub.mean()) / (sd / math.sqrt(n)))
    return float(np.median(ts)) if ts else 0.0


def newey_west_t(per_date: pd.Series, lag: int) -> float:
    """t-stat of the mean with a Bartlett-kernel HAC variance."""
    x = pd.Series(per_date, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 3:
        return 0.0
    u = x - x.mean()
    var = float(u @ u) / n
    for k in range(1, min(int(lag), n - 1) + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(u[k:] @ u[:-k]) / n
    return float(x.mean() / math.sqrt(var / n)) if var > 0 else 0.0


def ic_stats(frame: pd.DataFrame, h: int) -> dict[str, Any]:
    """Pooled Spearman IC with a date-clustered, non-overlapping t (and a Newey-West t)."""
    n = int(len(frame))
    if n < 3:
        return {"n": n, "n_dates": int(frame["date"].nunique()) if n else 0, "ic": float("nan"), "t": 0.0, "t_nw": 0.0}
    # mean of rank-z products == Spearman; averaging per date clusters the pairs
    prod = _rank_z(frame["signal"]) * _rank_z(frame["fwd"])
    per_date = prod.groupby(frame["date"].values).mean()
    return {
        "n": n,
        "n_dates": int(per_date.size),
        "ic": spearman(frame["signal"], frame["fwd"]),
        "t": nonoverlap_t(per_date, h),
        "t_nw": newey_west_t(per_date, max(0, int(h) - 1)),
    }


def signed_stats(frame: pd.DataFrame, h: int) -> dict[str, Any]:
    """Return in the bias direction: mean (bp of log return), hit rate, non-overlap t."""
    if frame.empty:
        return {"n": 0, "mean_bp": float("nan"), "hit": float("nan"), "t": 0.0}
    signed = np.sign(frame["signal"]) * frame["fwd"]
    per_date = signed.groupby(frame["date"].values).mean()
    return {
        "n": int(len(frame)),
        "mean_bp": float(signed.mean() * 1e4),
        "hit": float((signed > 0).mean()),
        "t": nonoverlap_t(per_date, h),
    }


def threshold_split(frame: pd.DataFrame, h: int, min_abs: float) -> dict[str, Any]:
    strong = frame["signal"].abs() >= float(min_abs)
    return {
        "all": signed_stats(frame, h),
        "above": signed_stats(frame[strong], h),
        "below": signed_stats(frame[~strong], h),
    }


def component_pair_signals(components: dict[str, pd.DataFrame], pairs: list[str]) -> dict[str, pd.DataFrame]:
    """comp[base] - comp[quote] per component.

    A currency missing from a component altogether (outside its beta map, e.g.
    USD for oil) has zero exposure; a pair with neither leg exposed is dropped.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, comp in components.items():
        absent = {c for c in comp.columns if comp[c].isna().all()}
        filled = comp.copy()
        for c in absent:
            filled[c] = 0.0
        cols = {}
        for p in pairs:
            b, q = p.upper()[:3], p.upper()[3:6]
            if b not in comp.columns or q not in comp.columns or (b in absent and q in absent):
                continue
            cols[p.upper()] = filled[b] - filled[q]
        if cols:
            out[name] = pd.DataFrame(cols, index=comp.index)
    return out


# ----------------------------------------------------------------- run


def run_macro_ic(
    panel: dict[str, pd.Series],
    strategy_cfg: dict[str, Any],
    fx: dict[str, pd.Series],
    horizons: tuple[int, ...] | list[int] = HORIZONS,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    macro_cfg = strategy_cfg.get("macro", {}) or {}
    universe = [p.upper() for p in strategy_cfg["universe"]]
    pairs = [p for p in universe if p in fx]
    # same currency set as the strategy, so the cross-sectional z's match what it trades
    currencies = sorted({p[:3] for p in universe} | {p[3:6] for p in universe})
    start = start if start is not None else (strategy_cfg.get("data") or {}).get("start")
    end = end if end is not None else (strategy_cfg.get("data") or {}).get("end")
    min_abs = float(strategy_cfg["signal"]["min_abs_bias"])
    horizons = tuple(int(h) for h in horizons)

    scored = build_currency_scores(panel, macro_cfg, currencies=currencies)
    bias = pair_bias(scored["scores"], pairs)
    comp_sigs = component_pair_signals(scored["components"], pairs)

    pooled, per_pair, components = [], [], []
    for h in horizons:
        fr = ic_frame(bias, fx, h, start, end)
        lagged = ic_stats(ic_frame(bias, fx, h, start, end, extra_lag=1), h)
        pooled.append({"h": h, **ic_stats(fr, h), "lag1_ic": lagged["ic"], "lag1_t": lagged["t"], **threshold_split(fr, h, min_abs)})
        for pair in pairs:
            g = fr[fr["pair"] == pair]
            per_pair.append({"pair": pair, "h": h, **ic_stats(g, h), "signed": signed_stats(g, h)})
        for name, sig in comp_sigs.items():
            components.append(
                {"component": name, "weight": scored["weights"].get(name), "h": h, "pairs": list(sig.columns), **ic_stats(ic_frame(sig, fx, h, start, end), h)}
            )

    fr_head = ic_frame(bias, fx, HEADLINE_H, start, end)
    by_year = []
    for year, g in fr_head.groupby(fr_head["date"].dt.year):
        d0, d1 = g["date"].min(), g["date"].max()
        partial = bool(d0 > pd.Timestamp(year=int(year), month=1, day=15, tz="UTC") or d1 < pd.Timestamp(year=int(year), month=12, day=15, tz="UTC"))
        by_year.append({"year": int(year), **ic_stats(g, HEADLINE_H), "partial": partial})

    head = next((r for r in pooled if r["h"] == HEADLINE_H), None)
    if head is None:
        head = {"h": HEADLINE_H, **ic_stats(fr_head, HEADLINE_H)}
    meta = {
        "pairs": pairs,
        "missing_fx": [p for p in universe if p not in fx],
        "currencies": currencies,
        "horizons": list(horizons),
        "min_abs_bias": min_abs,
        "weights": scored["weights"],
        "first_date": str(fr_head["date"].min().date()) if len(fr_head) else None,
        "last_date": str(fr_head["date"].max().date()) if len(fr_head) else None,
        "share_above_threshold": float((fr_head["signal"].abs() >= min_abs).mean()) if len(fr_head) else float("nan"),
        "fx_last": {p: str(fx[p].index.max().date()) for p in pairs},
    }
    return {
        "meta": meta,
        "pooled": pooled,
        "per_pair": per_pair,
        "components": components,
        "by_year": by_year,
        "verdict": phase_a_verdict(head, by_year),
    }


def phase_a_verdict(head: dict[str, Any], by_year: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the plan's A3 table to the pooled headline-horizon IC."""
    ic = float(head.get("ic", float("nan")))
    t = float(head.get("t", 0.0))
    full = [y for y in by_year if not y.get("partial") and not math.isnan(float(y["ic"]))]
    same = sum(1 for y in full if not math.isnan(ic) and np.sign(y["ic"]) == np.sign(ic))
    share = same / len(full) if full else 0.0
    if not math.isnan(ic) and ic >= GO_IC and t >= GO_T and share >= GO_YEAR_SHARE:
        verdict, action = "PROCEED", "Proceed to the bar backtest (Phase B)."
    elif not math.isnan(ic) and ic >= WEAK_IC:
        verdict, action = "WEAK", "Run Phase B only with trigger: none; expect a marginal result."
    else:
        verdict, action = (
            "NO-GO",
            "NO-GO for this component set. Do not retune weights; the next hypothesis needs a different input.",
        )
    return {
        "verdict": verdict,
        "action": action,
        "h": int(head.get("h", HEADLINE_H)),
        "ic": ic,
        "t": t,
        "years_same_sign": same,
        "years_full": len(full),
        "rule": (
            f"IC >= {GO_IC} with t >= {GO_T} and the same sign in >= 5 of 8 full years ({GO_YEAR_SHARE:.1%}) -> PROCEED; "
            f"IC {WEAK_IC} to {GO_IC} (or above {GO_IC} without the t / year support) -> WEAK; "
            f"IC < {WEAK_IC} or negative -> NO-GO"
        ),
    }


def signal_ic(
    panel: dict[str, pd.Series],
    cfg: dict[str, Any],
    fx: dict[str, pd.Series],
    start: str | None,
    end: str | None,
    pairs: list[str] | None = None,
    h: int = HEADLINE_H,
) -> dict[str, Any]:
    """Pooled h-day IC of a config's pair bias on [start, end], with the A3 verdict for that window."""
    universe = [p.upper() for p in cfg["universe"]]
    currencies = sorted({p[:3] for p in universe} | {p[3:6] for p in universe})
    scores = build_currency_scores(panel, cfg.get("macro", {}) or {}, currencies=currencies)["scores"]
    use = [p.upper() for p in (pairs or universe) if p.upper() in fx]
    fr = ic_frame(pair_bias(scores, use), fx, h, start, end)
    by_year = []
    for year, g in fr.groupby(fr["date"].dt.year):
        d0, d1 = g["date"].min(), g["date"].max()
        partial = bool(d0 > pd.Timestamp(year=int(year), month=1, day=15, tz="UTC") or d1 < pd.Timestamp(year=int(year), month=12, day=15, tz="UTC"))
        by_year.append({"year": int(year), **ic_stats(g, h), "partial": partial})
    head = {"h": h, **ic_stats(fr, h)}
    return {**head, "pairs": use, "by_year": by_year, "verdict": phase_a_verdict(head, by_year)["verdict"]}


def validate_variants(
    panel: dict[str, pd.Series],
    configs: dict[str, dict[str, Any]],
    fx: dict[str, pd.Series],
    baseline: str,
    dev: tuple[str | None, str | None],
    holdout: tuple[str | None, str | None],
    h: int = HEADLINE_H,
) -> dict[str, Any]:
    """Each config's IC on the dev and holdout windows; a variant is kept only if it beats
    the baseline on both and is positive on the holdout (rule fixed in the config)."""
    base_pairs = [p.upper() for p in configs[baseline]["universe"]]
    rows = []
    for name, cfg in configs.items():
        row: dict[str, Any] = {"variant": name, "n_pairs": len(cfg["universe"])}
        for label, (a, b) in (("dev", dev), ("holdout", holdout)):
            row[label] = signal_ic(panel, cfg, fx, a, b, h=h)
            row[f"{label}_base_pairs"] = signal_ic(panel, cfg, fx, a, b, pairs=base_pairs, h=h)
        rows.append(row)
    base = next(r for r in rows if r["variant"] == baseline)
    for r in rows:
        if r is base:
            r["kept"] = None
            continue
        r["kept"] = bool(
            r["dev"]["ic"] > base["dev"]["ic"] and r["holdout"]["ic"] > base["holdout"]["ic"] and r["holdout"]["ic"] > 0
        )
    return {"baseline": baseline, "dev": list(dev), "holdout": list(holdout), "h": h, "rows": rows}


def render_validation_markdown(res: dict[str, Any]) -> str:
    dev, hold = res["dev"], res["holdout"]
    lines = ["# Macro signal validation: dev vs holdout", ""]
    lines.append(f"- dev: {dev[0]} → {dev[1] or 'end'} · holdout: {hold[0]} → {hold[1] or 'end'} · pooled {res['h']}d Spearman IC")
    lines.append(
        f"- rule (fixed before the run): a variant is kept only if its IC beats `{res['baseline']}` on both windows "
        "and is positive on the holdout"
    )
    lines.append("- `5 base pairs`: the same IC restricted to the baseline's pairs (only differs for wider universes)")
    lines += ["", "| variant | pairs | dev IC (t) | holdout IC (t) | dev, 5 base pairs | holdout, 5 base pairs | A3 dev | A3 holdout | kept |"]
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        d, ho, db, hb = r["dev"], r["holdout"], r["dev_base_pairs"], r["holdout_base_pairs"]
        kept = "baseline" if r["kept"] is None else ("**yes**" if r["kept"] else "no")
        lines.append(
            f"| {r['variant']} | {r['n_pairs']} | {_fmt(d['ic'])} ({_fmt(d['t'], 2)}) | {_fmt(ho['ic'])} ({_fmt(ho['t'], 2)}) | "
            f"{_fmt(db['ic'])} | {_fmt(hb['ic'])} | {d['verdict']} | {ho['verdict']} | {kept} |"
        )
    lines += ["", "## By year (pooled 20d IC)", ""]
    years = sorted({y["year"] for r in res["rows"] for w in ("holdout", "dev") for y in r[w]["by_year"]})
    lines.append("| variant | " + " | ".join(str(y) for y in years) + " |")
    lines.append("|---|" + "---|" * len(years))
    for r in res["rows"]:
        ic = {y["year"]: y["ic"] for w in ("holdout", "dev") for y in r[w]["by_year"]}
        lines.append(f"| {r['variant']} | " + " | ".join(_fmt(ic.get(y), 2) for y in years) + " |")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------- report


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_markdown(result: dict[str, Any], title: str = "Macro IC check: macro_swing pair bias vs forward FX returns") -> str:
    meta, v = result["meta"], result["verdict"]
    lines = [f"# {title}", ""]
    lines.append(
        f"- decision dates {meta['first_date']} → {meta['last_date']} · pairs {', '.join(meta['pairs'])} · "
        f"FX: FRED H.10 noon buying rates (mid, no costs)"
    )
    if meta.get("missing_fx"):
        lines.append(f"- no FX series for: {', '.join(meta['missing_fx'])}")
    weights = ", ".join(f"{k} {w:g}" for k, w in meta["weights"].items())
    lines.append(f"- bias = score[base] − score[quote], config weights (priors, not fitted): {weights}")
    lines.append("- bias dated t uses data published by t−1; forward log return from the fix on t to the fix h business days later")
    lines.append(
        "- t: median over the h offsets of the t-stat on every h-th date, per-date mean across pairs "
        "(non-overlapping, clustered by date); t NW: Newey-West, lag h−1, same per-date series"
    )
    lines.append(f"- |bias| ≥ {_fmt(meta['min_abs_bias'], 2)} on {_fmt(meta['share_above_threshold'] * 100, 1)}% of pair-days (20d sample)")

    lines += ["", "## Verdict (plan A3)", ""]
    lines.append(
        f"**{v['verdict']}**: pooled {v['h']}d Spearman IC {_fmt(v['ic'])} (t {_fmt(v['t'], 2)}), "
        f"same sign in {v['years_same_sign']} of {v['years_full']} full years. {v['action']}"
    )
    lines.append("")
    lines.append(f"Rule: {v['rule']}.")

    lines += ["", "## Pooled IC across pairs", ""]
    lines.append("| h | obs | dates | IC | t | t NW | IC lag+1 | t lag+1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in result["pooled"]:
        lines.append(
            f"| {r['h']} | {r['n']} | {r['n_dates']} | {_fmt(r['ic'])} | {_fmt(r['t'], 2)} | {_fmt(r['t_nw'], 2)} | "
            f"{_fmt(r['lag1_ic'])} | {_fmt(r['lag1_t'], 2)} |"
        )
    lines.append("")
    lines.append("`lag+1`: bias shifted one more FX day, a guard against same-day leakage (e.g. CFTC data released Friday afternoon vs the Friday noon fix).")

    lines += ["", f"## Return in the bias direction (split at |bias| {_fmt(meta['min_abs_bias'], 2)})", ""]
    lines.append("| h | subset | n | mean bp | hit | t |")
    lines.append("|---|---|---|---|---|---|")
    for r in result["pooled"]:
        for key, label in (("above", f"≥ {_fmt(meta['min_abs_bias'], 2)}"), ("below", "below"), ("all", "all")):
            s = r[key]
            lines.append(f"| {r['h']} | {label} | {s['n']} | {_fmt(s['mean_bp'], 1)} | {_fmt(s['hit'])} | {_fmt(s['t'], 2)} |")

    hs = meta["horizons"]
    lines += ["", "## Per pair", ""]
    lines.append("| pair | " + " | ".join(f"IC {h}d" for h in hs) + f" | t {HEADLINE_H}d | signed {HEADLINE_H}d bp | hit {HEADLINE_H}d |")
    lines.append("|---|" + "---|" * (len(hs) + 3))
    for pair in meta["pairs"]:
        rows = {r["h"]: r for r in result["per_pair"] if r["pair"] == pair}
        head = rows.get(HEADLINE_H, {"t": float("nan"), "signed": {"mean_bp": float("nan"), "hit": float("nan")}})
        lines.append(
            f"| {pair} | "
            + " | ".join(_fmt(rows[h]["ic"]) if h in rows else "—" for h in hs)
            + f" | {_fmt(head['t'], 2)} | {_fmt(head['signed']['mean_bp'], 1)} | {_fmt(head['signed']['hit'])} |"
        )

    lines += ["", "## Per component (pair difference base − quote)", ""]
    lines.append("| component | weight | pairs | " + " | ".join(f"IC {h}d" for h in hs) + f" | t {HEADLINE_H}d | obs {HEADLINE_H}d |")
    lines.append("|---|---|---|" + "---|" * (len(hs) + 2))
    names = list(dict.fromkeys(r["component"] for r in result["components"]))
    for name in names:
        rows = {r["h"]: r for r in result["components"] if r["component"] == name}
        any_row = next(iter(rows.values()))
        head = rows.get(HEADLINE_H, {"t": float("nan"), "n": 0})
        pairs = "all" if len(any_row["pairs"]) == len(meta["pairs"]) else ", ".join(any_row["pairs"])
        lines.append(
            f"| {name} | {_fmt(any_row['weight'], 2)} | {pairs} | "
            + " | ".join(_fmt(rows[h]["ic"]) if h in rows else "—" for h in hs)
            + f" | {_fmt(head['t'], 2)} | {head['n']} |"
        )

    lines += ["", f"## By year (pooled, {HEADLINE_H}d)", ""]
    lines.append("| year | obs | IC | t | |")
    lines.append("|---|---|---|---|---|")
    for r in result["by_year"]:
        lines.append(f"| {r['year']} | {r['n']} | {_fmt(r['ic'])} | {_fmt(r['t'], 2)} | {'partial' if r['partial'] else ''} |")
    lines.append("")
    return "\n".join(lines)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def write_validation_report(res: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_json = out_dir / "macro_ic_validation.json"
    p_json.write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
    p_md = out_dir / "macro_ic_validation.md"
    p_md.write_text(render_validation_markdown(res), encoding="utf-8")
    return {"json": p_json, "md": p_md}


def write_macro_ic_report(result: dict[str, Any], out_dir: Path, tag: str | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"macro_ic_{tag}" if tag else "macro_ic"
    p_json = out_dir / f"{stem}.json"
    p_json.write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")
    p_md = out_dir / f"{stem}.md"
    p_md.write_text(render_markdown(result), encoding="utf-8")
    return {"json": p_json, "md": p_md}
