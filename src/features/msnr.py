"""MSNR (Malaysian Support & Resistance) zone helpers — no lookahead."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Zone:
    """Horizontal S/R zone from an HTF swing pivot."""

    side: int  # +1 support (long), -1 resistance (short)
    z_low: float
    z_high: float
    born_ts: pd.Timestamp
    confirmed_at: pd.Timestamp  # first time zone may be used (HTF confirm bar end)
    touches: int = 0
    flipped: bool = False
    active: bool = True
    zone_id: int = 0

    def intersects(self, bar_low: float, bar_high: float) -> bool:
        return bar_high >= self.z_low and bar_low <= self.z_high

    def rejection_long(self, close: float) -> bool:
        """Close back above the zone (support hold)."""
        return close > self.z_high

    def rejection_short(self, close: float) -> bool:
        """Close back below the zone (resistance hold)."""
        return close < self.z_low


def rule_to_timedelta(rule: str) -> pd.Timedelta:
    return pd.Timedelta(rule)


def swing_pivots(
    htf_ohlc: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Boolean pivot-high / pivot-low masks on HTF bars (marked at pivot index).

    A pivot at index i is only *known* after bar i+right completes; callers must
    gate usage on confirmation time (see ``zones_from_pivots``).
    """
    if left < 1 or right < 1:
        raise ValueError("pivot left/right must be >= 1")
    high = htf_ohlc["high"].to_numpy(dtype=float)
    low = htf_ohlc["low"].to_numpy(dtype=float)
    n = len(htf_ohlc)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        left_h = high[i - left : i]
        right_h = high[i + 1 : i + right + 1]
        left_l = low[i - left : i]
        right_l = low[i + 1 : i + right + 1]
        if high[i] > left_h.max() and high[i] > right_h.max():
            ph[i] = True
        if low[i] < left_l.min() and low[i] < right_l.min():
            pl[i] = True
    idx = htf_ohlc.index
    return pd.Series(ph, index=idx, name="pivot_high"), pd.Series(pl, index=idx, name="pivot_low")


def _zone_from_bar(row: pd.Series, side: int) -> tuple[float, float]:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    body_top = max(o, c)
    body_bot = min(o, c)
    if side < 0:
        # Resistance: body top -> wick high
        return body_top, h
    # Support: wick low -> body bottom
    return l, body_bot


def zones_from_pivots(
    htf_ohlc: pd.DataFrame,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    right: int,
    htf_rule: str = "4h",
) -> list[Zone]:
    """Build zones from pivots; ``confirmed_at`` is end of confirming HTF bar."""
    delta = rule_to_timedelta(htf_rule)
    times = list(htf_ohlc.index)
    zones: list[Zone] = []
    zid = 0
    for i, ts in enumerate(times):
        confirm_i = i + right
        if confirm_i >= len(times):
            continue
        confirmed_at = times[confirm_i] + delta
        row = htf_ohlc.iloc[i]
        if bool(pivot_high.iloc[i]):
            z_low, z_high = _zone_from_bar(row, side=-1)
            if z_high > z_low:
                zones.append(
                    Zone(
                        side=-1,
                        z_low=z_low,
                        z_high=z_high,
                        born_ts=pd.Timestamp(ts),
                        confirmed_at=pd.Timestamp(confirmed_at),
                        zone_id=zid,
                    )
                )
                zid += 1
        if bool(pivot_low.iloc[i]):
            z_low, z_high = _zone_from_bar(row, side=1)
            if z_high > z_low:
                zones.append(
                    Zone(
                        side=1,
                        z_low=z_low,
                        z_high=z_high,
                        born_ts=pd.Timestamp(ts),
                        confirmed_at=pd.Timestamp(confirmed_at),
                        zone_id=zid,
                    )
                )
                zid += 1
    return zones


def apply_htf_close_through(zones: list[Zone], close: float) -> None:
    """Invalidate or role-reverse zones on a completed HTF close (mutates in place)."""
    for z in zones:
        if not z.active:
            continue
        if z.side > 0 and close < z.z_low:
            if not z.flipped:
                z.side = -1
                z.flipped = True
                z.touches = 0
            else:
                z.active = False
        elif z.side < 0 and close > z.z_high:
            if not z.flipped:
                z.side = 1
                z.flipped = True
                z.touches = 0
            else:
                z.active = False


def register_touch(zone: Zone, max_touches: int) -> None:
    """Increment touch count; deactivate if over max."""
    zone.touches += 1
    if zone.touches >= max_touches:
        zone.active = False


def active_zones_at(
    ts: pd.Timestamp,
    zones: list[Zone],
    *,
    require_active: bool = True,
) -> list[Zone]:
    """Zones confirmed by ``ts`` (and optionally still marked active).

    Does not recompute invalidation/touches — caller must have applied those
    chronologically up to ``ts``.
    """
    out: list[Zone] = []
    for z in zones:
        if pd.Timestamp(ts) < pd.Timestamp(z.confirmed_at):
            continue
        if require_active and not z.active:
            continue
        out.append(z)
    return out


@dataclass
class ZoneBook:
    """Chronological zone state driven by completed HTF bars + M15 taps."""

    zones: list[Zone] = field(default_factory=list)
    _pending: list[Zone] = field(default_factory=list)
    _htf_times: list[pd.Timestamp] = field(default_factory=list)
    _htf_closes: list[float] = field(default_factory=list)
    _delta: pd.Timedelta = field(default_factory=lambda: pd.Timedelta("4h"))
    _next_htf: int = 0
    max_touches: int = 2
    _inside: dict[int, bool] = field(default_factory=dict)

    @classmethod
    def from_htf(
        cls,
        htf_ohlc: pd.DataFrame,
        left: int = 2,
        right: int = 2,
        htf_rule: str = "4h",
        max_touches: int = 2,
    ) -> ZoneBook:
        ph, pl = swing_pivots(htf_ohlc, left=left, right=right)
        pending = zones_from_pivots(htf_ohlc, ph, pl, right=right, htf_rule=htf_rule)
        book = cls(max_touches=max_touches)
        book._pending = sorted(pending, key=lambda z: z.confirmed_at)
        book._htf_times = [pd.Timestamp(t) for t in htf_ohlc.index]
        book._htf_closes = [float(c) for c in htf_ohlc["close"].to_numpy()]
        book._delta = rule_to_timedelta(htf_rule)
        return book

    def advance_to(self, ts: pd.Timestamp) -> None:
        """Apply HTF closes completed by ``ts`` and activate newly confirmed zones."""
        ts = pd.Timestamp(ts)
        while self._next_htf < len(self._htf_times):
            end = self._htf_times[self._next_htf] + self._delta
            if end > ts:
                break
            apply_htf_close_through(self.zones, self._htf_closes[self._next_htf])
            self._next_htf += 1
        while self._pending and self._pending[0].confirmed_at <= ts:
            self.zones.append(self._pending.pop(0))

    def active(self, ts: pd.Timestamp) -> list[Zone]:
        self.advance_to(ts)
        return active_zones_at(ts, self.zones, require_active=True)

    def note_m15_bar(self, ts: pd.Timestamp, bar_low: float, bar_high: float) -> None:
        """Advance clock and edge-trigger touch counts on active zones."""
        self.advance_to(ts)
        for z in self.zones:
            if not z.active:
                if z.zone_id in self._inside:
                    self._inside[z.zone_id] = False
                continue
            inside = z.intersects(bar_low, bar_high)
            was = self._inside.get(z.zone_id, False)
            if inside and not was:
                register_touch(z, self.max_touches)
            self._inside[z.zone_id] = inside
