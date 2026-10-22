"""Resolve strategy signal generators from config."""

from __future__ import annotations

from typing import Any, Callable

KNOWN_STRATEGIES: tuple[str, ...] = (
    "cssm",
    "orb",
    "csmo",
    "htf_pullback",
    "d1_mom",
    "fail_fade",
    "vwap_fade",
    "msnr",
    "donchian_atr",
    "xau_htf_pullback",
    "htf_adx_pullback",
    "event_drift",
    "macro_swing",
    "tom",
    "imom",
)


def list_strategies() -> list[str]:
    """Canonical strategy ids for batch runners and CLI help."""
    return list(KNOWN_STRATEGIES)


def strategy_id(strategy_cfg: dict[str, Any]) -> str:
    return str(strategy_cfg.get("strategy", "cssm")).lower()


def resolve_signal_fn(strategy_cfg: dict[str, Any]) -> Callable:
    sid = strategy_id(strategy_cfg)
    if sid == "orb":
        from src.strategy.orb import generate_signals

        return generate_signals
    if sid == "csmo":
        from src.strategy.csmo import generate_signals

        return generate_signals
    if sid in ("htf_pullback", "htf"):
        from src.strategy.htf_pullback import generate_signals

        return generate_signals
    if sid in ("d1_mom", "d1"):
        from src.strategy.d1_mom import generate_signals

        return generate_signals
    if sid in ("fail_fade", "fade"):
        from src.strategy.fail_fade import generate_signals

        return generate_signals
    if sid in ("vwap_fade", "vwap"):
        from src.strategy.vwap_fade import generate_signals

        return generate_signals
    if sid == "msnr":
        from src.strategy.msnr import generate_signals

        return generate_signals
    if sid in ("donchian_atr", "donchian"):
        from src.strategy.donchian_atr import generate_signals

        return generate_signals
    if sid in ("xau_htf_pullback", "xau_htf", "xau"):
        from src.strategy.xau_htf_pullback import generate_signals

        return generate_signals
    if sid in ("htf_adx_pullback", "htf_adx", "adx_pullback"):
        from src.strategy.htf_adx_pullback import generate_signals

        return generate_signals
    if sid in ("event_drift", "event"):
        from src.strategy.event_drift import generate_signals

        return generate_signals
    if sid in ("imom", "intraday_momentum"):
        from src.strategy.imom import generate_signals

        return generate_signals
    if sid in ("tom", "turn_of_month"):
        from src.strategy.tom import generate_signals

        return generate_signals
    if sid in ("macro_swing", "macro"):
        from src.strategy.macro_swing import generate_signals

        return generate_signals
    from src.strategy.cssm import generate_signals

    return generate_signals


def default_config_name(strategy: str | None = None) -> str:
    s = (strategy or "cssm").lower()
    if s == "orb":
        return "strategy_orb.yaml"
    if s == "csmo":
        return "strategy_csmo.yaml"
    if s in ("htf_pullback", "htf"):
        return "strategy_htf_pullback.yaml"
    if s in ("d1_mom", "d1"):
        return "strategy_d1_mom.yaml"
    if s in ("fail_fade", "fade"):
        return "strategy_fail_fade.yaml"
    if s in ("vwap_fade", "vwap"):
        return "strategy_vwap_fade.yaml"
    if s == "msnr":
        return "strategy_msnr.yaml"
    if s in ("donchian_atr", "donchian"):
        return "strategy_donchian_atr.yaml"
    if s in ("xau_htf_pullback", "xau_htf", "xau"):
        return "strategy_xau_htf_pullback.yaml"
    if s in ("htf_adx_pullback", "htf_adx", "adx_pullback"):
        return "strategy_htf_adx_pullback.yaml"
    if s in ("event_drift", "event"):
        return "strategy_event_drift.yaml"
    if s in ("macro_swing", "macro"):
        return "strategy_macro_swing.yaml"
    if s in ("tom", "turn_of_month"):
        return "strategy_tom.yaml"
    if s in ("imom", "intraday_momentum"):
        return "strategy_imom.yaml"
    return "strategy_cssm.yaml"
