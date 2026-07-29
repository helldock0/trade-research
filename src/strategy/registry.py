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
    "ebp",
    "nowick",
    "fib_swing",
    "judas",
)

_CONFIG_BY_ID: dict[str, str] = {
    "orb": "strategy_orb.yaml",
    "csmo": "strategy_csmo.yaml",
    "htf_pullback": "strategy_htf_pullback.yaml",
    "htf": "strategy_htf_pullback.yaml",
    "d1_mom": "strategy_d1_mom.yaml",
    "d1": "strategy_d1_mom.yaml",
    "fail_fade": "strategy_fail_fade.yaml",
    "fade": "strategy_fail_fade.yaml",
    "vwap_fade": "strategy_vwap_fade.yaml",
    "vwap": "strategy_vwap_fade.yaml",
    "msnr": "strategy_msnr.yaml",
    "donchian_atr": "strategy_donchian_atr.yaml",
    "donchian": "strategy_donchian_atr.yaml",
    "xau_htf_pullback": "strategy_xau_htf_pullback.yaml",
    "xau_htf": "strategy_xau_htf_pullback.yaml",
    "xau": "strategy_xau_htf_pullback.yaml",
    "htf_adx_pullback": "strategy_htf_adx_pullback.yaml",
    "htf_adx": "strategy_htf_adx_pullback.yaml",
    "adx_pullback": "strategy_htf_adx_pullback.yaml",
    "ebp": "strategy_ebp.yaml",
    "nowick": "strategy_nowick.yaml",
    "fib_swing": "strategy_fib_swing.yaml",
    "fib": "strategy_fib_swing.yaml",
    "judas": "strategy_judas.yaml",
}


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
    if sid == "ebp":
        from src.strategy.ebp import generate_signals

        return generate_signals
    if sid == "nowick":
        from src.strategy.nowick import generate_signals

        return generate_signals
    if sid in ("fib_swing", "fib"):
        from src.strategy.fib_swing import generate_signals

        return generate_signals
    if sid == "judas":
        from src.strategy.judas import generate_signals

        return generate_signals
    from src.strategy.cssm import generate_signals

    return generate_signals


def default_config_name(strategy: str | None = None) -> str:
    s = (strategy or "cssm").lower()
    return _CONFIG_BY_ID.get(s, "strategy_cssm.yaml")
