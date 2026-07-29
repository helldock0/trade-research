from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_strategy_config(name: str = "strategy_cssm.yaml") -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def resolve_strategy_config_name(strategy: str | None = None, config: str | None = None) -> str:
    """Prefer explicit --config; else map --strategy to yaml filename."""
    if config:
        return config
    from src.strategy.registry import default_config_name

    return default_config_name(strategy)


def load_ftmo_config(name: str = "ftmo_2step.yaml") -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def load_costs_config(name: str = "costs.yaml") -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def ensure_dirs() -> None:
    (ROOT / "data" / "parquet").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "mt5").mkdir(parents=True, exist_ok=True)
