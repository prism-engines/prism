"""
Orthon Configuration Management

Loads and provides access to YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Any, Optional

CONFIG_DIR = Path(__file__).parent


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file by name."""
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config(name: str, key: Optional[str] = None, default: Any = None) -> Any:
    """Get a config value, optionally by nested key."""
    config = load_config(name)

    if key is None:
        return config

    # Support nested keys like "cohort.signal_types.method"
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


# Lazy-loaded config accessors
_cache: dict[str, dict] = {}


def _get_cached(name: str) -> dict:
    if name not in _cache:
        _cache[name] = load_config(name)
    return _cache[name]


def engines() -> dict:
    """Get engine configuration."""
    return _get_cached("engines")


def cohort() -> dict:
    """Get cohort discovery configuration."""
    return _get_cached("cohort")


def report() -> dict:
    """Get report generation configuration."""
    return _get_cached("report")


def pipeline() -> dict:
    """Get pipeline configuration."""
    return _get_cached("pipeline")


__all__ = [
    "load_config",
    "get_config",
    "engines",
    "cohort",
    "report",
    "pipeline",
    "CONFIG_DIR",
]
