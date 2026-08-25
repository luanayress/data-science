"""Configuration management module for loading YAML config files."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    loaded = config if config is not None else {}
    return _apply_env_overrides(loaded)


def _coerce_env_value(raw_value: str) -> Any:
    """Best-effort coercion for env override values."""
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override config values from environment variables using CFG__KEY__PATH syntax."""
    merged = dict(config)

    for key, value in os.environ.items():
        if not key.startswith("CFG__"):
            continue

        path_parts = [part.lower() for part in key.split("__")[1:] if part]
        if not path_parts:
            continue

        target = merged
        for part in path_parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        target[path_parts[-1]] = _coerce_env_value(value)

    return merged


def load_training_config() -> Dict[str, Any]:
    """Load training configuration."""
    from .paths import get_config_path
    return load_config(get_config_path('training.yaml'))


def load_inference_config() -> Dict[str, Any]:
    """Load inference configuration."""
    from .paths import get_config_path
    return load_config(get_config_path('inference.yaml'))


def load_features_config() -> Dict[str, Any]:
    """Load features configuration."""
    from .paths import get_config_path
    return load_config(get_config_path('features.yaml'))
