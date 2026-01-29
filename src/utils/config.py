"""Configuration management module for loading YAML config files."""

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
    
    return config if config is not None else {}


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
