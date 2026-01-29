"""Path management utilities."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_path(filename: str = "") -> Path:
    """Get path to data directory."""
    data_dir = get_project_root() / "data"
    if filename:
        return data_dir / filename
    return data_dir


def get_raw_data_path(filename: str = "") -> Path:
    """Get path to raw data directory."""
    raw_dir = get_project_root() / "data" / "raw"
    if filename:
        return raw_dir / filename
    return raw_dir


def get_processed_data_path(filename: str = "") -> Path:
    """Get path to processed data directory."""
    processed_dir = get_project_root() / "data" / "processed"
    if filename:
        return processed_dir / filename
    return processed_dir


def get_models_path(version: str = "v1", filename: str = "") -> Path:
    """Get path to models directory."""
    models_dir = get_project_root() / "models" / version
    models_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        return models_dir / filename
    return models_dir


def get_config_path(filename: str = "") -> Path:
    """Get path to config directory."""
    config_dir = get_project_root() / "configs"
    if filename:
        return config_dir / filename
    return config_dir


def get_notebooks_path(filename: str = "") -> Path:
    """Get path to notebooks directory."""
    notebooks_dir = get_project_root() / "notebooks"
    if filename:
        return notebooks_dir / filename
    return notebooks_dir
