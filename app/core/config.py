"""Central configuration for the API and Streamlit HTTP client."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from src.utils.config import load_inference_config
from src.utils.paths import get_project_root

DEFAULT_ORIGINS = (
    "http://localhost:8501", "http://127.0.0.1:8501",
    "http://localhost:3000", "http://127.0.0.1:3000",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Resolved settings. Precedence is environment, YAML, then defaults."""

    environment: str
    model_version: str
    shadow_model_version: Optional[str]
    churn_threshold_profile: Optional[str]
    model_dir: Path
    api_host: str
    api_port: int
    api_url: str
    allowed_origins: Tuple[str, ...]
    prediction_threshold: float
    log_level: str
    http_timeout: float
    batch_http_timeout: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config = load_inference_config()
    model_config = config.get("model", {})
    prediction_config = config.get("prediction", {})
    api_config = config.get("api", {})
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    if not model_dir.is_absolute():
        model_dir = get_project_root() / model_dir
    origins_raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = tuple(item.strip() for item in origins_raw.split(",") if item.strip()) or DEFAULT_ORIGINS
    return Settings(
        environment=os.getenv("APP_ENV", config.get("environment", "development")),
        model_version=os.getenv("MODEL_VERSION", model_config.get("version", "v2")),
        shadow_model_version=os.getenv("SHADOW_MODEL_VERSION") or model_config.get("shadow_version") or None,
        churn_threshold_profile=os.getenv("CHURN_THRESHOLD_PROFILE") or prediction_config.get("threshold_profile") or None,
        model_dir=model_dir,
        api_host=os.getenv("API_HOST", api_config.get("host", "127.0.0.1")),
        api_port=_env_int("API_PORT", api_config.get("port", 8000)),
        api_url=os.getenv("API_URL", api_config.get("url", "http://localhost:8000")).rstrip("/"),
        allowed_origins=origins,
        prediction_threshold=_env_float("PREDICTION_THRESHOLD", prediction_config.get("threshold", 0.5)),
        log_level=os.getenv("LOG_LEVEL", config.get("logging", {}).get("level", "INFO")),
        http_timeout=_env_float("API_TIMEOUT", api_config.get("timeout", 5.0)),
        batch_http_timeout=_env_float("API_BATCH_TIMEOUT", api_config.get("batch_timeout", 30.0)),
    )
