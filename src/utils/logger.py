"""Logging configuration module."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name (typically __name__).
        log_level: Logging level (default: INFO).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # File handler (rotating)
    from logging.handlers import RotatingFileHandler
    log_dir = Path(__file__).parent.parent.parent / 'data' / 'monitoring'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'app.log'
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger."""
    return logging.getLogger(name)
