"""Data loading utilities."""

import pandas as pd
from pathlib import Path
from typing import Tuple
from ..utils.logger import get_logger
from ..utils.paths import get_raw_data_path, get_processed_data_path

logger = get_logger(__name__)


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw CSV data.
    
    Args:
        filename: Name of the CSV file.
        
    Returns:
        DataFrame with loaded data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    filepath = get_raw_data_path(filename)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Load processed CSV data.
    
    Args:
        filename: Name of the CSV file.
        
    Returns:
        DataFrame with loaded data.
    """
    filepath = get_processed_data_path(filename)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading processed data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def save_processed_data(df: pd.DataFrame, filename: str) -> Path:
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save.
        filename: Output filename.
        
    Returns:
        Path to saved file.
    """
    filepath = get_processed_data_path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {filepath}")
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
    
    return filepath


def load_customer_churn_data() -> pd.DataFrame:
    """
    Load the customer churn dataset (convenience function).
    
    Returns:
        DataFrame with raw customer churn data.
    """
    return load_raw_data("Customer-Churn-Records.csv")
