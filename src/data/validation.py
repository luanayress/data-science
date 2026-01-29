"""Data validation utilities."""

import pandas as pd
from typing import List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def validate_data_shape(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> bool:
    """
    Validate DataFrame dimensions.
    
    Args:
        df: DataFrame to validate.
        min_rows: Minimum number of rows required.
        min_cols: Minimum number of columns required.
        
    Returns:
        True if valid, raises ValueError otherwise.
    """
    if df.shape[0] < min_rows:
        raise ValueError(f"DataFrame has {df.shape[0]} rows, expected at least {min_rows}")
    
    if df.shape[1] < min_cols:
        raise ValueError(f"DataFrame has {df.shape[1]} columns, expected at least {min_cols}")
    
    return True


def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[bool, dict]:
    """
    Check for missing values in DataFrame.
    
    Args:
        df: DataFrame to check.
        threshold: Maximum allowed proportion of missing values (0-1).
        
    Returns:
        Tuple of (is_valid, missing_info_dict).
    """
    missing_info = {}
    invalid_cols = []
    
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        missing_info[col] = {
            'count': df[col].isna().sum(),
            'ratio': missing_ratio
        }
        
        if missing_ratio > threshold:
            invalid_cols.append(col)
    
    is_valid = len(invalid_cols) == 0
    
    if not is_valid:
        logger.warning(f"Columns with missing values > {threshold}: {invalid_cols}")
    
    return is_valid, missing_info


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate that all required columns are present.
    
    Args:
        df: DataFrame to validate.
        required_cols: List of required column names.
        
    Returns:
        True if all columns present, raises ValueError otherwise.
    """
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True


def validate_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> bool:
    """
    Validate that specified columns are numeric.
    
    Args:
        df: DataFrame to validate.
        numeric_cols: List of column names that should be numeric.
        
    Returns:
        True if all columns are numeric, raises ValueError otherwise.
    """
    non_numeric = []
    
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    
    if non_numeric:
        raise ValueError(f"Expected numeric columns but found non-numeric: {non_numeric}")
    
    return True


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of data characteristics.
    
    Args:
        df: DataFrame to summarize.
        
    Returns:
        Dictionary with summary statistics.
    """
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
