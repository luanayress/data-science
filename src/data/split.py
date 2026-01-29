"""Data splitting utilities."""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from ..utils.logger import get_logger

logger = get_logger(__name__)


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature DataFrame.
        y: Target Series (optional).
        test_size: Proportion of test set (0-1).
        random_state: Random seed for reproducibility.
        stratify: Whether to use stratified split (for classification).
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
        If y is None, returns only X_train, X_test.
    """
    stratify_arg = y if stratify else None
    
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )
        logger.info(f"Split data: train {len(X_train)} samples, test {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )
        logger.info(f"Split data: train {len(X_train)} samples, test {len(X_test)} samples")
        return X_train, X_test, None, None


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series = None,
    train_size: float = 0.6,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature DataFrame.
        y: Target Series (optional).
        train_size: Proportion of training set (0-1).
        val_size: Proportion of validation set (0-1).
        random_state: Random seed.
        stratify: Whether to use stratified split.
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    test_size = 1 - train_size - val_size
    
    # First split: train vs (val + test)
    stratify_arg = y if stratify else None
    if y is not None:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_arg
        )
    else:
        X_train, X_temp = train_test_split(
            X,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_arg
        )
        y_train = None
        y_temp = None
    
    # Second split: validation vs test
    val_ratio = val_size / (val_size + test_size)
    if y is not None:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
    else:
        X_val, X_test = train_test_split(
            X_temp,
            train_size=val_ratio,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
        y_val = None
        y_test = None
    
    logger.info(f"Split data: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    
    if y is not None:
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_val, X_test, None, None, None
