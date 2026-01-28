"""
validation.py - Data Splitting and Validation
==============================================
Functions for splitting time series data into train/test sets.
"""

from typing import List, Tuple
import torch
import polars as pl

from ..utils.common import to_tensor


def timeseries_split(t, test_size=0.25):
    """
    Split a tensor or array into train/test sets based on a proportion.

    Parameters
    ----------
    t : torch.Tensor or np.ndarray
        Time series data.
    test_size : float, default 0.25
        Proportion of data to use for testing. Must be between 0 and 1.

    Returns
    -------
    train, test : same type as t
        Train and test splits.

    Raises
    ------
    ValueError
        If test_size is not strictly between 0 and 1.
    """
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1 (got {test_size})")

    split_idx = int(len(t) * (1 - test_size))
    return t[:split_idx], t[split_idx:]


def timeseries_train_test_split(df: pl.DataFrame, features, target, test_size=0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split time series data into train/test sets and convert to PyTorch tensors.

    This is a convenience function that combines data cleaning, tensor conversion,
    and time series splitting in one step.

    Args:
        df: DataFrame containing features and target
        features: List of feature column names or single column name
        target: Target column name
        test_size: Proportion of data for testing (default: 0.25)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as PyTorch tensors

    Example:
        >>> features = ['lag_1', 'lag_2', 'lag_3']
        >>> target = 'close_log_return'
        >>> X_train, X_test, y_train, y_test = timeseries_train_test_split(
        ...     df, features, target, test_size=0.25
        ... )

    Note:
        - Automatically drops null values before splitting
        - Target is reshaped to (N, 1) for PyTorch compatibility
        - Preserves temporal order (earlier data in train, later in test)
    """
    df = df.drop_nulls()
    X = to_tensor(df[features])
    y = to_tensor(df[target]).reshape(-1, 1)
    X_train, X_test = timeseries_split(X, test_size)
    y_train, y_test = timeseries_split(y, test_size)
    return X_train, X_test, y_train, y_test


def _prepare_train_test_tensors(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    features: List[str],
    target: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert train/test DataFrames to PyTorch tensors (optimized).

    Fixes the "NumPy array is not writable" warning by copying arrays
    before converting to PyTorch tensors.

    Returns:
        X_train, X_test, y_train, y_test tensors
    """
    # Convert training features to tensor (make a copy first to avoid warning)
    X_train = torch.from_numpy(df_train[features].to_numpy().copy()).float()

    # Convert training target to tensor and reshape to (N,1)
    y_train = torch.from_numpy(df_train[target].to_numpy().copy()).float().reshape(-1, 1)

    # Convert test features to tensor (make a copy first to avoid warning)
    X_test = torch.from_numpy(df_test[features].to_numpy().copy()).float()

    # Convert test target to tensor and reshape to (N,1)
    y_test = torch.from_numpy(df_test[target].to_numpy().copy()).float().reshape(-1, 1)

    return X_train, X_test, y_train, y_test
