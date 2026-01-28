"""
inference.py - Model Inference
===============================
Functions for running inference with trained PyTorch models.
"""

from typing import List, Union
import torch
import torch.nn as nn
import polars as pl


def add_model_predictions(test_trades: pl.DataFrame, model: nn.Module, features: Union[str, List[str]]) -> pl.DataFrame:
    """
    Add model predictions as a new column to the trades DataFrame.

    This function runs inference on a trained model and adds the predictions
    as a 'y_hat' column, useful for strategy backtesting and analysis.

    Args:
        test_trades: DataFrame containing feature columns
        model: Trained PyTorch model
        features: Single feature name or list of feature names

    Returns:
        Original DataFrame with added 'y_hat' column containing predictions

    Example:
        >>> # Add predictions from trained model
        >>> model = LinearModel(3)
        >>> features = ['lag_1', 'lag_2', 'lag_3']
        >>> df_with_preds = add_model_predictions(test_df, model, features)
        >>> print(df_with_preds['y_hat'].head())

    Note:
        - Model is automatically set to eval mode (no gradients computed)
        - Predictions are converted to numpy then Polars Series
        - Handles both single feature and multiple features
    """
    if type(features) != list:
        features = [features]
    X_test = torch.tensor(test_trades[features].to_numpy(), dtype=torch.float32)
    y_hat = model(X_test).detach().cpu().numpy().squeeze()
    s = pl.Series('y_hat', y_hat)
    return test_trades.with_columns(s)
