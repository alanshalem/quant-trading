"""
trainer.py - Model Training
============================
Functions for training PyTorch models with batch gradient descent.
"""

from typing import Any, Dict, List, Optional, Tuple
import itertools
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..config import (
    SEED,
    DEFAULT_PARALLEL,
    DEFAULT_LBFGS_LR,
    DEFAULT_EPOCHS,
    LOG_INTERVAL_DIVISOR,
)
from ..utils.common import set_seed, init_weights
from .validation import timeseries_split, _prepare_train_test_tensors
from .inspection import get_linear_params

# Note: eval_model_performance is imported lazily in benchmark_reg_model
# to avoid circular import with backtest.engine


def batch_train_reg(
    model: nn.Module,
    X_train,
    X_test,
    y_train,
    y_test,
    no_epochs: int,
    criterion=None,
    optimizer=None,
    optimizer_type: str = 'lbfgs',
    logging=True,
    lr=None
):
    """Train a regression model with batch gradient descent.

    Args:
        model: PyTorch model to train
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        no_epochs: Number of training epochs
        criterion: Loss function (default: L1Loss)
        optimizer: Optional custom optimizer (overrides optimizer_type)
        optimizer_type: Type of optimizer - 'lbfgs' (default, better for small datasets)
                        or 'adam' (2-10x faster for large datasets)
        logging: Print training progress (default: True)
        lr: Learning rate (default: depends on optimizer_type)

    Returns:
        Test set predictions
    """
    if criterion is None:
        criterion = nn.L1Loss()

    # Default optimizer based on type
    if optimizer is None:
        if optimizer_type == 'adam':
            # Adam: Better for large datasets, faster convergence
            if lr is None:
                lr = 0.001  # Standard Adam learning rate
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:  # lbfgs (default)
            # LBFGS: Better convergence for small datasets
            if lr is None:
                lr = DEFAULT_LBFGS_LR
            optimizer = optim.LBFGS(
                model.parameters(),
                lr=lr,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-7,
                tolerance_change=1e-9
            )
    else:
        # If custom optimizer provided, use it
        if lr is not None:
            # Update learning rate if specified
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Logging model info
    if logging:
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
        print("Model architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel()} params)")
        print("\nTraining model...")

    train_loss = None
    log_tick_size = max(no_epochs // LOG_INTERVAL_DIVISOR, 1)  # avoid zero division

    # Training loop
    if isinstance(optimizer, torch.optim.LBFGS):
        # LBFGS requires a closure
        for epoch in range(no_epochs):
            def closure():
                optimizer.zero_grad()
                predictions = model(X_train)
                loss = criterion(predictions, y_train)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                train_loss = criterion(model(X_train), y_train).item()

            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch+1}/{no_epochs}], Loss: {train_loss:.6f}")

    else:
        # SGD/Adam loop
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch+1}/{no_epochs}], Loss: {loss.item():.6f}")

    # After training
    if logging:
        print("\nLearned parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}:\n{param.data.numpy()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
        test_loss = criterion(y_hat, y_test)
        if logging:
            print(f'\nTest Loss: {test_loss.item():.6f}, Train Loss: {train_loss:.6f}')

    return y_hat


def train_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate, test_size=0.25, loss=None, optimizer=None, optimizer_type: str = 'lbfgs', no_epochs=None, log=False, lr=None):
    """
    Train a regression model and return test set predictions.

    Args:
        df: DataFrame with features and target columns
        features: List of feature column names
        target: Target column name
        model: PyTorch model to train
        annualized_rate: Sharpe ratio annualization factor (not used in this function)
        test_size: Test set proportion (default: 0.25)
        loss: Optional loss function (default: L1Loss)
        optimizer: Optional optimizer (overrides optimizer_type)
        optimizer_type: Type of optimizer - 'lbfgs' or 'adam' (default: 'lbfgs')
        no_epochs: Training epochs (default: DEFAULT_EPOCHS)
        log: Whether to log training progress (default: False)
        lr: Learning rate (default: depends on optimizer_type)

    Returns:
        Test set predictions as PyTorch tensor

    Example:
        >>> model = LinearModel(3)
        >>> y_hat = train_reg_model(
        ...     df, ['lag_1', 'lag_2', 'lag_3'], 'close_log_return',
        ...     model, annualized_rate=10, no_epochs=1000
        ... )
    """
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = DEFAULT_EPOCHS
    X_train, X_test, y_train, y_test = _prepare_train_test_tensors(df_train, df_test, features, target)

    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, optimizer_type, lr=lr, logging=log)
    return y_hat


def benchmark_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate, test_size=0.25, loss=None, optimizer=None, optimizer_type: str = 'lbfgs', no_epochs=None, log=False, lr=None):
    """
    Train a regression model and return comprehensive performance metrics.

    This function trains a model and evaluates it using trading-focused metrics
    including Sharpe ratio, win rate, max drawdown, and more.

    Args:
        df: DataFrame with features and target columns
        features: List of feature column names
        target: Target column name
        model: PyTorch model to train
        annualized_rate: Annualization factor for Sharpe ratio
        test_size: Test set proportion (default: 0.25)
        loss: Optional loss function (default: L1Loss)
        optimizer: Optional optimizer (overrides optimizer_type)
        optimizer_type: Type of optimizer - 'lbfgs' or 'adam' (default: 'lbfgs')
        no_epochs: Training epochs (default: DEFAULT_EPOCHS)
        log: Whether to log training progress (default: False)
        lr: Learning rate (default: depends on optimizer_type)

    Returns:
        Dictionary with performance metrics:
        - features: Comma-separated feature names
        - win_rate: Proportion of profitable trades
        - sharpe: Annualized Sharpe ratio
        - max_drawdown: Maximum equity drawdown
        - weights: Model weights (for linear models)
        - biases: Model biases (for linear models)
        - And more...

    Example:
        >>> model = LinearModel(3)
        >>> perf = benchmark_reg_model(
        ...     df, ['lag_1', 'lag_2', 'lag_3'], 'close_log_return',
        ...     model, annualized_rate=10
        ... )
        >>> print(f"Sharpe: {perf['sharpe']:.2f}")
    """
    # Lazy import to avoid circular dependency with backtest.engine
    from ..backtest.performance import eval_model_performance

    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = DEFAULT_EPOCHS
    X_train, X_test, y_train, y_test = _prepare_train_test_tensors(df_train, df_test, features, target)

    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, optimizer_type, lr=lr, logging=log)

    perf = eval_model_performance(y_test, y_hat, features, target, annualized_rate)

    weights, biases = get_linear_params(model)
    perf['weights'] = str(weights)
    perf['biases'] = str(biases)

    return perf


def _train_single_model_config(args: Tuple) -> Dict[str, Any]:
    """Train a single model configuration (helper for parallel benchmarking).

    Args:
        args: Tuple of (features, ts_data, target, annualized_rate, n_epochs, test_size)

    Returns:
        Dictionary with performance metrics
    """
    features, ts_data, target, annualized_rate, n_epochs, test_size = args

    # Re-initialize seed for reproducibility in each process
    set_seed(SEED)

    # Create and train model
    import models
    m = models.LinearModel(len(features))
    m.apply(init_weights)

    return benchmark_reg_model(
        ts_data, list(features), target, m, annualized_rate,
        no_epochs=n_epochs, loss=None, test_size=test_size, log=False
    )


def benchmark_linear_models(
    ts: pl.DataFrame,
    target: str,
    feature_pool: List[str],
    annualized_rate: int,
    max_no_features: int = 1,
    no_epochs: int = 200,
    loss=None,
    test_size: float = 0.25,
    max_workers: Optional[int] = None,
    parallel: bool = DEFAULT_PARALLEL
) -> pl.DataFrame:
    """
    Benchmark all possible feature combinations using linear models (with optional parallelization).

    This function performs an exhaustive grid search over feature combinations,
    training a linear model for each combination and ranking by Sharpe ratio.

    Args:
        ts: Time series DataFrame with features and target
        target: Target column name (e.g., 'close_log_return')
        feature_pool: List of feature column names to combine
        annualized_rate: Annualization factor for Sharpe ratio
        max_no_features: Maximum number of features in combinations (default: 1)
        no_epochs: Training epochs per model (default: 200)
        loss: Optional loss function (default: L1Loss, currently not used in parallel mode)
        test_size: Test set proportion (default: 0.25)
        max_workers: Maximum number of parallel workers (None = use CPU count)
        parallel: Enable parallel processing (default: True for 4-8x speedup)

    Returns:
        DataFrame sorted by Sharpe ratio (descending) containing:
        - features: Comma-separated feature names
        - performance metrics: win_rate, sharpe, max_drawdown, etc.
        - model params: weights and biases

    Example:
        >>> feature_pool = ['lag_1', 'lag_2', 'lag_3', 'lag_4']
        >>> # Parallel benchmarking (default)
        >>> results = benchmark_linear_models(
        ...     ts, 'close_log_return', feature_pool,
        ...     annualized_rate=10, max_no_features=3
        ... )
        >>> print(results.head())  # Best performing combinations
        >>>
        >>> # Sequential benchmarking
        >>> results = benchmark_linear_models(
        ...     ts, 'close_log_return', feature_pool,
        ...     annualized_rate=10, max_no_features=3, parallel=False
        ... )

    Note:
        - Computational complexity: O(2^n) for n features
        - For max_no_features=3 and 4 features: tests 4+6+4 = 14 combinations
        - Results include both in-sample and out-of-sample metrics
        - Parallel mode provides 4-8x speedup on multi-core systems
    """
    ts = ts.drop_nulls()

    # Generate all feature combinations
    fs = []
    for i in range(1, max_no_features + 1):
        fs += list(itertools.combinations(feature_pool, i))

    if not parallel or len(fs) == 1:
        # Sequential processing
        from .architectures import LinearModel
        benchmarks = []
        for features in tqdm(fs, desc="Benchmarking models"):
            m = LinearModel(len(features))
            m.apply(init_weights)
            benchmarks.append(
                benchmark_reg_model(ts, list(features), target, m, annualized_rate,
                                  no_epochs=no_epochs, loss=loss, test_size=test_size, log=False)
            )
    else:
        # Parallel processing
        args_list = [(features, ts, target, annualized_rate, no_epochs, test_size) for features in fs]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            benchmarks = list(tqdm(
                executor.map(_train_single_model_config, args_list),
                total=len(args_list),
                desc="Benchmarking models (parallel)"
            ))

    benchmark = pl.DataFrame(benchmarks)
    return benchmark.sort('sharpe', descending=True)
