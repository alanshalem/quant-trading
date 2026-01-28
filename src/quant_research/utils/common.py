"""
common.py - Common Utilities and Reproducibility
=================================================
Transversal helper functions for Python, PyTorch, and general utilities.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import polars as pl

from ..config import SEED


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducible results across all libraries.

    This function ensures deterministic behavior by setting seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)
    - CUDNN (PyTorch's GPU backend)

    Args:
        seed: Integer seed value (default: SEED constant = 42)

    Returns:
        None

    Example:
        >>> set_seed(42)  # All subsequent random operations will be reproducible
        >>> # Train model - results will be identical on re-runs
        >>> model = train_model(...)

    Note:
        - Essential for reproducible research and debugging
        - May slightly reduce performance due to deterministic algorithms
        - Must be called before any random operations
        - For complete reproducibility, also control data loading order
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def to_tensor(x, dtype=None) -> torch.Tensor:
    """
    Convert a Polars Series or DataFrame column to a PyTorch tensor.

    Args:
        x: Polars Series or column to convert
        dtype: Optional PyTorch dtype. Defaults to torch.float32

    Returns:
        PyTorch tensor with the specified dtype

    Example:
        >>> # Convert column to tensor
        >>> features = to_tensor(df['close'])
        >>>
        >>> # Convert with specific dtype
        >>> labels = to_tensor(df['target'], dtype=torch.int64)
    """
    return torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)


def init_weights(m):
    """Initialize weights for linear layers with Xavier uniform initialization.

    Note: set_seed() should be called before applying this function to ensure
    reproducibility across the entire model.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
