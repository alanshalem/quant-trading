"""
models - Machine Learning Module
=================================
Functions for training, validating, and inspecting PyTorch models.
Includes neural network architectures for time series prediction.
"""

from .architectures import (
    AttentionModel,
    DeepModel,
    LinearModel,
    LSTMModel,
    NonLinearModel,
)
from .inference import (
    add_model_predictions,
)
from .inspection import (
    get_linear_params,
    print_model_complexity_ratio,
    print_model_info,
    print_model_params,
    total_model_params,
)
from .trainer import (
    _train_single_model_config,
    batch_train_reg,
    benchmark_linear_models,
    benchmark_reg_model,
    train_reg_model,
)
from .validation import (
    _prepare_train_test_tensors,
    timeseries_split,
    timeseries_train_test_split,
)

__all__ = [
    # validation.py
    'timeseries_split',
    'timeseries_train_test_split',
    '_prepare_train_test_tensors',
    # inspection.py
    'total_model_params',
    'print_model_info',
    'print_model_complexity_ratio',
    'get_linear_params',
    'print_model_params',
    # trainer.py
    'batch_train_reg',
    'train_reg_model',
    'benchmark_reg_model',
    'benchmark_linear_models',
    '_train_single_model_config',
    # inference.py
    'add_model_predictions',
    # architectures.py
    'LinearModel',
    'NonLinearModel',
    'DeepModel',
    'LSTMModel',
    'AttentionModel',
]
