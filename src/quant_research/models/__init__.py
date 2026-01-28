"""
models - Machine Learning Module
=================================
Functions for training, validating, and inspecting PyTorch models.
Includes neural network architectures for time series prediction.
"""

from .validation import (
    timeseries_split,
    timeseries_train_test_split,
    _prepare_train_test_tensors,
)

from .inspection import (
    total_model_params,
    print_model_info,
    print_model_complexity_ratio,
    get_linear_params,
    print_model_params,
)

from .trainer import (
    batch_train_reg,
    train_reg_model,
    benchmark_reg_model,
    benchmark_linear_models,
    _train_single_model_config,
)

from .inference import (
    add_model_predictions,
)

from .architectures import (
    LinearModel,
    NonLinearModel,
    DeepModel,
    LSTMModel,
    AttentionModel,
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
