"""
inspection.py - Model Inspection and Analysis
==============================================
Functions for analyzing and inspecting PyTorch model architectures and parameters.
"""

from typing import Tuple
import numpy as np
import torch.nn as nn


def total_model_params(model: nn.Module) -> int:
    """
    Count the total number of parameters in a PyTorch model.

    Args:
        model: PyTorch model (nn.Module)

    Returns:
        Total number of parameters (trainable + frozen)

    Example:
        >>> model = LinearModel(10)
        >>> param_count = total_model_params(model)
        >>> print(f"Model has {param_count:,} parameters")
    """
    return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module, model_name: str) -> None:
    """
    Print detailed information about a PyTorch model's architecture and parameters.

    This function helps you understand:
    - Model complexity (number of parameters)
    - Which parameters are trainable vs frozen
    - Overall model architecture

    Useful for:
    - Comparing different model architectures
    - Debugging training issues
    - Estimating memory requirements
    - Understanding model capacity

    Args:
        model: PyTorch model (nn.Module)
        model_name: Descriptive name for the model (e.g., 'LSTM Predictor')

    Returns:
        None (prints to console)

    Example:
        >>> model = MyTradingModel(input_size=10, hidden_size=64)
        >>> print_model_info(model, 'Trading LSTM v1')

        Output:
        Trading LSTM v1:
          Architecture: MyTradingModel(...)
          Total parameters: 15,234
          Trainable parameters: 15,234

    Note:
        Total parameters includes both trainable and frozen parameters.
        For transfer learning, trainable params may be less than total.
    """
    # Count parameters in a single pass (more efficient)
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        count = p.numel()
        total_params += count
        if p.requires_grad:
            trainable_params += count

    # Print formatted model information
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"\nArchitecture:")
    print(f"  {model}")
    print(f"\nParameter Count:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")

    # Warn if some parameters are frozen
    if total_params != trainable_params:
        frozen_params = total_params - trainable_params
        print(f"  Frozen parameters:     {frozen_params:,}")
        print(f"\n  Note: {frozen_params:,} parameters are frozen")

    print(f"{'='*60}\n")


def print_model_complexity_ratio(m1: nn.Module, m1_name: str, m2: nn.Module, m2_name: str) -> None:
    """
    Compare and print the complexity ratio between two PyTorch models.

    This function calculates the parameter count difference between two models
    and displays the complexity ratio, useful for comparing model architectures.

    Args:
        m1: First PyTorch model (baseline)
        m1_name: Name of the first model (e.g., 'Linear Model')
        m2: Second PyTorch model (comparison)
        m2_name: Name of the second model (e.g., 'Deep Network')

    Returns:
        None (prints comparison to console)

    Example:
        >>> linear_model = LinearModel(10)
        >>> deep_model = NonLinearModel(10)
        >>> print_model_complexity_ratio(linear_model, 'Linear', deep_model, 'NonLinear')
        Complexity Comparison:
            NonLinear has 5.2x more parameters than Linear
            Parametric difference: 1,234 additional parameters

    Note:
        Higher complexity doesn't always mean better performance.
        Consider the bias-variance tradeoff when choosing models.
    """
    m1_params = total_model_params(m1)
    m2_params = total_model_params(m2)
    complexity_ratio = m2_params / m1_params

    print(f"Complexity Comparison:")
    print(f"\t{m2_name} has {complexity_ratio:.1f}x more parameters than {m1_name}")
    print(f"\tParametric difference: {m2_params - m1_params:,} additional parameters")


def get_linear_params(model: nn.Module) -> Tuple[np.ndarray, float]:
    """Extract weights and bias from LinearModel as (w, b)."""
    weight = model.linear.weight.detach().cpu().numpy().flatten()
    bias = model.linear.bias.detach().cpu().numpy().item()
    return weight, bias


def print_model_params(model: nn.Module) -> None:
    """Print learned parameters of a PyTorch model."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:\n{param.data.numpy()}")
