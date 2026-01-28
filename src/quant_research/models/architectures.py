"""
architectures.py - Neural Network Model Architectures
=====================================================
PyTorch model architectures for quantitative trading predictions.

This module provides reusable model architectures for time series prediction,
starting with simple linear models and extending to more complex neural networks.
"""

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """
    Simple Linear Regression Model for time series prediction.

    This is an AR(n) (autoregressive) model implemented as a single
    linear layer. Maps n input features (lagged returns) to a single
    prediction of future return.

    Args:
        input_features: Number of input features (typically number of lags)

    Architecture:
        Input -> Linear(input_features, 1) -> Output

    The learned weights represent the AR coefficients:
    - Negative weights suggest mean reversion
    - Positive weights suggest momentum

    Example:
        >>> model = LinearModel(3)  # AR(3) model
        >>> x = torch.tensor([[0.01, -0.02, 0.015]])  # 3 lagged returns
        >>> prediction = model(x)  # Predicted next return
    """

    def __init__(self, input_features: int):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NonLinearModel(nn.Module):
    """
    Non-Linear Neural Network Model for time series prediction.

    A simple feed-forward neural network with one hidden layer and
    ReLU activation. Can capture non-linear relationships in the data
    that a linear model would miss.

    Args:
        input_features: Number of input features (typically number of lags)
        hidden_size: Number of neurons in the hidden layer (default: 64)

    Architecture:
        Input -> Linear(input_features, hidden_size) -> ReLU -> Linear(hidden_size, 1) -> Output

    Example:
        >>> model = NonLinearModel(3, hidden_size=32)
        >>> x = torch.tensor([[0.01, -0.02, 0.015]])
        >>> prediction = model(x)

    Note:
        Non-linear models are more prone to overfitting. Use proper
        regularization (dropout, weight decay) and validation.
    """

    def __init__(self, input_features: int, hidden_size: int = 64):
        super(NonLinearModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepModel(nn.Module):
    """
    Deep Neural Network Model with multiple hidden layers.

    A deeper architecture with multiple hidden layers, batch normalization,
    and dropout for regularization. Suitable for complex patterns but
    requires more data and careful tuning.

    Args:
        input_features: Number of input features
        hidden_sizes: List of hidden layer sizes (default: [64, 32])
        dropout: Dropout probability (default: 0.1)

    Architecture:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear -> Output

    Example:
        >>> model = DeepModel(3, hidden_sizes=[64, 32, 16], dropout=0.2)
        >>> x = torch.tensor([[0.01, -0.02, 0.015]])
        >>> prediction = model(x)
    """

    def __init__(self, input_features: int, hidden_sizes: list = None, dropout: float = 0.1):
        super(DeepModel, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        layers = []
        prev_size = input_features

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSTMModel(nn.Module):
    """
    LSTM (Long Short-Term Memory) Model for sequence prediction.

    Uses LSTM layers to capture temporal dependencies in the data.
    Good for capturing long-range dependencies in time series.

    Args:
        input_features: Number of features per time step
        hidden_size: LSTM hidden state size (default: 64)
        num_layers: Number of LSTM layers (default: 1)
        dropout: Dropout between LSTM layers (default: 0.0)

    Input Shape:
        (batch_size, sequence_length, input_features)

    Example:
        >>> model = LSTMModel(1, hidden_size=32, num_layers=2)
        >>> x = torch.randn(32, 10, 1)  # batch=32, seq_len=10, features=1
        >>> prediction = model(x)
    """

    def __init__(self, input_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class AttentionModel(nn.Module):
    """
    Simple Self-Attention Model for time series prediction.

    Uses a simplified attention mechanism to weight the importance
    of different time steps in the sequence.

    Args:
        input_features: Number of input features (typically sequence length)
        hidden_size: Size of query/key/value projections (default: 32)

    Example:
        >>> model = AttentionModel(10, hidden_size=16)
        >>> x = torch.randn(32, 10)  # batch=32, features=10
        >>> prediction = model(x)
    """

    def __init__(self, input_features: int, hidden_size: int = 32):
        super(AttentionModel, self).__init__()

        self.query = nn.Linear(input_features, hidden_size)
        self.key = nn.Linear(input_features, hidden_size)
        self.value = nn.Linear(input_features, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention scores
        scores = torch.matmul(q.unsqueeze(-1), k.unsqueeze(-2))
        weights = torch.softmax(scores / (k.shape[-1] ** 0.5), dim=-1)

        # Weighted values
        attended = torch.matmul(weights, v.unsqueeze(-1)).squeeze(-1)

        return self.fc(attended)
