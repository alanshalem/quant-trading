"""Tests for model architectures."""

import pytest
import torch

from src.quant_research import LinearModel, NonLinearModel, DeepModel


class TestLinearModel:
    """Tests for LinearModel architecture."""

    def test_initialization(self):
        """Test model can be initialized with different input sizes."""
        model = LinearModel(3)
        assert model.linear.in_features == 3
        assert model.linear.out_features == 1

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = LinearModel(3)
        x = torch.randn(10, 3)  # batch of 10, 3 features
        output = model(x)
        assert output.shape == (10, 1)

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = LinearModel(3)
        x = torch.randn(10, 3, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestNonLinearModel:
    """Tests for NonLinearModel architecture."""

    def test_initialization(self):
        """Test model can be initialized with custom hidden size."""
        model = NonLinearModel(3, hidden_size=32)
        assert isinstance(model.network, torch.nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = NonLinearModel(3, hidden_size=32)
        x = torch.randn(10, 3)
        output = model(x)
        assert output.shape == (10, 1)


class TestDeepModel:
    """Tests for DeepModel architecture."""

    def test_initialization(self):
        """Test model can be initialized with custom layers."""
        model = DeepModel(3, hidden_sizes=[64, 32, 16], dropout=0.2)
        assert isinstance(model.network, torch.nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = DeepModel(3, hidden_sizes=[32, 16])
        model.eval()  # Turn off dropout for testing
        x = torch.randn(10, 3)
        output = model(x)
        assert output.shape == (10, 1)
