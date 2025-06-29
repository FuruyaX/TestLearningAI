import pytest
import torch
from model.visual_nn import VisualNN

def test_visual_nn_forward_shape():
    model = VisualNN(hidden_sizes=[64, 32])
    dummy_input = torch.randn(1, 28*28)
    output, activations = model(dummy_input)

    assert output.shape == (1, 10)
    assert len(activations) == 3  # 2 hidden + 1 output layer
    assert all(isinstance(act, torch.Tensor) for act in activations)