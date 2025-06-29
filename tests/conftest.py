import pytest
import torch
from model.visual_nn import VisualNN

@pytest.fixture
def simple_model():
    model = VisualNN(hidden_sizes=[64, 32])
    model.eval()
    return model

@pytest.fixture
def dummy_input():
    return torch.randn(1, 28*28)