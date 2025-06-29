import pytest
import torch
from model.visual_nn import VisualNN
from model.inference import run_inference

def test_run_inference_output():
    model = VisualNN(hidden_sizes=[64])
    dummy_input = torch.randn(1, 28*28)
    label, activations = run_inference(model, dummy_input)
    
    assert isinstance(label, int)
    assert isinstance(activations, list)
    assert len(activations) == 2  # 1 hidden + 1 output layer
    assert all(isinstance(act, torch.Tensor) for act in activations)
def test_run_inference_shape():
    model = VisualNN(hidden_sizes=[64])
    dummy_input = torch.randn(1, 28*28)
    label, activations = run_inference(model, dummy_input)
    
    assert activations[0].shape == (1, 64)  # Hidden layer output
    assert activations[1].shape == (1, 10)  # Output layer logits
def test_run_inference_activation_values():
    model = VisualNN(hidden_sizes=[64])
    dummy_input = torch.randn(1, 28*28)
    label, activations = run_inference(model, dummy_input)
    
    # Check that activations are non-negative after ReLU
    assert (activations[0] >= 0).all()
    assert (activations[1] >= 0).all()  # Output layer logits can be negative
def test_run_inference_invalid_input():
    model = VisualNN(hidden_sizes=[64])
    with pytest.raises(RuntimeError):
        # Input should be of shape (1, 28*28)
        dummy_input = torch.randn(1, 28*27)  # Invalid shape
        run_inference(model, dummy_input)
        
    with pytest.raises(RuntimeError):
        # Input should be a tensor
        dummy_input = "invalid input"
        run_inference(model, dummy_input)