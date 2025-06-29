import pytest
from PIL import Image
import numpy as np
from utils.image_processing import preprocess_canvas_image

def test_preprocess_output_shape():
    dummy_img = Image.new('L', (280, 280), color=0)
    tensor = preprocess_canvas_image(dummy_img)
    assert tensor.shape == (1, 28*28)