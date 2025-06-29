import pytest
from ui.input_canvas import InputCanvas
import tkinter as tk

def test_canvas_export_image():
    root = tk.Tk()
    canvas = InputCanvas(master=root)
    image = canvas.export_image()
    assert image.size == (280, 280)  # 仕様に合わせて
    root.destroy()