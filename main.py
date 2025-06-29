# main.py
import tkinter as tk
from ui.input_canvas import InputCanvas
from model.visual_nn import VisualNN
from model.inference import run_inference
from visualization.layer_visualizer import visualize_network
from utils.image_processing import preprocess_canvas_image
import torch
import yaml 

# 設定読み込み
def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# モデルの初期化
def initialize_model(config):
    hidden_sizes = config['model']['hidden_sizes']
    model_path = config['model']['checkpoint']
    model = VisualNN(hidden_sizes=hidden_sizes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 推論＆可視化処理（ボタンから呼ばれる）
def process_input(canvas, model):
    pil_img = canvas.export_image()  # CanvasからPIL.Imageを取得
    input_tensor = preprocess_canvas_image(pil_img)
    pred_label, activations = run_inference(model, input_tensor)
    visualize_network(activations, pred_label)

def run_app():
    config = load_config()
    model = initialize_model(config)

    root = tk.Tk()
    root.title("手書き数字認識 - NN可視化教材")
    canvas = InputCanvas(master=root)

    button = tk.Button(root, text="認識・可視化", command=lambda: process_input(canvas, model))
    button.pack(pady=10)

    canvas.pack()
    root.mainloop()

if __name__ == '__main__':
    run_app()