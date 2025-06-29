from flask import Flask, render_template, request, jsonify
import torch
from model.visual_nn import VisualNN
from model.inference import run_inference
from utils.image_processing import preprocess_canvas_image
from PIL import Image
import io
import base64
import yaml
import traceback
import json
import os

app = Flask(__name__)

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
model = VisualNN(hidden_sizes=config["model"]["hidden_sizes"])
model.load_state_dict(torch.load(config["model"]["checkpoint"], map_location='cpu'))
model.eval()
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/overview")
def overview():
    return render_template("overview.html")

@app.route("/training_viewer")
def training_viewer():
    return render_template("training_viewer.html")

@app.route("/training_snapshot/<int:epoch>")
def training_snapshot(epoch):
    try:
        with open(f"static/training_snapshots/epoch_{epoch}.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"読み込み失敗: {e}"}), 404

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]
        _, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        input_tensor = preprocess_canvas_image(image)

        pred_label, activations = run_inference(model, input_tensor)
        input_flat = input_tensor.view(-1).detach().cpu()
        activations = [input_flat.unsqueeze(0)] + activations

        spacing_x, spacing_y = 2.0, 0.3
        max_per_column = 40
        positions = []
        nodes = []
        node_map = {}

        for l, layer in enumerate(activations):
            vals = layer[0].detach().cpu().tolist()
            layer_pos = []
            for j, val in enumerate(vals):
                x = l * spacing_x + (j // max_per_column) * 0.5
                y = -(j % max_per_column) * spacing_y
                nid = f"L{l}N{j}"
                node = {
                    "id": nid,
                    "label": nid,
                    "x": x,
                    "y": y,
                    "value": val
                }
                nodes.append(node)
                node_map[nid] = node
                layer_pos.append(nid)
            positions.append(layer_pos)

        # 重みからエッジ生成
        weight_tensors = [
            layer.weight.detach().cpu() for layer in model.layers if hasattr(layer, "weight")
        ]
        weight_tensors.append(model.output_layer.weight.detach().cpu())

        edges = []
        edge_threshold = 0.02
        for l, w in enumerate(weight_tensors):
            if l + 1 >= len(positions): continue
            inputs, outputs = positions[l], positions[l + 1]
            if w.size(0) != len(outputs) or w.size(1) != len(inputs):
                print(f"⚠️ shape mismatch: w={w.shape}, inputs={len(inputs)}, outputs={len(outputs)}")
                continue
            for j in range(w.size(0)):
                for i in range(w.size(1)):
                    weight = w[j][i].item()
                    if abs(weight) >= edge_threshold:
                        edges.append({
                            "source": inputs[i],
                            "target": outputs[j],
                            "weight": weight
                        })

        return jsonify({
            "prediction": pred_label,
            "nodes": nodes,
            "edges": edges
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)