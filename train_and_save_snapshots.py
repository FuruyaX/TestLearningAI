import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, json
from model.visual_nn import VisualNN  # 必要に応じて定義
from utils.image_processing import preprocess_tensor  # 必要なら調整

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_sizes = [128, 64]
model = VisualNN(hidden_sizes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# データセット（MNIST）
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

os.makedirs("static/training_snapshots", exist_ok=True)
def save_snapshot(epoch, model, input_tensor):
    model.eval()
    with torch.no_grad():
        activations = []
        x = input_tensor.clone().view(1, -1)
        for layer in model.layers:
            x = layer(x)
            activations.append(x.clone())
        output = model.output_layer(x)
        activations.append(output)

    # 入力層を先頭に追加
    input_flat = input_tensor.view(-1).cpu()
    activations = [input_flat.unsqueeze(0)] + [a.cpu() for a in activations]

    spacing_x, spacing_y = 2.0, 0.3
    max_per_col = 40
    positions = []
    nodes = []

    for l, layer in enumerate(activations):
        vals = layer[0].tolist()
        layer_pos = []
        for j, val in enumerate(vals):
            x = l * spacing_x + (j // max_per_col) * 0.5
            y = -(j % max_per_col) * spacing_y
            nid = f"L{l}N{j}"
            node = {
                "id": nid,
                "label": nid,
                "x": x,
                "y": y,
                "value": val
            }
            nodes.append(node)
            layer_pos.append(nid)
        positions.append(layer_pos)

    # 重みからエッジ作成
    weight_tensors = [layer.weight.detach().cpu() for layer in model.layers]
    weight_tensors.append(model.output_layer.weight.detach().cpu())
    edges = []

    for l, w in enumerate(weight_tensors):
        if l + 1 >= len(positions): continue
        inputs, outputs = positions[l], positions[l + 1]
        for j in range(w.size(0)):
            for i in range(w.size(1)):
                weight = w[j][i].item()
                edges.append({
                    "source": inputs[i],
                    "target": outputs[j],
                    "weight": weight
                })

    # 保存
    snapshot = {"epoch": epoch, "nodes": nodes, "edges": edges}
    out_path = f"static/training_snapshots/epoch_{epoch}.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"✅ Saved snapshot for epoch {epoch} → {out_path}")
    epochs = 10  # 必要に応じて
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            input_flat = data.view(data.size(0), -1)
            outputs = model(input_flat)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"📘 Epoch {epoch}: Loss = {running_loss:.4f}")

        # 代表サンプルを使って snapshot を保存（ここでは最初のバッチ先頭を使用）
        save_input = data[0].cpu()
        save_snapshot(epoch, model, save_input)