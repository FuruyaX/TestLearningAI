import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.visual_nn import VisualNN
import yaml
import os

# 設定ファイルの読み込み
def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    hidden_sizes = config['model']['hidden_sizes']
    model_path = config['model']['checkpoint']
    input_size = 28 * 28
    output_size = 10

    # モデル構築
    model = VisualNN(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)

    # データセット準備（MNIST）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # フラット化
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 最適化と損失関数
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 学習ループ
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - Loss: {total_loss:.4f}")

    # 保存先ディレクトリがなければ作成
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"学習済みモデルを保存しました: {model_path}")

if __name__ == '__main__':
    train_model()