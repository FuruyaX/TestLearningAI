import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualNN(nn.Module):
    """
    多層ニューラルネットワーク（視覚化・推論用）

    Args:
        input_size (int): 入力層ノード数（例: 28*28）
        hidden_sizes (List[int]): 隠れ層ノード数のリスト（例: [128, 64]）
        output_size (int): 出力層ノード数（例: 10）
    """

    def __init__(self, input_size=28*28, hidden_sizes=[128, 64], output_size=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # ModuleListで柔軟なレイヤー構成を定義
        self.layers = nn.ModuleList()

        # 入力層 → 最初の隠れ層
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # 隠れ層間
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # 出力層
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        """
        順伝播処理（各層出力を返す）

        Args:
            x (Tensor): (batch_size, input_size)

        Returns:
            output (Tensor): 最終出力（ロジット）
            activations (List[Tensor]): 各層の中間出力(ReLU後)+出力層
        """
        activations = []
        for layer in self.layers:
            x = F.relu(layer(x))
            activations.append(x)  # 中間層の出力（可視化用）

        output = self.output_layer(x)
        activations.append(output)  # 出力層のロジットも含む
        return output, activations