import torch
from model.visual_nn import VisualNN

def run_inference(model: VisualNN, input_tensor: torch.Tensor):
    """
    1枚の入力テンソルに対して推論を行い、予測ラベルと活性化値を返す

    Args:
        model (VisualNN): 学習済みのニューラルネットワークモデル
        input_tensor (torch.Tensor): (1, 784) 形式の入力画像テンソル

    Returns:
        pred_label (int): 予測された数字ラベル（0〜9）
        activations (List[torch.Tensor]): 各層の出力（可視化用）
    """
    model.eval()
    with torch.no_grad():
        output, activations = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()
        return pred_label, activations