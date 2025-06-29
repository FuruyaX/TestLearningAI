from PIL import Image, ImageOps
import numpy as np
import torch

def preprocess_canvas_image(image: Image.Image) -> torch.Tensor:
    """
    手書きキャンバス画像をPyTorchテンソルに変換（MNIST形式に整形）

    Args:
        image (PIL.Image): モノクロ（白背景、黒インク）の手書き画像

    Returns:
        torch.Tensor: (1, 784) の正規化済みテンソル
    """
    # 1. 縮小（28x28）＆グレースケール
    # PILのバージョンによってはResamplingフィルタが異なるため、try-exceptで対応
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS  # 古いバージョン向け
    img_resized = image.resize((28, 28), resample_filter)
    
    # 2. 色反転（白地に黒インク → MNIST準拠の黒地に白数字）
    img_inverted = ImageOps.invert(img_resized)

    # 3. numpy配列へ変換＆正規化（[0,1] float）
    img_array = np.array(img_inverted).astype(np.float32) / 255.0

    # 4. フラット化＆テンソル化
    tensor = torch.from_numpy(img_array).view(1, -1)

    return tensor