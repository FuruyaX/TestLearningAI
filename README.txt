
# Test
# pytest tests/ --maxfail=1 --disable-warnings -v

# Ready
python -m venv .venv
source .venv/bin/activate  # Windowsでは .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
# MAIN
python app.py
python main.py
# PYTHONPATH=. pytest tests/

project_root/
├── app.py
├── config/
│   └── config.yaml
├── model/
│   ├── __init__.py
│   ├── visual_nn.py
│   └── inference.py
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   ├── index.html
│   └── overview.html
├── utils/
│   ├── __init__.py
│   └── image_processing.py
└── checkpoints/
    └── (モデルファイル名).pt


# Steps
🛠 実装ステップ一覧
1. プロジェクト構造の初期セットアップ
- ディレクトリとモジュールの分割（前回提示した構成に基づく）
- 仮の main.py を置いて、全体の動作テストを可能に

2. ネットワーク構築（visual_nn.py）
- 可変な隠れ層（例：2~3層）を持つ VisualNN クラスを実装
- 各層の活性化値を保持できる forward 関数を設計

3. データ前処理（image_processing.py）
- 手書き画像を 28x28 にリサイズし、グレースケール＆正規化処理
- キャンバス画像（PIL形式）をTensorに変換する関数を提供

4. 推論ユニット（inference.py）
- モデルのロードと推論を行い、ラベルと活性化値を出力
- UIや可視化とスムーズに連携できるようインターフェース設計

5. ユーザー入力UI（input_canvas.py）
- tkinter による手書きキャンバス実装
- マウス操作で数字入力 → 画像保存 → 推論呼び出しまでを統合

6. 可視化機能（layer_visualizer.py）
- 各層のノード・接続を networkx / matplotlib で描画
- ノードの色や大きさを活性化強度で変化させ、視覚的に理解可能に

7. 統合と制御（main.py）
- UI起動、画像取得 → 推論 → 結果表示（数値＋ネットワーク）を制御
- ユーザーの操作に応じた動的更新（数字の書き直しや再推論）も実装

8. コンフィグファイル（config.yaml）
- 隠れ層構成やノード数、学習済みモデルパスなどの設定を外部化
- 将来的なチューニングや教材間の切り替えを容易に

9. （任意）モデル学習スクリプト
- 学習済みモデルがない場合に備え、MNISTでトレーニング可能なスクリプトも準備
