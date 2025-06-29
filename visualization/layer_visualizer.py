import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_network(activations, predicted_label=None):
    """
    ニューラルネットワークの活性化値をネットワーク図として可視化する

    Args:
        activations (List[Tensor]): 各層ごとの出力テンソル（forwardからの出力）
        predicted_label (int or None): 推論されたラベル（タイトルに表示）
    """
    G = nx.DiGraph()
    pos = {}         # ノードの座標情報
    node_colors = [] # 活性化強度
    layer_offset = 0
    node_labels = {}

    max_nodes = max([a.shape[1] for a in activations])  # ノード数の最大値（描画整列用）

    node_index = 0
    layer_node_ids = []

    for layer_idx, layer_act in enumerate(activations):
        layer_acts = layer_act[0].detach().cpu().numpy()
        n_nodes = len(layer_acts)
        ids = []
        for i in range(n_nodes):
            node_id = f"L{layer_idx}_N{i}"
            G.add_node(node_id)
            x = layer_idx * 2
            y = max_nodes / 2 - i
            pos[node_id] = (x, y)
            node_colors.append(layer_acts[i])
            node_labels[node_id] = f"{layer_acts[i]:.2f}"
            ids.append(node_id)
        layer_node_ids.append(ids)

        # ノード間のエッジ（前層との接続）
        if layer_idx > 0:
            for src in layer_node_ids[layer_idx - 1]:
                for dst in ids:
                    G.add_edge(src, dst)

    # 正規化：活性化を [0, 1] に
    node_colors = np.array(node_colors)
    if len(node_colors) > 0 and node_colors.max() > node_colors.min():
        node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())
    else:
        node_colors = np.zeros_like(node_colors)

    # 描画
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=500,
            node_color=node_colors, cmap=plt.cm.viridis,
            arrows=False)

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, font_color='black')
    if predicted_label is not None:
        plt.title(f"推論結果：{predicted_label}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()