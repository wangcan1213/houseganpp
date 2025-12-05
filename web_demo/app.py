import os
import sys
import io
import base64
import random

from flask import Flask, render_template, request, jsonify

import numpy as np
import torch

# ========== 让 Python 找到 houseganpp 的源码 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOUSEGAN_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, HOUSEGAN_ROOT)

from models.models import Generator
from misc.utils import _init_input, ROOM_CLASS, ID_COLOR, draw_masks  # :contentReference[oaicite:1]{index=1}

# ========== Flask 基本配置 ==========
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(HOUSEGAN_ROOT, "checkpoints", "pretrained.pth")

# ========== 加载预训练 Generator ==========
model = Generator()
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()

# ========= 从 ROOM_CLASS / ID_COLOR 构建前端用的房型列表 =========
# ROOM_CLASS: {"living_room":1, "kitchen":2, ...}
# ID_COLOR:   {1:"#...", 2:"#...", ...}  房间类型颜色

# 只展示主要房间类型，去掉门/unknown
EXCLUDE_LABEL_IDS = {6, 15, 16, 17}  # front door / unknown / interior_door
# EXCLUDE_LABEL_IDS = {16}  # unknown 
ROOM_TYPE_META = []

for name, label_id in ROOM_CLASS.items():
    if label_id not in ID_COLOR:
        continue
    if label_id in EXCLUDE_LABEL_IDS:
        continue
    color = ID_COLOR[label_id]
    display_label = name.replace("_", " ")  # "living_room" -> "living room"
    ROOM_TYPE_META.append(
        {
            "key": name,              # 传给后端的 key
            "label": display_label,   # 节点上显示的名字
            "label_id": int(label_id),
            "color": color,
        }
    )

# 18 通道：原代码 one_hot_embedding(..., num_classes=19)[:, 1:]
NUM_NODE_CHANNELS = 18  # 固定 18 维特征
MAX_LABEL_ID = max(ID_COLOR.keys())


# ========= 工具函数：把交互图编码成 HouseGAN++ 用的 nds / eds =========
def _infer(graph, model, prev_state=None):
    
    # configure input to the network
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
    # run inference model
    with torch.no_grad():
        masks = model(z.to('cuda'), given_masks_in.to('cuda'), given_nds.to('cuda'), given_eds.to('cuda'))
        masks = masks.detach().cpu().numpy()
    return masks


def encode_graph_to_tensors(nodes, edges):
    """
    nodes: [{"id": 0, "label": "living_room"}, ...]
    edges: [[src_id, dst_id], ...]

    新增功能：
    - 每条 A-B edge 增加 interior_door(17) 节点 M，并添加 A-M, M-B
    - 增加 front_door(16) 节点 F，并与第一个 living_room 连接
    """

    if not nodes:
        raise ValueError("Graph has no nodes")

    # 建 node_id -> index 映射
    id_to_idx = {n["id"]: idx for idx, n in enumerate(nodes)}

    # ----------------------------
    # ① 先复制原始 nodes，稍后追加门节点
    # ----------------------------
    expanded_nodes = nodes.copy()
    next_new_idx = len(expanded_nodes)

    # 记录 living room 的 index（第一个）
    living_idx = None
    for i, n in enumerate(nodes):
        if n["label"] == "living_room":
            living_idx = i
            break

    if living_idx is None:
        raise ValueError("图中没有 living_room，无法连接 front_door")

    # -----------------------------------
    # ② 先构建 adjacency 用于保持原有 A-B 信息
    # -----------------------------------
    adj = set()
    for e in edges:
        if len(e) != 2:
            continue
        s, t = e
        if s not in id_to_idx or t not in id_to_idx:
            continue
        i, j = id_to_idx[s], id_to_idx[t]
        if i != j:
            a, b = sorted((i, j))
            adj.add((a, b))

    # ==========================================================
    # ③ 为每条原始 edge (i,j) 添加 interior_door (type=17)
    # ==========================================================
    interior_nodes = []   # 保存新门节点的 index
    new_edges_extra = []  # A-M, M-B

    for (i, j) in adj:
        # 新建门节点
        door_node = {
            "id": f"intdoor_{i}_{j}",   # 给一个独特的 id
            "label": "interior_door"
        }
        expanded_nodes.append(door_node)

        M = next_new_idx
        next_new_idx += 1
        interior_nodes.append(M)

        # 添加 A-M, M-B
        new_edges_extra.append((i, M))
        new_edges_extra.append((M, j))

    # ==========================================================
    # ④ 创建 front_door 节点 (type=16)，连接 living room
    # ==========================================================
    front_door_node = {
        "id": "frontdoor",
        "label": "front door"
    }
    expanded_nodes.append(front_door_node)
    F = next_new_idx
    next_new_idx += 1

    new_edges_extra.append((F, living_idx))

    # ==========================================================
    # ⑤ 现在 expanded_nodes 包含：
    #   原节点 + interior doors + front door
    # 我们构建新的 adjacency（双向）
    # ==========================================================
    N = len(expanded_nodes)
    adj2 = set(adj)  # 原有边
    for (a, b) in new_edges_extra:
        x, y = sorted((a, b))
        adj2.add((x, y))

    # ==========================================================
    # ⑥ label_ids (1..17) & one-hot ([:,1:])
    # ==========================================================
    label_ids = []
    for n in expanded_nodes:
        type_key = n["label"]
        if type_key not in ROOM_CLASS:
            raise ValueError(f"未知房型 '{type_key}'")
        label_ids.append(ROOM_CLASS[type_key])

    label_ids = np.array(label_ids, dtype=np.int64)

    # one-hot (N,18)
    num_classes = 19
    eye = np.eye(num_classes, dtype=np.float32)
    nds = eye[label_ids][:, 1:]   # 丢掉第 0 类

    # real_nodes (0..16)
    real_nodes = label_ids - 1

    # ==========================================================
    # ⑦ 构建 triples (i, ±1, j)
    # ==========================================================
    triples = []
    for i in range(N):
        for j in range(i + 1, N):
            v = 1 if (i, j) in adj2 else -1
            triples.append([i, v, j])

    eds = np.array(triples, dtype=np.int64)
    nds_t = torch.from_numpy(nds)

    return nds_t, eds, real_nodes



# ========= 跑一次 HouseGAN++ =========
def run_housegan_once(nodes, edges, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    nds_t, eds_np, real_nodes = encode_graph_to_tensors(nodes, edges)
    graph = [nds_t, eds_np]

    _types = sorted(list(set(real_nodes)))
    selected_types = [_types[:k+1] for k in range(10)]

    state = {'masks': None, 'fixed_nodes': []}
    masks = _infer(graph, model, state)

    for _iter, _types in enumerate(selected_types):
        _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
            if len(_types) > 0 else np.array([]) 
        state = {'masks': masks, 'fixed_nodes': _fixed_nds}
        masks = _infer(graph, model, state)

    pil_im = draw_masks(masks.copy(), real_nodes)
    return pil_im
    


def run_housegan_multisample(nodes, edges, num_samples=4):
    """
    多次随机采样，返回多张 base64 编码的图片
    """
    images_b64 = []
    for _ in range(num_samples):
        seed = random.randint(0, 1_000_000_000)
        im = run_housegan_once(nodes, edges, seed=seed)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images_b64.append(b64)
    return images_b64


# ========= Flask 路由 =========
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/room_types", methods=["GET"])
def room_types():
    """
    给前端：房型列表 + 颜色信息
    """
    return jsonify({"room_types": ROOM_TYPE_META})


@app.route("/generate", methods=["POST"])
def generate():
    """
    前端 POST:
    {
      "nodes": [{"id":0, "label":"living_room"}, ...],
      "edges": [[0,1],[1,2],...],
      "num_samples": 4
    }

    返回:
    { "images": ["base64...", ...] }
    """
    data = request.get_json()
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    num_samples = int(data.get("num_samples", 4))

    if not nodes:
        return jsonify({"error": "图中没有任何节点"}), 400

    try:
        images_b64 = run_housegan_multisample(nodes, edges, num_samples=num_samples)
    except Exception as e:
        # 出错时，把错误消息返回前端，方便你调试
        return jsonify({"error": str(e)}), 500

    return jsonify({"images": images_b64})


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
