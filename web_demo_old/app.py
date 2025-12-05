from flask import Flask, request, jsonify
import torch
import numpy as np
import os
from infer_housegan import infer_housegan_graph

app = Flask(__name__)
os.makedirs("static/generated", exist_ok=True)


# ============================================================
# HouseGAN++ 官方房间分类（来自 misc/utils.py 的 ID_COLOR）
# ============================================================

ROOM_TYPES = [
    "Living",      # 0
    "Dining",      # 1
    "Kitchen",     # 2
    "Bedroom",     # 3
    "Bathroom",    # 4
    "Balcony",     # 5
    "Entrance",    # 6
    "Storage",     # 7
    "Corridor",    # 8
    "Other"        # 9
]


# ============================================================
# 前端 graph 转换成 HouseGAN++ 所需的 nds / eds
# ============================================================

def encode_graph_to_housegan(nodes, edges):
    """
    nodes: [{id:0, type:"Living", x:..., y:...}, ...]
    edges: [{from:0, to:1}, ...]
    """

    N = len(nodes)
    T = len(ROOM_TYPES)

    nds = torch.zeros((1, N, T), dtype=torch.float32)
    eds = torch.zeros((1, N, N), dtype=torch.float32)

    # ---- 节点 one-hot ----
    for i, node in enumerate(nodes):
        room_type = node["type"]
        if room_type in ROOM_TYPES:
            idx = ROOM_TYPES.index(room_type)
        else:
            idx = ROOM_TYPES.index("Other")
        nds[0, i, idx] = 1

    # ---- 邻接矩阵 ----
    for e in edges:
        i = e["from"]
        j = e["to"]
        eds[0, i, j] = 1
        eds[0, j, i] = 1

    return nds, eds


# ============================================================
# /generate 路由：真正调用 HouseGAN++
# ============================================================

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    nodes = data["nodes"]
    edges = data["edges"]

    # 1. 图结构编码
    nds, eds = encode_graph_to_housegan(nodes, edges)

    # 2. 调用推理（HouseGAN++）
    outfile = "static/generated/floorplan.png"
    abs_path = infer_housegan_graph(nds, eds, outfile)

    # 转成可以前端显示的 URL
    return jsonify({"image": "/" + abs_path.replace("\\", "/")})


@app.route("/")
def index():
    return "HouseGAN++ Web Demo Running"


if __name__ == "__main__":
    app.run(debug=True)
