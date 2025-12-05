import os
import numpy as np
import torch

from torchvision.utils import save_image
from models.models import Generator
from misc.utils import _init_input, draw_masks


# ============================================================
# 1.  HouseGAN++ 全局初始化（只加载一次，不重复加载）
# ============================================================

# 自动选择 GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[HouseGAN++] Using device:", device)

# ---- 修改为你的真实 checkpoint 路径 ----
CHECKPOINT = r"D:\00-codes\useful_tools\houseganpp\checkpoints\pretrained.pth"

# ---- 创建模型并加载权重 ----
model = Generator()
state = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state, strict=True)
model = model.to(device).eval()

print("[HouseGAN++] Model loaded.")


# ============================================================
# 2.  HouseGAN++ 推理核心函数（真正给前端用）
#     输入：自定义 graph（nodes + edges）
#     输出：生成 PNG 的路径
# ============================================================

def infer_housegan_graph(
    nds_tensor,
    eds_tensor,
    out_path="generated.png",
    refine_steps=10
):
    """
    nds_tensor: Tensor [1, N, T]   节点类型 one-hot（N=节点数，T=房间类型数）
    eds_tensor: Tensor [1, N, N]   邻接矩阵
    """

    # 保证在 batch 维度
    if nds_tensor.dim() == 2:
        nds_tensor = nds_tensor.unsqueeze(0)
    if eds_tensor.dim() == 2:
        eds_tensor = eds_tensor.unsqueeze(0)

    nds = nds_tensor.clone()
    eds = eds_tensor.clone()

    nds = nds.to(device)
    eds = eds.to(device)

    # ========================
    # step 1：initial layout
    # ========================
    state = {"masks": None, "fixed_nodes": []}
    z, masks_in, given_nds, given_eds = _init_input([nds, eds], state)

    with torch.no_grad():
        masks = model(
            z.to(device),
            masks_in.to(device),
            given_nds.to(device),
            given_eds.to(device)
        )
        masks = masks.detach().cpu().numpy()

    # ========================
    # step 2：increment refine
    # ========================
    # 找出所有房间类型
    real_nodes = np.where(nds.cpu() == 1)[-1]
    room_types = sorted(list(set(real_nodes)))

    steps = min(refine_steps, len(room_types))

    for k in range(steps):
        fixed_types = room_types[:k+1]

        fixed_idx = np.concatenate([
            np.where(real_nodes == t)[0]
            for t in fixed_types
        ]) if len(fixed_types) > 0 else np.array([])

        state = {"masks": masks, "fixed_nodes": fixed_idx}

        z, masks_in, given_nds, given_eds = _init_input([nds, eds], state)

        with torch.no_grad():
            masks = model(
                z.to(device),
                masks_in.to(device),
                given_nds.to(device),
                given_eds.to(device)
            )
            masks = masks.detach().cpu().numpy()

    # ========================
    # step 3：convert to image
    # ========================
    floorplan = draw_masks(masks.copy(), real_nodes)
    fp_arr = np.array(floorplan).transpose(2, 0, 1) / 255.0

    fp_tensor = torch.tensor(fp_arr)
    save_image(fp_tensor, out_path, nrow=1, normalize=False)

    return os.path.abspath(out_path)
