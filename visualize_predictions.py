import os
import sys
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# -------------------------
# 0. 确保 dust3r 可导入
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")
if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel


def move_batch_to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


def build_views_from_batch(batch):
    B = batch["img"].shape[0]

    view1 = {
        "img": batch["img"],
        "instance": [f"vis_{i}" for i in range(B)],
    }
    view2 = {
        "img": batch["img"],
        "instance": [f"vis_{i}" for i in range(B)],
    }
    return view1, view2


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = r"E:\Study\code\LightRecon3D\data\Structured3D"
    weights_path = r"checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ckpt_path = r"checkpoints\lightrecon3d\best.pth"

    # 1. 数据
    val_dataset = Structured3DDataset(
        root_dir=root_dir,
        split="val",
        train_ratio=0.9,
        image_size=(512, 512)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # 2. 模型
    dust3r_backbone = build_dust3r_backbone(weights_path, device=device)
    model = LightReconModel(dust3r_backbone, hidden_dim=768).to(device)

    # 3. 加载训练好的权重
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # 4. 拿一个 batch
    batch = next(iter(val_loader))
    batch = move_batch_to_device(batch, device)
    view1, view2 = build_views_from_batch(batch)

    # 5. 前向
    res1, res2 = model(view1, view2)

    # 6. 取出可视化内容
    img = batch["img"][0].permute(1, 2, 0).cpu().numpy()               # [H,W,3]
    gt_line = batch["gt_line"][0, 0].cpu().numpy()                     # [H,W]
    gt_plane = batch["gt_plane"][0].cpu().numpy()                      # [H,W]

    pred_line_prob = torch.sigmoid(res1["pred_line"][0, 0]).cpu().numpy()   # [H,W]
    pred_plane_cls = torch.argmax(res1["pred_plane"][0], dim=0).cpu().numpy()  # [H,W]

    # 7. 画图
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title("Input RGB")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("GT Line")
    plt.imshow(gt_line, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Pred Line (Prob)")
    plt.imshow(pred_line_prob, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("GT Plane")
    plt.imshow(gt_plane, cmap="jet")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Pred Plane (Argmax)")
    plt.imshow(pred_plane_cls, cmap="jet")
    plt.axis("off")

    # 第 6 张图留给你后面扩展
    plt.subplot(2, 3, 6)
    plt.title("Pred Line Binary")
    plt.imshow(pred_line_prob > 0.5, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()