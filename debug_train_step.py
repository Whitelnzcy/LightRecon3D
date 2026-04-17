import torch
from torch.utils.data import DataLoader

from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from loss.losses import total_loss_fn
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)


def move_batch_to_device(batch, device):
    """
    把 batch 字典里的 tensor 全部搬到 device 上。
    """
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
        "instance": [f"debug_{i}" for i in range(B)],
    }
    view2 = {
        "img": batch["img"],
        "instance": [f"debug_{i}" for i in range(B)],
    }
    return view1, view2


def main():
    # -------------------------
    # 1. 基础配置
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = r"checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    root_dir = r"F:\Structured3D_Data"

    print(f"Using device: {device}")

    # -------------------------
    # 2. 数据
    # -------------------------
    dataset = Structured3DDataset(
        root_dir=root_dir,
        scene_list=["scene_00000"],   # 当前先只用一个 scene 做闭环调试
        image_size=(512, 512)
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)}")

    # -------------------------
    # 3. backbone + model
    # -------------------------
    dust3r_backbone = build_dust3r_backbone(weights_path, device=device)
    model = LightReconModel(dust3r_backbone).to(device)
    model.train()

    # -------------------------
    # 4. optimizer
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # -------------------------
    # 5. 取一个 batch
    # -------------------------
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    print("Batch keys:", batch.keys())
    print("img shape:", batch["img"].shape)
    print("gt_line shape:", batch["gt_line"].shape)
    print("gt_plane shape:", batch["gt_plane"].shape)

    # -------------------------
    # 6. 构造 view1 / view2
    # -------------------------
    view1, view2 = build_views_from_batch(batch)

    # -------------------------
    # 7. forward
    # -------------------------
    optimizer.zero_grad()

    res1, res2 = model(view1, view2)

    print("Model forward success.")
    print("pred_line shape:", res1["pred_line"].shape)
    print("pred_plane shape:", res1["pred_plane"].shape)

    # -------------------------
    # 8. loss
    # -------------------------
    loss_total, loss_dict = total_loss_fn(
        res1,
        batch,
        line_weight=1.0,
        plane_weight=1.0,
        line_pos_weight=10.0,
        line_dice_weight=1.0
    )

    print("Loss computed.")
    print(loss_dict)

    # -------------------------
    # 9. backward
    # -------------------------
    loss_total.backward()
    optimizer.step()

    print("Backward + optimizer step success.")


if __name__ == "__main__":
    main()