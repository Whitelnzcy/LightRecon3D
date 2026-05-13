import os
import sys
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset


# ============================================================
# 0. Path setup
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DUST3R_REPO_ROOT = os.path.join(PROJECT_ROOT, "dust3r")

if DUST3R_REPO_ROOT not in sys.path:
    sys.path.insert(0, DUST3R_REPO_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from dataloaders.s3d_dataset import Structured3DDataset
from models.build_backbone import build_dust3r_backbone
from models.lightrecon_net import LightReconModel
from loss.losses import compute_losses


# ============================================================
# 1. Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("LightRecon3D Training")

    # -------------------------
    # paths
    # -------------------------
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/zhucy23u/datasets/Structured3D",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default="/data/zhucy23u/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/zhucy23u/checkpoints/lightrecon/plane_embedding_debug",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a LightRecon3D checkpoint.",
    )

    # -------------------------
    # dataset
    # -------------------------
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--small_train_size", type=int, default=-1)
    parser.add_argument("--small_val_size", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--input_mode",
        type=str,
        default="pair",
        choices=["pair", "single"],
        help=(
            "Use real Structured3D two-view pairs for DUSt3R by default. "
            "Set to 'single' to keep the old pseudo-pair behavior."
        ),
    )
    parser.add_argument(
        "--pair_strategy",
        type=str,
        default="adjacent",
        choices=["adjacent", "all"],
        help="How to form two-view samples inside each Structured3D space.",
    )
    parser.add_argument(
        "--pair_max_view_id_gap",
        type=int,
        default=0,
        help=(
            "Optional max numeric render-id gap for pair filtering. "
            "Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--print_pair_examples",
        type=int,
        default=0,
        help="Print the first N dataset pairs for sanity checking.",
    )

    # -------------------------
    # training
    # -------------------------
    parser.add_argument("--num_epochs", "--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Set <=0 to disable.",
    )

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--run_val", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # -------------------------
    # model
    # -------------------------
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--plane_embed_dim", type=int, default=16)
    parser.add_argument("--freeze_dust3r", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")

    # -------------------------
    # loss weights
    # -------------------------
    parser.add_argument("--line_weight", type=float, default=1.0)
    parser.add_argument("--plane_weight", type=float, default=1.0)

    parser.add_argument(
        "--geo_weight",
        type=float,
        default=0.0,
        help="Weight of 3D coplanarity loss. Keep 0.0 during plane embedding stage.",
    )

    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=None,
        help=(
            "Optional frozen teacher checkpoint for pointmap anchor. "
            "Usually set to the stable baseline checkpoint before geo training."
        ),
    )

    parser.add_argument(
        "--point_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for pointmap anchor loss. "
            "If > 0, teacher_ckpt should be provided."
        ),
    )

    parser.add_argument(
        "--point_anchor_beta",
        type=float,
        default=0.05,
        help="SmoothL1 beta for pointmap anchor loss.",
    )

    # -------------------------
    # line loss params
    # -------------------------
    parser.add_argument("--line_pos_weight", type=float, default=None)
    parser.add_argument("--line_bce_weight", type=float, default=1.0)
    parser.add_argument("--line_dice_weight", type=float, default=1.0)
    parser.add_argument(
        "--line_target_dilate",
        type=int,
        default=0,
        help=(
            "Dilate gt_line by this pixel radius before BCE+Dice. "
            "Use >0 to train a boundary-risk band instead of a 1-pixel line."
        ),
    )

    # -------------------------
    # plane embedding loss params
    # -------------------------
    parser.add_argument("--plane_min_pixels", type=int, default=64)
    parser.add_argument("--plane_max_pixels_per_plane", type=int, default=2048)
    parser.add_argument("--plane_max_planes_per_image", type=int, default=12)

    parser.add_argument("--plane_delta_var", type=float, default=0.5)
    parser.add_argument("--plane_delta_dist", type=float, default=1.5)
    parser.add_argument("--plane_var_weight", type=float, default=1.0)
    parser.add_argument("--plane_dist_weight", type=float, default=1.0)
    parser.add_argument("--plane_reg_weight", type=float, default=0.001)

    # -------------------------
    # coplanarity loss params
    # only used when geo_weight > 0
    # -------------------------
    parser.add_argument("--coplanarity_min_points", type=int, default=128)
    parser.add_argument("--coplanarity_max_points_per_plane", type=int, default=2048)
    parser.add_argument("--coplanarity_max_planes_per_image", type=int, default=8)

    parser.add_argument(
        "--coplanarity_normalize",
        action="store_true",
        help="Use normalized smallest eigenvalue for coplanarity loss.",
    )

    # -------------------------
    # swanlab
    # -------------------------
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="LightRecon3D")
    parser.add_argument("--swanlab_run_name", type=str, default=None)
    parser.add_argument(
        "--swanlab_mode",
        type=str,
        default="cloud",
        choices=["cloud", "local", "disabled"],
    )
    parser.add_argument(
        "--swanlab_logdir",
        type=str,
        default="/data/zhucy23u/logs/swanlab",
    )

    return parser.parse_args()


# ============================================================
# 2. Utility functions
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def move_batch_to_device(batch, device):
    out = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value

    return out


def build_views_from_batch(batch, prefix="train"):
    """
    Build DUSt3R-style view dictionaries from a dataloader batch.

    New paired batches contain img1/img2. Older single-view batches contain
    only img; for those, keep the previous pseudo-pair behavior.
    """
    img1 = batch.get("img1", batch["img"])
    img2 = batch.get("img2", batch["img"])
    batch_size = img1.shape[0]
    true_shape1 = torch.tensor(
        img1.shape[-2:],
        device=img1.device,
        dtype=torch.long,
    )[None].repeat(batch_size, 1)
    true_shape2 = torch.tensor(
        img2.shape[-2:],
        device=img2.device,
        dtype=torch.long,
    )[None].repeat(batch_size, 1)

    view1 = {
        "img": img1,
        "true_shape": true_shape1,
        "instance": [f"{prefix}_{i}_view1" for i in range(batch_size)],
    }

    view2 = {
        "img": img2,
        "true_shape": true_shape2,
        "instance": [f"{prefix}_{i}_view2" for i in range(batch_size)],
    }

    return view1, view2


def select_view_targets(batch, view_idx):
    """
    Return a batch-like target dict for one supervised view.

    View-specific labels are used when present. This keeps compute_losses()
    unchanged and preserves compatibility with older single-view batches.
    """
    if view_idx == 1:
        line_key = "gt_line1"
        plane_key = "gt_plane1"
        img_key = "img1"
    elif view_idx == 2:
        line_key = "gt_line2"
        plane_key = "gt_plane2"
        img_key = "img2"
    else:
        raise ValueError(f"view_idx must be 1 or 2, got {view_idx}")

    target_batch = dict(batch)
    target_batch["img"] = batch.get(img_key, batch["img"])
    target_batch["gt_line"] = batch.get(line_key, batch["gt_line"])
    target_batch["gt_plane"] = batch.get(plane_key, batch["gt_plane"])

    return target_batch


def make_subset(dataset, size):
    if size is None or size <= 0:
        return dataset

    size = min(size, len(dataset))
    indices = list(range(size))

    return Subset(dataset, indices)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")


def freeze_dust3r_encoder(model):
    """
    Freeze DUSt3R encoder-related parameters.

    Encoder names in DUSt3R/CroCo are usually:
    - patch_embed
    - enc_blocks
    - enc_norm

    Decoder and added heads remain trainable.
    """
    encoder_prefixes = (
        "patch_embed",
        "enc_blocks",
        "enc_norm",
    )

    frozen = 0
    total = 0

    for name, param in model.backbone.named_parameters():
        total += param.numel()

        if name.startswith(encoder_prefixes):
            param.requires_grad = False
            frozen += param.numel()

    print(f"Frozen DUSt3R encoder params: {frozen:,} / {total:,}")


def freeze_dust3r_backbone(model):
    frozen = 0
    total = 0

    for param in model.backbone.parameters():
        total += param.numel()
        param.requires_grad = False
        frozen += param.numel()

    print(f"Frozen full DUSt3R backbone params: {frozen:,} / {total:,}")


def init_swanlab(args):
    if not args.use_swanlab:
        return None

    import swanlab

    os.makedirs(args.swanlab_logdir, exist_ok=True)

    run = swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_run_name,
        mode=args.swanlab_mode,
        logdir=args.swanlab_logdir,
        config=vars(args),
        description="LightRecon3D training: line supervision + plane embedding supervision.",
    )

    return run


def swanlab_log(run, metrics, step=None):
    if run is None:
        return

    clean_metrics = {}

    for key, value in metrics.items():
        if value is None:
            continue

        try:
            clean_metrics[key] = to_float(value)
        except Exception:
            continue

    if not clean_metrics:
        return

    try:
        run.log(clean_metrics, step=step)
    except AttributeError:
        import swanlab
        swanlab.log(clean_metrics, step=step)


def finish_swanlab(run):
    if run is None:
        return

    try:
        run.finish()
    except AttributeError:
        import swanlab
        swanlab.finish()


def save_checkpoint(
    save_dir,
    filename,
    model,
    optimizer,
    epoch,
    args,
    best_val=None,
):
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "args": vars(args),
        "best_val": best_val,
    }

    torch.save(ckpt, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    print(f"Loading checkpoint from: {path}")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[Resume] Missing keys: {missing[:10]}")
        if len(missing) > 10:
            print(f"... and {len(missing) - 10} more missing keys")

    if unexpected:
        print(f"[Resume] Unexpected keys: {unexpected[:10]}")
        if len(unexpected) > 10:
            print(f"... and {len(unexpected) - 10} more unexpected keys")

    if optimizer is not None and isinstance(ckpt, dict):
        opt_state = ckpt.get("optimizer", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    start_epoch = 1
    best_val = None

    if isinstance(ckpt, dict):
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = ckpt.get("best_val", None)

    print(f"Resume start_epoch = {start_epoch}")
    print(f"Resume best_val    = {best_val}")

    return start_epoch, best_val


def load_teacher_model(args, device):
    """
    Load a frozen teacher model from args.teacher_ckpt.

    The teacher should usually be the stable baseline checkpoint before geo training.
    """
    if args.teacher_ckpt is None or args.point_anchor_weight <= 0:
        return None

    print("=" * 80)
    print("Loading frozen teacher model for pointmap anchor")
    print("=" * 80)
    print(f"teacher_ckpt       : {args.teacher_ckpt}")
    print(f"point_anchor_weight: {args.point_anchor_weight}")
    print(f"point_anchor_beta  : {args.point_anchor_beta}")
    print("=" * 80)

    teacher_backbone = build_dust3r_backbone(args.weights_path, device=device)

    teacher_model = LightReconModel(
        dust3r_backbone=teacher_backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)

    try:
        ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.teacher_ckpt, map_location=device)

    state_dict = ckpt.get("model", ckpt)

    missing, unexpected = teacher_model.load_state_dict(state_dict, strict=False)

    print(f"[Teacher] missing keys   : {len(missing)}")
    print(f"[Teacher] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[Teacher] first missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("[Teacher] first unexpected keys:", unexpected[:10])

    teacher_model.eval()

    for p in teacher_model.parameters():
        p.requires_grad_(False)

    return teacher_model


def compute_loss_wrapper(res, batch, args):
    """
    Only place that calls compute_losses.

    This avoids train / val using different loss APIs.
    """
    return compute_losses(
        res=res,
        batch=batch,

        line_weight=args.line_weight,
        plane_weight=args.plane_weight,
        geo_weight=args.geo_weight,

        line_pos_weight=args.line_pos_weight,
        line_bce_weight=args.line_bce_weight,
        line_dice_weight=args.line_dice_weight,
        line_target_dilate=args.line_target_dilate,

        plane_min_pixels=args.plane_min_pixels,
        plane_max_pixels_per_plane=args.plane_max_pixels_per_plane,
        plane_max_planes_per_image=args.plane_max_planes_per_image,

        plane_delta_var=args.plane_delta_var,
        plane_delta_dist=args.plane_delta_dist,
        plane_var_weight=args.plane_var_weight,
        plane_dist_weight=args.plane_dist_weight,
        plane_reg_weight=args.plane_reg_weight,

        coplanarity_min_points=args.coplanarity_min_points,
        coplanarity_max_points_per_plane=args.coplanarity_max_points_per_plane,
        coplanarity_max_planes_per_image=args.coplanarity_max_planes_per_image,
        coplanarity_normalize=args.coplanarity_normalize,
    )


def average_loss_dicts(loss_dicts):
    """
    Average scalar logging dictionaries from multiple supervised outputs.
    """
    if not loss_dicts:
        return {}

    keys = set()
    for loss_dict in loss_dicts:
        keys.update(loss_dict.keys())

    averaged = {}
    for key in keys:
        values = []
        for loss_dict in loss_dicts:
            if key in loss_dict:
                values.append(to_float(loss_dict[key]))

        if values:
            averaged[key] = sum(values) / len(values)

    return averaged


def compute_two_view_loss(res1, res2, batch, args):
    """
    Supervise both DUSt3R outputs.

    Paired batches use view-specific targets. Single-view batches fall back to
    shared pseudo-pair targets through select_view_targets().
    """
    batch1 = select_view_targets(batch, view_idx=1)
    batch2 = select_view_targets(batch, view_idx=2)

    loss1, loss_dict1 = compute_loss_wrapper(res1, batch1, args)
    loss2, loss_dict2 = compute_loss_wrapper(res2, batch2, args)

    loss_total = 0.5 * (loss1 + loss2)
    loss_dict = average_loss_dicts([loss_dict1, loss_dict2])
    loss_dict["loss_total"] = to_float(loss_total)

    return loss_total, loss_dict


def get_pts3d_from_res_for_anchor(res):
    """
    Extract pointmap from model output and return it as [B, H, W, 3].

    This helper is intentionally local to train.py so that the anchor loss does
    not depend on evaluation code.
    """
    candidate_keys = [
        "pts3d",
        "pts3d_in_other_view",
        "pointmap",
        "pred_pts3d",
    ]

    pts = None
    used_key = None

    for key in candidate_keys:
        if key in res:
            pts = res[key]
            used_key = key
            break

    if pts is None:
        raise KeyError(
            f"Cannot find pointmap in model output for anchor loss. "
            f"Available keys: {list(res.keys())}"
        )

    if pts.ndim != 4:
        raise ValueError(
            f"Expected 4D pointmap from key={used_key}, got shape={tuple(pts.shape)}"
        )

    # [B, 3, H, W] -> [B, H, W, 3]
    if pts.shape[1] == 3:
        pts = pts.permute(0, 2, 3, 1).contiguous()

    if pts.shape[-1] != 3:
        raise ValueError(
            f"Expected pointmap with last dim 3 from key={used_key}, "
            f"got shape={tuple(pts.shape)}"
        )

    return pts


def point_anchor_loss(
    student_res,
    teacher_res,
    beta=0.05,
    max_abs_coord=1e4,
):
    """
    Keep student pointmap close to a frozen teacher pointmap.

    Purpose:
        Prevent coplanarity loss from collapsing the pointmap scale.

    student_res:
        trainable model output

    teacher_res:
        frozen teacher model output

    Returns:
        SmoothL1(student_pts, teacher_pts) on valid points.
    """
    student_pts = get_pts3d_from_res_for_anchor(student_res)
    teacher_pts = get_pts3d_from_res_for_anchor(teacher_res).detach()

    valid = (
        torch.isfinite(student_pts).all(dim=-1)
        & torch.isfinite(teacher_pts).all(dim=-1)
        & (student_pts.abs().amax(dim=-1) < max_abs_coord)
        & (teacher_pts.abs().amax(dim=-1) < max_abs_coord)
    )

    if valid.sum() == 0:
        return student_pts.sum() * 0.0

    return torch.nn.functional.smooth_l1_loss(
        student_pts[valid],
        teacher_pts[valid],
        beta=beta,
        reduction="mean",
    )


@torch.no_grad()
def compute_teacher_outputs(teacher_model, view1, view2):
    """
    Forward frozen teacher model without gradient.
    """
    teacher_model.eval()
    teacher_res1, teacher_res2 = teacher_model(view1, view2)
    return teacher_res1, teacher_res2


def add_point_anchor_loss(
    loss_total,
    loss_dict,
    res1,
    res2,
    teacher_res1,
    teacher_res2,
    args,
):
    """
    Add two-view pointmap anchor loss to total loss and loss_dict.
    """
    if args.point_anchor_weight <= 0:
        return loss_total, loss_dict

    anchor1 = point_anchor_loss(
        student_res=res1,
        teacher_res=teacher_res1,
        beta=args.point_anchor_beta,
    )

    anchor2 = point_anchor_loss(
        student_res=res2,
        teacher_res=teacher_res2,
        beta=args.point_anchor_beta,
    )

    loss_anchor = 0.5 * (anchor1 + anchor2)

    loss_total = loss_total + args.point_anchor_weight * loss_anchor

    loss_dict["loss_point_anchor"] = to_float(loss_anchor)
    loss_dict["loss_total"] = to_float(loss_total)

    return loss_total, loss_dict


def add_to_meters(meters, loss_dict):
    for key, value in loss_dict.items():
        try:
            meters[key] += to_float(value)
        except Exception:
            pass


def average_meters(meters, denom):
    if denom <= 0:
        return {}

    return {key: value / denom for key, value in meters.items()}


# ============================================================
# 3. Train / val
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    epoch,
    swanlab_run=None,
    teacher_model=None,
):
    model.train()

    meters = defaultdict(float)
    valid_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix="train")

        res1, res2 = model(view1, view2)

        loss_total, loss_dict = compute_two_view_loss(res1, res2, batch, args)

        if teacher_model is not None and args.point_anchor_weight > 0:
            teacher_res1, teacher_res2 = compute_teacher_outputs(
                teacher_model,
                view1,
                view2,
            )

            loss_total, loss_dict = add_point_anchor_loss(
                loss_total=loss_total,
                loss_dict=loss_dict,
                res1=res1,
                res2=res2,
                teacher_res1=teacher_res1,
                teacher_res2=teacher_res2,
                args=args,
            )

        if epoch == 1 and batch_idx == 1:
            print("[Debug] loss_dict keys:", sorted(loss_dict.keys()))
            if "loss_coplanarity" in loss_dict:
                print("[Debug] loss_coplanarity:", to_float(loss_dict["loss_coplanarity"]))
            if "num_geo_planes" in loss_dict:
                print("[Debug] num_geo_planes:", to_float(loss_dict["num_geo_planes"]))
            if "loss_point_anchor" in loss_dict:
                print("[Debug] loss_point_anchor:", to_float(loss_dict["loss_point_anchor"]))

        if not torch.isfinite(loss_total):
            print(
                f"[Warning] non-finite train loss at "
                f"epoch={epoch}, batch={batch_idx}: {loss_total.item()}"
            )
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        loss_total.backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if args.grad_clip is not None and args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=args.grad_clip,
            )
        else:
            grad_norm = torch.norm(
                torch.stack([
                    p.grad.detach().norm()
                    for p in trainable_params
                    if p.grad is not None
                ])
            )

        if not torch.isfinite(grad_norm):
            print(
                f"[Warning] non-finite grad norm at "
                f"epoch={epoch}, batch={batch_idx}: {to_float(grad_norm)}"
            )
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue

        optimizer.step()

        valid_batches += 1
        add_to_meters(meters, loss_dict)

        global_step = (epoch - 1) * len(loader) + batch_idx

        log_metrics = {
            "train/batch_total_loss": to_float(loss_total),
            "train/batch_line_loss": loss_dict.get("loss_line", 0.0),
            "train/batch_plane_loss": loss_dict.get("loss_plane", 0.0),
            "train/lr": optimizer.param_groups[0]["lr"],
        }

        extra_keys = [
            "loss_line_bce",
            "loss_line_dice",
            "line_target_positive_ratio",
            "loss_plane_embedding",
            "loss_plane_var",
            "loss_plane_dist",
            "loss_plane_reg",
            "loss_plane_pull",
            "loss_plane_push",
            "num_planes",
            "loss_coplanarity",
            "num_geo_planes",
            "loss_point_anchor",
        ]

        for key in extra_keys:
            if key in loss_dict:
                log_metrics[f"train/{key}"] = loss_dict[key]

        swanlab_log(swanlab_run, log_metrics, step=global_step)

        if (
            batch_idx == 1
            or batch_idx == len(loader)
            or batch_idx % args.log_every == 0
        ):
            print(
                f"[Train] batch {batch_idx}/{len(loader)} | "
                f"total={to_float(loss_total):.4f}, "
                f"line={to_float(loss_dict.get('loss_line', 0.0)):.4f}, "
                f"plane={to_float(loss_dict.get('loss_plane', 0.0)):.4f}, "
                f"geo={to_float(loss_dict.get('loss_coplanarity', 0.0)):.6f}, "
                f"anchor={to_float(loss_dict.get('loss_point_anchor', 0.0)):.6f}, "
                f"num_geo={to_float(loss_dict.get('num_geo_planes', 0.0)):.1f}"
            )

    stats = average_meters(meters, valid_batches)

    if valid_batches == 0:
        stats = {
            "loss_total": float("nan"),
            "loss_line": float("nan"),
            "loss_plane": float("nan"),
        }

    stats["skipped_batches"] = skipped_batches

    print(
        f"Train | total: {stats.get('loss_total', float('nan')):.4f}, "
        f"line: {stats.get('loss_line', float('nan')):.4f}, "
        f"plane: {stats.get('loss_plane', float('nan')):.4f}, "
        f"geo: {stats.get('loss_coplanarity', 0.0):.6f}, "
        f"anchor: {stats.get('loss_point_anchor', 0.0):.6f}, "
        f"num_geo: {stats.get('num_geo_planes', 0.0):.1f}, "
        f"skipped: {skipped_batches}"
    )

    epoch_metrics = {
        "train/epoch_total_loss": stats.get("loss_total", float("nan")),
        "train/epoch_line_loss": stats.get("loss_line", float("nan")),
        "train/epoch_plane_loss": stats.get("loss_plane", float("nan")),
        "train/epoch_skipped_batches": skipped_batches,
    }

    for key in [
        "loss_line_bce",
        "loss_line_dice",
        "loss_plane_embedding",
        "loss_plane_var",
        "loss_plane_dist",
        "loss_plane_reg",
        "loss_plane_pull",
        "loss_plane_push",
        "num_planes",
        "loss_coplanarity",
        "num_geo_planes",
        "loss_point_anchor",
    ]:
        if key in stats:
            epoch_metrics[f"train/epoch_{key}"] = stats[key]

    swanlab_log(swanlab_run, epoch_metrics, step=epoch)

    return stats


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    args,
    epoch,
    swanlab_run=None,
    teacher_model=None,
):
    model.eval()

    meters = defaultdict(float)
    valid_batches = 0
    skipped_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        view1, view2 = build_views_from_batch(batch, prefix="val")

        res1, res2 = model(view1, view2)

        loss_total, loss_dict = compute_two_view_loss(res1, res2, batch, args)

        if teacher_model is not None and args.point_anchor_weight > 0:
            teacher_res1, teacher_res2 = compute_teacher_outputs(
                teacher_model,
                view1,
                view2,
            )

            loss_total, loss_dict = add_point_anchor_loss(
                loss_total=loss_total,
                loss_dict=loss_dict,
                res1=res1,
                res2=res2,
                teacher_res1=teacher_res1,
                teacher_res2=teacher_res2,
                args=args,
            )

        if not torch.isfinite(loss_total):
            print(
                f"[Warning] non-finite val loss at "
                f"epoch={epoch}, batch={batch_idx}: {loss_total.item()}"
            )
            skipped_batches += 1
            continue

        valid_batches += 1
        add_to_meters(meters, loss_dict)

        global_step = (epoch - 1) * len(loader) + batch_idx

        log_metrics = {
            "val/batch_total_loss": to_float(loss_total),
            "val/batch_line_loss": loss_dict.get("loss_line", 0.0),
            "val/batch_plane_loss": loss_dict.get("loss_plane", 0.0),
        }

        extra_keys = [
            "loss_line_bce",
            "loss_line_dice",
            "line_target_positive_ratio",
            "loss_plane_embedding",
            "loss_plane_var",
            "loss_plane_dist",
            "loss_plane_reg",
            "loss_plane_pull",
            "loss_plane_push",
            "num_planes",
            "loss_coplanarity",
            "num_geo_planes",
            "loss_point_anchor",
        ]

        for key in extra_keys:
            if key in loss_dict:
                log_metrics[f"val/{key}"] = loss_dict[key]

        swanlab_log(swanlab_run, log_metrics, step=global_step)

        if (
            batch_idx == 1
            or batch_idx == len(loader)
            or batch_idx % args.log_every == 0
        ):
            print(
                f"[Val] batch {batch_idx}/{len(loader)} | "
                f"total={to_float(loss_total):.4f}, "
                f"line={to_float(loss_dict.get('loss_line', 0.0)):.4f}, "
                f"plane={to_float(loss_dict.get('loss_plane', 0.0)):.4f}, "
                f"geo={to_float(loss_dict.get('loss_coplanarity', 0.0)):.6f}, "
                f"anchor={to_float(loss_dict.get('loss_point_anchor', 0.0)):.6f}, "
                f"num_geo={to_float(loss_dict.get('num_geo_planes', 0.0)):.1f}"
            )

    stats = average_meters(meters, valid_batches)

    if valid_batches == 0:
        stats = {
            "loss_total": float("nan"),
            "loss_line": float("nan"),
            "loss_plane": float("nan"),
        }

    stats["skipped_batches"] = skipped_batches

    print(
        f"Val   | total: {stats.get('loss_total', float('nan')):.4f}, "
        f"line: {stats.get('loss_line', float('nan')):.4f}, "
        f"plane: {stats.get('loss_plane', float('nan')):.4f}, "
        f"geo: {stats.get('loss_coplanarity', 0.0):.6f}, "
        f"anchor: {stats.get('loss_point_anchor', 0.0):.6f}, "
        f"num_geo: {stats.get('num_geo_planes', 0.0):.1f}, "
        f"skipped: {skipped_batches}"
    )

    epoch_metrics = {
        "val/epoch_total_loss": stats.get("loss_total", float("nan")),
        "val/epoch_line_loss": stats.get("loss_line", float("nan")),
        "val/epoch_plane_loss": stats.get("loss_plane", float("nan")),
        "val/epoch_skipped_batches": skipped_batches,
    }

    for key in [
        "loss_line_bce",
        "loss_line_dice",
        "loss_plane_embedding",
        "loss_plane_var",
        "loss_plane_dist",
        "loss_plane_reg",
        "loss_plane_pull",
        "loss_plane_push",
        "num_planes",
        "loss_coplanarity",
        "num_geo_planes",
        "loss_point_anchor",
    ]:
        if key in stats:
            epoch_metrics[f"val/epoch_{key}"] = stats[key]

    swanlab_log(swanlab_run, epoch_metrics, step=epoch)

    return stats


# ============================================================
# 4. Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("LightRecon3D Training")
    print("=" * 80)
    print(f"device      : {device}")
    print(f"root_dir    : {args.root_dir}")
    print(f"weights_path: {args.weights_path}")
    print(f"save_dir    : {args.save_dir}")
    print(f"line_weight : {args.line_weight}")
    print(f"plane_weight: {args.plane_weight}")
    print(f"geo_weight  : {args.geo_weight}")
    print(f"teacher_ckpt: {args.teacher_ckpt}")
    print(f"point_anchor_weight: {args.point_anchor_weight}")
    print(f"point_anchor_beta  : {args.point_anchor_beta}")
    print(f"input_mode  : {args.input_mode}")
    print(f"pair_strategy: {args.pair_strategy}")
    print(f"pair_max_view_id_gap: {args.pair_max_view_id_gap}")
    print(f"line_target_dilate: {args.line_target_dilate}")
    print(f"freeze_dust3r: {args.freeze_dust3r}")
    print(f"freeze_encoder: {args.freeze_encoder}")
    print("=" * 80)

    # -------------------------
    # dataset
    # -------------------------
    train_dataset_full = Structured3DDataset(
        root_dir=args.root_dir,
        split="train",
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode=args.input_mode,
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )

    val_dataset_full = Structured3DDataset(
        root_dir=args.root_dir,
        split="val",
        train_ratio=args.train_ratio,
        image_size=(args.image_size, args.image_size),
        input_mode=args.input_mode,
        pair_strategy=args.pair_strategy,
        pair_max_view_id_gap=args.pair_max_view_id_gap,
    )

    print(f"Train scenes: {len(train_dataset_full.scenes)}")
    print(f"Val scenes  : {len(val_dataset_full.scenes)}")
    print(f"Full train samples: {len(train_dataset_full)}")
    print(f"Full val samples  : {len(val_dataset_full)}")

    train_dataset = make_subset(train_dataset_full, args.small_train_size)
    val_dataset = make_subset(val_dataset_full, args.small_val_size)

    print(f"Used train samples: {len(train_dataset)}")
    print(f"Used val samples  : {len(val_dataset)}")

    if args.print_pair_examples > 0:
        print("=" * 80)
        print("Pair examples")
        print("=" * 80)
        num_examples = min(args.print_pair_examples, len(train_dataset_full))
        for idx in range(num_examples):
            print(train_dataset_full.format_pair_sample(idx))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # model
    # -------------------------
    dust3r_backbone = build_dust3r_backbone(
        args.weights_path,
        device=device,
    )

    model = LightReconModel(
        dust3r_backbone=dust3r_backbone,
        hidden_dim=args.hidden_dim,
        plane_embed_dim=args.plane_embed_dim,
    ).to(device)

    if args.freeze_dust3r:
        freeze_dust3r_backbone(model)
    elif args.freeze_encoder:
        freeze_dust3r_encoder(model)

    count_parameters(model)

    teacher_model = load_teacher_model(args, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val = None

    if args.resume is not None:
        start_epoch, best_val = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    swanlab_run = init_swanlab(args)

    # -------------------------
    # training loop
    # -------------------------
    for epoch in range(start_epoch, args.num_epochs + 1):
        print()
        print(f"========== Epoch {epoch}/{args.num_epochs} ==========")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            swanlab_run=swanlab_run,
            teacher_model=teacher_model,
        )

        val_stats = None

        if args.run_val:
            val_stats = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                args=args,
                epoch=epoch,
                swanlab_run=swanlab_run,
                teacher_model=teacher_model,
            )

        # Always save latest.
        save_checkpoint(
            save_dir=args.save_dir,
            filename="latest.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            best_val=best_val,
        )

        # Best checkpoint is based on val total loss if validation is enabled.
        if args.run_val and val_stats is not None:
            current_val = val_stats.get("loss_total", float("nan"))

            if current_val == current_val:  # not NaN
                if best_val is None or current_val < best_val:
                    best_val = current_val

                    save_checkpoint(
                        save_dir=args.save_dir,
                        filename="best.pth",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        best_val=best_val,
                    )

                    print(f"Best checkpoint updated at epoch {epoch}: val={best_val:.4f}")
        else:
            # If no validation, use train total loss for best.
            current_train = train_stats.get("loss_total", float("nan"))

            if current_train == current_train:
                if best_val is None or current_train < best_val:
                    best_val = current_train

                    save_checkpoint(
                        save_dir=args.save_dir,
                        filename="best.pth",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        best_val=best_val,
                    )

                    print(f"Best checkpoint updated at epoch {epoch}: train={best_val:.4f}")

    finish_swanlab(swanlab_run)

    print()
    print("Training finished.")


if __name__ == "__main__":
    main()
