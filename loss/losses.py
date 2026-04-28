import torch

from .line_loss import bce_dice_line_loss
from .plane_loss import (
    plane_embedding_loss,
    coplanarity_loss_from_gt_plane,
    get_pts3d_from_res,
)


def _to_float(x):
    """
    Convert tensor / scalar to Python float for logging.
    """
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())

    return float(x)


def compute_losses(
    res,
    batch,

    # Main weights
    line_weight=1.0,
    plane_weight=1.0,
    geo_weight=0.0,

    # Compatible aliases
    lambda_line=None,
    lambda_plane=None,

    # Line loss params
    line_pos_weight=None,
    line_bce_weight=1.0,
    line_dice_weight=1.0,

    # Plane embedding params
    plane_min_pixels=64,
    plane_max_pixels_per_plane=2048,
    plane_max_planes_per_image=12,

    plane_delta_var=0.5,
    plane_delta_dist=1.5,
    plane_var_weight=1.0,
    plane_dist_weight=1.0,
    plane_reg_weight=0.001,

    # Coplanarity params
    coplanarity_min_points=128,
    coplanarity_max_points_per_plane=2048,
    coplanarity_max_planes_per_image=8,
    coplanarity_normalize=True,

    # Old unused args, kept for compatibility
    plane_pos_weight=None,
    plane_bce_weight=1.0,
    plane_dice_weight=1.0,
    plane_boundary_dilation=2,

    **kwargs,
):
    """
    Main LightRecon3D loss.

    Current objective:

        L_total =
            line_weight  * L_line_2d
          + plane_weight * L_plane_embedding
          + geo_weight   * L_coplanarity

    Current model outputs:

        pred_line:
            [B, 1, H, W], line logits

        pred_plane:
            [B, C, H, W], plane embedding map

    Important:
        pred_plane is NOT class logits.
        pred_plane is NOT boundary logits.
        pred_plane is a per-pixel plane embedding.

    gt_plane is used only as within-image instance grouping.
    It does not require cross-image class consistency.
    """
    if lambda_line is not None:
        line_weight = lambda_line

    if lambda_plane is not None:
        plane_weight = lambda_plane

    if "pred_line" not in res:
        raise KeyError("res does not contain pred_line")

    if "pred_plane" not in res:
        raise KeyError("res does not contain pred_plane / plane embedding")

    if "gt_line" not in batch:
        raise KeyError("batch does not contain gt_line")

    if "gt_plane" not in batch:
        raise KeyError("batch does not contain gt_plane")

    pred_line = res["pred_line"]
    pred_plane_embedding = res["pred_plane"]

    gt_line = batch["gt_line"]
    gt_plane = batch["gt_plane"]

    # -------------------------
    # 1. 2D line loss
    # -------------------------
    loss_line, line_stats = bce_dice_line_loss(
        pred_line=pred_line,
        gt_line=gt_line,
        pos_weight=line_pos_weight,
        bce_weight=line_bce_weight,
        dice_weight=line_dice_weight,
    )

    # -------------------------
    # 2. Plane embedding loss
    # -------------------------
    loss_plane, plane_stats = plane_embedding_loss(
        pred_plane_embedding=pred_plane_embedding,
        gt_plane=gt_plane,
        min_pixels=plane_min_pixels,
        max_pixels_per_plane=plane_max_pixels_per_plane,
        max_planes_per_image=plane_max_planes_per_image,

        delta_var=plane_delta_var,
        delta_dist=plane_delta_dist,
        var_weight=plane_var_weight,
        dist_weight=plane_dist_weight,
        reg_weight=plane_reg_weight,
    )

    # -------------------------
    # 3. Optional 3D coplanarity loss
    # -------------------------
    if geo_weight is not None and float(geo_weight) > 0.0:
        pts3d = get_pts3d_from_res(res)

        loss_geo, geo_stats = coplanarity_loss_from_gt_plane(
            pts3d=pts3d,
            gt_plane=gt_plane,
            min_points=coplanarity_min_points,
            max_points_per_plane=coplanarity_max_points_per_plane,
            max_planes_per_image=coplanarity_max_planes_per_image,
            normalize=coplanarity_normalize,
        )
    else:
        loss_geo = loss_plane * 0.0
        geo_stats = {
            "loss_coplanarity": loss_geo,
            "num_geo_planes": loss_geo,
        }

    # -------------------------
    # Total
    # -------------------------
    loss_total = (
        line_weight * loss_line
        + plane_weight * loss_plane
        + geo_weight * loss_geo
    )

    zero = loss_total * 0.0

    # This dict is for logging.
    # Keep all values detached as Python floats to avoid graph retention.
    loss_dict = {
        "loss_total": _to_float(loss_total),

        # Main logs
        "loss_line": _to_float(loss_line),
        "loss_plane": _to_float(loss_plane),

        # Line details
        "loss_line_bce": _to_float(line_stats["loss_line_bce"]),
        "loss_line_dice": _to_float(line_stats["loss_line_dice"]),

        # Plane embedding details
        "loss_plane_embedding": _to_float(plane_stats["loss_plane_embedding"]),
        "loss_plane_var": _to_float(plane_stats["loss_plane_var"]),
        "loss_plane_dist": _to_float(plane_stats["loss_plane_dist"]),
        "loss_plane_reg": _to_float(plane_stats["loss_plane_reg"]),

        # Aliases for older names
        "loss_plane_pull": _to_float(plane_stats["loss_plane_pull"]),
        "loss_plane_push": _to_float(plane_stats["loss_plane_push"]),

        "num_planes": _to_float(plane_stats["num_planes"]),

        # Geometry details
        "loss_coplanarity": _to_float(geo_stats["loss_coplanarity"]),
        "num_geo_planes": _to_float(geo_stats["num_geo_planes"]),

        # Old compatibility keys
        "loss_plane_bce": _to_float(zero),
        "loss_plane_dice": _to_float(zero),
        "loss_plane_boundary": _to_float(zero),
        "loss_plane_boundary_bce": _to_float(zero),
        "loss_plane_boundary_dice": _to_float(zero),
    }

    return loss_total, loss_dict


# Compatibility aliases
def compute_loss(*args, **kwargs):
    return compute_losses(*args, **kwargs)


def compute_lightrecon_loss(*args, **kwargs):
    return compute_losses(*args, **kwargs)


def total_loss_fn(*args, **kwargs):
    return compute_losses(*args, **kwargs)