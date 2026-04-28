import torch
import torch.nn.functional as F


# ============================================================
# 1. Plane embedding discriminative loss
# ============================================================

def resize_label_to_pred(gt_plane, pred_embedding):
    """
    gt_plane:
        [B, H, W]

    pred_embedding:
        [B, C, H_pred, W_pred]

    return:
        [B, H_pred, W_pred]
    """
    if gt_plane.ndim != 3:
        raise ValueError(f"Expected gt_plane [B,H,W], got {gt_plane.shape}")

    H_pred, W_pred = pred_embedding.shape[-2:]

    if gt_plane.shape[-2:] == (H_pred, W_pred):
        return gt_plane.long()

    gt = gt_plane.unsqueeze(1).float()
    gt = F.interpolate(
        gt,
        size=(H_pred, W_pred),
        mode="nearest",
    )

    return gt[:, 0].long()


def _sample_embeddings(embeddings, max_pixels):
    """
    embeddings:
        [N, C]
    """
    N = embeddings.shape[0]

    if max_pixels is None or N <= max_pixels:
        return embeddings

    idx = torch.randperm(N, device=embeddings.device)[:max_pixels]
    return embeddings[idx]


def plane_embedding_loss_single_image(
    embedding_map,
    plane_map,
    min_pixels=64,
    max_pixels_per_plane=2048,
    max_planes=12,
    ignore_ids=(-1, 255),

    # discriminative loss parameters
    delta_var=0.5,
    delta_dist=1.5,
    var_weight=1.0,
    dist_weight=1.0,
    reg_weight=0.001,
    eps=1e-6,
):
    """
    Stable discriminative plane embedding loss for one image.

    embedding_map:
        [C, H, W]

    plane_map:
        [H, W]

    Loss:
        L = var_weight  * L_var
          + dist_weight * L_dist
          + reg_weight  * L_reg

    L_var:
        same-plane pixels should stay close to their plane center,
        but only penalize distances larger than delta_var.

    L_dist:
        different plane centers should be separated by at least 2 * delta_dist.

    L_reg:
        weak regularization to prevent centers from drifting too far.
    """
    if embedding_map.ndim != 3:
        raise ValueError(f"Expected embedding_map [C,H,W], got {embedding_map.shape}")

    C, H, W = embedding_map.shape
    emb = embedding_map.permute(1, 2, 0).contiguous()  # [H, W, C]

    plane_ids = torch.unique(plane_map)

    valid_ids = []
    areas = []

    for pid in plane_ids:
        pid_int = int(pid.item())

        if pid_int in ignore_ids:
            continue

        area = (plane_map == pid).sum()

        if int(area.item()) < min_pixels:
            continue

        valid_ids.append(pid)
        areas.append(area)

    if len(valid_ids) == 0:
        zero = embedding_map.sum() * 0.0
        return zero, {
            "loss_plane_embedding": zero,
            "loss_plane_var": zero,
            "loss_plane_dist": zero,
            "loss_plane_reg": zero,
            "loss_plane_pull": zero,
            "loss_plane_push": zero,
            "num_planes": torch.tensor(0.0, device=embedding_map.device, dtype=embedding_map.dtype),
        }

    # Prefer larger planes.
    if max_planes is not None and len(valid_ids) > max_planes:
        areas_tensor = torch.stack(areas)
        topk = torch.topk(
            areas_tensor,
            k=max_planes,
            largest=True,
        ).indices
        valid_ids = [valid_ids[i] for i in topk.tolist()]

    centers = []
    var_losses = []

    for pid in valid_ids:
        mask = plane_map == pid
        plane_emb = emb[mask]  # [N, C]

        # Remove non-finite embeddings.
        finite_mask = torch.isfinite(plane_emb).all(dim=1)
        plane_emb = plane_emb[finite_mask]

        if plane_emb.shape[0] < min_pixels:
            continue

        plane_emb = _sample_embeddings(plane_emb, max_pixels_per_plane)

        if plane_emb.shape[0] < min_pixels:
            continue

        center = plane_emb.mean(dim=0)  # [C]

        if not torch.isfinite(center).all():
            continue

        centers.append(center)

        # Distance from pixels to center.
        dist = torch.sqrt(((plane_emb - center.unsqueeze(0)) ** 2).sum(dim=1) + eps)

        # Only penalize distances larger than delta_var.
        var_loss = F.relu(dist - delta_var).pow(2).mean()
        var_losses.append(var_loss)

    if len(centers) == 0:
        zero = embedding_map.sum() * 0.0
        return zero, {
            "loss_plane_embedding": zero,
            "loss_plane_var": zero,
            "loss_plane_dist": zero,
            "loss_plane_reg": zero,
            "loss_plane_pull": zero,
            "loss_plane_push": zero,
            "num_planes": torch.tensor(0.0, device=embedding_map.device, dtype=embedding_map.dtype),
        }

    centers = torch.stack(centers, dim=0)  # [P, C]
    var_loss = torch.stack(var_losses).mean()

    # Inter-plane center distance.
    if centers.shape[0] >= 2:
        dist_matrix = torch.cdist(centers, centers, p=2)  # [P, P]

        P = centers.shape[0]
        upper_mask = torch.triu(
            torch.ones((P, P), device=centers.device, dtype=torch.bool),
            diagonal=1,
        )

        center_dists = dist_matrix[upper_mask]

        # Different centers should be farther than 2 * delta_dist.
        dist_loss = F.relu(2.0 * delta_dist - center_dists).pow(2).mean()
    else:
        dist_loss = var_loss * 0.0

    reg_loss = torch.norm(centers, p=2, dim=1).mean()

    total = (
        var_weight * var_loss
        + dist_weight * dist_loss
        + reg_weight * reg_loss
    )

    # Last defense: if loss is non-finite, return zero to avoid poisoning training.
    if not torch.isfinite(total):
        zero = embedding_map.sum() * 0.0
        total = zero
        var_loss = zero
        dist_loss = zero
        reg_loss = zero

    stats = {
        "loss_plane_embedding": total,
        "loss_plane_var": var_loss,
        "loss_plane_dist": dist_loss,
        "loss_plane_reg": reg_loss,

        # aliases for older logging names
        "loss_plane_pull": var_loss,
        "loss_plane_push": dist_loss,

        "num_planes": torch.tensor(
            float(centers.shape[0]),
            device=embedding_map.device,
            dtype=embedding_map.dtype,
        ),
    }

    return total, stats


def plane_embedding_loss(
    pred_plane_embedding,
    gt_plane,
    min_pixels=64,
    max_pixels_per_plane=2048,
    max_planes_per_image=12,
    ignore_ids=(-1, 255),

    delta_var=0.5,
    delta_dist=1.5,
    var_weight=1.0,
    dist_weight=1.0,
    reg_weight=0.001,
):
    """
    Plane embedding loss over batch.

    pred_plane_embedding:
        [B, C, H, W]

    gt_plane:
        [B, H, W]
    """
    if pred_plane_embedding.ndim != 4:
        raise ValueError(
            f"Expected pred_plane_embedding [B,C,H,W], got {pred_plane_embedding.shape}"
        )

    gt_plane = resize_label_to_pred(gt_plane, pred_plane_embedding)

    B = pred_plane_embedding.shape[0]

    losses = []
    var_losses = []
    dist_losses = []
    reg_losses = []
    num_planes = []

    for b in range(B):
        loss_b, stats_b = plane_embedding_loss_single_image(
            embedding_map=pred_plane_embedding[b],
            plane_map=gt_plane[b],
            min_pixels=min_pixels,
            max_pixels_per_plane=max_pixels_per_plane,
            max_planes=max_planes_per_image,
            ignore_ids=ignore_ids,

            delta_var=delta_var,
            delta_dist=delta_dist,
            var_weight=var_weight,
            dist_weight=dist_weight,
            reg_weight=reg_weight,
        )

        losses.append(loss_b)
        var_losses.append(stats_b["loss_plane_var"])
        dist_losses.append(stats_b["loss_plane_dist"])
        reg_losses.append(stats_b["loss_plane_reg"])
        num_planes.append(stats_b["num_planes"])

    loss = torch.stack(losses).mean()

    if not torch.isfinite(loss):
        zero = pred_plane_embedding.sum() * 0.0
        loss = zero

    stats = {
        "loss_plane_embedding": loss,
        "loss_plane_var": torch.stack(var_losses).mean(),
        "loss_plane_dist": torch.stack(dist_losses).mean(),
        "loss_plane_reg": torch.stack(reg_losses).mean(),

        # aliases
        "loss_plane_pull": torch.stack(var_losses).mean(),
        "loss_plane_push": torch.stack(dist_losses).mean(),

        "num_planes": torch.stack(num_planes).mean(),
    }

    return loss, stats


# ============================================================
# 2. Optional GT-guided 3D coplanarity loss
# ============================================================

def get_pts3d_from_res(res):
    """
    Get DUSt3R pointmap from model output.

    Possible keys:
        pts3d
        pts3d_in_other_view
        pointmap
        pred_pts3d

    Return:
        [B, H, W, 3]
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
            "Cannot find pointmap in model output. "
            f"Tried keys: {candidate_keys}. "
            f"Available keys: {list(res.keys())}"
        )

    if not torch.is_tensor(pts):
        raise TypeError(f"res['{used_key}'] is not a tensor, got {type(pts)}")

    if pts.ndim != 4:
        raise ValueError(
            f"Expected pointmap [B,H,W,3] or [B,3,H,W], got {pts.shape}"
        )

    if pts.shape[-1] == 3:
        return pts

    if pts.shape[1] == 3:
        return pts.permute(0, 2, 3, 1).contiguous()

    raise ValueError(
        f"Cannot interpret pointmap shape {pts.shape} from key '{used_key}'."
    )


def resize_plane_to_pts(gt_plane, pts3d):
    """
    gt_plane:
        [B, H, W]

    pts3d:
        [B, H_pts, W_pts, 3]

    return:
        [B, H_pts, W_pts]
    """
    if gt_plane.ndim != 3:
        raise ValueError(f"Expected gt_plane [B,H,W], got {gt_plane.shape}")

    H_pts, W_pts = pts3d.shape[1], pts3d.shape[2]

    if gt_plane.shape[-2:] == (H_pts, W_pts):
        return gt_plane.long()

    gt = gt_plane.unsqueeze(1).float()
    gt = F.interpolate(
        gt,
        size=(H_pts, W_pts),
        mode="nearest",
    )

    return gt[:, 0].long()


def _sample_points(points, max_points):
    N = points.shape[0]

    if max_points is None or N <= max_points:
        return points

    idx = torch.randperm(N, device=points.device)[:max_points]
    return points[idx]


def coplanarity_loss_single_plane(
    points,
    eps=1e-6,
    normalize=True,
):
    """
    PCA-based coplanarity loss for one plane.

    points:
        [N, 3]
    """
    valid = torch.isfinite(points).all(dim=1)
    points = points[valid]

    if points.shape[0] < 3:
        return None

    center = points.mean(dim=0, keepdim=True)
    x = points - center

    cov = x.transpose(0, 1) @ x / (points.shape[0] + eps)

    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)

    smallest = eigvals[0]

    if normalize:
        return smallest / (eigvals.sum() + eps)

    return smallest


def coplanarity_loss_from_gt_plane(
    pts3d,
    gt_plane,
    min_points=128,
    max_points_per_plane=2048,
    max_planes_per_image=8,
    ignore_ids=(-1, 255),
    normalize=True,
):
    """
    GT-guided 3D coplanarity loss.

    pts3d:
        [B, H, W, 3]

    gt_plane:
        [B, H, W]
    """
    if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
        raise ValueError(f"Expected pts3d [B,H,W,3], got {pts3d.shape}")

    gt_plane = resize_plane_to_pts(gt_plane, pts3d)

    B = pts3d.shape[0]

    losses = []
    num_valid_planes = []

    for b in range(B):
        plane_map = gt_plane[b]
        points_map = pts3d[b]

        plane_ids = torch.unique(plane_map)

        valid_ids = []
        areas = []

        for pid in plane_ids:
            pid_int = int(pid.item())

            if pid_int in ignore_ids:
                continue

            area = (plane_map == pid).sum()

            if int(area.item()) < min_points:
                continue

            valid_ids.append(pid)
            areas.append(area)

        if len(valid_ids) == 0:
            continue

        if max_planes_per_image is not None and len(valid_ids) > max_planes_per_image:
            areas_tensor = torch.stack(areas)
            topk = torch.topk(
                areas_tensor,
                k=max_planes_per_image,
                largest=True,
            ).indices
            valid_ids = [valid_ids[i] for i in topk.tolist()]

        count_b = 0

        for pid in valid_ids:
            mask = plane_map == pid
            points = points_map[mask]

            valid = torch.isfinite(points).all(dim=1)
            points = points[valid]

            if points.shape[0] < min_points:
                continue

            points = _sample_points(points, max_points_per_plane)

            loss_i = coplanarity_loss_single_plane(
                points,
                normalize=normalize,
            )

            if loss_i is None:
                continue

            if not torch.isfinite(loss_i):
                continue

            losses.append(loss_i)
            count_b += 1

        num_valid_planes.append(
            torch.tensor(
                float(count_b),
                device=pts3d.device,
                dtype=pts3d.dtype,
            )
        )

    if len(losses) == 0:
        zero = pts3d.sum() * 0.0
        return zero, {
            "loss_coplanarity": zero,
            "num_geo_planes": torch.tensor(
                0.0,
                device=pts3d.device,
                dtype=pts3d.dtype,
            ),
        }

    loss = torch.stack(losses).mean()

    if not torch.isfinite(loss):
        loss = pts3d.sum() * 0.0

    if len(num_valid_planes) == 0:
        num_geo = torch.tensor(
            0.0,
            device=pts3d.device,
            dtype=pts3d.dtype,
        )
    else:
        num_geo = torch.stack(num_valid_planes).mean()

    stats = {
        "loss_coplanarity": loss,
        "num_geo_planes": num_geo,
    }

    return loss, stats