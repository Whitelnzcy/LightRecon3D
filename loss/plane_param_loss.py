import torch
import torch.nn.functional as F


def _resize_like(x, target_hw, mode):
    if x.shape[-2:] == target_hw:
        return x
    return F.interpolate(x, size=target_hw, mode=mode)


def _resize_label_like(x, target_hw):
    if x.shape[-2:] == target_hw:
        return x.long()
    return F.interpolate(x.unsqueeze(1).float(), size=target_hw, mode="nearest")[:, 0].long()


def _safe_normalize(x, dim, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def plane_parameter_loss_single_image(
    pred_params,
    gt_normal,
    gt_offset,
    gt_plane,
    valid_mask,
    min_pixels=64,
    max_pixels_per_plane=2048,
    max_planes=8,
    normal_weight=1.0,
    offset_weight=0.25,
    consistency_weight=0.1,
    ignore_ids=(-1, 0, 255),
):
    """
    Plane parameter supervision for one image.

    pred_params:
        [4,H,W], first 3 channels are normal logits, last channel is normalized offset.

    gt_normal:
        [3,H,W], unit normal from Structured3D layout.json.

    gt_offset:
        [1,H,W], layout offset divided by dataset plane_offset_scale.

    gt_plane:
        [H,W], plane instance ids, where i+1 corresponds to layout.json planes[i].

    valid_mask:
        [1,H,W], pixels covered by a layout plane.

    The loss supervises a per-plane aggregate parameter, not every pixel independently:
    this matches the actual target, a compact parameter equation per structure plane.
    """
    _, h, w = pred_params.shape
    pred = pred_params.permute(1, 2, 0).contiguous()
    gt_n = gt_normal.permute(1, 2, 0).contiguous()
    gt_d = gt_offset[0]
    valid = valid_mask[0] > 0.5

    plane_ids = torch.unique(gt_plane)
    candidates = []
    for pid in plane_ids:
        pid_int = int(pid.item())
        if pid_int in ignore_ids:
            continue
        mask = (gt_plane == pid) & valid
        count = int(mask.sum().item())
        if count >= min_pixels:
            candidates.append((pid, count))

    if len(candidates) == 0:
        zero = pred_params.sum() * 0.0
        return zero, {
            "loss_plane_param": zero,
            "loss_param_normal": zero,
            "loss_param_offset": zero,
            "loss_param_consistency": zero,
            "num_param_planes": torch.tensor(0.0, device=pred_params.device, dtype=pred_params.dtype),
        }

    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:max_planes]

    normal_losses = []
    offset_losses = []
    consistency_losses = []

    for pid, _ in candidates:
        mask = (gt_plane == pid) & valid
        pred_plane = pred[mask]
        gt_n_plane = gt_n[mask]
        gt_d_plane = gt_d[mask]

        finite = torch.isfinite(pred_plane).all(dim=1)
        pred_plane = pred_plane[finite]
        gt_n_plane = gt_n_plane[finite]
        gt_d_plane = gt_d_plane[finite]

        if pred_plane.shape[0] < min_pixels:
            continue

        if pred_plane.shape[0] > max_pixels_per_plane:
            idx = torch.randperm(pred_plane.shape[0], device=pred_plane.device)[:max_pixels_per_plane]
            pred_plane = pred_plane[idx]
            gt_n_plane = gt_n_plane[idx]
            gt_d_plane = gt_d_plane[idx]

        pred_normal_px = _safe_normalize(pred_plane[:, :3], dim=1)
        pred_offset_px = pred_plane[:, 3]

        pred_normal = _safe_normalize(pred_normal_px.mean(dim=0), dim=0)
        pred_offset = pred_offset_px.mean()

        target_normal = _safe_normalize(gt_n_plane.mean(dim=0), dim=0)
        target_offset = gt_d_plane.mean()

        # Normal direction has a sign ambiguity. Align the predicted equation to
        # the target before offset supervision.
        sign = torch.where((pred_normal * target_normal).sum() < 0, -1.0, 1.0)
        pred_normal = pred_normal * sign
        pred_offset = pred_offset * sign

        cosine = (pred_normal * target_normal).sum().clamp(-1.0, 1.0)
        normal_losses.append(1.0 - cosine)
        offset_losses.append(F.smooth_l1_loss(pred_offset, target_offset))

        consistency_normal = (1.0 - (pred_normal_px * pred_normal.detach()).sum(dim=1).clamp(-1.0, 1.0)).mean()
        consistency_offset = F.smooth_l1_loss(pred_offset_px, pred_offset.detach().expand_as(pred_offset_px))
        consistency_losses.append(consistency_normal + consistency_offset)

    if len(normal_losses) == 0:
        zero = pred_params.sum() * 0.0
        return zero, {
            "loss_plane_param": zero,
            "loss_param_normal": zero,
            "loss_param_offset": zero,
            "loss_param_consistency": zero,
            "num_param_planes": torch.tensor(0.0, device=pred_params.device, dtype=pred_params.dtype),
        }

    loss_normal = torch.stack(normal_losses).mean()
    loss_offset = torch.stack(offset_losses).mean()
    loss_consistency = torch.stack(consistency_losses).mean()

    total = normal_weight * loss_normal + offset_weight * loss_offset + consistency_weight * loss_consistency

    if not torch.isfinite(total):
        zero = pred_params.sum() * 0.0
        total = zero
        loss_normal = zero
        loss_offset = zero
        loss_consistency = zero

    return total, {
        "loss_plane_param": total,
        "loss_param_normal": loss_normal,
        "loss_param_offset": loss_offset,
        "loss_param_consistency": loss_consistency,
        "num_param_planes": torch.tensor(
            float(len(normal_losses)),
            device=pred_params.device,
            dtype=pred_params.dtype,
        ),
    }


def plane_parameter_loss(
    pred_params,
    gt_normal,
    gt_offset,
    gt_plane,
    valid_mask,
    min_pixels=64,
    max_pixels_per_plane=2048,
    max_planes_per_image=8,
    normal_weight=1.0,
    offset_weight=0.25,
    consistency_weight=0.1,
):
    if pred_params.ndim != 4 or pred_params.shape[1] != 4:
        raise ValueError(f"Expected pred_params [B,4,H,W], got {pred_params.shape}")

    target_hw = pred_params.shape[-2:]
    gt_normal = _resize_like(gt_normal, target_hw, mode="nearest")
    gt_offset = _resize_like(gt_offset, target_hw, mode="nearest")
    valid_mask = _resize_like(valid_mask, target_hw, mode="nearest")
    gt_plane = _resize_label_like(gt_plane, target_hw)

    losses = []
    stats_accum = {
        "loss_plane_param": [],
        "loss_param_normal": [],
        "loss_param_offset": [],
        "loss_param_consistency": [],
        "num_param_planes": [],
    }

    for b in range(pred_params.shape[0]):
        loss_b, stats_b = plane_parameter_loss_single_image(
            pred_params=pred_params[b],
            gt_normal=gt_normal[b],
            gt_offset=gt_offset[b],
            gt_plane=gt_plane[b],
            valid_mask=valid_mask[b],
            min_pixels=min_pixels,
            max_pixels_per_plane=max_pixels_per_plane,
            max_planes=max_planes_per_image,
            normal_weight=normal_weight,
            offset_weight=offset_weight,
            consistency_weight=consistency_weight,
        )
        losses.append(loss_b)
        for key in stats_accum:
            stats_accum[key].append(stats_b[key])

    loss = torch.stack(losses).mean()
    stats = {key: torch.stack(values).mean() for key, values in stats_accum.items()}
    return loss, stats
