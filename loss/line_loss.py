import torch
import torch.nn.functional as F


def dice_loss_with_logits(logits, targets, eps=1e-6):
    """
    Binary Dice loss.

    logits:
        [B, 1, H, W]

    targets:
        [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)

    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def resize_binary_target_if_needed(target, pred):
    """
    target:
        [B, 1, H, W]

    pred:
        [B, 1, H, W]
    """
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(
            target.float(),
            size=pred.shape[-2:],
            mode="nearest",
        )
    return target


def bce_dice_line_loss(
    pred_line,
    gt_line,
    pos_weight=None,
    bce_weight=1.0,
    dice_weight=1.0,
):
    """
    Line loss = BCEWithLogits + Dice.

    pred_line:
        [B, 1, H, W], logits

    gt_line:
        [B, 1, H, W] or [B, H, W]
    """
    if gt_line.ndim == 3:
        gt_line = gt_line.unsqueeze(1)

    gt_line = resize_binary_target_if_needed(gt_line, pred_line)
    gt_line = gt_line.float()

    if pos_weight is not None:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(
                [float(pos_weight)],
                device=pred_line.device,
                dtype=pred_line.dtype,
            )
        else:
            pos_weight = pos_weight.to(
                device=pred_line.device,
                dtype=pred_line.dtype,
            )

    bce = F.binary_cross_entropy_with_logits(
        pred_line,
        gt_line,
        pos_weight=pos_weight,
    )

    dice = dice_loss_with_logits(pred_line, gt_line)

    loss = bce_weight * bce + dice_weight * dice

    stats = {
        "loss_line": loss,
        "loss_line_bce": bce,
        "loss_line_dice": dice,
    }

    return loss, stats