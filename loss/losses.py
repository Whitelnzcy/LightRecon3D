import torch
import torch.nn.functional as F


def dice_loss_with_logits(logits, targets, smooth=1.0):
    targets = targets.float()
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss.mean()


def line_loss_fn(pred_line, gt_line, dice_weight=1.0, pos_weight=10.0):
    """
    pred_line: [B, 1, H, W]
    gt_line:   [B, 1, H, W]
    """
    gt_line = gt_line.float()

    pos_weight = torch.tensor(
        [pos_weight],
        device=pred_line.device,
        dtype=pred_line.dtype
    )

    loss_bce = F.binary_cross_entropy_with_logits(
        pred_line,
        gt_line,
        pos_weight=pos_weight
    )

    loss_dice = dice_loss_with_logits(pred_line, gt_line)

    loss_line = loss_bce + dice_weight * loss_dice

    loss_dict = {
        "loss_line_total": loss_line.item(),
        "loss_line_bce": loss_bce.item(),
        "loss_line_dice": loss_dice.item(),
    }

    return loss_line, loss_dict


def plane_loss_fn(pred_plane, gt_plane):
    """
    pred_plane: [B, C, H, W]
    gt_plane:   [B, H, W]
    """
    gt_plane = gt_plane.long()

    num_classes = pred_plane.shape[1]
    gt_plane = gt_plane.clamp(0, num_classes - 1)

    loss_plane = F.cross_entropy(pred_plane, gt_plane, reduction='mean')
    return loss_plane


def total_loss_fn(
    outputs,
    batch,
    line_weight=1.0,
    plane_weight=1.0,
    line_pos_weight=10.0,
    line_dice_weight=1.0
):
    """
    outputs:
        outputs["pred_line"]  -> [B, 1, H, W]
        outputs["pred_plane"] -> [B, C, H, W]

    batch:
        batch["gt_line"]  -> [B, 1, H, W]
        batch["gt_plane"] -> [B, H, W]
    """
    pred_line = outputs["pred_line"]
    pred_plane = outputs["pred_plane"]

    gt_line = batch["gt_line"]
    gt_plane = batch["gt_plane"]

    loss_line, line_dict = line_loss_fn(
        pred_line,
        gt_line,
        dice_weight=line_dice_weight,
        pos_weight=line_pos_weight
    )

    loss_plane = plane_loss_fn(pred_plane, gt_plane)

    loss_total = line_weight * loss_line + plane_weight * loss_plane

    loss_dict = {
        "loss_total": loss_total.item(),
        "loss_line": loss_line.item(),
        "loss_plane": loss_plane.item(),
        "loss_line_bce": line_dict["loss_line_bce"],
        "loss_line_dice": line_dict["loss_line_dice"],
    }

    return loss_total, loss_dict