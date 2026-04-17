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


def line_loss_fn(pred_line, gt_line):
    gt_line = gt_line.float()

    with torch.no_grad():
        pos = gt_line.sum()
        neg = gt_line.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=50.0)

    loss_bce = F.binary_cross_entropy_with_logits(
        pred_line,
        gt_line,
        pos_weight=pos_weight
    )

    loss_dice = dice_loss_with_logits(pred_line, gt_line)

    dice_weight = 1.0
    loss_line = loss_bce + dice_weight * loss_dice

    loss_dict = {
        "loss_line_total": loss_line.item(),
        "loss_line_bce": loss_bce.item(),
        "loss_line_dice": loss_dice.item(),
    }

    return loss_line, loss_dict