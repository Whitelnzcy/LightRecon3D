import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss_with_logits(logits, targets, smooth=1.0):
    # GPT 写的这个函数没问题，保留
    targets = targets.float()
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()

def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """
    专门针对极度不平衡的线检测任务设计的 Focal Loss
    """
    targets = targets.float()
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    probs = torch.sigmoid(logits)
    # p_t 是模型对正确类别的预测概率
    p_t = probs * targets + (1 - probs) * (1 - targets)
    
    # alpha 权重平衡
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # 核心：(1 - p_t)**gamma 会大幅削弱那些已经预测得很准的空白墙面的梯度
    focal_weight = alpha_t * (1 - p_t) ** gamma
    focal_loss = focal_weight * bce_loss
    
    return focal_loss.mean()

def line_loss_fn(pred_line, gt_line, lambda_focal=1.0, lambda_dice=1.0):
    """
    pred_line: (B, 1, H, W) 未经过 sigmoid 的 logits
    gt_line: (B, 1, H, W) 0或1的真实掩码
    """
    # 丢弃动态 pos_weight，改用 Focal Loss 压制背景噪声
    loss_focal = focal_loss_with_logits(pred_line, gt_line)
    
    # 结合 Dice Loss 保证线条的连贯性和结构完整性
    loss_dice = dice_loss_with_logits(pred_line, gt_line)
    
    loss_line = lambda_focal * loss_focal + lambda_dice * loss_dice
    
    loss_dict = {
        "loss_line_total": loss_line.item(),
        "loss_line_focal": loss_focal.item(),
        "loss_line_dice": loss_dice.item(),
    }
    
    return loss_line, loss_dict