import torch
import torch.nn.functional as F

def plane_loss_fn(pred_plane, gt_plane):
    """
    pred_plane: [B, C, H, W]，模型输出的 logits
    gt_plane:   [B, H, W]，像素级整数标签
    """

    # 1. CE 的 target 应该是什么 dtype？
    gt_plane = gt_plane.long()

    # 2. 防止标签值超过类别范围
    num_classes = pred_plane.shape[1]
    gt_plane = gt_plane.clamp(0, num_classes - 1)

    # 3. 计算 CE loss
    loss_plane = F.cross_entropy(pred_plane, gt_plane, reduction='mean')

    return loss_plane