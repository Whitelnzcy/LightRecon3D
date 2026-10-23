"""M1: spatial RGB/mask/geometry baseline, no line or GT inference features."""
import torch
from torch import nn
import torch.nn.functional as F


def block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(4, out_channels), nn.GELU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(4, out_channels), nn.GELU())


class SpatialSupportRefiner(nn.Module):
    """Shared candidate scorer: candidate reordering only reorders output logits.

    common: B x 4 x H x W (RGB [-1,1], observed registry mask).
    candidates: B x K x 2 x H x W (original support, log geometric residual).
    """
    def __init__(self):
        super().__init__()
        self.image = block(4, 16)
        self.context = block(16, 32)
        self.decode = block(48, 16)
        self.candidate = block(18, 24)
        self.score = nn.Conv2d(24, 1, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, common, candidates):
        if common.ndim != 4 or candidates.ndim != 5:
            raise ValueError('Expected BCHW common and BKCHW candidate inputs')
        b, k, c, h, w = candidates.shape
        if common.shape != (b, 4, h, w) or c != 2 or k < 1:
            raise ValueError('Input dimensions disagree')
        fine = self.image(common)
        context = self.context(F.avg_pool2d(fine, 2))
        context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=False)
        shared = self.decode(torch.cat([fine, context], 1))
        shared = shared[:, None].expand(-1, k, -1, -1, -1)
        features = torch.cat([shared, candidates], 2).reshape(b*k, 18, h, w)
        delta = self.score(self.candidate(features)).reshape(b, k, h, w)
        return delta + 2*candidates[:, :, 0]


def ownership_loss(logits, target, original_columns, dice_weight=.5):
    """Missing candidates/invalid/unassigned targets use -100, never a fake class.

    Loss ignores these targets; full-cloud evaluation must still count them.
    CE is doubled for initially correct pixels; Dice macro-averages present
    target classes per patch. Empty-target patches return finite zero loss.
    """
    valid = target >= 0
    if not valid.any():
        return logits.sum()*0
    ce = F.cross_entropy(logits, target, ignore_index=-100, reduction='none')
    weight = 1 + (original_columns == target).float()
    ce = (ce*weight)[valid].mean()
    k = logits.shape[1]
    gt = F.one_hot(target.clamp_min(0), k).permute(0, 3, 1, 2).to(logits.dtype)
    mask = valid[:, None]
    gt = gt*mask
    probability = logits.softmax(1)*mask
    present = gt.sum((2, 3)) > 0
    dice = 1-(2*(gt*probability).sum((2, 3))+1)/(gt.sum((2, 3))+probability.sum((2, 3))+1)
    return ce + dice_weight*dice[present].mean()
