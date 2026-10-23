"""Matched M1/M2/M3 arms; line evidence gates an actual candidate-score term."""
import torch
from torch import nn
import torch.nn.functional as F
from models.spatial_support_refiner import SpatialSupportRefiner,block


class SpatialLineRefiner(SpatialSupportRefiner):
    def __init__(self, arm):
        super().__init__()
        if arm not in ['M1','M2','M3','MC','MS']:raise ValueError('Unknown arm')
        self.arm=arm
        self.line_branch=block(24,24)
        self.line_score=nn.Conv2d(24,1,1)
        self.line_gate=nn.Conv2d(24,1,1)
        nn.init.zeros_(self.line_score.weight);nn.init.zeros_(self.line_score.bias)
        nn.init.zeros_(self.line_gate.weight);nn.init.zeros_(self.line_gate.bias)

    def forward(self,common,candidates,lines,return_gate=False):
        b,k,c,h,w=candidates.shape
        if common.shape!=(b,4,h,w) or c!=2 or lines.shape!=(b,k,6,h,w):
            raise ValueError('Input shape mismatch')
        fine=self.image(common)
        context=F.interpolate(self.context(F.avg_pool2d(fine,2)),size=(h,w),mode='bilinear',align_corners=False)
        shared=self.decode(torch.cat([fine,context],1))[:,None].expand(-1,k,-1,-1,-1)
        base=torch.cat([shared,candidates],2).reshape(b*k,18,h,w)
        logits=self.score(self.candidate(base)).reshape(b,k,h,w)+2*candidates[:,:,0]
        # M1 computes the same branch with zero line channels, but zero
        # availability makes it observationally equivalent to the plain model.
        evidence=lines if self.arm!='M1' else torch.zeros_like(lines)
        if self.arm=='MC':
            # Capacity control: all six extra channels contain only existing
            # RGB/observed/mask/residual inputs, never line availability/geometry.
            evidence=torch.cat([common[:,None].expand(-1,k,-1,-1,-1),candidates],2)
        extra=self.line_branch(torch.cat([base,evidence.reshape(b*k,6,h,w)],1))
        availability=evidence[:,:,0]
        if self.arm=='MC':
            availability=(common[:,3:4]>0).to(common.dtype).expand(-1,k,-1,-1)
        if self.arm in ['M3','MC']:
            gate=self.line_gate(extra).sigmoid().reshape(b,k,h,w)
        elif self.arm=='MS':
            # Reuse one existing bias parameter: old checkpoint keys stay valid.
            # No feature-dependent gate, and no gradient to line_gate.weight.
            gate=self.line_gate.bias[0].sigmoid().expand_as(availability)
        else:
            gate=torch.full_like(availability,.5)
        contribution=availability*gate*self.line_score(extra).reshape(b,k,h,w)
        result=logits+contribution
        return (result,gate*availability) if return_gate else result
