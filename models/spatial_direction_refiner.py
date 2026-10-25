"""Small PushBoundary-inspired 3D directional propagation on registered grids.

Not full PointNet++/KPConv reproduction: local grid neighbors, same-resolution
features, frozen RGB prior. No point movement; no GT in forward.
"""
import torch
from torch import nn
import torch.nn.functional as F
from models.spatial_support_refiner import SpatialSupportRefiner,block


def neighbor(t,dy,dx):
    h,w=t.shape[-2:];pad=max(1,abs(dy),abs(dx));p=F.pad(t,(pad,pad,pad,pad))
    return p[...,pad+dy:pad+dy+h,pad+dx:pad+dx+w]


def direction_agreement_weight(unit,other):
    agreement=(1+(unit*other).sum(2)).clamp(0,2)*.5
    reliable=(torch.linalg.vector_norm(unit,dim=2)>.5)&(torch.linalg.vector_norm(other,dim=2)>.5)
    return torch.where(reliable,agreement,torch.ones_like(agreement))


class SpatialDirectionRefiner(SpatialSupportRefiner):
    def __init__(self,arm='D0',include_xyz=True,grid_step=1,support_bias=2.,direction_agreement=False):
        super().__init__();self.arm=arm
        if not isinstance(grid_step,int) or grid_step<1:raise ValueError('grid_step must be positive integer')
        self.grid_step=grid_step
        self.support_bias=float(support_bias)
        self.direction_agreement=bool(direction_agreement)
        self.include_xyz=include_xyz
        self.line_direction=nn.Sequential(block(22 if include_xyz else 19,16),nn.Conv2d(16,3,1),nn.Tanh())
        self.line_fuse=block(48,24)

    def forward(self,common,candidates,lines,return_gate=False,return_details=False,
                propagation='default',direction_override=None):
        if propagation not in ['default','none']:raise ValueError('Unknown propagation mode')
        if self.arm not in ['D0','DS','DC']:raise ValueError('Unknown direction arm')
        b,k,_,h,w=candidates.shape
        fine=self.image(common);context=F.interpolate(self.context(F.avg_pool2d(fine,2)),size=(h,w),mode='bilinear',align_corners=False)
        shared=self.decode(torch.cat([fine,context],1))[:,None].expand(-1,k,-1,-1,-1)
        features=self.candidate(torch.cat([shared,candidates],2).reshape(b*k,18,h,w))
        condition=candidates if self.arm=='DC' else candidates.mean(1,keepdim=True).expand_as(candidates)
        direction_inputs=[shared,condition,lines[:,:,4:5]]
        if self.include_xyz:direction_inputs.append(lines[:,:,1:4])
        d=self.line_direction(torch.cat(direction_inputs,2).reshape(b*k,22 if self.include_xyz else 19,h,w)).reshape(b,k,3,h,w)
        if direction_override is not None:
            if direction_override.shape!=d.shape:raise ValueError('Direction override shape mismatch')
        unit=F.normalize(d if direction_override is None else direction_override,dim=2);xyz=lines[0:,:,1:4];observed=lines[:,:,0];prob=lines[:,:,4]
        total=torch.zeros_like(features);denom=features.new_zeros((b*k,1,h,w))
        # Center included: nonzero weight whenever the center is observed.
        step=self.grid_step
        for dy,dx in [(0,0),(0,step),(0,-step),(step,0),(-step,0)]:
            offset=neighbor(xyz,dy,dx)-xyz
            spatial=torch.exp(-torch.linalg.vector_norm(offset,dim=2)/.02)
            cosine=(F.normalize(offset,dim=2)*unit).sum(2)
            weight=spatial if self.arm=='D0' else (spatial+torch.exp(prob-1)*cosine).clamp_min(0)
            if self.direction_agreement:
                other=neighbor(unit,dy,dx)
                weight=weight*direction_agreement_weight(unit,other)
            weight=(weight*observed*neighbor(observed,dy,dx)).reshape(b*k,1,h,w)
            total=total+weight*neighbor(features,dy,dx);denom=denom+weight
        propagated=total/denom.clamp_min(1e-12)
        if propagation=='none':propagated=features
        logits=self.score(self.line_fuse(torch.cat([features,propagated],1))).reshape(b,k,h,w)+self.support_bias*candidates[:,:,0]
        if return_details:return logits,d
        return (logits,torch.zeros_like(logits)) if return_gate else logits
