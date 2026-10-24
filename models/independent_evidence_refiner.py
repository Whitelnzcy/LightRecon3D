"""Four input ablations with identical active layers and observed-pixel scope."""
import torch
import torch.nn.functional as F
from models.spatial_line_refiner import SpatialLineRefiner


class IndependentEvidenceRefiner(SpatialLineRefiner):
    def __init__(self,arm='N'):
        super().__init__('MC'); self.arm=arm

    def evidence_channels(self,common,candidates,evidence):
        if self.arm not in ['N','B','G','BG']: raise ValueError('Unknown arm')
        base=torch.cat([common[:,None].expand(-1,candidates.shape[1],-1,-1,-1),candidates],2)
        mask=evidence.new_tensor([int('G' in self.arm)]*5+[int('B' in self.arm)])[None,None,:,None,None]
        return base+evidence*mask

    def forward(self,common,candidates,lines,return_gate=False):
        b,k,_,h,w=candidates.shape; fine=self.image(common)
        context=F.interpolate(self.context(F.avg_pool2d(fine,2)),size=(h,w),mode='bilinear',align_corners=False)
        shared=self.decode(torch.cat([fine,context],1))[:,None].expand(-1,k,-1,-1,-1)
        base=torch.cat([shared,candidates],2).reshape(b*k,18,h,w)
        logits=self.score(self.candidate(base)).reshape(b,k,h,w)+2*candidates[:,:,0]
        extra=self.line_branch(torch.cat([base,self.evidence_channels(common,candidates,lines).reshape(b*k,6,h,w)],1))
        gate=self.line_gate(extra).sigmoid().reshape(b,k,h,w)*(common[:,3:4]>0)
        result=logits+gate*self.line_score(extra).reshape(b,k,h,w)
        return (result,gate) if return_gate else result
