"""TRAIN-only gate supervision; does not change prediction inputs or geometry."""
import torch
import torch.nn.functional as F


def line_permission_loss(gate, availability, target, original):
    if gate.shape!=availability.shape or target.shape!=original.shape or gate.shape[0:1]+gate.shape[2:]!=target.shape:
        raise ValueError('Gate/target shape mismatch')
    k=gate.shape[1];columns=torch.arange(k,device=gate.device)[None,:,None,None]
    valid=(target>=0)&(target<k)&(original>=0)&(original<k)
    mask=(availability>0)&valid[:,None]&(columns!=original[:,None])
    positive=mask&(columns==target[:,None]);negative=mask&~positive
    terms=[]
    # Balance positive/negative available candidate-pixels. Unmatched GT and
    # absent lines contribute no loss; no distance-to-GT threshold is used.
    if positive.any():terms.append(F.binary_cross_entropy(gate[positive].clamp(1e-6,1-1e-6),torch.ones_like(gate[positive])))
    if negative.any():terms.append(F.binary_cross_entropy(gate[negative].clamp(1e-6,1-1e-6),torch.zeros_like(gate[negative])))
    return torch.stack(terms).mean() if terms else gate.sum()*0
