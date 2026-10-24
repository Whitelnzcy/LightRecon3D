"""PushBoundary Eq2-inspired operator; adaptation, not full paper reproduction."""
import torch
import torch.nn.functional as F


def direction_propagation(features, offsets, direction, boundary_probability, radius, alpha=1.):
    """features [...,J,C], offsets [...,J,3], direction [...,3], probability [...].

    XYZ offsets and positive radius MUST share units. Zero directions reduce to
    distance interpolation. Caller may add candidate axis for conditioned heads.
    No GT, learned weights or coordinate edits in this operator.
    """
    if radius<=0 or alpha<0:raise ValueError('Invalid propagation scale')
    if offsets.shape[-1]!=3 or direction.shape!=offsets.shape[:-2]+(3,):raise ValueError('Direction dimensions')
    if features.shape[:-1]!=offsets.shape[:-1]:raise ValueError('Neighbor dimensions')
    if boundary_probability.shape!=direction.shape[:-1]:raise ValueError('Boundary dimensions')
    spatial=torch.exp(-torch.linalg.vector_norm(offsets,dim=-1)/radius)
    cosine=(F.normalize(offsets,dim=-1)*F.normalize(direction,dim=-1)[...,None,:]).sum(-1)
    weights=(spatial+alpha*torch.exp(boundary_probability[...,None]-1)*cosine).clamp_min(0)
    # Explicit adaptation for an all-rejected neighborhood; stable spatial fallback.
    weights=torch.where(weights.sum(-1,keepdim=True)>1e-12,weights,spatial)
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    return (features*weights[...,None]).sum(-2),weights
