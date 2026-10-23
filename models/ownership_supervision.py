"""Scene-level identity matching for training/evaluation only, never inference."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def supervision_only(original_labels, dense_gt, candidate_ids):
    """Raw GT IDs may include zero; missing candidate IDs are not forced to a plane.

    Match once over whole scene and reuse for each view, never independently
    rematch a minibatch. This function must not run inside inference.
    """
    valid=dense_gt>=0;gids=np.unique(dense_gt[valid])
    overlap=np.array([[np.sum((original_labels==p)&(dense_gt==g)) for g in gids] for p in candidate_ids])
    if not len(gids) or not len(candidate_ids):raise ValueError('No valid matching identities')
    rows,cols=linear_sum_assignment(-overlap)
    target=np.full(len(dense_gt),-100,np.int64);mapping={}
    for r,k in zip(rows,cols):
        if overlap[r,k]<=0:continue
        mapping[int(candidate_ids[r])]=int(gids[k]);target[dense_gt==gids[k]]=int(candidate_ids[r])
    editable=original_labels>=0
    return dict(target_plane_id=target,mapping=mapping,valid=valid,editable=editable,
                missing_candidate=valid&(target==-100),
                keep_target=valid&(target==original_labels))
