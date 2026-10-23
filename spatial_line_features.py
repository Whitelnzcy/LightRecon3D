"""Predicted plane-pair-conditioned line coordinates; no GT arguments."""
import numpy as np
from models.line_support_patches import canonical_line,line_coordinates
from audit_line_support_refinement import read_json
from pathlib import Path


def pair_features(labels,candidate_ids,lines):
    h,w=labels.shape;yy,xx=np.indices((h,w));xy=np.stack([xx,yy],-1)
    features=np.zeros((len(candidate_ids),6,h,w),np.float32)
    lookup={int(pid):k for k,pid in enumerate(candidate_ids)}
    seen=set()
    for row in lines:
        a,b=map(int,row['pair']);key=tuple(sorted([a,b]))
        if a==b or key in seen:raise ValueError('Ambiguous/duplicate candidate pair')
        seen.add(key)
        if a not in lookup or b not in lookup:raise ValueError('Unknown candidate in line pair')
        if not np.any(labels==a) or not np.any(labels==b):continue
        line=canonical_line(row['after']);coords=line_coordinates(xy,line)
        along,signed=coords[...,0],coords[...,1]
        fa=float(np.mean(signed[labels==a]>=0));fb=float(np.mean(signed[labels==b]>=0))
        for source,dest,fraction in [(a,b,fb),(b,a,fa)]:
            selected=labels==source;column=lookup[dest];orientation=1 if fraction>=.5 else -1
            values=[np.ones((h,w)),np.tanh(signed*orientation/16),np.exp(-np.abs(signed)/16),
                    along/np.hypot(h,w),np.full((h,w),abs(fa-fb)),np.full((h,w),row.get('score_after',0))]
            for ch,value in enumerate(values):features[column,ch,selected]=value[selected]
    return features


def attach_predicted_lines(scene):
    """Use cached train line predictions or frozen DEV prediction provenance."""
    name=scene['name']
    if name.startswith('train_'):
        path=Path('local_outputs/intersection_prior_refine20_v1')/f'{name}_predictions.json'
        rows={int(k):v for k,v in read_json(path)['views'].items()}
    else:
        path=Path('local_outputs/intersection_frozen_dev2_v1')/f'{name}_provenance.json'
        rows={int(v['view']):v['lines'] for v in read_json(path)['views']}
    for v in scene['views']:
        labels=np.full(v['registry'].shape,-1,int);ok=v['registry']>=0
        labels[ok]=scene['old']['point_plane_ids'][v['registry'][ok]]
        v['lines']=pair_features(labels,scene['old']['plane_ids'],rows[int(v['transform']['alignment_view_index'])])
    return path


def line_batch(view,boxes,device):
    import torch
    return torch.from_numpy(np.stack([view['lines'][:,:,y0:y1,x0:x1] for x0,y0,x1,y1 in boxes])).to(device)
