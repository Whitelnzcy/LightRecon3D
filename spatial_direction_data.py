"""Prediction-only XYZ/RGB-prior inputs; separately constructed TRAIN directions."""
from pathlib import Path
import numpy as np
import torch
from audit_line_support_refinement import read_json,sha,write_json
from audit_pushboundary_targets import boundary_targets,directions


class DirectionInputs:
    def __init__(self,output):self.output=Path(output)
    def attach(self,s):
        root=Path('local_outputs/independent_evidence_train4_dev2_20261024_v1')
        path=root/f'{s["name"]}_evidence_sources.json';ev=read_json(path)
        prior=read_json(root/('prediction_inputs.json'))['scenes'] if s['name'].startswith('train_') else [read_json(root/f'{s["name"]}_prediction_inputs.json')]
        source=next(r for r in prior if r['scene']==s['name'])
        assert source['source_sha256']==s['source_sha256']
        assert sha(ev['cache'])==ev['cache_sha256']
        xyz=s['old']['points'];center=xyz.mean(0);scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-6)
        with np.load(ev['cache']) as z:
            for v in s['views']:
                vi=v['transform']['alignment_view_index'];registry=v['registry'];ok=registry>=0
                recorded=next(t['transform'] for t in ev['views'] if t['view']==vi)
                assert recorded==v['transform']
                base=np.zeros((6,*registry.shape),np.float32)
                base[0]=ok; base[1:4,ok]=((xyz[registry[ok]]-center)/scale).T
                base[4]=z[f'view{vi}_boundary']
                v['lines']=np.broadcast_to(base,(len(s['old']['plane_ids']),*base.shape))
        receipt=self.output/f'{s["name"]}_direction_sources.json'
        write_json(receipt,dict(schema_version=1,source_sha256=s['source_sha256'],prior_receipt=str(path),prior_sha256=sha(path),
            prior_cache=ev['cache'],prior_cache_sha256=ev['cache_sha256'],center=center.tolist(),scale=scale,
            channels='observed, normalized_global_xyz3, frozen_RGB_boundary, reserved0',radius=.02))
        return receipt


def attach_direction_supervision(s):
    valid=s['truth']>=0;points=s['old']['points'][valid];labels=s['truth'][valid]
    boundary=boundary_targets(points,labels)
    target,ok=directions(points,points[boundary]);target[boundary]=0
    full=np.zeros((len(valid),3),np.float32);full[valid]=target
    for v in s['views']:
        registry=v['registry'];observed=registry>=0
        grid=np.zeros((3,*registry.shape),np.float32);grid[:,observed]=full[registry[observed]].T
        v['direction_target']=grid
    s['direction_supervision_summary']=dict(valid=int(valid.sum()),boundary=int(boundary.sum()),
        scope='TRAIN only: k4 GT instance transition in predicted globalXYZ; global nearest boundary, zero at boundary')


def direction_batch(v,boxes,device):
    return torch.from_numpy(np.stack([v['direction_target'][:,y0:y1,x0:x1] for x0,y0,x1,y1 in boxes])).to(device)


def direction_loss(prediction,target,ownership):
    # Correct candidate selection is supervision only; never used in forward.
    valid=ownership>=0
    if not valid.any():return prediction.sum()*0
    selected=prediction.gather(1,ownership.clamp_min(0)[:,None,None].expand(-1,1,3,-1,-1)).squeeze(1)
    return (selected-target).square().mean(1)[valid].mean()
