"""TRAIN-only frozen direction diagnostic; GT never changes saved assignments."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from spatial_support_data import load_prediction_scene,attach_supervision
from spatial_direction_data import DirectionInputs,attach_direction_supervision,direction_batch
from spatial_context_data import context_batch
from models.spatial_direction_refiner import SpatialDirectionRefiner
from audit_line_support_refinement import write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=Path(args.run)
    out=run/'direction_quality';out.mkdir(exist_ok=False);threadpool_limits(1);torch.set_num_threads(4)
    s=load_prediction_scene('local_outputs/gct3/items/train_000_scene_00000')
    DirectionInputs(out).attach(s);attach_supervision(s);attach_direction_supervision(s)
    results={}
    for arm in ['D0','DS','DC']:
        checkpoint=torch.load(run/f'{arm}.pt',map_location='cpu',weights_only=False)
        m=SpatialDirectionRefiner(arm).cuda().eval();m.load_state_dict(checkpoint['model'])
        angles=[];norms=[];sse=zero=0.;count=0
        with torch.inference_mode():
            for v in s['views']:
                # Fixed first16 GT-free proposals/view; no outcome-based crop selection.
                boxes=[p['box_xyxy'] for p in v['patches'][:16]]
                for start in range(0,len(boxes),4):
                    batch=boxes[start:start+4];c,k,e,target,_=context_batch(v,batch,'cuda')
                    _,d=m(c,k,e,return_details=True);truth=direction_batch(v,batch,'cuda')
                    selected=d.gather(1,target.clamp_min(0)[:,None,None].expand(-1,1,3,-1,-1)).squeeze(1)
                    valid=target>=0;eligible=valid&(torch.linalg.vector_norm(truth,dim=1)>.5)
                    sse+=float((selected-truth).square().mean(1)[valid].sum().cpu())
                    zero+=float(truth.square().mean(1)[valid].sum().cpu());count+=int(valid.sum())
                    cosine=torch.nn.functional.cosine_similarity(selected,truth,dim=1)
                    angles.extend(torch.rad2deg(cosine[eligible].clamp(-1,1).acos()).cpu().tolist())
                    norms.extend(torch.linalg.vector_norm(selected,dim=1)[eligible].cpu().tolist())
        results[arm]=dict(sample_occurrences=count,mse=sse/count,zero_predictor_mse=zero/count,
            nonzero_target_angle_median=float(np.median(angles)),predicted_norm_median=float(np.median(norms)))
    write_json(out/'report.json',dict(scope='TRAIN scene0,80 fixed patches; overlaps counted, no independent evaluation',results=results));print(results)


if __name__=='__main__':main()
