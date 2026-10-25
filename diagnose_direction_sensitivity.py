"""Frozen-model feature/logit sensitivity on fixed GT-free selected DEV patches."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from models.spatial_direction_refiner import SpatialDirectionRefiner
from spatial_support_data import load_prediction_scene,tile_boxes
from spatial_direction_data import DirectionInputs
from spatial_context_data import context_batch
from audit_line_support_refinement import write_json,read_json,sha


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--checkpoint',default='local_outputs/direction_control_20261024_v1/learned.pt');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    threadpool_limits(1);torch.set_num_threads(4)
    checkpoint=Path(args.checkpoint);saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
    model=SpatialDirectionRefiner('DS',grid_step=saved.get('grid_step',1),support_bias=saved.get('support_bias',2.),direction_agreement=saved.get('direction_agreement',False)).cuda().eval();model.load_state_dict(saved['model'])
    provider=DirectionInputs(out);results={};root=Path('local_outputs/dense_boundary_competition_20260918_v1')
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        s=load_prediction_scene(Path('local_outputs/b12r2/items')/name);provider.attach(s)
        collected=[]
        for v in s['views']:
            boxes=tile_boxes(v['registry'].shape)[:16]
            for start in range(0,len(boxes),4):
                batch=boxes[start:start+4];c,k,e=context_batch(v,batch,'cuda',0,False);capture=[]
                hook=model.line_fuse.register_forward_pre_hook(lambda m,i:capture.append(i[0]))
                base=model(c,k,e);hook.remove();f,p=capture[0].chunk(2,dim=1)
                zero=model(c,k,e,direction_override=torch.zeros(k.shape[0],k.shape[1],3,*k.shape[-2:],device='cuda'))
                identity=model(c,k,e,propagation='none')
                # No retraining: post-hoc bias sensitivity, not a selectable model.
                bias0=base-model.support_bias*k[:,:,0]
                feature_abs=(p-f).abs().mean(1).reshape(k.shape[0],k.shape[1],*k.shape[-2:]).mean(1)
                feature_scale=f.abs().mean(1).reshape(k.shape[0],k.shape[1],*k.shape[-2:]).mean(1)
                changes={tag:((other-base).abs().mean(1).cpu().numpy(),(other.argmax(1)!=base.argmax(1)).cpu().numpy())
                    for tag,other in [('zero_direction',zero),('identity_propagation',identity),('remove_bias',bias0)]}
                for j,box in enumerate(batch):
                    x0,y0,x1,y1=box;indices=v['registry'][y0:y1,x0:x1];ok=indices>=0
                    collected.append(dict(indices=indices[ok],feature_abs=feature_abs[j].cpu().numpy()[ok],feature_scale=feature_scale[j].cpu().numpy()[ok],
                        **{tag+'_logit':values[0][j][ok] for tag,values in changes.items()},
                        **{tag+'_flip':values[1][j][ok] for tag,values in changes.items()}))
        # GT read only after all model forwards; used for descriptive strata.
        data={key:np.concatenate([r[key] for r in collected]) for key in collected[0]}
        with np.load(root/f'{name}_eval_only.npz') as z:
            np.testing.assert_array_equal(z['pixel_xy'],s['old']['pixel_xy']);np.testing.assert_array_equal(z['view_indices'],s['old']['source_views'])
            valid=z['raw_dense_gt'][data['indices']]>=0;distance=z['boundary_distance'][data['indices']]
        rows={}
        for label,mask in [('all_valid',valid),('boundary_5px',valid&(distance<=5)),('interior',valid&(distance>5))]:
            row=dict(sample_occurrences=int(mask.sum()))
            for key,val in data.items():
                if key!='indices':row[key+'_mean']=float(val[mask].mean()) if mask.any() else None
            row['feature_relative_mean']=row['feature_abs_mean']/max(row['feature_scale_mean'],1e-12)
            rows[label]=row
        assert sha(s['source'])==s['source_sha256'];results[name]=rows;print(name,rows,flush=True)
        write_json(out/'results.json',dict(checkpoint_sha256=sha(checkpoint),scope='first16 GT-free tiles/view, overlapping occurrences not full-domain metrics; frozen interventions',results=results))


if __name__=='__main__':main()
