"""Frozen direction quality, explicitly stratified by GT boundary/interior."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from models.spatial_direction_refiner import SpatialDirectionRefiner
from spatial_support_data import load_prediction_scene,attach_supervision,tile_boxes
from spatial_direction_data import DirectionInputs,extend_boundary_directions,registered_grid_boundaries
from spatial_context_data import context_batch
from audit_pushboundary_targets import boundary_targets,directions
from audit_line_support_refinement import read_json,write_json,sha


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--represented-only',action='store_true')
    ap.add_argument('--inward-checkpoint',default='local_outputs/direction_inward_learned_20261025_s42/learned.pt')
    ap.add_argument('--target-boundary',choices=['knn','grid'],default='knn');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);threadpool_limits(1);torch.set_num_threads(4)
    checkpoints={'zero':'local_outputs/direction_budget_20261024_s42/learned800.pt',
                 'inward':args.inward_checkpoint}
    models={}
    for a,path in checkpoints.items():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=SpatialDirectionRefiner('DS',grid_step=saved.get('grid_step',1),support_bias=saved.get('support_bias',2.),direction_agreement=saved.get('direction_agreement',False)).cuda().eval()
        model.load_state_dict(saved['model']);models[a]=model
    roots=[Path('local_outputs/gct3/items/train_000_scene_00000')]+[Path('local_outputs/b12r2/items')/n for n in ['final_016_scene_00199','final_008_scene_00190']]
    provider=DirectionInputs(out);report={};gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    for root in roots:
        s=load_prediction_scene(root);provider.attach(s);indices=[];predictions={a:[] for a in models}
        for v in s['views']:
            boxes=tile_boxes(v['registry'].shape)[:16]
            for start in range(0,len(boxes),4):
                batch=boxes[start:start+4];c,k,e=context_batch(v,batch,'cuda',0,False)
                ds={}
                for a,m in models.items():_,d=m(c,k,e,return_details=True);ds[a]=d[:,0].cpu().numpy()
                for j,(x0,y0,x1,y1) in enumerate(batch):
                    registry=v['registry'][y0:y1,x0:x1];ok=registry>=0;indices.append(registry[ok])
                    for a in models:predictions[a].append(ds[a][j].transpose(1,2,0)[ok])
        ix=np.concatenate(indices);predictions={a:np.concatenate(values) for a,values in predictions.items()}
        np.savez_compressed(out/f'{s["name"]}_before_GT.npz',schema_version=1,indices=ix,**predictions)
        if s['name'].startswith('train_'):attach_supervision(s)
        else:
            with np.load(gtroot/f'{s["name"]}_eval_only.npz') as z:
                np.testing.assert_array_equal(z['pixel_xy'],s['old']['pixel_xy']);np.testing.assert_array_equal(z['view_indices'],s['old']['source_views'])
                s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=read_json(gtroot/'report.json')[s['name']]['fixed_identity_map'])
        valid=s['truth']>=0;xyz=s['old']['points'][valid];labels=s['truth'][valid]
        boundary=(registered_grid_boundaries([v['registry'] for v in s['views']],s['truth'])[valid] if args.target_boundary=='grid' else boundary_targets(xyz,labels))
        target,_=directions(xyz,xyz[boundary]);target[boundary]=0;target,filled=extend_boundary_directions(xyz,labels,boundary,target)
        full=np.zeros_like(s['old']['points']);full[valid]=target;is_boundary=np.zeros(len(valid),bool);is_boundary[valid]=boundary
        target=full[ix];eligible=valid[ix]&(np.linalg.norm(target,axis=1)>.5);rows={}
        if args.represented_only:
            represented=np.isin(s['truth'],list(map(int,s['mapping'].values())))&(s['old']['point_plane_ids']>=0)
            eligible &= represented[ix]
        for a,p in predictions.items():
            norm=np.linalg.norm(p,axis=1);unit=p/np.maximum(norm[:,None],1e-12);angle=np.degrees(np.arccos(np.clip((unit*target).sum(1),-1,1)))
            rows[a]={}
            for name,mask in [('boundary',eligible&is_boundary[ix]),('interior',eligible&~is_boundary[ix]),('near_boundary_5px',eligible&(s['distance'][ix]<=5))]:
                rows[a][name]=dict(occurrences=int(mask.sum()),median_angle=float(np.median(angle[mask])) if mask.any() else None,
                    mean_norm=float(norm[mask].mean()) if mask.any() else None,
                    fraction_within30deg=float((angle[mask]<=30).mean()) if mask.any() else None,
                    mse_to_inward_target=float(((p[mask]-target[mask])**2).mean()) if mask.any() else None)
        assert sha(s['source'])==s['source_sha256'];report[s['name']]=rows
        write_json(out/'report.json',dict(scope='first16 GT-free tiles/view, overlapping occurrences; both evaluated against completed inward field; no GT in model forward',represented_editable_only=args.represented_only,
            target_boundary=args.target_boundary,checkpoint_hashes={a:sha(p) for a,p in checkpoints.items()},scenes=report));print(s['name'],rows,flush=True)


if __name__=='__main__':main()
