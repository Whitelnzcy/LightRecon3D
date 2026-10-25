"""GT diagnostic: does the fixed oracle operator suppress cross-plane edges?"""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from models.line_support_patches import point_index_grid
from audit_pushboundary_targets import boundary_targets,directions
from audit_line_support_refinement import read_json,write_json,sha
from spatial_direction_data import extend_boundary_directions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--boundary-target',choices=['zero','inward'],default='zero');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);threadpool_limits(1)
    base=Path('local_outputs/independent_evidence_train4_dev2_20261024_v1');gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1');report={}
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        receipt=read_json(base/f'{name}_prediction_inputs.json');assert sha(receipt['source'])==receipt['source_sha256']
        with np.load(receipt['source']) as z:xyz=z['points'];xy=z['pixel_xy'];views=z['source_views']
        with np.load(gtroot/f'{name}_eval_only.npz') as z:
            truth=z['raw_dense_gt'];np.testing.assert_array_equal(xy,z['pixel_xy']);np.testing.assert_array_equal(views,z['view_indices'])
        valid=truth>=0;boundary=boundary_targets(xyz[valid],truth[valid]);target=np.zeros_like(xyz)
        target[valid],_=directions(xyz[valid],xyz[valid][boundary]);target[np.flatnonzero(valid)[boundary]]=0
        if args.boundary_target=='inward':target[valid],_=extend_boundary_directions(xyz[valid],truth[valid],boundary,target[valid])
        scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-6);records=[]
        ev=read_json(base/f'{name}_evidence_sources.json');assert sha(ev['cache'])==ev['cache_sha256']
        with np.load(ev['cache']) as prior:
            for v in receipt['views']:
                t=v['transform'];vi=t['alignment_view_index'];ix=np.flatnonzero(views==vi)
                grid=point_index_grid(xy[ix],ix,t['pointmap_hw']);p=prior[f'view{vi}_boundary']
                for axis in [0,1]:
                    left=grid[:-1,:] if axis==0 else grid[:,:-1];right=grid[1:,:] if axis==0 else grid[:,1:]
                    pl=p[:-1,:] if axis==0 else p[:,:-1];pr=p[1:,:] if axis==0 else p[:,1:]
                    observed=(left>=0)&(right>=0);a=left[observed];b=right[observed];pa=pl[observed];pb=pr[observed]
                    ok=(truth[a]>=0)&(truth[b]>=0);a=a[ok];b=b[ok];pa=pa[ok];pb=pb[ok]
                    for src,dst,prob in [(a,b,pa),(b,a,pb)]:
                        offset=(xyz[dst]-xyz[src])/scale;norm=np.linalg.norm(offset,axis=1);unit=offset/np.maximum(norm[:,None],1e-12)
                        spatial=np.exp(-norm/.02);oracle=np.maximum(0,spatial+np.exp(prob-1)*(unit*target[src]).sum(1))
                        both=(np.linalg.norm(target[src],axis=1)>.5)&(np.linalg.norm(target[dst],axis=1)>.5)
                        consistency=np.where(both,np.clip((1+(target[src]*target[dst]).sum(1))*.5,0,1),1)
                        records.append(dict(cross=truth[src]!=truth[dst],spatial=spatial,oracle=oracle,agreement=oracle*consistency,
                            direction_dot=(target[src]*target[dst]).sum(1),both=both))
        values={k:np.concatenate([r[k] for r in records]) for k in records[0]};cross=values['cross'];row={}
        for mode in ['spatial','oracle','agreement']:
            w=values[mode];row[mode]=dict(cross_mass_fraction=float(w[cross].sum()/max(w.sum(),1e-12)),
                cross_mean_weight=float(w[cross].mean()),same_mean_weight=float(w[~cross].mean()))
        row['edge_counts']=dict(directed_total=len(cross),cross=int(cross.sum()),
            cross_both_nonzero=int((cross&values['both']).sum()),
            cross_opposing_directions=int((cross&values['both']&(values['direction_dot']<0)).sum()))
        report[name]=row;write_json(out/'report.json',dict(boundary_target=args.boundary_target,scope='GT diagnostic raw unnormalized weights on valid directed grid edges; excludes self; not accuracy or deployable performance',scenes=report));print(name,row,flush=True)


if __name__=='__main__':main()
