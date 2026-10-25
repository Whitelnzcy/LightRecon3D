"""GT-only diagnostic: does the 3D inward target fit image-grid propagation?

No training or predictions. Full registered pixels; offsets stay in fixed global
XYZ. Cross-view nearest anchors are not intrinsically wrong: report, don't infer.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from spatial_support_data import load_prediction_scene,attach_supervision
from spatial_direction_data import attach_direction_supervision
from audit_line_support_refinement import read_json,write_json,sha


def edge_diagnostic(registry,xyz,labels,target,boundary):
    rows=[]
    for dy,dx in [(0,1),(1,0)]:
        a=registry[:registry.shape[0]-dy or None,:registry.shape[1]-dx or None]
        b=registry[dy:,dx:]
        ok=(a>=0)&(b>=0);a=a[ok];b=b[ok]
        ok=(labels[a]>=0)&(labels[b]>=0);a=a[ok];b=b[ok]
        for i,j in [(a,b),(b,a)]:
            offset=xyz[j]-xyz[i];length=np.linalg.norm(offset,axis=1)
            unit=offset/np.maximum(length[:,None],1e-12)
            cosine=(target[i]*unit).sum(1)
            coherence=(target[i]*target[j]).sum(1)
            active=(np.linalg.norm(target[i],axis=1)>.5)&(length>1e-12)
            rows.append((i,j,cosine,coherence,active))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);threadpool_limits(1)
    roots=[Path('local_outputs/gct3/items')/f'train_{i:03d}_scene_{i:05d}' for i in range(3)]
    roots += [Path('local_outputs/gct4/items/train_003_scene_00003')]
    roots += [Path('local_outputs/b12r2/items')/n for n in ['final_016_scene_00199','final_008_scene_00190']]
    report={};gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    for root in roots:
        s=load_prediction_scene(root)
        if s['name'].startswith('train_'):attach_supervision(s)
        else:
            with np.load(gtroot/f'{s["name"]}_eval_only.npz') as z:
                np.testing.assert_array_equal(z['pixel_xy'],s['old']['pixel_xy'])
                np.testing.assert_array_equal(z['view_indices'],s['old']['source_views'])
                s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=read_json(gtroot/'report.json')[s['name']]['fixed_identity_map'])
        attach_direction_supervision(s,'inward')
        xyz=s['old']['points'];labels=s['truth'];views=s['old']['source_views']
        target=np.zeros_like(xyz);boundary=np.zeros(len(xyz),bool)
        for v in s['views']:
            r=v['registry'];ok=r>=0;target[r[ok]]=v['direction_target'][:,ok].T;boundary[r[ok]]=v['direction_boundary'][ok]
        eligible=(labels>=0)&(s['old']['point_plane_ids']>=0)&np.isin(labels,list(map(int,s['mapping'].values())))
        cross=[];distances=[];cosines={k:[] for k in ['same','cross']};coherences=[]
        best_same=np.full(len(xyz),-2.);best_cross=best_same.copy()
        for pid in np.unique(labels[labels>=0]):
            q=np.flatnonzero(boundary&(labels==pid)&eligible);anchors=np.flatnonzero(~boundary&(labels==pid))
            if not len(q) or not len(anchors):continue
            length,ix=cKDTree(xyz[anchors]).query(xyz[q]);cross.extend((views[q]!=views[anchors[ix]]).tolist());distances.extend(length.tolist())
        for v in s['views']:
            for i,j,c,coherence,active in edge_diagnostic(v['registry'],xyz,labels,target,boundary):
                ok=active&boundary[i]&eligible[i]
                same=labels[i]==labels[j]
                for key,mask in [('same',ok&same),('cross',ok&~same)]:
                    cosines[key].extend(c[mask].tolist())
                    np.maximum.at(best_same if key=='same' else best_cross,i[mask],c[mask])
                mask=ok&same&boundary[j]&(np.linalg.norm(target[j],axis=1)>.5)
                coherences.extend(coherence[mask].tolist())
        b=eligible&boundary;both=b&(best_same>-2)&(best_cross>-2)
        row=dict(eligible_boundary_points=int(b.sum()),near_2d_boundary_5px_fraction=float((s['distance'][b]<=5).mean()),
            cross_view_nearest_interior_fraction=float(np.mean(cross)),anchor_distance_median=float(np.median(distances)),
            anchor_distance_units='fixed predicted global XYZ, not metric calibrated',
            boundary_with_both_neighbor_types=int(both.sum()),cross_neighbor_more_aligned_fraction=float((best_cross[both]>best_same[both]).mean()) if both.any() else None,
            same_boundary_edge_count=len(coherences),same_boundary_target_cosine_median=float(np.median(coherences)),
            same_boundary_target_opposed_fraction=float((np.asarray(coherences)<0).mean()),
            target_offset_cosine={k:dict(edges=len(v),median=float(np.median(v)),positive_fraction=float((np.asarray(v)>0).mean())) for k,v in cosines.items()})
        assert sha(s['source'])==s['source_sha256'];report[s['name']]=row
        write_json(out/'report.json',dict(scope='GT-only target diagnostic, full domain, directed 4-neighbor edges; not model efficacy',scenes=report))
        print(s['name'],row,flush=True)


if __name__=='__main__':main()
