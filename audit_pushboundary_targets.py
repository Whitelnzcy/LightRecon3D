"""Evaluation-only adaptation of boundary direction targets, NOT inference."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from audit_line_support_refinement import read_json,write_json,sha,convert


def directions(points,anchors):
    if not len(anchors):return np.zeros_like(points),np.zeros(len(points),bool)
    _,index=cKDTree(anchors).query(points)
    delta=points-anchors[index];norm=np.linalg.norm(delta,axis=1)
    return delta/np.maximum(norm[:,None],1e-12),norm>1e-12


def boundary_targets(points,labels):
    if len(points)<4:raise ValueError('At least4 points required')
    _,neighbors=cKDTree(points).query(points,k=4)
    return np.any(labels[neighbors]!=labels[:,None],axis=1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    base=Path('local_outputs/independent_evidence_train4_dev2_20261024_v1')
    gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1');frozen=read_json(gtroot/'report.json')
    report={}
    write_json(out/'protocol.json',dict(schema_version=1,scope='GT supervision feasibility only; NOT reconstruction results',
        baseline='paper-intent unit vector from global nearest boundary; boundary points zero',
        variant='nearest boundary restricted to same GT plane identity; boundary points still zero',
        neighbors=4,sampling='all valid full-domain points, no subsampling',
        caveats='3D GT labels attached to predicted XYZ; duplicate multiview samples affect kNN; no semantic labels used; not original benchmark reproduction'))
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        receipt=read_json(base/f'{name}_prediction_inputs.json')
        assert sha(receipt['source'])==receipt['source_sha256']
        with np.load(receipt['source']) as z:
            xyz=z['points'];original=z['point_plane_ids'];xy=z['pixel_xy'];views=z['source_views']
        with np.load(gtroot/f'{name}_eval_only.npz') as z:
            truth=z['raw_dense_gt'];distance=z['boundary_distance']
            np.testing.assert_array_equal(xy,z['pixel_xy']);np.testing.assert_array_equal(views,z['view_indices'])
        valid=truth>=0;points=xyz[valid];labels=truth[valid];boundary=boundary_targets(points,labels)
        shared,shared_valid=directions(points,points[boundary]);specific=np.zeros_like(shared);specific_valid=np.zeros(len(points),bool)
        _,nearest=cKDTree(points[boundary]).query(points)
        nearest_ids=labels[boundary][nearest]
        for pid in np.unique(labels):
            select=labels==pid
            specific[select],specific_valid[select]=directions(points[select],points[boundary&(labels==pid)])
        shared[boundary]=0;specific[boundary]=0
        eligible=~boundary&shared_valid&specific_valid
        cosine=np.clip((shared*specific).sum(1),-1,1);angle=np.degrees(np.arccos(cosine))
        mapping=frozen[name]['fixed_identity_map'];wrong=convert(original,mapping)[valid]!=labels
        represented=np.isin(labels,list(map(int,mapping.values())))
        editable=original[valid]>=0
        groups={}
        for label,mask in dict(all=eligible,near_boundary=eligible&(distance[valid]<=5),
                               wrong_near_boundary=eligible&wrong&(distance[valid]<=5)).items():
            groups[label]=dict(count=int(mask.sum()),nearest_boundary_other_plane=int((mask&(nearest_ids!=labels)).sum()),
                direction_difference_over30deg=int((mask&(angle>30)).sum()),
                angle_median=float(np.median(angle[mask])) if mask.any() else None)
        report[name]=dict(valid_points=len(points),boundary_points=int(boundary.sum()),
            wrong_points=int(wrong.sum()),wrong_represented_editable=int((wrong&represented&editable).sum()),
            missing_candidate_wrong=int((wrong&~represented).sum()),groups=groups,
            source_sha256=receipt['source_sha256'])
        np.savez_compressed(out/f'{name}_supervision_diagnostic.npz',schema_version=1,
            source_indices=np.flatnonzero(valid),boundary=boundary,eligible=eligible,
            direction_angle_degrees=angle,nearest_boundary_other_plane=nearest_ids!=labels)
        write_json(out/'report.json',report);print(name,report[name],flush=True)


if __name__=='__main__':main()
