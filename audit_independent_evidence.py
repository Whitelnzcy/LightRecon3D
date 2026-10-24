"""Recompute raw-line provenance and DEV metrics, without selecting predictions."""
import argparse
from pathlib import Path
import numpy as np
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics
from recover_cache_projection import intersect_project
from models.line_support_patches import canonical_line


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); args=ap.parse_args()
    run=Path(args.run); report=read_json(run/'report.json')
    receipts=read_json(run/'prediction_inputs.json')['scenes']
    receipts += [read_json(run/f'{name}_prediction_inputs.json') for name,r in report.items() if r['split']=='historical_DEV']
    results=[]
    for r in receipts:
        assert sha(r['source'])==r['source_sha256']
        assert sha(r['lines_path'])==r['lines_sha256']
        evidence=read_json(r['lines_path'])
        assert sha(evidence['cache'])==evidence['cache_sha256']
        assert sha(evidence['checkpoint'])==evidence['checkpoint_sha256']
        with np.load(r['source']) as z:
            old={k:z[k] for k in ['plane_ids','plane_normals','plane_offsets','point_plane_ids','pixel_xy','source_views']}
        lookup={int(pid):i for i,pid in enumerate(old['plane_ids'])}; count=0
        assert len(r['views'])==len(evidence['views'])
        for view,ev in zip(r['views'],evidence['views']):
            assert view['transform']==ev['transform']; assert sha(ev['image'])==ev['image_sha256']
            p=np.asarray(view['projection_from_predicted_xyz'])
            for line in ev['raw_intersections']:
                a,b=[lookup[x] for x in line['pair']]
                expected=intersect_project(old['plane_normals'][a],old['plane_offsets'][a],old['plane_normals'][b],old['plane_offsets'][b],p)
                np.testing.assert_allclose(canonical_line(expected),canonical_line(line['coefficients']),atol=1e-6)
                count+=1
        name=r['scene']
        with np.load(run/f'{name}_predictions.npz') as z:
            np.testing.assert_array_equal(z['original_ids'],old['point_plane_ids'])
            np.testing.assert_array_equal(z['source_indices'],np.arange(len(z['original_ids'])))
            preds={a:z[a] for a in report[name]['metrics']}
        for pred in preds.values():
            assert np.isin(pred[pred>=0],old['plane_ids']).all()
            np.testing.assert_array_equal(pred[old['point_plane_ids']<0],old['point_plane_ids'][old['point_plane_ids']<0])
        if report[name]['split']=='historical_DEV':
            target=Path('local_outputs/dense_boundary_competition_20260918_v1')
            mapping=read_json(target/'report.json')[name]['fixed_identity_map']
            with np.load(target/f'{name}_eval_only.npz') as z:
                np.testing.assert_array_equal(z['pixel_xy'],old['pixel_xy'])
                np.testing.assert_array_equal(z['view_indices'],old['source_views'])
                before=convert(old['point_plane_ids'],mapping)
                for arm,pred in preds.items():
                    m=metrics(before,convert(pred,mapping),z['raw_dense_gt'],z['boundary_distance'])
                    m['changed']=int((pred!=old['point_plane_ids']).sum())
                    assert m==report[name]['metrics'][arm]
        results.append(dict(scene=name,raw_intersections_checked=count,rows_and_sources_checked=True,
                            dev_metrics_recomputed=report[name]['split']=='historical_DEV'))
    write_json(run/'independent_audit.json',dict(schema_version=1,scenes=results))
    print(results)


if __name__=='__main__': main()
