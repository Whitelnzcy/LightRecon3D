"""Verify frozen direction pilot receipts and full-domain DEV predictions."""
import argparse
from pathlib import Path
import numpy as np
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=Path(args.run)
    report=read_json(run/'report.json');receipts=read_json(run/'prediction_inputs.json')['scenes']
    receipts += [read_json(run/f'{n}_prediction_inputs.json') for n,r in report.items() if r['split']=='historical_DEV']
    output={};gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    for r in receipts:
        name=r['scene'];assert sha(r['source'])==r['source_sha256'];assert sha(r['lines_path'])==r['lines_sha256']
        e=read_json(r['lines_path']);assert e['source_sha256']==r['source_sha256']
        assert sha(e['prior_receipt'])==e['prior_sha256'];assert sha(e['prior_cache'])==e['prior_cache_sha256']
        with np.load(r['source']) as z:old={k:z[k] for k in ['point_plane_ids','plane_ids','pixel_xy','source_views','points']}
        np.testing.assert_allclose(e['center'],old['points'].mean(0))
        np.testing.assert_allclose(e['scale'],np.linalg.norm(np.ptp(old['points'],axis=0)))
        with np.load(run/f'{name}_predictions.npz') as z:
            np.testing.assert_array_equal(z['original_ids'],old['point_plane_ids'])
            np.testing.assert_array_equal(z['source_indices'],np.arange(len(z['original_ids'])))
            pred={a:z[a] for a in report[name]['metrics']}
        for p in pred.values():
            assert np.isin(p[p>=0],old['plane_ids']).all()
            np.testing.assert_array_equal(p[old['point_plane_ids']<0],old['point_plane_ids'][old['point_plane_ids']<0])
        if report[name]['split']=='historical_DEV':
            mapping=read_json(gtroot/'report.json')[name]['fixed_identity_map']
            with np.load(gtroot/f'{name}_eval_only.npz') as z:
                np.testing.assert_array_equal(z['pixel_xy'],old['pixel_xy']);np.testing.assert_array_equal(z['view_indices'],old['source_views'])
                before=convert(old['point_plane_ids'],mapping)
                for arm,p in pred.items():
                    m=metrics(before,convert(p,mapping),z['raw_dense_gt'],z['boundary_distance'])
                    m['changed']=int((p!=old['point_plane_ids']).sum());assert m==report[name]['metrics'][arm]
        output[name]=dict(source_mapping_verified=True,unchanged_unassigned=True,dev_metrics_recomputed=report[name]['split']=='historical_DEV')
    profile=read_json(run/'profile.json')
    assert all(any(v>0 for n,v in profile['gate_gradients_step3'][a].items() if n.startswith('line_direction')) for a in ['D0','DS','DC'])
    write_json(run/'audit.json',output);print(output)


if __name__=='__main__':main()
