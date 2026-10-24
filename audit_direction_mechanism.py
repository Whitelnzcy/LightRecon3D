"""Independent metric and input-invariance check for GT diagnostic artifacts."""
import argparse
from pathlib import Path
import numpy as np
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=Path(args.run)
    report=read_json(run/'interventions.json');gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1');checked={}
    for name,row in report.items():
        r=read_json(run/f'{name}_prediction_inputs.json');assert sha(r['source'])==r['source_sha256']
        with np.load(r['source']) as z:original=z['point_plane_ids'];ids=z['plane_ids'];xy=z['pixel_xy'];views=z['source_views']
        with np.load(gtroot/f'{name}_eval_only.npz') as z:
            truth=z['raw_dense_gt'];dist=z['boundary_distance'];np.testing.assert_array_equal(z['pixel_xy'],xy);np.testing.assert_array_equal(z['view_indices'],views)
        mapping=read_json(gtroot/'report.json')[name]['fixed_identity_map'];before=convert(original,mapping)
        with np.load(run/f'{name}_GT_DIAGNOSTIC.npz') as z:
            np.testing.assert_array_equal(z['original_ids'],original);np.testing.assert_array_equal(z['source_indices'],np.arange(len(original)))
            for key,value in row.items():
                p=z[key];assert metrics(before,convert(p,mapping),truth,dist)==value
                np.testing.assert_array_equal(p[original<0],original[original<0]);assert np.isin(p[p>=0],ids).all()
        checked[name]=dict(full_domain_metrics_recomputed=len(row),source_and_indices_verified=True)
    write_json(run/'audit.json',checked);print(checked)


if __name__=='__main__':main()
