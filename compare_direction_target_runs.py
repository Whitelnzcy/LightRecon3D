"""Compare saved GT-free predictions for zero/inward x none/learned controls."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import read_json,write_json,convert,metrics
from spatial_support_data import load_prediction_scene
from train_spatial_line_ablation import draw_comparison


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--baseline',default='local_outputs/direction_budget_20261024_s42');args=ap.parse_args()
    run=Path(args.run);out=run/'target_comparison';out.mkdir(exist_ok=False);threadpool_limits(1)
    old=Path(args.baseline);gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    current_protocol=read_json(run/'protocol.json');old_protocol=read_json(old/'protocol.json')
    assert current_protocol['seed']==old_protocol['seed']
    assert current_protocol['steps']==old_protocol['steps']==800
    assert current_protocol['boundary_target']=='inward' and old_protocol.get('boundary_target','zero')=='zero'
    old_keys={'zero_none':'none800','zero_learned':'learned800'} if 'none800' in old_protocol['arms'] else {'zero_none':'none','zero_learned':'learned'}
    report={}
    for name in read_json(run/'interventions.json'):
        s=load_prediction_scene(Path('local_outputs/b12r2/items')/name);pred={}
        for path,keys in [(old,old_keys),(run,{'inward_none':'none','inward_learned':'learned'})]:
            assert read_json(path/f'{name}_inputs.json')['source_sha256']==s['source_sha256']
            with np.load(path/f'{name}_before_GT.npz') as z:
                pred.update({tag:z[key].copy() for tag,key in keys.items()})
        with np.load(gtroot/f'{name}_eval_only.npz') as z:
            np.testing.assert_array_equal(z['pixel_xy'],s['old']['pixel_xy']);np.testing.assert_array_equal(z['view_indices'],s['old']['source_views'])
            s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=read_json(gtroot/'report.json')[name]['fixed_identity_map'])
        before=convert(s['old']['point_plane_ids'],s['mapping'])
        report[name]={tag:metrics(before,convert(p,s['mapping']),s['truth'],s['distance']) for tag,p in pred.items()}
        draw_comparison(s,pred,out/f'{name}.png',f'800steps, seed{current_protocol["seed"]}; all predictions saved before GT access',
            'Zero/inward: TRAIN direction target. None/learned: propagation. Same XYZ, candidates, budget and view.')
    aggregate={a:{key:sum(r[a][key] for r in report.values()) for key in next(iter(report.values()))[a]} for a in next(iter(report.values()))}
    write_json(out/'report.json',dict(scope='historicalDEV2, single seed, no independent-test claim',scenes=report,aggregate=aggregate))
    print(aggregate)


if __name__=='__main__':main()
