"""Post-run receipts and damage localization; never selects or changes outputs."""
import argparse
from pathlib import Path
import numpy as np
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    run=Path(args.run);reports=read_json(run/'report.json');receipts=read_json(run/'prediction_inputs.json')['scenes']
    result=dict(schema_version=1,training_line_provenance=[],development={})
    old_protocol=read_json('local_outputs/intersection_finite_loo4_v1/protocol.json')
    historical_hashes={str(Path(r['source']).resolve()):r['sha256'] for r in old_protocol['inputs']}
    for r in receipts:
        line=read_json(r['lines_path']);proposal_path=Path(line['source']);proposal=read_json(proposal_path)
        assert Path(proposal['source']).resolve()==Path(r['source']).resolve()
        assert proposal['coordinate']=='aligned pointmap xy integer centers; n.X+d=0; cam_to_world'
        assert sha(r['source'])==r['source_sha256']==historical_hashes[str(Path(r['source']).resolve())]
        assert sha(r['lines_path'])==r['lines_sha256']
        assert set(line['views'])=={str(v['transform']['alignment_view_index']) for v in r['views']}
        checked=0
        for view,rows in line['views'].items():
            lookup={tuple(row['pair']):row for row in proposal['views'][view]}
            for row in rows:
                np.testing.assert_allclose(row['before'],lookup[tuple(row['pair'])]['line'])
                checked+=1
        result['training_line_provenance'].append(dict(scene=r['scene'],verified_line_pairs=checked,
            proposal_sha256=sha(proposal_path),coordinate=proposal['coordinate']))
    targets=Path('local_outputs/dense_boundary_competition_20260918_v1')
    frozen=read_json(targets/'report.json')
    for item,row in reports.items():
        if row['split']!='historical_DEV':continue
        mapping=frozen[item]['fixed_identity_map']
        with np.load(targets/f'{item}_eval_only.npz') as z:truth=z['raw_dense_gt'];distance=z['boundary_distance']
        with np.load(run/f'{item}_predictions.npz') as z:
            original=z['original_ids'];predictions={a:z[a] for a in ['M1','M2','M3']}
            assert np.array_equal(z['source_indices'],np.arange(len(original)))
        with np.load(Path('local_outputs/intersection_frozen_dev2_v1')/f'{item}_predictions.npz') as z:
            assert np.array_equal(z['original_ids'],original);previous=z['finite_support']
        before=convert(original,mapping);valid=truth>=0;summary={}
        for arm,pred in predictions.items():
            after=convert(pred,mapping);m=metrics(before,after,truth,distance);m['changed']=int((pred!=original).sum())
            assert m==row['metrics'][arm]
            assert np.array_equal(pred[original<0],original[original<0])
            damaged=valid&(before==truth)&(after!=truth)
            bins={key:int((damaged&mask).sum()) for key,mask in {
                'within_5px':distance<=5,'5_to_20px':(distance>5)&(distance<=20),'beyond_20px':distance>20}.items()}
            transitions,count=np.unique(np.stack([original[damaged],pred[damaged]],axis=1),axis=0,return_counts=True)
            top=np.argsort(-count)[:5]
            summary[arm]=dict(recomputed_exact=True,damage_distance=bins,
                              top_damage_pairs=[dict(source=int(transitions[k,0]),destination=int(transitions[k,1]),count=int(count[k])) for k in top])
        old=convert(previous,mapping);new=convert(predictions['M3'],mapping)
        summary['M3_vs_historical_finite']=dict(newly_correct=int((valid&(old!=truth)&(new==truth)).sum()),
            newly_wrong=int((valid&(old==truth)&(new!=truth)).sum()))
        result['development'][item]=summary
    write_json(out,result);print(result,flush=True)


if __name__=='__main__':main()
