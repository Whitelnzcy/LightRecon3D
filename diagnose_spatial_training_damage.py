"""TRAIN-only diagnosis of edits; no thresholds are selected and no outputs edited."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics
from spatial_support_data import load_prediction_scene,attach_supervision
from spatial_line_features import attach_predicted_lines


def summarize_edits(scene,pred):
    old=scene['old'];original=old['point_plane_ids'];truth=scene['truth']
    before=convert(original,scene['mapping']);after=convert(pred,scene['mapping'])
    valid=truth>=0;corrected=valid&(before!=truth)&(after==truth);damaged=valid&(before==truth)&(after!=truth)
    covered=np.zeros(len(pred),bool);available=np.zeros(len(pred),bool)
    agrees=np.zeros(len(pred),bool);far_line=np.zeros(len(pred),bool)
    old_residual=np.zeros(len(pred),np.float32);new_residual=np.zeros_like(old_residual)
    for v in scene['views']:
        reg=v['registry'];ok=reg>=0;index=reg[ok]
        for patch in v['patches']:
            x0,y0,x1,y1=patch['box_xyxy'];indices=reg[y0:y1,x0:x1];covered[indices[indices>=0]]=True
        for k,pid in enumerate(old['plane_ids']):
            selected=pred[index]==pid;selected_indices=index[selected]
            available[selected_indices]=v['lines'][k,0][ok][selected]>0
            agrees[selected_indices]=v['lines'][k,1][ok][selected]>=0
            far_line[selected_indices]=v['lines'][k,2][ok][selected]<np.exp(-20/16)
            new_residual[selected_indices]=v['candidates'][k,1][ok][selected]
            source=original[index]==pid;old_residual[index[source]]=v['candidates'][k,1][ok][source]
    result={}
    for label,mask in [('corrected',corrected),('damaged',damaged)]:
        active=mask&available
        result[label]=dict(total=int(mask.sum()),within_5px_gt=int((mask&(scene['distance']<=5)).sum()),
            gt_5_to_20px=int((mask&(scene['distance']>5)&(scene['distance']<=20)).sum()),
            beyond_20px_gt=int((mask&(scene['distance']>20)).sum()),covered_by_proposed_train_centers=int((mask&covered).sum()),
            destination_line_available=int(active.sum()),line_side_agrees_with_destination=int((active&agrees).sum()),
            beyond_20px_predicted_line=int((active&far_line).sum()),
            new_plane_residual_greater=int((mask&(new_residual>old_residual)).sum()))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    threadpool_limits(1);cv2.setNumThreads(1);report=read_json(run/'report.json');rows={}
    receipts=read_json(run/'prediction_inputs.json')['scenes']
    for r in receipts:
        assert report[r['scene']]['split']=='TRAIN_fit'
        assert sha(r['source'])==r['source_sha256'] and sha(r['lines_path'])==r['lines_sha256']
        s=load_prediction_scene(Path(r['source']).parent.parent);attach_predicted_lines(s);attach_supervision(s)
        before=convert(s['old']['point_plane_ids'],s['mapping']);arms={}
        with np.load(run/f'{s["name"]}_predictions.npz') as z:
            assert np.array_equal(z['source_indices'],np.arange(len(before)))
            assert np.array_equal(z['original_ids'],s['old']['point_plane_ids'])
            for arm in report[s['name']]['metrics']:
                m=metrics(before,convert(z[arm],s['mapping']),s['truth'],s['distance'])
                m['changed']=int((z[arm]!=z['original_ids']).sum())
                assert m==report[s['name']]['metrics'][arm]
                arms[arm]=summarize_edits(s,z[arm])
        rows[s['name']]=arms;print('TRAIN_DAMAGE',s['name'],arms,flush=True)
    aggregate={arm:{kind:{key:sum(row[arm][kind][key] for row in rows.values())
                for key in next(iter(rows.values()))[arm][kind]} for kind in ['corrected','damaged']}
                for arm in next(iter(rows.values()))}
    write_json(out,dict(schema_version=1,scope='TRAIN only; GT used for post-run diagnosis, not inference; no parameter selection',
        caveats='Line proximity is measured to predicted infinite projected line, not finite segment. Residuals are existing clipped/log-scaled values. Associations are NOT causal attribution to the line branch.',
        scenes=rows,aggregate=aggregate,code_sha256=sha(__file__)))


if __name__=='__main__':main()
