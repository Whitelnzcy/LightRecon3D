"""Frozen M3 gate intervention; TRAIN-calibrated constant, not a trained baseline.

No DEV calibration/selection; fixed XYZ and candidate labels. The constant
replaces the gate only, retaining the already spatial line-score branch.
"""
import argparse
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics
from models.spatial_line_refiner import SpatialLineRefiner
from spatial_support_data import load_prediction_scene,patch_batch,tile_boxes
from spatial_line_features import attach_predicted_lines,line_batch
from train_spatial_line_ablation import predict


@torch.inference_mode()
def gate_moments(model,scene):
    total=squares=count=0;low=1.;high=0.
    for v in scene['views']:
        boxes=tile_boxes(v['registry'].shape)
        for start in range(0,len(boxes),4):
            batch=boxes[start:start+4];common,candidates=patch_batch(v,batch,'cuda',False)
            line=line_batch(v,batch,'cuda');_,gate=model(common,candidates,line,True)
            values=gate[line[:,:,0]>0].double()
            if values.numel():
                total+=float(values.sum().cpu());squares+=float(values.square().sum().cpu())
                count+=values.numel();low=min(low,float(values.min().cpu()));high=max(high,float(values.max().cpu()))
    return dict(sum=total,squares=squares,count=count,min=low,max=high,
                mean=total/max(count,1),std=max(0,squares/max(count,1)-(total/max(count,1))**2)**.5)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    threadpool_limits(1);torch.set_num_threads(4);cv2.setNumThreads(1)
    checkpoint=run/'M3.pt';digest=sha(checkpoint)
    model=SpatialLineRefiner('M3').cuda();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
    protocol=dict(schema_version=1,checkpoint_sha256=digest,source_run=str(run),
        intervention='Freeze M3 weights, replace only gate by mean measured over TRAIN available tiled pixels. No retraining; diagnostic, NOT matched trained constant-gate baseline.',
        calibration='TRAIN4 all views, 64px tiles stride48, repeated tile coverage counted, no GT',
        code_sha256=sha(__file__))
    write_json(out/'protocol.json',protocol);calibration={}
    for r in read_json(run/'prediction_inputs.json')['scenes']:
        assert sha(r['source'])==r['source_sha256'] and sha(r['lines_path'])==r['lines_sha256']
        s=load_prediction_scene(Path(r['source']).parent.parent);attach_predicted_lines(s)
        calibration[s['name']]=gate_moments(model,s);print('TRAIN_GATE',s['name'],calibration[s['name']],flush=True)
    count=sum(v['count'] for v in calibration.values());mean=sum(v['sum'] for v in calibration.values())/count
    write_json(out/'calibration.json',dict(scenes=calibration,constant=mean))
    constant=copy.deepcopy(model);constant.arm='MS'
    with torch.no_grad():constant.line_gate.bias.fill_(float(np.log(mean/(1-mean))))
    report={};targets=Path('local_outputs/dense_boundary_competition_20260918_v1');frozen=read_json(targets/'report.json')
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        s=load_prediction_scene(Path('local_outputs/b12r2/items')/name);attach_predicted_lines(s)
        prediction,_=predict(constant,s)
        np.savez_compressed(out/f'{name}_predictions.npz',schema_version=1,source_indices=np.arange(len(prediction)),
                            original_ids=s['old']['point_plane_ids'],M3_constant=prediction)
        # Read evaluation targets only after the complete prediction is saved.
        with np.load(targets/f'{name}_eval_only.npz') as z:
            assert np.array_equal(z['pixel_xy'],s['old']['pixel_xy']) and np.array_equal(z['view_indices'],s['old']['source_views'])
            truth=z['raw_dense_gt'];distance=z['boundary_distance']
        mapping=frozen[name]['fixed_identity_map'];original=s['old']['point_plane_ids'];before=convert(original,mapping)
        with np.load(run/f'{name}_predictions.npz') as z:spatial=z['M3'];assert np.array_equal(z['original_ids'],original)
        row={}
        for arm,p in [('M3',spatial),('M3_constant',prediction)]:
            row[arm]=metrics(before,convert(p,mapping),truth,distance);row[arm]['changed']=int((p!=original).sum())
            assert np.array_equal(p[original<0],original[original<0])
        assert row['M3']==read_json(run/'report.json')[name]['metrics']['M3']
        report[name]=dict(metrics=row,gate=gate_moments(model,s),source_sha256=sha(s['source']))
        write_json(out/'report.json',report);print('DEV',name,row,flush=True)
    assert sha(checkpoint)==digest
    write_json(out/'completion.json',dict(status='complete',constant=mean,aggregate={a:{k:sum(r['metrics'][a][k] for r in report.values())
        for k in next(iter(report.values()))['metrics'][a]} for a in ['M3','M3_constant']}))


if __name__=='__main__':main()
