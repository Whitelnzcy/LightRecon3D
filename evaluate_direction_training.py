"""Full TRAIN-fit metrics for frozen direction budget checkpoints."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from models.spatial_direction_refiner import SpatialDirectionRefiner
from train_direction_control import NoPropagation
from train_spatial_line_ablation import predict
from spatial_support_data import load_prediction_scene,attach_supervision
from spatial_direction_data import DirectionInputs,attach_direction_supervision,direction_batch
from spatial_context_data import context_batch
from diagnose_direction_mechanism import quality
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=Path(args.run)
    out=run/'training_fit';out.mkdir(exist_ok=False);threadpool_limits(1);torch.set_num_threads(4)
    protocol=read_json(run/'protocol.json');models={}
    for a in protocol['arms']:
        checkpoint=torch.load(run/f'{a}.pt',map_location='cpu',weights_only=False)
        cls=NoPropagation if a.startswith('none') else SpatialDirectionRefiner
        if a.startswith('oracle'):raise ValueError('This TRAIN-fit evaluator excludes oracle arms')
        m=cls('DS',grid_step=checkpoint.get('grid_step',1),support_bias=checkpoint.get('support_bias',2.),direction_agreement=checkpoint.get('direction_agreement',False),width=checkpoint.get('width',1)).cuda().eval()
        m.load_state_dict(checkpoint['model']);models[a]=m
    roots=[Path('local_outputs/gct3/items')/f'train_{i:03d}_scene_{i:05d}' for i in range(3)]+[Path('local_outputs/gct4/items/train_003_scene_00003')]
    provider=DirectionInputs(out);report={}
    for root in roots:
        s=load_prediction_scene(root);provider.attach(s)
        predictions={a:predict(m,s)[0] for a,m in models.items()}
        np.savez_compressed(out/f'{s["name"]}_predictions.npz',schema_version=1,source_indices=np.arange(len(s['old']['points'])),**predictions)
        attach_supervision(s);attach_direction_supervision(s,protocol.get('boundary_target','zero'));before=convert(s['old']['point_plane_ids'],s['mapping'])
        rows={a:metrics(before,convert(p,s['mapping']),s['truth'],s['distance']) for a,p in predictions.items()}
        v=s['views'][0];boxes=[p['box_xyxy'] for p in v['patches'][:4]];c,k,e,y,_=context_batch(v,boxes,'cuda');target=direction_batch(v,boxes,'cuda')
        angles={}
        for a,m in models.items():_,d=m(c,k,e,return_details=True);angles[a]=quality(d,target,y)
        assert sha(s['source'])==s['source_sha256'];report[s['name']]=dict(metrics=rows,direction_first4patches_view0=angles)
        write_json(out/'report.json',report);print('TRAIN_FIT',s['name'],rows,angles,flush=True)
    aggregate={a:{key:sum(r['metrics'][a][key] for r in report.values()) for key in next(iter(report.values()))['metrics'][a]} for a in models}
    write_json(out/'completion.json',dict(status='complete',scope='training fit only; not independent generalization',aggregate=aggregate))


if __name__=='__main__':main()
