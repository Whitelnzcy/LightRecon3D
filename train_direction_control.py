"""Matched propagation control. Oracle is explicitly GT-assisted diagnostic only."""
import argparse
import copy
import subprocess
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from models.spatial_direction_refiner import SpatialDirectionRefiner
from models.spatial_support_refiner import ownership_loss
from spatial_support_data import load_prediction_scene,attach_supervision
from spatial_direction_data import DirectionInputs,attach_direction_supervision,direction_batch,direction_loss
from spatial_context_data import context_batch
from train_spatial_line_ablation import predict,draw_comparison
from diagnose_direction_mechanism import intervened
from train_spatial_support_pilot import input_receipt
from audit_line_support_refinement import write_json,read_json,sha,convert,metrics


class NoPropagation(SpatialDirectionRefiner):
    def forward(self,*args,**kwargs):
        return super().forward(*args,**kwargs,propagation='none')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=200);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    threadpool_limits(1);torch.set_num_threads(4);torch.manual_seed(42)
    provider=DirectionInputs(out);scenes=[]
    roots=[Path('local_outputs/gct3/items')/f'train_{i:03d}_scene_{i:05d}' for i in range(3)]+[Path('local_outputs/gct4/items/train_003_scene_00003')]
    for root in roots:
        s=load_prediction_scene(root);provider.attach(s)
        write_json(out/f'{s["name"]}_inputs.json',input_receipt(s))
        attach_supervision(s);attach_direction_supervision(s);scenes.append(s);print('PREPARED',s['name'],flush=True)
    torch.manual_seed(42);base=SpatialDirectionRefiner('DS').cuda()
    models={a:copy.deepcopy(base) for a in ['none','spatial','learned','oracle']}
    models['none'].__class__=NoPropagation;models['spatial'].arm='D0'
    opts={a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4) for a,m in models.items()}
    protocol=dict(schema_version=1,steps=args.steps,seed=42,arms=list(models),batch=4,
        scope='TRAIN4 historicalDEV2, one seed; oracle uses GT at TRAIN and DEV, NOT deployable or upper bound',
        initialization='identical state_dict; none concatenates original feature twice, same fusion capacity',
        loss='same ownership loss + .3 directionMSE all arms; oracle override detached and broadcast to ALL candidates',
        fixed='same batch sequence, 4-neighbor radius.02, original logit+2, frozen RGB prior, fixed XYZ',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hashes={p:sha(p) for p in ['train_direction_control.py','models/spatial_direction_refiner.py','spatial_direction_data.py']})
    write_json(out/'protocol.json',protocol);rng=np.random.default_rng(42);views=[v for s in scenes for v in s['views']];history=[]
    for step in range(args.steps):
        v=views[int(rng.integers(len(views)))];q=rng.choice(len(v['patches']),4,replace=len(v['patches'])<4)
        boxes=[v['patches'][i]['box_xyxy'] for i in q];c,k,e,y,o=context_batch(v,boxes,'cuda');t=direction_batch(v,boxes,'cuda')
        row=dict(step=step+1)
        for a,m in models.items():
            m.train();opts[a].zero_grad();kw={}
            if a=='oracle':kw['direction_override']=t[:,None].expand(-1,k.shape[1],-1,-1,-1).detach()
            logits,d=m(c,k,e,return_details=True,**kw);seg=ownership_loss(logits,y,o);aux=direction_loss(d,t,y);loss=seg+.3*aux
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();row[a]=dict(ownership=float(seg.detach()),direction=float(aux.detach()),
                fusion_grad=sum(float(p.grad.norm()) for p in m.line_fuse.parameters() if p.grad is not None))
            opts[a].step()
        if step%25==0 or step+1==args.steps:history.append(row);write_json(out/'history.json',history);print('TRAIN',row,flush=True)
    for a,m in models.items():m.eval();torch.save(dict(model=m.state_dict(),protocol=protocol,arm=a),out/f'{a}.pt')
    del scenes,views,c,k,e,y,o,t
    report={};gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        s=load_prediction_scene(Path('local_outputs/b12r2/items')/name);provider.attach(s);write_json(out/f'{name}_inputs.json',input_receipt(s))
        predictions={a:predict(m,s)[0] for a,m in models.items() if a!='oracle'}
        np.savez_compressed(out/f'{name}_before_GT.npz',schema_version=1,**predictions)
        with np.load(gtroot/f'{name}_eval_only.npz') as z:
            np.testing.assert_array_equal(z['pixel_xy'],s['old']['pixel_xy']);np.testing.assert_array_equal(z['view_indices'],s['old']['source_views'])
            s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=read_json(gtroot/'report.json')[name]['fixed_identity_map'])
        attach_direction_supervision(s);predictions['oracle']=intervened(models['oracle'],s,'oracle')
        assert sha(s['source'])==s['source_sha256'];before=convert(s['old']['point_plane_ids'],s['mapping']);row={}
        for a,p in predictions.items():
            assert np.isin(p[p>=0],s['old']['plane_ids']).all()
            np.testing.assert_array_equal(p[s['old']['point_plane_ids']<0],s['old']['point_plane_ids'][s['old']['point_plane_ids']<0])
            row[a]=metrics(before,convert(p,s['mapping']),s['truth'],s['distance'])
        np.savez_compressed(out/f'{name}_GT_DIAGNOSTIC.npz',schema_version=1,source_indices=np.arange(len(before)),original_ids=s['old']['point_plane_ids'],**predictions)
        report[name]=row;write_json(out/'interventions.json',report)
        draw_comparison(s,predictions,out/'figures'/f'{name}.png','controlled pilot; oracle GT-assisted',
            'Same initialization/batches/budget. Oracle trained/evaluated with GT direction; NOT deployable or upper bound.')
        print('EVAL',name,row,flush=True)
    aggregate={a:{key:sum(r[a][key] for r in report.values()) for key in next(iter(report.values()))[a]} for a in models}
    write_json(out/'completion.json',dict(status='complete',aggregate=aggregate,scope=protocol['scope']))


if __name__=='__main__':main()
