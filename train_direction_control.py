"""Matched propagation control. Oracle is explicitly GT-assisted diagnostic only."""
import argparse
import copy
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from models.spatial_direction_refiner import SpatialDirectionRefiner
from models.spatial_support_refiner import ownership_loss
from spatial_support_data import load_prediction_scene,attach_supervision
from spatial_direction_data import DirectionInputs,attach_direction_supervision,direction_batch,direction_loss,boundary_direction_batch,balanced_direction_loss
from spatial_context_data import context_batch
from train_spatial_line_ablation import predict,draw_comparison
from diagnose_direction_mechanism import intervened
from train_spatial_support_pilot import input_receipt
from audit_line_support_refinement import write_json,read_json,sha,convert,metrics


class NoPropagation(SpatialDirectionRefiner):
    def forward(self,*args,**kwargs):
        return super().forward(*args,**kwargs,propagation='none')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=200)
    ap.add_argument('--experiment',choices=['control','reach','replay','bias','budget'],default='control');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--deterministic',action='store_true')
    ap.add_argument('--control-arms',nargs='+',choices=['none','spatial','learned','oracle','agreement','oracle_agreement'],default=['none','spatial','learned','oracle'])
    ap.add_argument('--boundary-target',choices=['zero','inward','grid_inward'],default='zero')
    ap.add_argument('--direction-balance',choices=['uniform','boundary'],default='uniform');args=ap.parse_args()
    if args.direction_balance=='boundary' and args.boundary_target!='inward':raise ValueError('Boundary balance protocol requires inward targets')
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    threadpool_limits(1);torch.set_num_threads(4);torch.manual_seed(args.seed)
    provider=DirectionInputs(out);scenes=[]
    roots=[Path('local_outputs/gct3/items')/f'train_{i:03d}_scene_{i:05d}' for i in range(3)]+[Path('local_outputs/gct4/items/train_003_scene_00003')]
    for root in roots:
        s=load_prediction_scene(root);provider.attach(s)
        write_json(out/f'{s["name"]}_inputs.json',input_receipt(s))
        attach_supervision(s);attach_direction_supervision(s,args.boundary_target);scenes.append(s);print('PREPARED',s['name'],flush=True)
    torch.manual_seed(args.seed);base=SpatialDirectionRefiner('DS').cuda()
    arms=args.control_arms if args.experiment=='control' else ['reach1','repeat1'] if args.experiment=='replay' else ['reach1','reach4','reach8']
    if args.experiment=='bias':arms=['bias0','bias1','bias2','bias2repeat']
    if args.experiment=='budget':
        if args.steps!=800:raise ValueError('Budget protocol fixes maximum800 steps')
        arms=['none200','learned200','none800','learned800']
    models={a:copy.deepcopy(base) for a in arms}
    if args.experiment=='control':
        if 'none' in models:models['none'].__class__=NoPropagation
        if 'spatial' in models:models['spatial'].arm='D0'
        for a,m in models.items():m.direction_agreement=a.endswith('agreement')
    elif args.experiment=='bias':
        for a,m in models.items():m.support_bias=float(a[4])
    elif args.experiment=='budget':
        for a,m in models.items():
            if a.startswith('none'):m.__class__=NoPropagation
    else:
        for a,m in models.items():m.grid_step=1 if a=='repeat1' else int(a.removeprefix('reach'))
    opts={a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4) for a,m in models.items()}
    protocol=dict(schema_version=1,steps=args.steps,seed=args.seed,arms=list(models),batch=4,
        scope='TRAIN4 historicalDEV2, one seed; oracle uses GT at TRAIN and DEV, NOT deployable or upper bound',
        initialization='identical state_dict; none concatenates original feature twice, same fusion capacity',
        loss='same ownership loss + .3 directionMSE all arms; oracle override detached and broadcast to ALL candidates',
        fixed='same batch sequence, 4-neighbor radius.02, original logit+2, frozen RGB prior, fixed XYZ',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hashes={p:sha(p) for p in ['train_direction_control.py','models/spatial_direction_refiner.py','spatial_direction_data.py']})
    if args.experiment!='control':protocol.update(scope='TRAIN4 historicalDEV2, one seed, NO GT in prediction',
        fixed='Only grid step changes 1/4/8; four neighbors+center, same 3D radius.02, bias+2, shared direction, loss, initialization and batches',
        initialization='identical state_dict; grid step has no parameters',loss='same ownership+.3 directionMSE; no oracle override')
    protocol.update(deterministic=args.deterministic,experiment=args.experiment,torch_version=torch.__version__)
    if not any(a.startswith('oracle') for a in arms):
        protocol['scope']='TRAIN4 historicalDEV2, one seed; NO GT in prediction; GT used for TRAIN supervision and evaluation only'
    protocol['direction_agreement']={a:m.direction_agreement for a,m in models.items()}
    protocol['boundary_target']=args.boundary_target
    protocol['direction_balance']=args.direction_balance
    if args.direction_balance=='boundary':protocol['loss']='same ownership + .3 * equal-mean boundary/interior direction MSE; absent strata omitted'
    protocol['target_summaries']={s['name']:s['direction_supervision_summary'] for s in scenes}
    if any(m.direction_agreement for m in models.values()):protocol['agreement_rule']='Multiply propagation weight by (1+dot(unit_direction_i,unit_direction_j))/2; zero directions neutral; no extra parameters. Other settings unchanged.'
    if args.experiment=='replay':protocol['fixed']='Two identical reach1 arms; assert bitwise equal parameters after EVERY update. Not an efficacy experiment.'
    if args.experiment=='bias':protocol.update(fixed='Only output support bias0/1/2 changes; grid1/radius.02/loss/candidate inputs unchanged. bias2repeat asserts bitwise replay each step.',
        initialization='identical state_dict, different nonparameter output bias; candidate original-support input remains present')
    if args.experiment=='budget':protocol.update(fixed='Grid1,bias2, same weights/loss/batches. None vs learned direction, each at200/800 updates. All frozen before DEV.',
        initialization='identical state_dict; shorter arms stop at200, longer arms share exact first200updates',budgets={a:200 if a.endswith('200') else 800 for a in arms})
    write_json(out/'protocol.json',protocol);rng=np.random.default_rng(args.seed);views=[v for s in scenes for v in s['views']];history=[]
    for step in range(args.steps):
        v=views[int(rng.integers(len(views)))];q=rng.choice(len(v['patches']),4,replace=len(v['patches'])<4)
        boxes=[v['patches'][i]['box_xyxy'] for i in q];c,k,e,y,o=context_batch(v,boxes,'cuda');t=direction_batch(v,boxes,'cuda')
        row=dict(step=step+1)
        for a,m in models.items():
            if args.experiment=='budget' and a.endswith('200') and step>=200:continue
            m.train();opts[a].zero_grad();kw={}
            if a.startswith('oracle'):kw['direction_override']=t[:,None].expand(-1,k.shape[1],-1,-1,-1).detach()
            logits,d=m(c,k,e,return_details=True,**kw);seg=ownership_loss(logits,y,o)
            aux=balanced_direction_loss(d,t,y,boundary_direction_batch(v,boxes,'cuda')) if args.direction_balance=='boundary' else direction_loss(d,t,y)
            loss=seg+.3*aux
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();row[a]=dict(ownership=float(seg.detach()),direction=float(aux.detach()),
                fusion_grad=sum(float(p.grad.norm()) for p in m.line_fuse.parameters() if p.grad is not None))
            opts[a].step()
        if args.experiment=='replay':
            for key,value in models['reach1'].state_dict().items():
                if not torch.equal(value,models['repeat1'].state_dict()[key]):raise AssertionError(f'Replay diverged at step {step+1}: {key}')
        if args.experiment=='bias':
            for key,value in models['bias2'].state_dict().items():
                if not torch.equal(value,models['bias2repeat'].state_dict()[key]):raise AssertionError(f'Bias replay diverged at step {step+1}: {key}')
        if args.experiment=='budget' and step<200:
            for prefix in ['none','learned']:
                for key,value in models[prefix+'200'].state_dict().items():
                    if not torch.equal(value,models[prefix+'800'].state_dict()[key]):raise AssertionError(f'Budget prefix diverged at step {step+1}: {key}')
        if step%25==0 or step+1==args.steps:history.append(row);write_json(out/'history.json',history);print('TRAIN',row,flush=True)
    for a,m in models.items():m.eval();torch.save(dict(schema_version=3,model=m.state_dict(),protocol=protocol,arm=a,grid_step=m.grid_step,support_bias=m.support_bias,direction_agreement=m.direction_agreement),out/f'{a}.pt')
    del scenes,views,c,k,e,y,o,t
    report={};gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        s=load_prediction_scene(Path('local_outputs/b12r2/items')/name);provider.attach(s);write_json(out/f'{name}_inputs.json',input_receipt(s))
        predictions={a:predict(m,s)[0] for a,m in models.items() if not a.startswith('oracle')}
        if args.experiment=='replay':np.testing.assert_array_equal(predictions['reach1'],predictions['repeat1'])
        if args.experiment=='bias':np.testing.assert_array_equal(predictions['bias2'],predictions['bias2repeat'])
        np.savez_compressed(out/f'{name}_before_GT.npz',schema_version=1,**predictions)
        with np.load(gtroot/f'{name}_eval_only.npz') as z:
            np.testing.assert_array_equal(z['pixel_xy'],s['old']['pixel_xy']);np.testing.assert_array_equal(z['view_indices'],s['old']['source_views'])
            s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=read_json(gtroot/'report.json')[name]['fixed_identity_map'])
        if any(a.startswith('oracle') for a in models):
            attach_direction_supervision(s,args.boundary_target)
            for a,m in models.items():
                if a.startswith('oracle'):predictions[a]=intervened(m,s,'oracle')
        assert sha(s['source'])==s['source_sha256'];before=convert(s['old']['point_plane_ids'],s['mapping']);row={}
        for a,p in predictions.items():
            assert np.isin(p[p>=0],s['old']['plane_ids']).all()
            np.testing.assert_array_equal(p[s['old']['point_plane_ids']<0],s['old']['point_plane_ids'][s['old']['point_plane_ids']<0])
            row[a]=metrics(before,convert(p,s['mapping']),s['truth'],s['distance'])
        np.savez_compressed(out/f'{name}_GT_DIAGNOSTIC.npz',schema_version=1,source_indices=np.arange(len(before)),original_ids=s['old']['point_plane_ids'],**predictions)
        report[name]=row;write_json(out/'interventions.json',report)
        draw_comparison(s,predictions,out/'figures'/f'{name}.png',protocol['scope'],
            ('Same initialization/batches/budget. Oracle uses GT; diagnostic only.' if any(a.startswith('oracle') for a in arms) else
             'Same initialization/batches/budget. Predicted directions only; GT evaluation only. Target: '+args.boundary_target) if args.experiment=='control' else
            'Only support bias0/1/2 changes. Fixed grid1, XYZ, loss. bias2repeat exact replay.' if args.experiment=='bias' else
            'None vs learned direction at200/800 steps; same grid1,bias2,inputs. No GT in prediction.' if args.experiment=='budget' else
            'Only grid reach1/4/8 changes (replay uses identical grid1). No GT in prediction.')
        print('EVAL',name,row,flush=True)
    aggregate={a:{key:sum(r[a][key] for r in report.values()) for key in next(iter(report.values()))[a]} for a in models}
    write_json(out/'completion.json',dict(status='complete',aggregate=aggregate,scope=protocol['scope']))


if __name__=='__main__':main()
