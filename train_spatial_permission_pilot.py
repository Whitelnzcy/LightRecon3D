"""One pre-specified auxiliary-loss control, fixed64 input, no DEV tuning."""
import argparse
import copy
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics
from models.spatial_line_refiner import SpatialLineRefiner
from models.spatial_support_refiner import ownership_loss
from models.line_permission_loss import line_permission_loss
from spatial_support_data import load_prediction_scene,attach_supervision,patch_batch
from spatial_line_features import attach_predicted_lines,line_batch
from train_spatial_support_pilot import input_receipt
from train_spatial_line_ablation import predict,draw_comparison


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    threadpool_limits(1);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(42)
    roots=[Path('local_outputs/gct3/items')/f'train_{i:03d}_scene_{i:05d}' for i in range(3)]+[Path('local_outputs/gct4/items/train_003_scene_00003')]
    scenes=[];receipts=[]
    for root in roots:
        s=load_prediction_scene(root);path=attach_predicted_lines(s)
        receipts.append(dict(**input_receipt(s),lines_path=str(path),lines_sha256=sha(path)))
        write_json(out/'prediction_inputs.json',dict(schema_version=1,scenes=receipts))
        attach_supervision(s);scenes.append(s);print('PREPARED',s['name'],flush=True)
    base=SpatialLineRefiner('M3').cuda();models={key:copy.deepcopy(base) for key in ['M3','M3P']}
    opts={key:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for key,m in models.items()}
    protocol=dict(schema_version=1,steps=200,batch=4,seed=42,input_size=64,arms=list(models),
        auxiliary_weight={'M3':0.,'M3P':.1},
        hypothesis='Explicit candidate edit-permission supervision reduces damage without discarding correction benefit',
        controls='Same initialization,96GT-free patches/view,TRAIN4,point order,geometry,loss except auxiliary,optimizer and steps. Historical DEV only after freezing both.',
        auxiliary='Balanced BCE on available alternate candidate-pixels. Positive iff candidate is matched training GT; other available alternatives negative. Ignore unmatched,unassigned and absent lines. No GT in inference.',
        loss='Original weighted CE+0.5Dice, M3P adds0.1 permission BCE. Weight fixed before run, no sweep.',
        limitation='Gate scales signed line logits, not a guaranteed edit probability; base branch can still edit. Mechanistic supervision is a hypothesis, not safety guarantee.',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={p:sha(p) for p in ['train_spatial_permission_pilot.py','models/line_permission_loss.py','models/spatial_line_refiner.py','models/spatial_support_refiner.py','train_spatial_line_ablation.py','spatial_context_data.py','spatial_support_data.py','spatial_line_features.py']})
    write_json(out/'protocol.json',protocol);views=[v for s in scenes for v in s['views']];rng=np.random.default_rng(42)
    history=[];gradients={};torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for step in range(200):
        v=views[int(rng.integers(len(views)))];q=rng.choice(len(v['patches']),4,replace=len(v['patches'])<4)
        boxes=[v['patches'][i]['box_xyxy'] for i in q];common,candidates,target,original=patch_batch(v,boxes,'cuda');lines=line_batch(v,boxes,'cuda')
        row=dict(step=step+1)
        for arm,m in models.items():
            m.train();opts[arm].zero_grad()
            if arm=='M3P':
                logits,gate=m(common,candidates,lines,True)
                extra=line_permission_loss(gate,lines[:,:,0],target,original)
                loss=ownership_loss(logits,target,original)+.1*extra;row['permission_loss']=float(extra.detach().cpu())
            else:loss=ownership_loss(m(common,candidates,lines),target,original)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if step==2:gradients[arm]=float(m.line_gate.weight.grad.norm().cpu())
            opts[arm].step();row[arm]=float(loss.detach().cpu())
        if step%20==0 or step==199:
            history.append(row);write_json(out/'history.json',history);print('TRAIN',row,flush=True)
    torch.cuda.synchronize();profile=dict(train_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gate_gradient_step3=gradients)
    assert all(g>0 for g in gradients.values());write_json(out/'profile.json',profile)
    for arm,m in models.items():
        m.eval();torch.save(dict(schema_version=1,model=m.state_dict(),arm='M3',context_margin=0,protocol=protocol),out/f'{arm}.pt')
    report={}
    for item,split in [(s,'TRAIN_fit') for s in scenes]+[(n,'historical_DEV') for n in ['final_016_scene_00199','final_008_scene_00190']]:
        if isinstance(item,str):
            s=load_prediction_scene(Path('local_outputs/b12r2/items')/item);path=attach_predicted_lines(s)
            write_json(out/f'{item}_prediction_inputs.json',dict(**input_receipt(s),lines_path=str(path),lines_sha256=sha(path)))
        else:s=item
        predictions={};gates={}
        for arm,m in models.items():predictions[arm],gates[arm]=predict(m,s)
        np.savez_compressed(out/f'{s["name"]}_predictions.npz',schema_version=1,source_indices=np.arange(len(s['old']['points'])),original_ids=s['old']['point_plane_ids'],**predictions)
        assert sha(s['source'])==s['source_sha256']
        if split=='historical_DEV':
            eroot=Path('local_outputs/dense_boundary_competition_20260918_v1');frozen=read_json(eroot/'report.json')[s['name']]
            with np.load(eroot/f'{s["name"]}_eval_only.npz') as z:
                assert np.array_equal(z['pixel_xy'],s['old']['pixel_xy']) and np.array_equal(z['view_indices'],s['old']['source_views'])
                s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=frozen['fixed_identity_map'])
        before=convert(s['old']['point_plane_ids'],s['mapping']);rows={}
        for arm,p in predictions.items():
            rows[arm]=metrics(before,convert(p,s['mapping']),s['truth'],s['distance']);rows[arm]['changed']=int((p!=s['old']['point_plane_ids']).sum())
        report[s['name']]=dict(split=split,metrics=rows,mean_gate_on_available_tiled_pixels=gates,source_sha256=s['source_sha256'])
        write_json(out/'report.json',report)
        draw_comparison(s,{'M3 baseline':predictions['M3'],'M3P + permission loss':predictions['M3P']},out/'figures'/f'{s["name"]}_view0.png',split,
                        footer_text='Same64px input,200steps,loss only differs by0.1 auxiliary gate supervision. GT-free zoom; XYZ unchanged.')
        print('EVAL',s['name'],rows,flush=True)
    aggregate={split:{arm:{k:sum(v['metrics'][arm][k] for v in report.values() if v['split']==split) for k in next(iter(report.values()))['metrics'][arm]} for arm in models} for split in ['TRAIN_fit','historical_DEV']}
    write_json(out/'completion.json',dict(status='complete',aggregate=aggregate,profile=profile,checkpoint_sha256={a:sha(out/f'{a}.pt') for a in models}))


if __name__=='__main__':main()
