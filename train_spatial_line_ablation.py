"""First fixed-budget M1/M2/M3 comparison; one seed, TRAIN4 and historical DEV2."""
import argparse
import copy
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics,palette
from models.spatial_line_refiner import SpatialLineRefiner
from models.spatial_support_refiner import ownership_loss
from models.line_support_patches import fuse_patch_logits
from spatial_support_data import load_prediction_scene,attach_supervision,tile_boxes,patch_batch
from spatial_line_features import attach_predicted_lines,line_batch
from train_spatial_support_pilot import input_receipt
from spatial_context_data import context_batch,center_crop


@torch.inference_mode()
def predict(model,scene,context_margin=0):
    contributions=[];gate_sum=0.;gate_count=0
    for v in scene['views']:
        boxes=tile_boxes(v['registry'].shape)
        for start in range(0,len(boxes),4):
            batch=boxes[start:start+4];common,candidates,line=context_batch(v,batch,'cuda',context_margin,False)
            logits,gate=model(common,candidates,line,True)
            logits=center_crop(logits,context_margin);gate=center_crop(gate,context_margin)
            active=center_crop(line[:,:,0],context_margin)>0
            gate_sum+=float(gate[active].sum().cpu());gate_count+=int(active.sum().cpu())
            for box,score in zip(batch,logits.cpu().numpy()):
                x0,y0,x1,y1=box;index=v['registry'][y0:y1,x0:x1];ok=index>=0
                contributions.append((index[ok],score.transpose(1,2,0)[ok]))
    return fuse_patch_logits(scene['old']['point_plane_ids'],scene['old']['plane_ids'],contributions),gate_sum/max(1,gate_count)


def draw_comparison(scene,predictions,path,split,footer_text=None):
    v=scene['views'][0];registry=v['registry'];ok=registry>=0;h,w=registry.shape
    panels=[]
    for label,raw in [('Original',scene['old']['point_plane_ids'])]+list(predictions.items()):
        values=convert(raw,scene['mapping']);grid=np.full((h,w),-1,int);grid[ok]=values[registry[ok]]
        panels.append((label,palette(grid)))
    gt=np.full((h,w),-1,int);gt[ok]=scene['truth'][registry[ok]];panels.append(('GT: evaluation only',palette(gt)))
    sheet=Image.new('RGB',(len(panels)*w,2*h+112),'white');d=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);box=v['patches'][0]['box_xyxy']
    d.text((8,5),scene['name']+f' | {split}; same points, candidate pool, camera and colors',font=font,fill='black')
    for j,(label,pixels) in enumerate(panels):
        img=Image.fromarray(pixels);sheet.paste(img,(j*w,56));d.text((j*w+8,31),label,font=font,fill='black')
        d.rectangle([j*w+box[0],56+box[1],j*w+box[2],56+box[3]],outline='black',width=2)
        sheet.paste(img.crop(box).resize((h,h),Image.Resampling.NEAREST),(j*w+(w-h)//2,h+72))
    footer='MC: active no-line branch; MS: scalar line gate; M3: spatial line gate. Zoom selected without GT.'
    if any('_c' in key for key in predictions):footer='c64/c128: input context; same 64px supervised/output center. MC: no line; M3: spatial line gate. GT-free zoom.'
    elif 'MC' not in predictions:footer='M1: no line; M2: fixed 0.5 line weight; M3: learned line weight. Zoom selected without GT.'
    d.text((8,2*h+82),footer_text or footer,font=font,fill='black')
    sheet.save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=200)
    ap.add_argument('--arms',nargs='+',choices=['M1','M2','M3','MC','MS'],default=['M1','M2','M3'])
    ap.add_argument('--context-margins',nargs='+',type=int,choices=[0,32],default=[0])
    args=ap.parse_args()
    if args.steps<3:raise ValueError('At least three steps required')
    if len(set(args.arms))!=len(args.arms):raise ValueError('Duplicate arms')
    if len(set(args.context_margins))!=len(args.context_margins):raise ValueError('Duplicate margins')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    threadpool_limits(1);torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(42)
    roots=[Path('local_outputs/gct3/items')/f'train_{i:03d}_scene_{i:05d}' for i in range(3)]+[Path('local_outputs/gct4/items/train_003_scene_00003')]
    scenes=[];receipts=[]
    for root in roots:
        s=load_prediction_scene(root);path=attach_predicted_lines(s)
        receipts.append(dict(**input_receipt(s),lines_path=str(path),lines_sha256=sha(path)))
        write_json(out/'prediction_inputs.json',dict(schema_version=1,scenes=receipts))
        attach_supervision(s);scenes.append(s);print('PREPARED',s['name'],flush=True)
    views=[v for s in scenes for v in s['views']];rng=np.random.default_rng(42)
    specs={(f'{a}_c{64+2*margin}' if args.context_margins!=[0] else a):(a,margin)
           for margin in args.context_margins for a in args.arms}
    base=SpatialLineRefiner('M1').cuda();models={a:copy.deepcopy(base) for a in specs}
    for a,m in models.items():m.arm=specs[a][0]
    opts={a:torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4) for a,m in models.items()}
    protocol=dict(schema_version=1,steps=args.steps,seed=42,batch=4,arms=list(models),
        scope='One-seed development pilot; no independent test or statistical claim',
        initialization='identical state_dict for all arms; zero score heads; original identity logit2',
        sampling='same TRAIN4 views, same fixed96 GT-free patches/view, same batch sequence',
        context_specs={a:dict(model_arm=specs[a][0],margin=specs[a][1],input_size=64+2*specs[a][1],target_size=64) for a in specs},
        context_protocol='Larger halo only: same64px center targets and output tiles stride48; zero outside-image padding with observed=0; no interpolation. Larger input changes convolution boundary context AND GroupNorm statistics. Same update/target budget, NOT same FLOPs.',
        loss='same weighted CE +0.5 present-class Dice',optimizer='AdamW lr1e-3 weight_decay1e-4',
        line_input='predicted original/refined plane-pair lines; current/destination-conditioned availability, oriented distance, proximity, along coordinate, side confidence, refinement score',
        gate='M2 fixed0.5; M3 local sigmoid; MS global scalar sigmoid; MC local no-line sigmoid; all initialized0.5',
        capacity_note='MC/M3 identical active architecture; MC repeats RGB/observed/mask/residual into six extra channels and enables branch on observed pixels, without line availability. M1 disables branch; M2 gate unused; MS only one gate bias active. Different applicability masks remain an intentional input/mechanism difference.',
        parameters=sum(p.numel() for p in base.parameters()),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={p:sha(p) for p in ['train_spatial_line_ablation.py','models/spatial_line_refiner.py','spatial_line_features.py',
                    'models/spatial_support_refiner.py','spatial_support_data.py','spatial_context_data.py','models/line_support_patches.py','models/ownership_supervision.py']})
    write_json(out/'protocol.json',protocol);history=[];start=time.perf_counter();gradient={};torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        v=views[int(rng.integers(len(views)))];q=rng.choice(len(v['patches']),4,replace=len(v['patches'])<4)
        boxes=[v['patches'][i]['box_xyxy'] for i in q]
        batches={margin:context_batch(v,boxes,'cuda',margin) for margin in args.context_margins}
        targets=[batch[3] for batch in batches.values()]
        assert all(torch.equal(targets[0],target) for target in targets)
        row=dict(step=step+1)
        for arm,m in models.items():
            margin=specs[arm][1];common,candidates,lines,target,original=batches[margin]
            m.train();opts[arm].zero_grad();logits=center_crop(m(common,candidates,lines),margin)
            loss=ownership_loss(logits,target,original)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward()
            if step==2:gradient[arm]={n:float(p.grad.norm().cpu()) for n,p in m.named_parameters() if p.grad is not None and n.startswith('line_')}
            opts[arm].step();row[arm]=float(loss.detach().cpu())
        if step%20==0 or step+1==args.steps:
            row['elapsed_seconds']=time.perf_counter()-start;history.append(row);write_json(out/'history.json',history);print('TRAIN',row,flush=True)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start
    for arm in models:
        if specs[arm][0] in {'M3','MC','MS'} and not any(x>0 for n,x in gradient[arm].items() if n.startswith('line_gate')):
            raise ValueError('Gate received no gradient: '+arm)
    for arm,m in models.items():
        m.eval();torch.save(dict(schema_version=1,model=m.state_dict(),step=args.steps,arm=m.arm,
                               context_margin=specs[arm][1],protocol=protocol),out/f'{arm}.pt')
    profile=dict(train_seconds=train_seconds,seconds_per_step_all_arms=train_seconds/args.steps,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 gate_gradients_step3=gradient,
                 scalar_gate={a:float(m.line_gate.bias[0].sigmoid().detach().cpu()) for a,m in models.items() if m.arm=='MS'})
    write_json(out/'profile.json',profile)
    report={}
    # Train metrics first; DEV is loaded only after every model is frozen.
    items=[(s,'TRAIN_fit') for s in scenes]+[(name,'historical_DEV') for name in ['final_016_scene_00199','final_008_scene_00190']]
    for item,split in items:
        if isinstance(item,str):
            s=load_prediction_scene(Path('local_outputs/b12r2/items')/item);path=attach_predicted_lines(s)
            write_json(out/f'{item}_prediction_inputs.json',dict(**input_receipt(s),lines_path=str(path),lines_sha256=sha(path)))
        else:s=item
        predictions={};gates={}
        for arm,m in models.items():predictions[arm],gates[arm]=predict(m,s,specs[arm][1])
        np.savez_compressed(out/f'{s["name"]}_predictions.npz',schema_version=1,
            source_indices=np.arange(len(s['old']['points'])),original_ids=s['old']['point_plane_ids'],**predictions)
        assert sha(s['source'])==s['source_sha256']
        if split=='historical_DEV':
            eroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
            frozen=read_json(eroot/'report.json')[s['name']]
            with np.load(eroot/f'{s["name"]}_eval_only.npz') as z:
                assert np.array_equal(z['pixel_xy'],s['old']['pixel_xy']) and np.array_equal(z['view_indices'],s['old']['source_views'])
                s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=frozen['fixed_identity_map'])
        before=convert(s['old']['point_plane_ids'],s['mapping']);rows={}
        for arm,pred in predictions.items():
            rows[arm]=metrics(before,convert(pred,s['mapping']),s['truth'],s['distance'])
            rows[arm]['changed']=int((pred!=s['old']['point_plane_ids']).sum())
        report[s['name']]=dict(split=split,metrics=rows,mean_gate_on_available_tiled_pixels=gates,
                               mapping=s['mapping'],source_sha256=s['source_sha256'])
        write_json(out/'report.json',report);draw_comparison(s,predictions,out/'figures'/f'{s["name"]}_view0.png',split)
        print('EVAL',s['name'],rows,flush=True)
    aggregate={split:{arm:{k:sum(v['metrics'][arm][k] for v in report.values() if v['split']==split)
                          for k in next(iter(report.values()))['metrics'][arm]} for arm in models}
               for split in ['TRAIN_fit','historical_DEV']}
    write_json(out/'completion.json',dict(status='complete',aggregate=aggregate,profile=profile,
        checkpoint_sha256={arm:sha(out/f'{arm}.pt') for arm in models}))


if __name__=='__main__':main()
