"""Bounded M1 implementation/fit pilot: four previously inspected TRAIN scenes.

No test-set selection, no DUSt3R training, no point/candidate geometry changes.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import convert, metrics, palette, sha, write_json
from models.line_support_patches import fuse_patch_logits
from models.spatial_support_refiner import SpatialSupportRefiner, ownership_loss
from spatial_support_data import load_prediction_scene, attach_supervision, tile_boxes, patch_batch


def input_receipt(scene):
    return dict(scene=scene['name'],source=scene['source'],source_sha256=scene['source_sha256'],
                candidate_ids=scene['old']['plane_ids'].tolist(),
                views=[dict(transform=v['transform'],patches=v['patches'],
                            projection_from_predicted_xyz=v['projection'].tolist(),
                            projection_quality=v['projection_quality']) for v in scene['views']])


@torch.inference_mode()
def predict_scene(model,scene,device):
    """GT-free full-view tiling, all candidates; overlap average before argmax."""
    model.eval();old=scene['old'];contributions=[]
    for view in scene['views']:
        boxes=tile_boxes(view['registry'].shape)
        for start in range(0,len(boxes),4):
            batch=boxes[start:start+4]
            common,candidates=patch_batch(view,batch,device,with_targets=False)
            logits=model(common,candidates).cpu().numpy()
            for box,score in zip(batch,logits):
                x0,y0,x1,y1=box;index=view['registry'][y0:y1,x0:x1];valid=index>=0
                contributions.append((index[valid],score.transpose(1,2,0)[valid]))
    result=fuse_patch_logits(old['point_plane_ids'],old['plane_ids'],contributions)
    assert np.array_equal(result[old['point_plane_ids']<0],old['point_plane_ids'][old['point_plane_ids']<0])
    return result


def preview(scene,prediction,path):
    v=scene['views'][0];index=v['registry'];observed=index>=0;h,w=index.shape
    before=convert(scene['old']['point_plane_ids'],scene['mapping'])
    after=convert(prediction,scene['mapping']);truth=scene['truth']
    grids=[]
    for values in [before,after,truth]:
        g=np.full(index.shape,-1,int);g[observed]=values[index[observed]];grids.append(g)
    b,a,g=grids;error=np.full((h,w,3),238,np.uint8);valid=observed&(g>=0)
    error[valid&(b!=g)&(a==g)]=[0,150,95]
    error[valid&(b==g)&(a!=g)]=[220,45,55]
    error[valid&(b!=g)&(a!=g)]=[235,165,35]
    rgb=np.clip((v['common'][:3].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)
    panels=[('RGB',rgb),('Original planes',palette(b)),('M1: no lines',palette(a)),('GT: evaluation only',palette(g)),('Fixed / damaged / remaining',error)]
    sheet=Image.new('RGB',(w*5,h*2+112),'white');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    draw.text((8,5),scene['name']+' | TRAIN fit diagnostic, not held-out performance | XYZ unchanged',fill='black',font=font)
    box=v['patches'][0]['box_xyxy']
    for j,(label,arr) in enumerate(panels):
        img=Image.fromarray(arr);sheet.paste(img,(j*w,56));draw.text((j*w+8,31),label,fill='black',font=font)
        draw.rectangle([j*w+box[0],56+box[1],j*w+box[2],56+box[3]],outline='black',width=2)
        sheet.paste(img.crop(box).resize((h,h),Image.Resampling.NEAREST),(j*w+(w-h)//2,h+72))
    draw.text((8,2*h+82),'Local window chosen from predicted boundary, not GT errors. Green: fixed; red: damaged; orange: still wrong.',fill='black',font=font)
    sheet.save(path)


@torch.inference_mode()
def probe_loss(model,views,device):
    model.eval();losses=[]
    for v in views:
        batch=patch_batch(v,[p['box_xyxy'] for p in v['patches'][:4]],device)
        losses.append(float(ownership_loss(model(*batch[:2]),*batch[2:]).cpu()))
    return float(np.mean(losses))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=200);ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--device',default='cuda');args=ap.parse_args()
    if args.steps<1 or args.batch<1:raise ValueError('Positive steps/batch required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    threadpool_limits(1);cv2.setNumThreads(1);torch.set_num_threads(4)
    torch.manual_seed(42);rng=np.random.default_rng(42)
    roots=[Path('local_outputs/gct3/items')/f'train_{i:03d}_scene_{i:05d}' for i in range(3)]
    roots.append(Path('local_outputs/gct4/items/train_003_scene_00003'))
    scenes=[];receipts=[]
    for root in roots:
        s=load_prediction_scene(root);receipts.append(input_receipt(s))
        # Prediction input manifest is saved before supervision is opened.
        write_json(out/'prediction_inputs.json',dict(schema_version=1,scenes=receipts))
        attach_supervision(s);scenes.append(s);print('PREPARED',s['name'],s['coverage'],flush=True)
    views=[v for s in scenes for v in s['views']]
    model=SpatialSupportRefiner().to(args.device)
    protocol=dict(schema_version=1,arm='M1_no_line',steps=args.steps,batch=args.batch,seed=42,
        scope='Implementation/fit pilot on 4 previously inspected TRAIN scenes; no generalization claim',
        optimizer='AdamW lr1e-3 weight_decay1e-4, no schedule, no checkpoint selection',
        loss='CE weighted2 on initially correct points +0.5 present-class Dice; -100 ignored for loss only',
        inputs='RGB, observed registry, original candidate masks, log1p(100*normalized plane residual); no lines',
        inference='Full-view64px tiles stride48, average overlapping logits; all scene candidates; preserve unassigned',
        sampling='Uniform view, then 4 patches from fixed96 proposals/view (64 predicted-boundary+32 ordinary)',
        parameters=sum(p.numel() for p in model.parameters()),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={p:sha(p) for p in ['train_spatial_support_pilot.py','spatial_support_data.py','models/spatial_support_refiner.py','models/line_support_patches.py','models/ownership_supervision.py']},
        coverage={s['name']:s['coverage'] for s in scenes})
    write_json(out/'protocol.json',protocol)
    initial=probe_loss(model,views,args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    history=[];start=time.perf_counter();gradient_receipt=None
    for step in range(args.steps):
        model.train();v=views[int(rng.integers(len(views)))];q=rng.choice(len(v['patches']),args.batch,replace=len(v['patches'])<args.batch)
        batch=patch_batch(v,[v['patches'][i]['box_xyxy'] for i in q],args.device)
        loss=ownership_loss(model(*batch[:2]),*batch[2:]);opt.zero_grad();loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        if step==2:
            gradient_receipt={name:float(p.grad.norm().cpu()) for name,p in model.named_parameters() if p.grad is not None}
            if not any(value>0 for name,value in gradient_receipt.items() if name.startswith('image.')):
                raise RuntimeError('Spatial image encoder has no gradient')
        opt.step()
        if step%20==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach().cpu()),elapsed_seconds=time.perf_counter()-start)
            history.append(row);write_json(out/'history.json',history);print('TRAIN',row,flush=True)
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    train_seconds=time.perf_counter()-start;final=probe_loss(model,views,args.device)
    torch.save(dict(schema_version=1,model=model.state_dict(),optimizer=opt.state_dict(),step=args.steps,protocol=protocol),out/'last.pt')
    profile=dict(initial_fixed_probe_loss=initial,final_fixed_probe_loss=final,train_seconds=train_seconds,
                 seconds_per_step=train_seconds/args.steps,gradient_step3=gradient_receipt,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                 device=torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device)
    write_json(out/'profile.json',profile);print('PROFILE',profile,flush=True)
    reports={};evaluation_start=time.perf_counter()
    for s in scenes:
        predicted=predict_scene(model,s,args.device)
        # Save predictions before evaluating, with an explicit source-order registry.
        np.savez_compressed(out/f'{s["name"]}_predictions.npz',schema_version=1,
            source_indices=np.arange(len(predicted)),original_ids=s['old']['point_plane_ids'],M1_no_line=predicted)
        assert sha(s['source'])==s['source_sha256']
        before=convert(s['old']['point_plane_ids'],s['mapping']);after=convert(predicted,s['mapping'])
        report=metrics(before,after,s['truth'],s['distance'])
        report['changed']=int((predicted!=s['old']['point_plane_ids']).sum())
        reports[s['name']]=dict(metrics=report,missing_candidate=s['missing_candidate'],mapping=s['mapping'],source_sha256=s['source_sha256'])
        preview(s,predicted,out/'figures'/f'{s["name"]}_view0.png')
        write_json(out/'report.json',reports);print('EVAL',s['name'],report,flush=True)
    aggregate={k:sum(r['metrics'][k] for r in reports.values()) for k in next(iter(reports.values()))['metrics']}
    write_json(out/'completion.json',dict(status='complete',scope=protocol['scope'],aggregate=aggregate,
        evaluation_seconds=time.perf_counter()-evaluation_start,checkpoint_sha256=sha(out/'last.pt'),profile=profile))


if __name__=='__main__':main()
