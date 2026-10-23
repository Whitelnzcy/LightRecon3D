"""Frozen M1 transfer to historical DEV2; no training or validation selection."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image,ImageDraw,ImageFont
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics,palette
from models.spatial_support_refiner import SpatialSupportRefiner
from spatial_support_data import load_prediction_scene
from train_spatial_support_pilot import predict_scene,input_receipt


def comparison(scene,prediction,historical,truth,mapping,path):
    v=scene['views'][0];index=v['registry'];observed=index>=0;h,w=index.shape
    values=[convert(scene['old']['point_plane_ids'],mapping),convert(historical,mapping),
            convert(prediction,mapping),truth]
    grids=[]
    for value in values:
        g=np.full(index.shape,-1,int);g[observed]=value[index[observed]];grids.append(g)
    b,old,a,g=grids;error=np.full((h,w,3),238,np.uint8);valid=observed&(g>=0)
    error[valid&(b!=g)&(a==g)]=[0,150,95];error[valid&(b==g)&(a!=g)]=[220,45,55]
    error[valid&(b!=g)&(a!=g)]=[235,165,35]
    rgb=np.clip((v['common'][:3].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)
    panels=[('RGB',rgb),('Original planes',palette(b)),('Historical finite model',palette(old)),
            ('M1: no lines, 200-step pilot',palette(a)),('GT: evaluation only',palette(g)),('M1 fixed / damaged / remaining',error)]
    sheet=Image.new('RGB',(w*6,h*2+112),'white');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    draw.text((8,5),scene['name']+' | Previously inspected DEV, not blind test | No XYZ changes',fill='black',font=font)
    box=v['patches'][0]['box_xyxy']
    for j,(label,arr) in enumerate(panels):
        img=Image.fromarray(arr);sheet.paste(img,(j*w,56));draw.text((j*w+8,31),label,fill='black',font=font)
        draw.rectangle([j*w+box[0],56+box[1],j*w+box[2],56+box[3]],outline='black',width=2)
        sheet.paste(img.crop(box).resize((h,h),Image.Resampling.NEAREST),(j*w+(w-h)//2,h+72))
    draw.text((8,2*h+82),'Zoom proposed without GT. Green: fixed; red: damaged; orange: still wrong. Historical model differs in training budget and inputs.',fill='black',font=font)
    sheet.save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    threadpool_limits(1);cv2.setNumThreads(1);torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=SpatialSupportRefiner().cuda().eval();model.load_state_dict(checkpoint['model'])
    protocol=dict(schema_version=1,scope='Frozen transfer, historically inspected DEV2; not independent test',
                  checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),steps=checkpoint['step'],
                  training_protocol=checkpoint['protocol'],code_sha256={p:sha(p) for p in [
                      'evaluate_spatial_support_dev.py','train_spatial_support_pilot.py','spatial_support_data.py','models/spatial_support_refiner.py']})
    write_json(out/'protocol.json',protocol)
    reports={};started=time.perf_counter()
    for item in ['final_016_scene_00199','final_008_scene_00190']:
        scene=load_prediction_scene(Path('local_outputs/b12r2/items')/item)
        write_json(out/f'{item}_prediction_inputs.json',input_receipt(scene))
        prediction=predict_scene(model,scene,'cuda')
        np.savez_compressed(out/f'{item}_predictions.npz',schema_version=1,
            source_indices=np.arange(len(prediction)),original_ids=scene['old']['point_plane_ids'],M1_no_line=prediction)
        assert sha(scene['source'])==scene['source_sha256']
        # Only now access labels and the pre-existing frozen evaluation map.
        eroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
        row=read_json(eroot/'report.json')[item];mapping=row['fixed_identity_map']
        with np.load(eroot/f'{item}_eval_only.npz') as z:
            assert np.array_equal(z['pixel_xy'],scene['old']['pixel_xy'])
            assert np.array_equal(z['view_indices'],scene['old']['source_views'])
            truth,distance=z['raw_dense_gt'],z['boundary_distance']
        before=convert(scene['old']['point_plane_ids'],mapping);after=convert(prediction,mapping)
        measured=metrics(before,after,truth,distance)
        measured['changed']=int((prediction!=scene['old']['point_plane_ids']).sum())
        assert measured['before_errors']==row['variants']['before']['total_errors']
        assert measured['boundary_before']==row['variants']['before']['boundary_errors']
        frozen=Path('local_outputs/intersection_frozen_dev2_v1')
        with np.load(frozen/f'{item}_predictions.npz') as z:
            assert np.array_equal(z['original_ids'],scene['old']['point_plane_ids'])
            historical=z['finite_support']
        reports[item]=dict(metrics=measured,source_sha256=scene['source_sha256'],
                           fixed_identity_map=mapping,unrepresented_identity_points=row['unrepresented_identity_points'],
                           historical_finite=read_json(frozen/'report.json')[item]['metrics']['finite_support'])
        comparison(scene,prediction,historical,truth,mapping,out/'figures'/f'{item}_view0.png')
        write_json(out/'report.json',reports);print('EVAL',item,measured,flush=True)
    aggregate={k:sum(r['metrics'][k] for r in reports.values()) for k in next(iter(reports.values()))['metrics']}
    write_json(out/'completion.json',dict(status='complete',aggregate=aggregate,elapsed_seconds=time.perf_counter()-started))


if __name__=='__main__':main()
