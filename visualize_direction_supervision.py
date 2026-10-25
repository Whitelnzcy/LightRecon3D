"""Actual registered TRAIN0 boundaries: supervision diagnostic, not result gain."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from threadpoolctl import threadpool_limits
from spatial_support_data import load_prediction_scene,attach_supervision
from spatial_direction_data import attach_direction_supervision
from models.line_support_patches import valid_transitions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);threadpool_limits(1)
    s=load_prediction_scene('local_outputs/gct3/items/train_000_scene_00000')
    attach_supervision(s);attach_direction_supervision(s,'inward')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);panels=[]
    for v in s['views'][:2]:
        r=v['registry'];ok=r>=0;labels=np.full(r.shape,-1,int);labels[ok]=s['truth'][r[ok]]
        grid=valid_transitions(labels);knn=v['direction_boundary'];rgb=np.clip((v['common'][:3].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)
        views=[rgb.copy() for _ in range(4)]
        views[1][grid]=[0,180,80];views[2][knn]=[220,0,180]
        views[3][grid&~knn]=[0,180,80];views[3][knn&~grid]=[220,0,180];views[3][grid&knn]=[0,160,255]
        h,w=r.shape;panel=Image.new('RGB',(4*w,h+90),'white');draw=ImageDraw.Draw(panel)
        draw.text((8,4),f'TRAIN scene00000 view{v["transform"]["alignment_view_index"]} | GT supervision diagnostic, NOT predictions',font=font,fill='black')
        for k,(array,title) in enumerate(zip(views,['RGB','2D label transitions (green)','3D kNN label transitions (magenta)','Overlap blue; exact pixels only'])):
            draw.text((k*w+8,32),title,font=font,fill='black');panel.paste(Image.fromarray(array),(k*w,60))
        draw.text((8,h+64),'Same registered pixels and fixed predicted XYZ. 3D kNN mixes all views; 2D uses each image grid.',font=font,fill='black');panels.append(panel)
    sheet=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)),'white');y=0
    for p in panels:sheet.paste(p,(0,y));y+=p.height
    sheet.save(out/'train0_boundary_targets.png')


if __name__=='__main__':main()
