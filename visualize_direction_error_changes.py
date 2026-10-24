"""Fixed full-view GT-assisted error visualization; no point displacement."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_line_support_refinement import read_json,convert
from models.line_support_patches import rgb_in_pointmap,point_index_grid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--baseline',required=True);ap.add_argument('--method',required=True);args=ap.parse_args()
    run=Path(args.run);gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    target=run/f'error_changes_{args.baseline}_vs_{args.method}';target.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for name in read_json(run/'interventions.json'):
        receipt=read_json(run/f'{name}_inputs.json');t=receipt['views'][0]['transform'];mapping=read_json(gtroot/'report.json')[name]['fixed_identity_map']
        with np.load(receipt['source']) as z:
            indices=np.flatnonzero(z['source_views']==t['alignment_view_index'])
            registry=point_index_grid(z['pixel_xy'][indices],indices,t['pointmap_hw'])
        with np.load(gtroot/f'{name}_eval_only.npz') as z:truth=z['raw_dense_gt']
        with np.load(run/f'{name}_GT_DIAGNOSTIC.npz') as z:before=convert(z[args.baseline],mapping);after=convert(z[args.method],mapping)
        observed=registry>=0;valid=observed.copy();valid[observed]=truth[registry[observed]]>=0
        b=np.zeros_like(valid);a=b.copy();b[valid]=before[registry[valid]]==truth[registry[valid]];a[valid]=after[registry[valid]]==truth[registry[valid]]
        rgb=rgb_in_pointmap(np.asarray(Image.open(t['image_path']).convert('RGB')),t);h,w=registry.shape
        panels=[('Input RGB',rgb)]
        for title,correct in [(args.baseline,b),(args.method,a)]:
            image=np.full((h,w,3),255,np.uint8);image[valid]=[205,210,215];image[valid&~correct]=[215,60,55];panels.append((title+' errors',image))
        changes=np.full((h,w,3),255,np.uint8);changes[valid]=[225,228,232]
        changes[valid&~b&a]=[0,160,95];changes[valid&b&~a]=[190,30,160]
        panels.append(('Green: corrected; magenta: damaged',changes))
        sheet=Image.new('RGB',(4*w,h+110),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),name+' | fixed full view0, same point IDs; GT used for evaluation only',font=font,fill='black')
        for i,(title,image) in enumerate(panels):draw.text((i*w+6,30),title,font=font,fill='black');sheet.paste(Image.fromarray(image),(i*w,57))
        draw.text((8,h+70),f'View0 only: corrected {int((valid&~b&a).sum())}, damaged {int((valid&b&~a).sum())}. Red = wrong ownership; gray = valid unchanged/correct.',font=font,fill='black')
        sheet.save(target/f'{name}.png')


if __name__=='__main__':main()
