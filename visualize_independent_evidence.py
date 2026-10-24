"""Show independent RGB prior versus raw geometry, no GT-selected crop."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_line_support_refinement import read_json
from models.line_support_patches import rgb_in_pointmap
from models.line_geometry import clip_line


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    run=Path(args.run);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);panels=[]
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        r=read_json(run/f'{name}_evidence_sources.json');v=r['views'][0];t=v['transform']
        rgb=rgb_in_pointmap(np.asarray(Image.open(t['image_path']).convert('RGB')),t)
        h,w=rgb.shape[:2]
        with np.load(r['cache']) as z:p=z[f'view{v["view"]}_boundary']
        tint=np.zeros_like(rgb);tint[:,:,0]=255;tint[:,:,1]=80
        alpha=(p*.8)[:,:,None]
        boundary=Image.fromarray((rgb*(1-alpha)+tint*alpha).astype('uint8'))
        raw=Image.fromarray(rgb.copy());d=ImageDraw.Draw(raw)
        for line in v['raw_intersections']:
            ends=clip_line(np.asarray(line['coefficients']),h,w)
            if ends is not None:d.line([tuple(ends[0]),tuple(ends[1])],fill=(0,210,255),width=2)
        row=Image.new('RGB',(3*w,h+82),'white');d=ImageDraw.Draw(row)
        d.text((8,3),name+' | first view, no GT selection',font=font,fill='black')
        for j,(title,img) in enumerate([('RGB',Image.fromarray(rgb)),('B: frozen RGB boundary probability',boundary),('G: RAW infinite plane intersections',raw)]):
            row.paste(img,(j*w,56));d.text((j*w+5,30),title,font=font,fill='black')
        panels.append(row)
    sheet=Image.new('RGB',(panels[0].width,sum(x.height for x in panels)),'white');y=0
    for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(run/'figures'/'independent_inputs.png')


if __name__=='__main__':main()
