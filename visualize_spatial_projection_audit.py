"""Visual check of cached global XYZ to explicit cropped-pointmap coordinates.

Reads saved prediction receipts, never camera GT; does not imply depth accuracy.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_line_support_refinement import read_json, sha, write_json
from models.line_support_patches import rgb_in_pointmap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    records=[];panels=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        receipt=run/f'{name}_prediction_inputs.json'
        if not receipt.exists():receipt=run/f'{name}_inputs.json'
        r=read_json(receipt);assert sha(r['source'])==r['source_sha256']
        v=r['views'][0];t=v['transform'];p=np.asarray(v['projection_from_predicted_xyz'])
        with np.load(r['source']) as z:
            ix=np.flatnonzero(z['source_views']==t['alignment_view_index'])
            xyz=z['points'][ix];xy=z['pixel_xy'][ix]
        homogeneous=np.c_[xyz,np.ones(len(xyz))]@p.T
        projected=homogeneous[:,:2]/homogeneous[:,2:3]
        distance=np.linalg.norm(projected-xy,axis=1)
        rgb=rgb_in_pointmap(np.asarray(Image.open(t['image_path']).convert('RGB')),t)
        h,w=rgb.shape[:2];canvas=Image.new('RGB',(2*w,h+100),'white');d=ImageDraw.Draw(canvas)
        d.text((8,4),name+' | fixed global XYZ; no GT camera',font=font,fill='black')
        selected=np.arange(0,len(ix),max(1,len(ix)//450))
        for col,points in enumerate([xy,projected]):
            overlay=Image.fromarray(rgb);q=ImageDraw.Draw(overlay)
            for x,y in points[selected]:q.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=(255,70,30))
            canvas.paste(overlay,(col*w,55))
            d.text((col*w+8,30),['Stored integer-center pixels','Projected global XYZ (same indices)'][col],font=font,fill='black')
        quality=v['projection_quality']
        d.text((8,h+65),f'Disjoint DLT check P95 {quality["p95"]:.4f}px. Coordinate consistency, NOT depth accuracy.',font=font,fill='black')
        panels.append(canvas);records.append(dict(scene=name,view=t['alignment_view_index'],points=len(ix),
            disjoint_projection_quality=quality,all_point_p95=float(np.percentile(distance,95)),source_sha256=r['source_sha256']))
    sheet=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'white');y=0
    for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(out/'global_xyz_projection.png');write_json(out/'audit.json',dict(schema_version=1,views=records))


if __name__=='__main__':main()
