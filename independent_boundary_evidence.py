"""Frozen RGB-only prior and RAW plane intersections; no GT inputs."""
from itertools import combinations
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from train_rgb_structure_boundary import BoundaryNet
from recover_cache_projection import intersect_project
from models.line_geometry import clip_line
from spatial_line_features import pair_features
from audit_line_support_refinement import sha, write_json

CHECKPOINT = Path('local_outputs/rgb_structure_boundary_20260908_full3997_e50_v1/best.pt')


def align_probability(prob, t):
    if prob.shape != (256,256): raise ValueError('Expected full-image256 probability')
    h,w=t['pointmap_hw']; rh,rw=t['resized_hw']; x0,y0,_,_=t['crop_xyxy']
    yy,xx=np.indices((h,w))
    return cv2.remap(prob,((xx+x0+.5)*256/rw-.5).astype('float32'),
        ((yy+y0+.5)*256/rh-.5).astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


class IndependentEvidence:
    def __init__(self, output):
        self.output=Path(output); self.net=BoundaryNet().cuda().eval()
        self.net.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model'])
        self.checkpoint_hash=sha(CHECKPOINT)

    @torch.inference_mode()
    def attach(self, scene):
        old=scene['old']; records=[]; arrays={}
        for v in scene['views']:
            t=v['transform']; vi=int(t['alignment_view_index'])
            rgb=cv2.cvtColor(cv2.imread(t['image_path']),cv2.COLOR_BGR2RGB)
            assert list(rgb.shape[:2])==t['original_hw']
            image=torch.from_numpy(cv2.resize(rgb,(512,512))).permute(2,0,1).float()[None]/255
            prob=self.net(F.interpolate(image,size=(256,256),mode='bilinear',align_corners=False).cuda()).sigmoid()[0,0].cpu().numpy()
            aligned=align_probability(prob,t)
            labels=np.full(v['registry'].shape,-1,int); ok=v['registry']>=0
            labels[ok]=old['point_plane_ids'][v['registry'][ok]]; rows=[]
            for ai,bi in combinations(range(len(old['plane_ids'])),2):
                a,b=int(old['plane_ids'][ai]),int(old['plane_ids'][bi])
                if not np.any(labels==a) or not np.any(labels==b): continue
                line=intersect_project(old['plane_normals'][ai],old['plane_offsets'][ai],old['plane_normals'][bi],old['plane_offsets'][bi],v['projection'])
                if line is None or clip_line(line,*labels.shape) is None: continue
                # Helper's legacy 'after' key receives RAW line; no refinement/score.
                rows.append(dict(pair=[a,b],after=line.tolist()))
            evidence=pair_features(labels,old['plane_ids'],rows); evidence[:,5]=aligned[None]
            v['lines']=evidence; arrays[f'view{vi}_boundary']=aligned; arrays[f'view{vi}_geometry']=evidence[:,:5]
            records.append(dict(view=vi,image=t['image_path'],image_sha256=sha(t['image_path']),transform=t,
                raw_intersections=[dict(pair=r['pair'],coefficients=r['after']) for r in rows]))
        cache=self.output/f'{scene["name"]}_independent_evidence.npz'
        np.savez_compressed(cache,schema_version=1,**arrays)
        receipt=self.output/f'{scene["name"]}_evidence_sources.json'
        write_json(receipt,dict(schema_version=1,checkpoint=str(CHECKPOINT),checkpoint_sha256=self.checkpoint_hash,
            training_protocol_sha256=sha(CHECKPOINT.parent/'protocol.json'),cache=str(cache),cache_sha256=sha(cache),
            geometry='RAW predicted intersections, no RGB search or GT',
            boundary='RGB512->bilinear256->explicit halfpixel resized/cropped pointmap mapping',views=records))
        return receipt
