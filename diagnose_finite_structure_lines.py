"""Prediction-only 3D finite-line diagnostic; never changes plane ownership.

Fixed diagnostic thresholds, not a trained contact/visibility model. No GT read.
"""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_line_support_refinement import read_json,write_json,sha


def intersect3d(n1,d1,n2,d2):
    n1=np.asarray(n1,float);n2=np.asarray(n2,float)
    cosine=abs(n1@n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
    if np.degrees(np.arccos(np.clip(cosine,0,1)))<10:return None
    v=np.cross(n1,n2);v/=np.linalg.norm(v)
    p=np.linalg.solve(np.array([n1,n2,v]),[-d1,-d2,0.])
    return p,v


def runs(mask):
    padded=np.r_[False,mask,False].astype(int);starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
    return [(int(a),int(b)) for a,b in zip(starts,ends) if b-a>=2]


def sample_evidence(xyz,projection,prob,view_xyz,view_xy,bandwidth):
    """Observed-point proximity proxy, NOT physical occlusion/ray-depth test."""
    q=np.c_[xyz,np.ones(len(xyz))]@projection.T
    valid=np.abs(q[:,2])>1e-9;uv=np.zeros((len(xyz),2));uv[valid]=q[valid,:2]/q[valid,2:3]
    h,w=prob.shape;valid &= (uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    distance,idx=cKDTree(view_xy).query(uv)
    valid &= (distance<=1.5)&(np.linalg.norm(view_xyz[idx]-xyz,axis=1)<=bandwidth)
    values=cv2.remap(prob,uv[:,0].astype('float32')[None],uv[:,1].astype('float32')[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
    return values,valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema_version=1,samples_per_pair=256,support_bandwidth_fraction=.02,boundary_threshold=.5,
        interval='overlap of each predicted support along-line1%-99% quantiles',
        support='both plane point sets within2% robust scene diagonal',
        rgb='at least one view with probability>=.5 and nearest observed pixel<=1.5px and XYZ distance<=bandwidth',
        caveat='diagnostic heuristic only; no calibrated physical visibility, no GT, no label/XYZ edits, thresholds not selected on outcome')
    write_json(out/'protocol.json',protocol);report={}
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        receipt=read_json(run/f'{name}_prediction_inputs.json');ev=read_json(receipt['lines_path'])
        assert sha(receipt['source'])==receipt['source_sha256'];assert sha(ev['cache'])==ev['cache_sha256']
        with np.load(receipt['source']) as z:
            old={k:z[k] for k in ['points','point_plane_ids','plane_ids','plane_normals','plane_offsets','pixel_xy','source_views']}
        points=old['points'];labels=old['point_plane_ids'];ids=old['plane_ids']
        lo,hi=np.quantile(points,[.01,.99],axis=0);scale=float(np.linalg.norm(hi-lo));band=.02*scale
        supports={int(pid):points[labels==pid] for pid in ids}
        trees={pid:cKDTree(x) for pid,x in supports.items() if len(x)}
        views=[]
        with np.load(ev['cache']) as z:
            for view in receipt['views']:
                vi=view['transform']['alignment_view_index'];ix=old['source_views']==vi
                views.append((np.asarray(view['projection_from_predicted_xyz']),z[f'view{vi}_boundary'],points[ix],old['pixel_xy'][ix]))
        records=[];arrays={};all_lines=[]
        for ai,bi in combinations(range(len(ids)),2):
            a,b=int(ids[ai]),int(ids[bi]);line=intersect3d(old['plane_normals'][ai],old['plane_offsets'][ai],old['plane_normals'][bi],old['plane_offsets'][bi])
            if line is None or a not in trees or b not in trees:continue
            p,v=line;bounds=[np.quantile((supports[i]-p)@v,[.01,.99]) for i in [a,b]]
            left=max(x[0] for x in bounds);right=min(x[1] for x in bounds)
            if right<=left:
                records.append(dict(pair=[a,b],status='no_along_support_overlap'));continue
            t=np.linspace(left,right,256);xyz=p+t[:,None]*v
            da,ia=trees[a].query(xyz);db,ib=trees[b].query(xyz);near=(da<=band)&(db<=band)
            closest=int(np.argmin(np.maximum(da,db)))
            observed=np.zeros(256,int);votes=np.zeros(256,int);scores=np.zeros(256)
            for proj,prob,vxyz,vxy in views:
                score,valid=sample_evidence(xyz,proj,prob,vxyz,vxy,band)
                observed+=valid;votes+=valid&(score>=.5);scores=np.maximum(scores,np.where(valid,score,0))
            selected=near&(votes>=1);key=f'pair_{a}_{b}'
            arrays[key+'_xyz']=xyz;arrays[key+'_near_support']=near;arrays[key+'_rgb_supported']=selected
            segments=[dict(t_min=float(t[i]),t_max=float(t[j-1]),endpoints=[xyz[i].tolist(),xyz[j-1].tolist()]) for i,j in runs(selected)]
            records.append(dict(pair=[a,b],status='sampled',point=p.tolist(),direction=v.tolist(),
                support_fraction=float(near.mean()),observed_fraction=float((observed>0).mean()),
                minimum_joint_support_distance=float(max(da[closest],db[closest])),
                minimum_joint_support_distance_over_band=float(max(da[closest],db[closest])/band),
                closest_sample=xyz[closest].tolist(),nearest_support_a=supports[a][ia[closest]].tolist(),nearest_support_b=supports[b][ib[closest]].tolist(),
                rgb_supported_fraction=float(selected.mean()),multi_view_supported_samples=int((near&(votes>=2)).sum()),
                segments=segments,max_observed_probability=float(scores.max())))
            all_lines.append((xyz,near,selected))
        np.savez_compressed(out/f'{name}_segments.npz',schema_version=1,**arrays)
        fig=plt.figure(figsize=(15,5));axes=[fig.add_subplot(1,3,j+1,projection='3d') for j in range(3)]
        ix=np.arange(0,len(points),max(1,len(points)//12000));color=plt.get_cmap('tab10')
        colors=np.array([color(int(pid)%10) if pid>=0 else (.6,.6,.6,1) for pid in labels[ix]])
        center=(lo+hi)/2;radius=max(hi-lo)/2
        for j,ax in enumerate(axes):
            ax.scatter(*points[ix].T,c=colors,s=.35,alpha=.22,rasterized=True)
            for xyz,near,selected in all_lines:
                mask=np.ones(len(xyz),bool) if j==0 else near if j==1 else selected
                for a,b in runs(mask):ax.plot(*xyz[a:b].T,color='#d62728',linewidth=2)
            ax.view_init(elev=22,azim=-65);ax.set_box_aspect((1,1,1));ax.set_axis_off()
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_title(['3D intersections: overlap interval','Near BOTH plane supports','+ frozen RGB boundary evidence'][j],fontsize=11)
        fig.suptitle(name+' | fixed predicted XYZ, same camera/colors; red = 3D line samples',fontsize=12)
        fig.text(.02,.025,'Diagnostic thresholds only; no ownership edits. Point colors = predicted planes. RGB agreement is not GT correctness.',fontsize=10)
        fig.subplots_adjust(left=0,right=1,bottom=.07,top=.87,wspace=0)
        fig.savefig(out/f'{name}_3d.png',dpi=150);plt.close(fig)
        report[name]=dict(source=receipt['source'],source_sha256=receipt['source_sha256'],bandwidth=band,
            coordinate='same global XYZ as cached predicted pointmaps; scale not asserted metric',
            candidate_pairs=len(records),sampled_pairs=len(all_lines),
            support_pairs=sum(bool(r.get('support_fraction',0)>0) for r in records),
            rgb_pairs=sum(bool(r.get('segments')) for r in records),records=records)
        write_json(out/'report.json',report)
        print(name,{k:v for k,v in report[name].items() if k!='records'},flush=True)


if __name__=='__main__':main()
