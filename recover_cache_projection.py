"""Recover camera projection from registered predicted XYZ/pixels, never GT."""
import numpy as np


def fit_projection(xyz,xy):
    xyz=np.asarray(xyz,float);xy=np.asarray(xy,float)
    center=xyz.mean(0);scale=np.sqrt(3)/max(np.linalg.norm(xyz-center,axis=1).mean(),1e-12)
    T=np.eye(4);T[:3,:3]*=scale;T[:3,3]=-scale*center
    c=xy.mean(0);s=np.sqrt(2)/max(np.linalg.norm(xy-c,axis=1).mean(),1e-12)
    U=np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1.]])
    X=np.c_[xyz,np.ones(len(xyz))]@T.T;uv=(xy-c)*s
    if np.linalg.matrix_rank(X)<4:raise ValueError('Degenerate coplanar fit points')
    A=np.zeros((len(xyz)*2,12));A[::2,:4]=X;A[::2,8:]=-uv[:,0,None]*X
    A[1::2,4:8]=X;A[1::2,8:]=-uv[:,1,None]*X
    _,sing,V=np.linalg.svd(A,full_matrices=False)
    P=np.linalg.inv(U)@V[-1].reshape(3,4)@T
    return P/np.linalg.norm(P),float(sing[-2]/max(sing[-1],1e-12))


def recover_checked(xyz,xy):
    ids=np.arange(len(xyz));fit=ids[::max(1,len(ids)//2048)][:2048]
    check=np.setdiff1d(ids[::max(1,len(ids)//4096)],fit)
    if len(check)<100:raise ValueError('Insufficient disjoint camera validation samples')
    P,ratio=fit_projection(xyz[fit],xy[fit]);q=np.c_[xyz[check],np.ones(len(check))]@P.T
    uv=q[:,:2]/q[:,2,None];err=np.linalg.norm(uv-xy[check],axis=1)
    report=dict(fit_points=len(fit),check_points=len(check),median=float(np.median(err)),p95=float(np.percentile(err,95)),singular_ratio=ratio)
    if not np.isfinite(err).all() or report['p95']>2:raise ValueError(f'Projection rejected: {report}')
    return P,report


def intersect_project(n1,d1,n2,d2,P):
    direction=np.cross(n1,n2);norm=np.linalg.norm(direction)
    angle=np.degrees(np.arccos(np.clip(abs(np.dot(n1,n2))/(np.linalg.norm(n1)*np.linalg.norm(n2)),0,1)))
    if angle<10:return None
    direction=direction/norm
    x=np.linalg.solve(np.array([n1,n2,direction]),[-d1,-d2,0])
    a=P@np.r_[x,1];b=P@np.r_[x+direction,1];l=np.cross(a,b)
    return None if np.linalg.norm(l[:2])<1e-10 else l/np.linalg.norm(l[:2])
