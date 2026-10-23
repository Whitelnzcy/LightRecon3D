import numpy as np
import unittest
from recover_cache_projection import recover_checked, fit_projection, intersect_project


def test_disjoint_projection_recovery():
    xyz=np.random.default_rng(3).normal(size=(10000,3));xyz[:,2]+=7
    P=np.array([[500,0,256,0],[0,500,144,0],[0,0,1,0.]])
    q=np.c_[xyz,np.ones(len(xyz))]@P.T;xy=q[:,:2]/q[:,2,None]
    fitted,report=recover_checked(xyz,xy)
    assert report['p95']<1e-8
    l=intersect_project(np.array([1.,0,0]),-1,np.array([0.,1,0]),-1,fitted)
    points=np.array([[1,1,5,1],[1,1,9,1]])@P.T
    assert np.max(abs((points/points[:,2,None])@l))<1e-8


def test_coplanar_rejected():
    xyz=np.random.default_rng(2).normal(size=(100,3));xyz[:,2]=2
    with unittest.TestCase().assertRaisesRegex(ValueError,'coplanar'):fit_projection(xyz,xyz[:,:2])


if __name__=='__main__':
    test_disjoint_projection_recovery()
    test_coplanar_rejected()
    print('2 projection checks passed')
