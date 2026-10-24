import unittest
import torch
from models.spatial_direction_refiner import SpatialDirectionRefiner
from spatial_direction_data import direction_loss


class DirectionRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.c=torch.randn(1,4,8,8);self.c[:,3]=1
        self.k=torch.randn(1,3,2,8,8);self.e=torch.randn(1,3,6,8,8)
        self.e[:, :,0]=1;self.e[:,:,1:4]=self.e[:,:1,1:4];self.e[:,:,4]=.5
    def test_candidate_permutation(self):
        m=SpatialDirectionRefiner('DC');torch.nn.init.normal_(m.score.weight)
        order=[2,0,1]
        torch.testing.assert_close(m(self.c,self.k,self.e)[:,order],m(self.c,self.k[:,order],self.e[:,order]))
    def test_shared_direction(self):
        m=SpatialDirectionRefiner('DS');_,d=m(self.c,self.k,self.e,return_details=True)
        torch.testing.assert_close(d[:,0],d[:,1])
    def test_xyz_reaches_direction_head(self):
        m=SpatialDirectionRefiner('DS');_,a=m(self.c,self.k,self.e,return_details=True)
        e=self.e.clone();e[:,:,1:4]*=-1
        _,b=m(self.c,self.k,e,return_details=True)
        self.assertGreater(float((a-b).detach().abs().max()),1e-5)
    def test_gradient_and_ignore(self):
        m=SpatialDirectionRefiner('DC');y,d=m(self.c,self.k,self.e,return_details=True)
        target=torch.zeros(1,8,8,dtype=torch.long)
        loss=direction_loss(d,torch.ones(1,3,8,8),target);loss.backward()
        self.assertGreater(float(m.line_direction[1].weight.grad.norm()),0)
        self.assertEqual(float(direction_loss(d,torch.ones(1,3,8,8),target-100)),0)


if __name__=='__main__':unittest.main()
