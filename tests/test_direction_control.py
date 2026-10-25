import unittest
import torch
from models.spatial_direction_refiner import SpatialDirectionRefiner,neighbor,direction_agreement_weight
from train_direction_control import NoPropagation


class ControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);torch.set_num_threads(1)
        self.m=SpatialDirectionRefiner('DS').eval()
        torch.nn.init.normal_(self.m.score.weight)
        self.c=torch.randn(1,4,8,8);self.k=torch.randn(1,2,2,8,8)
        self.e=torch.randn(1,2,6,8,8);self.e[:,:,0]=1;self.e[:,:,4]=.5

    def test_override_matches_hook(self):
        t=torch.randn(1,2,3,8,8)
        explicit=self.m(self.c,self.k,self.e,direction_override=t)
        hook=self.m.line_direction.register_forward_hook(lambda m,i,o:t.reshape(2,3,8,8))
        try:hooked=self.m(self.c,self.k,self.e)
        finally:hook.remove()
        torch.testing.assert_close(explicit,hooked)

    def test_none_fuses_identical_features(self):
        captured=[]
        hook=self.m.line_fuse.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach()))
        try:self.m(self.c,self.k,self.e,propagation='none')
        finally:hook.remove()
        torch.testing.assert_close(captured[0][:,:24],captured[0][:,24:])
        other=NoPropagation('DS').eval();other.load_state_dict(self.m.state_dict())
        torch.testing.assert_close(other(self.c,self.k,self.e),self.m(self.c,self.k,self.e,propagation='none'))

    def test_override_shape_checked(self):
        with self.assertRaises(ValueError):self.m(self.c,self.k,self.e,direction_override=torch.zeros(1,3,8,8))

    def test_dilated_neighbor_no_wrap(self):
        x=torch.arange(25).reshape(1,1,5,5).float()
        shifted=neighbor(x,0,3)
        torch.testing.assert_close(shifted[...,:2],x[...,3:])
        self.assertEqual(float(shifted[...,2:].abs().sum()),0)
        self.assertEqual(float(neighbor(x,8,0).abs().sum()),0)

    def test_reach_shape_finite_and_candidate_order(self):
        m=SpatialDirectionRefiner('DS',grid_step=4).eval();m.load_state_dict(self.m.state_dict())
        result=m(self.c,self.k,self.e)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[:,[1,0]],m(self.c,self.k[:,[1,0]],self.e[:,[1,0]]))

    def test_support_bias_is_only_additive(self):
        old=self.m(self.c,self.k,self.e)
        self.m.support_bias=0.
        torch.testing.assert_close(old-self.m(self.c,self.k,self.e),2*self.k[:,:,0])

    def test_direction_agreement_geometry(self):
        x=torch.tensor([1.,0.,0.]).reshape(1,1,3,1,1)
        y=torch.tensor([0.,1.,0.]).reshape(1,1,3,1,1)
        self.assertEqual(float(direction_agreement_weight(x,x)),1.)
        self.assertEqual(float(direction_agreement_weight(x,-x)),0.)
        self.assertEqual(float(direction_agreement_weight(x,y)),.5)
        self.assertEqual(float(direction_agreement_weight(x,torch.zeros_like(x))),1.)

    def test_constant_direction_agreement_matches_original(self):
        direction=torch.zeros(1,2,3,8,8);direction[:,:,0]=1
        a=self.m(self.c,self.k,self.e,direction_override=direction)
        self.m.direction_agreement=True
        b=self.m(self.c,self.k,self.e,direction_override=direction)
        torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
