import unittest
import torch
from spatial_direction_data import balanced_direction_loss,direction_loss


class BalancedLossTests(unittest.TestCase):
    def test_equal_strata_not_pixel_frequency(self):
        pred=torch.zeros(1,1,3,1,4,requires_grad=True);target=torch.zeros(1,3,1,4);target[:,:,:,0]=2
        y=torch.zeros(1,1,4,dtype=torch.long);boundary=torch.tensor([[[True,False,False,False]]])
        loss=balanced_direction_loss(pred,target,y,boundary)
        self.assertAlmostEqual(float(loss.detach()),2.);loss.backward();self.assertTrue(torch.isfinite(pred.grad).all())
    def test_single_stratum_matches_uniform(self):
        p=torch.randn(1,2,3,3,3);t=torch.randn(1,3,3,3);y=torch.zeros(1,3,3,dtype=torch.long)
        for b in [torch.zeros_like(y,dtype=torch.bool),torch.ones_like(y,dtype=torch.bool)]:
            torch.testing.assert_close(balanced_direction_loss(p,t,y,b),direction_loss(p,t,y))
    def test_ignored_finite_zero(self):
        p=torch.randn(1,2,3,2,2,requires_grad=True);t=torch.zeros(1,3,2,2);y=torch.full((1,2,2),-100)
        loss=balanced_direction_loss(p,t,y,torch.ones_like(y,dtype=torch.bool));self.assertEqual(float(loss.detach()),0.)
        loss.backward();self.assertEqual(float(p.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
