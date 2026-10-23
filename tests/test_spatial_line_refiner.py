import unittest
import numpy as np
import torch
from models.spatial_line_refiner import SpatialLineRefiner
from models.spatial_support_refiner import ownership_loss
from spatial_line_features import pair_features


class SpatialLineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(42)

    def test_feature_sign_and_candidate_permutation(self):
        labels=np.zeros((16,16),int);labels[:,8:]=1
        row=dict(pair=[0,1],after=[1,0,-8],score_after=.7)
        f=pair_features(labels,[0,1],[row]);row['after']=[-2,0,16]
        np.testing.assert_allclose(f,pair_features(labels,[0,1],[row]))
        np.testing.assert_allclose(f[::-1],pair_features(labels,[1,0],[row]))
        self.assertTrue((f[1,0,:,:8]==1).all());self.assertTrue((f[0,0,:,:8]==0).all())
        self.assertFalse(pair_features(labels,[0,1],[]).any())

    def test_m1_ignores_line_inputs_and_gate_controls_scores(self):
        x=torch.randn(1,4,16,16);c=torch.rand(1,2,2,16,16);l=torch.rand(1,2,6,16,16);l[:,:,0]=1
        m=SpatialLineRefiner('M1').eval()
        torch.testing.assert_close(m(x,c,l),m(x,c,l*0))
        m.arm='M3'
        with torch.no_grad():m.line_score.bias.fill_(1)
        a=m(x,c,l)
        with torch.no_grad():m.line_gate.bias.fill_(-10)
        b=m(x,c,l)
        self.assertGreater(float((a-b).abs().mean().detach()),.4)

    def test_gate_learns_and_candidate_equivariance(self):
        m=SpatialLineRefiner('M3');x=torch.randn(1,4,16,16);c=torch.rand(1,2,2,16,16)
        l=torch.rand(1,2,6,16,16);l[:,:,0]=1;y=torch.zeros(1,16,16,dtype=torch.long)
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad();ownership_loss(m(x,c,l),y,y).backward();opt.step()
        self.assertGreater(float(m.line_gate.weight.grad.abs().sum()),0)
        m.eval();torch.testing.assert_close(m(x,c[:,[1,0]],l[:,[1,0]]),m(x,c,l)[:,[1,0]])


if __name__=='__main__':unittest.main()
