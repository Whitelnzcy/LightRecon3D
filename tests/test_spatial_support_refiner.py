import unittest
import numpy as np
import torch
from models.spatial_support_refiner import SpatialSupportRefiner,ownership_loss
from spatial_support_data import tile_boxes,patch_batch


class SpatialRefinerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2)

    def test_candidate_permutation_and_initial_identity(self):
        net=SpatialSupportRefiner().eval();rgb=torch.randn(2,4,16,16)
        masks=torch.zeros(2,3,2,16,16);masks[:,1,0]=1
        logits=net(rgb,masks)
        self.assertTrue((logits.argmax(1)==1).all())
        with torch.no_grad():net.score.weight.normal_(0,.01)
        torch.testing.assert_close(net(rgb,masks[:,[2,0,1]]),net(rgb,masks)[:,[2,0,1]])

    def test_loss_ignores_missing_and_backpropagates(self):
        net=SpatialSupportRefiner();rgb=torch.randn(1,4,16,16)
        masks=torch.rand(1,2,2,16,16);target=torch.ones(1,16,16,dtype=torch.long);target[:,:3]=-100
        original=torch.zeros_like(target);opt=torch.optim.SGD(net.parameters(),lr=.01)
        for _ in range(2):
            opt.zero_grad();loss=ownership_loss(net(rgb,masks),target,original);loss.backward();opt.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(net.image[0].weight.grad.abs().sum()),0)
        empty=ownership_loss(net(rgb,masks),torch.full_like(target,-100),original)
        self.assertEqual(float(empty.detach()),0)

    def test_loss_invariant_to_ignored_pixel_logits(self):
        logits=torch.randn(1,2,8,8);target=torch.zeros(1,8,8,dtype=torch.long);target[:,:,0]=-100
        original=torch.zeros_like(target);changed=logits.clone();changed[:,:,:,0]+=torch.randn(1,2,8)*100
        torch.testing.assert_close(ownership_loss(logits,target,original),ownership_loss(changed,target,original))

    def test_tiles_cover_full_domain(self):
        covered=np.zeros((288,512),bool)
        for x0,y0,x1,y1 in tile_boxes(covered.shape): covered[y0:y1,x0:x1]=True
        self.assertTrue(covered.all())
        with self.assertRaises(ValueError):tile_boxes((64,64),stride=65)

    def test_inference_batch_has_no_supervision_dependency(self):
        v=dict(common=np.zeros((4,64,64),np.float32),candidates=np.zeros((2,2,64,64),np.float32))
        batch=patch_batch(v,[[0,0,64,64]],'cpu',with_targets=False)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[1].shape,(1,2,2,64,64))


if __name__=='__main__':unittest.main()
